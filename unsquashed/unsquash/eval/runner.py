"""Attribution A/B evaluation: one forward pass per case, every method scored
against HotpotQA supporting facts.

Teacher-forced: the gold answer is fed as the assistant turn, so attribution
is computed for the tokens of the *correct* answer and no output verification
is needed. (This matches the repo's "don't generate when evaluating" setup.)

Results are written incrementally to ``results.jsonl`` (one record per case)
and aggregated into ``summary.json``. Runs resume from where they stopped.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import torch
from tqdm import tqdm

from unsquash.eval.hotpot import EvalCase
from unsquash.eval.metrics import MeanMetrics
from unsquash.prior import PriorConfig
from unsquash.rollout import METHODS, RolloutAttributor
from unsquash.spans import prepare_case, score_spans

logger = logging.getLogger(__name__)


@dataclass
class EvalSettings:
    methods: Sequence[str] = ("attention_sum", "rollout", "unsquashed")
    ks: Sequence[int] = (1, 2, 5, 10)
    reduction: str = "mean"
    max_context_tokens: int = 2000
    max_cases: int | None = None
    residual: bool | None = None  # None -> per-method default
    out_dir: str = "unsquash_eval_results"
    extra: dict = field(default_factory=dict)  # recorded into summary.json


def _results_path(out_dir: str) -> str:
    return os.path.join(out_dir, "results.jsonl")


def _summary_path(out_dir: str) -> str:
    return os.path.join(out_dir, "summary.json")


def _completed_cases(out_dir: str) -> int:
    path = _results_path(out_dir)
    if not os.path.exists(path):
        return 0
    with open(path) as fd:
        return sum(1 for line in fd if line.strip())


def evaluate(
    attributor: RolloutAttributor,
    cases: Iterable[EvalCase],
    settings: EvalSettings,
) -> dict:
    """Run the evaluation; returns the summary dict (also written to disk)."""
    for m in settings.methods:
        if m not in METHODS:
            raise ValueError(f"Unknown method {m!r}; expected one of {METHODS}")

    os.makedirs(settings.out_dir, exist_ok=True)
    skip = _completed_cases(settings.out_dir)
    if skip:
        logger.info("Resuming: %d cases already evaluated.", skip)

    metrics = {m: MeanMetrics(settings.ks) for m in settings.methods}
    evaluated = considered = 0

    with open(_results_path(settings.out_dir), "a") as results_fd:
        progress = tqdm(cases, desc="evaluating", unit="case")
        for case in progress:
            considered += 1
            if considered <= skip:
                continue
            if settings.max_cases is not None and evaluated >= settings.max_cases:
                break

            record: dict = {"question": case.question, "answer": case.answer}
            try:
                prepared = prepare_case(
                    attributor.tokenizer,
                    context=case.context,
                    answer=case.answer,
                    sentence_char_spans=case.sentence_char_spans,
                )
                n_tokens = int(prepared.input_ids.shape[0])
                if n_tokens > settings.max_context_tokens:
                    record["skipped"] = f"too long ({n_tokens} tokens)"
                    results_fd.write(json.dumps(record) + "\n")
                    continue

                attentions = attributor.capture(prepared.input_ids)
                record["n_tokens"] = n_tokens
                record["supporting"] = sorted(case.supporting)
                record["methods"] = {}
                for method in settings.methods:
                    Y = attributor.attribute(
                        attentions, method, residual=settings.residual
                    )
                    scores = score_spans(
                        Y,
                        prepared.answer_span,
                        prepared.sentence_spans,
                        reduction=settings.reduction,
                    )
                    ranked = sorted(
                        range(len(scores)), key=scores.__getitem__, reverse=True
                    )
                    case_metrics = metrics[method].update(ranked, case.supporting)
                    record["methods"][method] = {
                        "ranked": ranked[: max(settings.ks)],
                        "metrics": case_metrics,
                    }
                del attentions
                evaluated += 1
            except Exception:
                logger.exception("Failed on case %r", case.question[:80])
                record["error"] = True
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            results_fd.write(json.dumps(record) + "\n")
            results_fd.flush()
            progress.set_postfix(
                {
                    m: f"r@{max(settings.ks)}={metrics[m].means().get(f'recall@{max(settings.ks)}', 0):.2f}"
                    for m in settings.methods
                }
            )

    summary = {
        "evaluated_cases": max(m.n for m in metrics.values()) if metrics else 0,
        "settings": {
            "methods": list(settings.methods),
            "ks": list(settings.ks),
            "reduction": settings.reduction,
            "max_context_tokens": settings.max_context_tokens,
            "residual": settings.residual,
            **settings.extra,
        },
        "metrics": {m: metrics[m].means() for m in settings.methods},
    }
    with open(_summary_path(settings.out_dir), "w") as fd:
        json.dump(summary, fd, indent=2)
    return summary


def print_summary(summary: dict) -> None:
    methods = list(summary["metrics"].keys())
    names = sorted({n for m in methods for n in summary["metrics"][m]})
    width = max((len(n) for n in names), default=10) + 2
    print(f"\nEvaluated cases: {summary['evaluated_cases']}")
    header = " " * width + "".join(f"{m:>16}" for m in methods)
    print(header)
    for name in names:
        row = f"{name:<{width}}"
        for m in methods:
            v = summary["metrics"][m].get(name)
            row += f"{v:>16.4f}" if v is not None else f"{'-':>16}"
        print(row)


def build_attributor(
    *,
    model_name_or_path: str,
    dtype: str = "float32",
    device_map: str | None = None,
    trust_remote_code: bool = False,
    head_weighting: str = "o_proj_norm",
    prior: PriorConfig | None | str = "auto",
) -> RolloutAttributor:
    """Load a model for attribution capture.

    ``prior="auto"`` looks for an ``unsquash_prior.json`` next to a local
    checkpoint (written by ``unsquash.train``) and applies it automatically.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if prior == "auto":
        prior = PriorConfig.load(model_name_or_path)
        if prior is not None:
            logger.info(
                "Applying prior from checkpoint: k=%s lambda=%s", prior.k, prior.lam
            )

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=getattr(torch, dtype),
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        attn_implementation="eager",  # required for output_attentions
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=trust_remote_code
    )
    return RolloutAttributor(
        model, tokenizer, head_weighting=head_weighting, prior=prior
    )
