"""The long-context ladder: one harness, every model, swept over length.

Tiers (cheap and diagnostic first, real-ish last):

- ``ppl``: held-out language-model loss on packed fineweb-edu blocks at each
  length. Necessary-but-weak — flat perplexity past the training length can
  coexist with an inability to *use* far context (ALiBi is the canonical
  example), which is exactly what the retrieval tiers detect.
- ``passkey`` / ``kv`` / ``copy``: teacher-forced retrieval and copying
  probes from :mod:`unsquash.ladder.tasks`, swept over length x depth.

Lengths in ``lengths`` run on every tier; ``extreme_lengths`` (the length-
extension stress test, e.g. 64k-512k) run passkey only, at three depths.
All scoring goes through ``PriorLlama.chunked_hidden`` — bias slices are
built on the fly, never the [n, n] mask — so the same code path covers 2k
and 512k, and full-vocab logits are only ever computed for the rows being
scored.

Rows append to ``results.jsonl`` (resumable: existing (tier, length, depth,
idx) cells are skipped on re-entry, matching the eval harness's contract);
``summary.json`` aggregates exact-match and gold log-probability per cell.

Metrics per case:

- ``exact``: every gold token is the argmax at its position (retrieval hit).
- ``gold_logprob``: mean log-probability of the gold tokens — a graded
  signal that moves before accuracy does.
- ``ppl`` tier instead reports mean cross-entropy over the block.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, asdict, field

import torch

from unsquash.ladder.tasks import BUILDERS, LadderCase

logger = logging.getLogger(__name__)

DEFAULT_LENGTHS = (2048, 4096, 8192, 16384, 32768)
DEFAULT_EXTREME_LENGTHS = (65536, 131072, 262144)
DEFAULT_DEPTHS = (0.0, 0.25, 0.5, 0.75, 1.0)
EXTREME_DEPTHS = (0.1, 0.5, 0.9)


@dataclass
class LadderSettings:
    model: str  # checkpoint dir (HF export + unsquash_prior.json sidecar)
    out_dir: str = "ladder_out"
    tiers: tuple[str, ...] = ("ppl", "passkey", "kv", "copy")
    lengths: tuple[int, ...] = DEFAULT_LENGTHS
    depths: tuple[float, ...] = DEFAULT_DEPTHS
    extreme_lengths: tuple[int, ...] = DEFAULT_EXTREME_LENGTHS
    cases_per_cell: int = 5
    extreme_cases_per_cell: int = 2
    ppl_blocks: int = 4  # held-out blocks per length
    seed: int = 0
    # RoPE overlay: >1 rescales theta at load (NTK-style position
    # interpolation), separating "the bias failed" from "RoPE failed" when
    # extrapolating. 1.0 = the checkpoint's native theta.
    rope_theta_scale: float = 1.0
    device: str | None = None
    # ppl tier data (defaults match pretraining's held-out distribution)
    dataset: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: str | None = "sample-10BT"
    split: str = "train"
    text_file: str | None = None
    extra: dict = field(default_factory=dict)


@torch.no_grad()
def _score_case(model, case: LadderCase, device) -> dict:
    """Teacher-forced gold-token scoring through the chunked path."""
    ids = torch.tensor(case.input_ids, dtype=torch.long, device=device)
    hidden = model.chunked_hidden(ids)
    # Row t predicts token t+1: gold tokens are predicted by rows
    # [gold_start-1, len-2].
    rows = hidden[0, case.gold_start - 1 : -1]
    logits = model.lm_head(rows).float()
    logprobs = torch.log_softmax(logits, dim=-1)
    gold = torch.tensor(case.gold_ids, dtype=torch.long, device=device)
    gold_lp = logprobs.gather(-1, gold[:, None]).squeeze(-1)
    exact = bool((logits.argmax(-1) == gold).all())
    return {
        "tier": case.tier,
        "length": case.length,
        "depth": case.depth,
        "idx": case.idx,
        "actual_tokens": len(case.input_ids),
        "exact": exact,
        "gold_logprob": float(gold_lp.mean()),
        "first_gold_rank": int(
            (logits[0] > logits[0, gold[0]]).sum()
        ),  # 0 = argmax
    }


@torch.no_grad()
def _score_ppl_block(model, block: torch.Tensor, device) -> float:
    """Mean cross-entropy over one ``[1, n]`` block, LM-head applied in row
    chunks so full logits are never materialized."""
    ids = block.to(device)
    hidden = model.chunked_hidden(ids)
    n = ids.shape[1]
    total, count = 0.0, 0
    for i0 in range(0, n - 1, 4096):
        i1 = min(n - 1, i0 + 4096)
        logits = model.lm_head(hidden[0, i0:i1]).float()
        loss = torch.nn.functional.cross_entropy(
            logits, ids[0, i0 + 1 : i1 + 1], reduction="sum"
        )
        total += float(loss)
        count += i1 - i0
    return total / max(1, count)


def _ppl_blocks(settings: LadderSettings, tokenizer, length: int):
    """``ppl_blocks`` held-out ``[1, length]`` blocks; same packing as the
    pretraining data pipeline, fixed seed distinct from training's."""
    from unsquash.train.data import packed_batches

    batches = packed_batches(
        tokenizer,
        seq_len=length,
        batch_size=1,
        dataset=settings.dataset,
        dataset_config=settings.dataset_config,
        split=settings.split,
        text_column="text",
        text_file=settings.text_file,
        seed=settings.seed + 1_000_003,  # never the training stream
    )
    return [next(batches) for _ in range(settings.ppl_blocks)]


def _cells(settings: LadderSettings):
    """Yield (tier, length, depths, cases) for every cell of the sweep."""
    for tier in settings.tiers:
        if tier == "ppl":
            continue
        depths = settings.depths if tier != "copy" else (0.0,)
        for length in settings.lengths:
            yield tier, length, depths, settings.cases_per_cell
    if "passkey" in settings.tiers:
        for length in settings.extreme_lengths:
            yield "passkey", length, EXTREME_DEPTHS, settings.extreme_cases_per_cell


def run_ladder_with(model, tokenizer, settings: LadderSettings) -> dict:
    """Run the ladder against an already-loaded model (the testable core)."""
    device = settings.device or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = model.to(device)

    os.makedirs(settings.out_dir, exist_ok=True)
    with open(os.path.join(settings.out_dir, "ladder_settings.json"), "w") as fd:
        json.dump(asdict(settings), fd, indent=2, default=str)

    results_path = os.path.join(settings.out_dir, "results.jsonl")
    done: set[str] = set()
    if os.path.exists(results_path):
        with open(results_path) as fd:
            for line in fd:
                r = json.loads(line)
                done.add(f"{r['tier']}:{r['length']}:{r['depth']}:{r['idx']}")
        logger.info("Resuming: %d rows already scored", len(done))

    out_fd = open(results_path, "a")

    def emit(row: dict) -> None:
        out_fd.write(json.dumps(row) + "\n")
        out_fd.flush()

    t0 = time.time()
    if "ppl" in settings.tiers:
        for length in settings.lengths:
            key = f"ppl:{length}:0.0:0"
            if key in done:
                continue
            losses = [
                _score_ppl_block(model, block, device)
                for block in _ppl_blocks(settings, tokenizer, length)
            ]
            loss = sum(losses) / len(losses)
            emit({
                "tier": "ppl", "length": length, "depth": 0.0, "idx": 0,
                "loss": loss, "ppl": math.exp(min(loss, 20.0)),
                "blocks": len(losses),
            })
            logger.info("ppl @ %d: loss %.4f", length, loss)

    for tier, length, depths, cases in _cells(settings):
        builder = BUILDERS[tier]
        for depth in depths:
            for idx in range(cases):
                key = f"{tier}:{length}:{depth}:{idx}"
                if key in done:
                    continue
                kwargs = dict(length=length, idx=idx, seed=settings.seed)
                if tier != "copy":
                    kwargs["depth"] = depth
                case = builder(tokenizer, **kwargs)
                row = _score_case(model, case, device)
                emit(row)
                logger.info(
                    "%s @ %d depth %.2f #%d: exact=%s gold_lp=%.3f (%.0fs)",
                    tier, length, depth, idx, row["exact"],
                    row["gold_logprob"], time.time() - t0,
                )
    out_fd.close()

    summary = summarize(results_path, settings)
    with open(os.path.join(settings.out_dir, "summary.json"), "w") as fd:
        json.dump(summary, fd, indent=2)
    return summary


def run_ladder(settings: LadderSettings) -> dict:
    """Load the checkpoint (bias auto-applied from its sidecar) and run."""
    from transformers import AutoTokenizer

    from unsquash.pretrain.model import PriorLlama

    max_len = max(
        (*settings.lengths,
         *(settings.extreme_lengths if "passkey" in settings.tiers else (0,)))
    )
    rope_theta = None
    if settings.rope_theta_scale != 1.0:
        from transformers import AutoConfig

        base = getattr(AutoConfig.from_pretrained(settings.model), "rope_theta", 1e4)
        rope_theta = float(base) * settings.rope_theta_scale
    # +64 slack: builders overshoot the requested length by up to a filler
    # unit plus the gold continuation.
    model = PriorLlama.from_pretrained(
        settings.model, max_seq_len=max_len + 64, rope_theta=rope_theta
    )
    logger.info(
        "Loaded %s: alibi=%s unsquashed_k=%s rope_theta=%s max_seq_len=%d",
        settings.model, model.spec.alibi, model.spec.unsquashed_k,
        model.spec.rope_theta, model.spec.max_seq_len,
    )
    tokenizer = AutoTokenizer.from_pretrained(settings.model)
    return run_ladder_with(model, tokenizer, settings)


def summarize(results_path: str, settings: LadderSettings) -> dict:
    """Aggregate per (tier, length, depth): exact rate + mean gold logprob;
    plus a RULER-style effective context length per retrieval tier (longest
    length whose depth-averaged exact rate is >= 0.5)."""
    rows = []
    with open(results_path) as fd:
        for line in fd:
            rows.append(json.loads(line))

    cells: dict = {}
    for r in rows:
        if r["tier"] == "ppl":
            cells.setdefault("ppl", {})[str(r["length"])] = {
                "loss": r["loss"], "ppl": r["ppl"],
            }
            continue
        tier = cells.setdefault(r["tier"], {})
        cell = tier.setdefault(str(r["length"]), {}).setdefault(
            str(r["depth"]), {"n": 0, "exact": 0, "gold_logprob": 0.0}
        )
        cell["n"] += 1
        cell["exact"] += int(r["exact"])
        cell["gold_logprob"] += r["gold_logprob"]

    for tier, lengths in cells.items():
        if tier == "ppl":
            continue
        for depths in lengths.values():
            for cell in depths.values():
                n = max(1, cell["n"])
                cell["exact_rate"] = cell.pop("exact") / n
                cell["gold_logprob"] = cell["gold_logprob"] / n

    effective: dict = {}
    for tier, lengths in cells.items():
        if tier == "ppl":
            continue
        best = 0
        for length_s, depths in lengths.items():
            rates = [c["exact_rate"] for c in depths.values() if c["n"] > 0]
            if rates and sum(rates) / len(rates) >= 0.5:
                best = max(best, int(length_s))
        effective[tier] = best

    return {
        "model": settings.model,
        "rope_theta_scale": settings.rope_theta_scale,
        "cells": cells,
        "effective_context_length": effective,
        "rows": len(rows),
        "extra": settings.extra,
    }
