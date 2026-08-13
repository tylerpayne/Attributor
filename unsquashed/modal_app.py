"""Run the unsquash experiments on Modal GPUs.

Setup (once):

    pip install modal
    modal setup

Run from this directory (``unsquashed/``):

    # Experiment 1: post-hoc A/B eval on a frozen model
    modal run modal_app.py::evaluate --max-cases 500

    # Experiment 2: continued-pretraining retrofit (detach for long runs)
    modal run --detach modal_app.py::retrofit --steps 5000

    # Evaluate the retrofitted checkpoint (prior auto-applied)
    modal run modal_app.py::evaluate --model /results/train/smollm-unsquashed/final

    # Or the whole thing: base eval + retrofit in parallel, then checkpoint eval
    modal run --detach modal_app.py::pipeline

Artifacts (results.jsonl, summary.json, checkpoints, train_log.jsonl) persist
in the ``unsquash-results`` volume:

    modal volume ls unsquash-results
    modal volume get unsquash-results eval/<name>/summary.json .

Evaluation runs are resumable: re-running with the same ``--out-name`` picks up
where the previous run stopped. GPU types can be overridden at launch, e.g.
``UNSQUASH_TRAIN_GPU=A100 modal run --detach modal_app.py::retrofit``.
"""

from __future__ import annotations

import os

import modal

DEFAULT_MODEL = "HuggingFaceTB/SmolLM-135M-Instruct"
EVAL_GPU = os.environ.get("UNSQUASH_EVAL_GPU", "L4")
TRAIN_GPU = os.environ.get("UNSQUASH_TRAIN_GPU", "A10G")

app = modal.App("unsquash")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1",
        "transformers>=4.44",
        "accelerate>=0.30",
        "datasets>=2.19",
        "tokenizers>=0.19",
        "tqdm",
        "hf-transfer",
    )
    .env({"HF_HOME": "/cache/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("unsquash")
)

results_volume = modal.Volume.from_name("unsquash-results", create_if_missing=True)
cache_volume = modal.Volume.from_name("unsquash-hf-cache", create_if_missing=True)
VOLUMES = {"/results": results_volume, "/cache": cache_volume}


def _slug(model: str) -> str:
    return model.strip("/").replace("/", "__")


@app.function(image=image, gpu=EVAL_GPU, timeout=6 * 60 * 60, volumes=VOLUMES)
def run_eval(
    model: str,
    out_name: str,
    methods: str = "attention_sum,rollout,unsquashed",
    max_cases: int = 500,
    split: str = "validation",
    reduction: str = "mean",
    max_context_tokens: int = 2000,
    head_weighting: str = "o_proj_norm",
    prior_k: float = 0.0,
    prior_lambda: float = 1.0,
    no_prior: bool = False,
) -> dict:
    import logging

    from unsquash.eval import hotpot
    from unsquash.eval.runner import EvalSettings, build_attributor, evaluate
    from unsquash.prior import PriorConfig

    logging.basicConfig(level=logging.INFO)

    if no_prior:
        prior = None
    elif prior_k > 0:
        prior = PriorConfig(k=prior_k, lam=prior_lambda)
    else:
        prior = "auto"  # applies a checkpoint's recorded unsquash_prior.json

    attributor = build_attributor(
        model_name_or_path=model,
        dtype="float32",
        device_map="cuda",
        head_weighting=head_weighting,
        prior=prior,
    )
    settings = EvalSettings(
        methods=tuple(m.strip() for m in methods.split(",") if m.strip()),
        max_cases=max_cases if max_cases > 0 else None,
        reduction=reduction,
        max_context_tokens=max_context_tokens,
        out_dir=f"/results/eval/{out_name}",
        extra={"model": model, "split": split, "head_weighting": head_weighting},
    )
    cases = hotpot.load_cases(split=split)
    summary = evaluate(attributor, cases, settings)
    results_volume.commit()
    return summary


@app.function(image=image, gpu=TRAIN_GPU, timeout=24 * 60 * 60, volumes=VOLUMES)
def run_train(
    model: str,
    out_name: str,
    steps: int = 5000,
    prior_warmup_steps: int = 1000,
    seq_len: int = 1024,
    # Eager attention (needed for the 4D prior mask) materializes full
    # per-layer attention matrices, so batch 8 OOMs a 24GB A10G at seq 1024;
    # batch 4 x accum 8 keeps the same effective batch within memory.
    batch_size: int = 4,
    grad_accum: int = 8,
    lr: float = 3e-5,
    lr_warmup_steps: int = 100,
    dataset: str = "HuggingFaceFW/fineweb-edu",
    dataset_config: str = "sample-10BT",
    prior_k: float = 0.0,
    eval_every: int = 200,
    save_every: int = 1000,
    seed: int = 0,
) -> str:
    import logging

    from unsquash.train.runner import TrainSettings, train

    logging.basicConfig(level=logging.INFO)

    settings = TrainSettings(
        model=model,
        out_dir=f"/results/train/{out_name}",
        steps=steps,
        prior_warmup_steps=prior_warmup_steps,
        seq_len=seq_len,
        batch_size=batch_size,
        grad_accum=grad_accum,
        lr=lr,
        lr_warmup_steps=lr_warmup_steps,
        dataset=dataset,
        dataset_config=dataset_config or None,
        prior_k=prior_k if prior_k > 0 else None,
        eval_every=eval_every,
        save_every=save_every,
        seed=seed,
        autocast_bf16=True,
    )
    final = train(settings)
    results_volume.commit()
    return final


def _print_summary(summary: dict, title: str) -> None:
    print(f"\n=== {title} ===")
    print(f"Evaluated cases: {summary['evaluated_cases']}")
    metrics = summary["metrics"]
    methods = list(metrics)
    names = sorted({n for m in methods for n in metrics[m]})
    width = max((len(n) for n in names), default=10) + 2
    print(" " * width + "".join(f"{m:>16}" for m in methods))
    for name in names:
        row = f"{name:<{width}}"
        for m in methods:
            v = metrics[m].get(name)
            row += f"{v:>16.4f}" if v is not None else f"{'-':>16}"
        print(row)


@app.local_entrypoint()
def evaluate(
    model: str = DEFAULT_MODEL,
    max_cases: int = 500,
    methods: str = "attention_sum,rollout,unsquashed",
    split: str = "validation",
    reduction: str = "mean",
    max_context_tokens: int = 2000,
    head_weighting: str = "o_proj_norm",
    prior_k: float = 0.0,
    prior_lambda: float = 1.0,
    no_prior: bool = False,
    out_name: str = "",
):
    """Experiment 1: post-hoc attribution A/B on HotpotQA."""
    out_name = out_name or f"{_slug(model)}__{split}"
    summary = run_eval.remote(
        model=model,
        out_name=out_name,
        methods=methods,
        max_cases=max_cases,
        split=split,
        reduction=reduction,
        max_context_tokens=max_context_tokens,
        head_weighting=head_weighting,
        prior_k=prior_k,
        prior_lambda=prior_lambda,
        no_prior=no_prior,
    )
    _print_summary(summary, f"{model} ({split})")
    print(f"\nArtifacts: modal volume get unsquash-results eval/{out_name}/summary.json .")


@app.local_entrypoint()
def retrofit(
    model: str = DEFAULT_MODEL,
    steps: int = 5000,
    prior_warmup_steps: int = 1000,
    seq_len: int = 1024,
    batch_size: int = 4,
    grad_accum: int = 8,
    lr: float = 3e-5,
    dataset: str = "HuggingFaceFW/fineweb-edu",
    dataset_config: str = "sample-10BT",
    prior_k: float = 0.0,
    out_name: str = "",
):
    """Experiment 2: continued pretraining with the annealed prior."""
    out_name = out_name or f"{_slug(model)}__unsquashed"
    final = run_train.remote(
        model=model,
        out_name=out_name,
        steps=steps,
        prior_warmup_steps=prior_warmup_steps,
        seq_len=seq_len,
        batch_size=batch_size,
        grad_accum=grad_accum,
        lr=lr,
        dataset=dataset,
        dataset_config=dataset_config,
        prior_k=prior_k,
    )
    print(f"\nFinal checkpoint (in the unsquash-results volume): {final}")
    print("Evaluate it (prior auto-applied) with:")
    print(f"  modal run modal_app.py::evaluate --model {final}")


@app.local_entrypoint()
def pipeline(
    model: str = DEFAULT_MODEL,
    max_cases: int = 500,
    steps: int = 5000,
    prior_warmup_steps: int = 1000,
):
    """The full A/B: frozen-model eval and retrofit run in parallel, then the
    retrofitted checkpoint is evaluated with plain rollout + its prior."""
    base_name = f"{_slug(model)}__validation"
    train_name = f"{_slug(model)}__unsquashed"

    base_eval = run_eval.spawn(model=model, out_name=base_name, max_cases=max_cases)
    final = run_train.remote(
        model=model,
        out_name=train_name,
        steps=steps,
        prior_warmup_steps=prior_warmup_steps,
    )
    retro_summary = run_eval.remote(
        model=final,
        out_name=f"{train_name}__eval",
        max_cases=max_cases,
        methods="attention_sum,rollout,unsquashed",
    )
    base_summary = base_eval.get()

    _print_summary(base_summary, f"Frozen {model}")
    _print_summary(retro_summary, f"Retrofitted {final} (prior applied at capture)")
    print(
        "\nThe comparison that matters: 'rollout' on the retrofitted model vs "
        "'unsquashed' vs 'rollout' on the frozen model."
    )
