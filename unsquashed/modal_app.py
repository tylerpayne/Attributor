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

    # Experiment 3: from-scratch pretraining arms (unsquashed / alibi / none)
    modal run --detach modal_app.py::pretrain_from_scratch --prior alibi

    # Long-context ladder (ppl, passkey, kv, copy + extreme lengths)
    modal run --detach modal_app.py::ladder --models /results/pretrain/<run>/final

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
DEFAULT_PRETRAIN_CONFIG = "HuggingFaceTB/SmolLM2-135M"
EVAL_GPU = os.environ.get("UNSQUASH_EVAL_GPU", "L4")
TRAIN_GPU = os.environ.get("UNSQUASH_TRAIN_GPU", "A10G")
PRETRAIN_GPU = os.environ.get("UNSQUASH_PRETRAIN_GPU", "H100")

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
    # torch.compile / triton need a host C compiler for kernel launcher stubs
    .apt_install("build-essential")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("unsquash")
)

results_volume = modal.Volume.from_name("unsquash-results", create_if_missing=True)
cache_volume = modal.Volume.from_name("unsquash-hf-cache", create_if_missing=True)
VOLUMES = {"/results": results_volume, "/cache": cache_volume}


def _slug(model: str) -> str:
    return model.strip("/").replace("/", "__")


# Worker preemptions surface as cancelled inputs; retries restart the call.
# run_eval resumes from results.jsonl on re-entry; run_train restarts fresh.
RETRIES = modal.Retries(max_retries=3, initial_delay=10.0)


@app.function(image=image, gpu=EVAL_GPU, timeout=6 * 60 * 60, volumes=VOLUMES,
              retries=RETRIES)
def run_eval(
    model: str,
    out_name: str,
    methods: str = "attention_sum,rollout,unsquashed",
    max_cases: int = 500,
    split: str = "validation",
    reduction: str = "mean",
    max_context_tokens: int = 2000,
    head_weighting: str = "o_proj_norm",
    unsquashed_k: float = 0.0,
    unsquashed_lambda: float = 1.0,
    no_prior: bool = False,
) -> dict:
    import logging

    from unsquash.eval import hotpot
    from unsquash.eval.runner import EvalSettings, build_attributor, evaluate
    from unsquash.prior import PriorConfig

    logging.basicConfig(level=logging.INFO)

    if no_prior:
        prior = None
    elif unsquashed_k > 0:
        prior = PriorConfig(k=unsquashed_k, lam=unsquashed_lambda)
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


@app.function(image=image, gpu=TRAIN_GPU, timeout=24 * 60 * 60, volumes=VOLUMES,
              retries=RETRIES)
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
    unsquashed_k: float = 0.0,
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
        unsquashed_k=unsquashed_k if unsquashed_k > 0 else None,
        eval_every=eval_every,
        save_every=save_every,
        seed=seed,
        autocast_bf16=True,
    )
    final = train(settings)
    results_volume.commit()
    return final


@app.function(image=image, gpu=PRETRAIN_GPU, timeout=12 * 60 * 60, volumes=VOLUMES,
              retries=RETRIES)
def run_pretrain(
    out_name: str,
    model_config: str = DEFAULT_PRETRAIN_CONFIG,
    tokens: float = 2.7e9,
    seq_len: int = 2048,
    batch_size: int = 32,
    grad_accum: int = 2,
    lr: float = 1e-3,
    prior: str = "unsquashed",  # "unsquashed" | "alibi" | "none"
    unsquashed_k: float = 0.0,
    unsquashed_lam: float = 1.0,
    eval_every: int = 500,
    save_every: int = 5000,
    seed: int = 0,
) -> str:
    import logging

    from unsquash.pretrain.runner import PretrainSettings, pretrain

    logging.basicConfig(level=logging.INFO)

    settings = PretrainSettings(
        model_config=model_config,
        out_dir=f"/results/pretrain/{out_name}",
        tokens=tokens,
        seq_len=seq_len,
        batch_size=batch_size,
        grad_accum=grad_accum,
        lr=lr,
        prior=prior,
        unsquashed_k=(unsquashed_k if unsquashed_k > 0 else None),
        unsquashed_lam=unsquashed_lam,
        eval_every=eval_every,
        save_every=save_every,
        seed=seed,
    )
    final = pretrain(settings)
    results_volume.commit()
    return final


@app.function(image=image, gpu=PRETRAIN_GPU, timeout=12 * 60 * 60, volumes=VOLUMES,
              retries=RETRIES)
def run_ladder(
    model: str,
    out_name: str,
    tiers: str = "ppl,passkey,kv,copy",
    lengths: str = "2048,4096,8192,16384,32768",
    depths: str = "0.0,0.25,0.5,0.75,1.0",
    extreme_lengths: str = "65536,131072,262144",
    cases_per_cell: int = 5,
    extreme_cases_per_cell: int = 2,
    ppl_blocks: int = 4,
    rope_theta_scale: float = 1.0,
    seed: int = 0,
) -> dict:
    import logging

    from unsquash.ladder.runner import LadderSettings, run_ladder as _run

    logging.basicConfig(level=logging.INFO)

    settings = LadderSettings(
        model=model,
        out_dir=f"/results/ladder/{out_name}",
        tiers=tuple(t.strip() for t in tiers.split(",") if t.strip()),
        lengths=tuple(int(n) for n in lengths.split(",") if n.strip()),
        depths=tuple(float(d) for d in depths.split(",") if d.strip()),
        extreme_lengths=tuple(
            int(n) for n in extreme_lengths.split(",") if n.strip() and int(n) > 0
        ),
        cases_per_cell=cases_per_cell,
        extreme_cases_per_cell=extreme_cases_per_cell,
        ppl_blocks=ppl_blocks,
        rope_theta_scale=rope_theta_scale,
        seed=seed,
        extra={"model": model},
    )
    summary = _run(settings)
    results_volume.commit()
    return summary


@app.local_entrypoint()
def ladder(
    models: str,
    tiers: str = "ppl,passkey,kv,copy",
    lengths: str = "2048,4096,8192,16384,32768",
    depths: str = "0.0,0.25,0.5,0.75,1.0",
    extreme_lengths: str = "65536,131072,262144",
    cases_per_cell: int = 5,
    rope_theta_scale: float = 1.0,
    out_suffix: str = "",
):
    """The long-context ladder over one or more checkpoints (comma-separated
    volume paths), one detached GPU job per model. Results land in
    ``ladder/<checkpoint-slug><suffix>/`` in the unsquash-results volume.

    Example (all three arms, native RoPE):

        modal run --detach modal_app.py::ladder --models \\
            /results/pretrain/A__scratch_unsquashed/final,\\
            /results/pretrain/A__scratch_control/final,\\
            /results/pretrain/A__scratch_alibi/final
    """
    calls = []
    for model in (m.strip() for m in models.split(",") if m.strip()):
        out_name = _slug(model.removeprefix("/results/pretrain/")) + out_suffix
        call = run_ladder.spawn(
            model=model,
            out_name=out_name,
            tiers=tiers,
            lengths=lengths,
            depths=depths,
            extreme_lengths=extreme_lengths,
            cases_per_cell=cases_per_cell,
            rope_theta_scale=rope_theta_scale,
        )
        calls.append((model, out_name, call))
        print(f"Spawned ladder for {model}: {call.object_id}")
    for model, out_name, call in calls:
        summary = call.get()
        eff = summary.get("effective_context_length", {})
        print(f"\n=== {model} ===")
        print(f"effective context length by tier: {eff}")
        print(f"  modal volume get unsquash-results ladder/{out_name}/summary.json .")


PRETRAIN_TAGS = {"unsquashed": "unsquashed", "alibi": "alibi", "none": "control"}


@app.local_entrypoint()
def pretrain_from_scratch(
    tokens: float = 2.7e9,
    model_config: str = DEFAULT_PRETRAIN_CONFIG,
    prior: str = "unsquashed",
    out_name: str = "",
):
    """From-scratch SmolLM2-135M-shaped pretraining, prior on from step 0.

    ``--prior`` selects the arm: ``unsquashed`` (log-distance prior),
    ``alibi`` (linear-distance prior), or ``none`` (plain causal control) —
    all three on the same data order and init seed.
    """
    if prior not in PRETRAIN_TAGS:
        raise SystemExit(f"--prior must be one of {sorted(PRETRAIN_TAGS)}")
    out_name = out_name or f"{_slug(model_config)}__scratch_{PRETRAIN_TAGS[prior]}"
    call = run_pretrain.spawn(
        out_name=out_name,
        model_config=model_config,
        tokens=tokens,
        prior=prior,
    )
    print(f"Spawned pretraining (survives client death): {call.object_id}")
    final = call.get()
    print(f"\nFinal checkpoint (in the unsquash-results volume): {final}")
    print("Evaluate it (bias auto-applied) with:")
    print(f"  modal run modal_app.py::evaluate --model {final}")


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
    unsquashed_k: float = 0.0,
    unsquashed_lambda: float = 1.0,
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
        unsquashed_k=unsquashed_k,
        unsquashed_lambda=unsquashed_lambda,
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
    unsquashed_k: float = 0.0,
    out_name: str = "",
):
    """Experiment 2: continued pretraining with the annealed prior."""
    out_name = out_name or f"{_slug(model)}__unsquashed"
    # spawn, not .remote(): a blocking call is cancelled server-side if this
    # local client dies, killing the training task mid-run.
    call = run_train.spawn(
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
        unsquashed_k=unsquashed_k,
    )
    print(f"Spawned training (survives client death): {call.object_id}")
    final = call.get()
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
    # spawn + get, not .remote(): blocking calls are cancelled server-side if
    # this local client dies; spawned calls run to completion regardless.
    train_call = run_train.spawn(
        model=model,
        out_name=train_name,
        steps=steps,
        prior_warmup_steps=prior_warmup_steps,
    )
    final = train_call.get()
    retro_summary = run_eval.spawn(
        model=final,
        out_name=f"{train_name}__eval",
        max_cases=max_cases,
        methods="attention_sum,rollout,unsquashed",
    ).get()
    base_summary = base_eval.get()

    _print_summary(base_summary, f"Frozen {model}")
    _print_summary(retro_summary, f"Retrofitted {final} (prior applied at capture)")
    print(
        "\nThe comparison that matters: 'rollout' on the retrofitted model vs "
        "'unsquashed' vs 'rollout' on the frozen model."
    )
