"""From-scratch pretraining with the unsquash prior active from step 0.

The retrofit experiment showed a pretrained model defends its attention sink
when the prior is annealed in. This runner tests the cleaner hypothesis: a
model that has *never* trained without the prior shouldn't need to form sinks
at all. The prior runs at full strength from the first step (no anneal), via
the SDPA additive mask in ``unsquash.pretrain.model``.

Logging matches the retrofit trainer: train loss, held-out eval loss, and
sink mass to ``train_log.jsonl``. Checkpoints export to HF format with the
bias recorded in ``unsquash_prior.json``, so ``unsquash.eval`` runs on them
unchanged. ``attn_bias`` selects the arm: "prior" (unsquash), "alibi"
(linear-distance control), or "none" (plain causal control) — all three on
identical data order and init seed.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, asdict

import torch

from unsquash.prior import PriorConfig
from unsquash.pretrain.model import PriorLlama, ModelSpec

logger = logging.getLogger(__name__)


@dataclass
class PretrainSettings:
    # architecture + tokenizer source (weights are NOT loaded)
    model_config: str = "HuggingFaceTB/SmolLM2-135M"
    out_dir: str = "unsquash_pretrain_out"
    # data
    dataset: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: str | None = "sample-10BT"
    split: str = "train"
    text_column: str = "text"
    text_file: str | None = None
    seq_len: int = 2048
    batch_size: int = 32
    grad_accum: int = 2
    # optimization
    tokens: float = 2.7e9  # ~Chinchilla for 135M
    lr: float = 1e-3
    weight_decay: float = 0.1
    lr_warmup_steps: int = 500
    grad_clip: float = 1.0
    autocast_bf16: bool = True
    compile: bool = True
    device: str | None = None
    seed: int = 0
    # attention bias: "prior" (unsquash log-distance), "alibi" (linear-
    # distance control), or "none" (plain causal control)
    attn_bias: str = "prior"
    prior_k: float | None = None  # default: num_layers (attn_bias="prior")
    prior_lam: float = 1.0
    # bookkeeping
    log_every: int = 20
    eval_every: int = 500
    eval_batches: int = 8
    save_every: int = 5000
    trust_remote_code: bool = False

    @property
    def tokens_per_step(self) -> int:
        return self.seq_len * self.batch_size * self.grad_accum

    @property
    def steps(self) -> int:
        return max(1, math.ceil(self.tokens / self.tokens_per_step))


def pretrain(settings: PretrainSettings) -> str:
    """Run from-scratch pretraining; returns the final checkpoint directory."""
    from transformers import AutoConfig, AutoTokenizer

    from unsquash.train.data import packed_batches

    torch.manual_seed(settings.seed)
    device = settings.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if settings.attn_bias not in ("prior", "alibi", "none"):
        raise ValueError(f"Unknown attn_bias: {settings.attn_bias!r}")
    spec = ModelSpec.from_hf(settings.model_config, max_seq_len=settings.seq_len)
    spec.prior_k = None
    if settings.attn_bias == "prior":
        spec.prior_k = settings.prior_k or float(spec.num_layers)
        spec.prior_lam = settings.prior_lam
    spec.alibi = settings.attn_bias == "alibi"

    logger.info(
        "From-scratch %s: %d layers, attn_bias=%s (prior_k=%s), "
        "%d steps of %d tokens (%.2fB total)",
        settings.model_config, spec.num_layers, settings.attn_bias, spec.prior_k,
        settings.steps, settings.tokens_per_step,
        settings.steps * settings.tokens_per_step / 1e9,
    )

    model = PriorLlama(spec).to(device)
    hf_config = AutoConfig.from_pretrained(settings.model_config)
    tokenizer = AutoTokenizer.from_pretrained(
        settings.model_config, trust_remote_code=settings.trust_remote_code
    )

    step_model = torch.compile(model) if settings.compile else model

    batches = packed_batches(
        tokenizer,
        seq_len=settings.seq_len,
        batch_size=settings.batch_size,
        dataset=settings.dataset,
        dataset_config=settings.dataset_config,
        split=settings.split,
        text_column=settings.text_column,
        text_file=settings.text_file,
        seed=settings.seed,
    )
    # Held-out batches (drawn first, never trained on) + a sink probe.
    eval_batches = [next(batches) for _ in range(settings.eval_batches)]
    probe = eval_batches[0][:1]

    decay, no_decay = [], []
    for p in model.parameters():
        (decay if p.dim() >= 2 else no_decay).append(p)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": settings.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=settings.lr,
        betas=(0.9, 0.95),
    )

    def lr_lambda(step: int) -> float:
        if step < settings.lr_warmup_steps:
            return (step + 1) / settings.lr_warmup_steps
        t = (step - settings.lr_warmup_steps) / max(
            1, settings.steps - settings.lr_warmup_steps
        )
        return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    os.makedirs(settings.out_dir, exist_ok=True)
    with open(os.path.join(settings.out_dir, "pretrain_settings.json"), "w") as fd:
        json.dump(asdict(settings), fd, indent=2)

    def save_checkpoint(name: str) -> str:
        path = os.path.join(settings.out_dir, name)
        model.to_hf(hf_config).save_pretrained(path)
        tokenizer.save_pretrained(path)
        # Record the training-time bias so eval/attribution re-applies it
        # ("auto" consumers read this sidecar; absent = plain causal).
        if settings.attn_bias == "prior":
            PriorConfig(k=spec.prior_k, lam=spec.prior_lam).save(path)
        elif settings.attn_bias == "alibi":
            PriorConfig(
                k=0.0, kind="alibi", num_heads=spec.num_heads
            ).save(path)
        logger.info("Saved checkpoint %s", path)
        return path

    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if settings.autocast_bf16 and device == "cuda"
        else None
    )

    @torch.no_grad()
    def eval_loss() -> float:
        step_model.eval()
        total = 0.0
        for batch in eval_batches:
            batch = batch.to(device)
            if autocast is not None:
                with autocast:
                    total += float(step_model(batch, labels=batch))
            else:
                total += float(step_model(batch, labels=batch))
        step_model.train()
        return total / max(1, len(eval_batches))

    log_path = os.path.join(settings.out_dir, "train_log.jsonl")
    log_fd = open(log_path, "a")
    running_loss, tokens_seen, t0 = 0.0, 0, time.time()

    step_model.train()
    for step in range(settings.steps):
        optimizer.zero_grad(set_to_none=True)
        micro_loss = 0.0
        for _ in range(settings.grad_accum):
            batch = next(batches).to(device)
            if autocast is not None:
                with autocast:
                    loss = step_model(batch, labels=batch) / settings.grad_accum
            else:
                loss = step_model(batch, labels=batch) / settings.grad_accum
            loss.backward()
            micro_loss += float(loss.detach())
            tokens_seen += batch.numel()

        torch.nn.utils.clip_grad_norm_(model.parameters(), settings.grad_clip)
        optimizer.step()
        scheduler.step()
        running_loss += micro_loss

        if (step + 1) % settings.log_every == 0:
            elapsed = time.time() - t0
            entry = {
                "step": step + 1,
                "loss": running_loss / settings.log_every,
                "lr": scheduler.get_last_lr()[0],
                "tokens_seen": tokens_seen,
                "tokens_per_s": tokens_seen / max(elapsed, 1e-9),
            }
            logger.info(
                "step %d/%d  loss %.4f  lr %.2e  %.0f tok/s",
                entry["step"], settings.steps, entry["loss"], entry["lr"],
                entry["tokens_per_s"],
            )
            log_fd.write(json.dumps(entry) + "\n")
            log_fd.flush()
            running_loss = 0.0

        if (step + 1) % settings.eval_every == 0:
            ev = eval_loss()
            sink = model.sink_mass(probe.to(device))
            entry = {
                "step": step + 1,
                "eval_loss": ev,
                "eval_ppl": math.exp(min(ev, 20.0)),
                "sink_mass": sink,
            }
            logger.info(
                "step %d  eval_loss %.4f (ppl %.2f)  sink_mass %.4f",
                entry["step"], ev, entry["eval_ppl"], sink,
            )
            log_fd.write(json.dumps(entry) + "\n")
            log_fd.flush()

        if (step + 1) % settings.save_every == 0 and (step + 1) < settings.steps:
            save_checkpoint(f"step-{step + 1}")

    path = save_checkpoint("final")
    log_fd.close()
    return path
