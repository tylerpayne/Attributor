"""Continued pretraining with the annealed unsquash prior.

The prior — a fixed additive attention-logit bias ``lambda * ln(c_{i-j})`` —
is ramped in linearly over ``prior_warmup_steps`` so the pretrained model
adapts rather than breaks, then held at ``lambda = 1``. Checkpoints record the
prior (``unsquash_prior.json``) so ``unsquash.eval`` applies it automatically
at attribution time.

Also monitored: attention-sink mass (mean attention on position 0). The prior
penalizes exactly the long-lag edges sinks live on, so sink re-formation is
the early signal that the model is adapting rather than degrading.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, asdict

import torch

from unsquash.prior import PriorConfig, linear_anneal, prior_attention_bias

logger = logging.getLogger(__name__)


@dataclass
class TrainSettings:
    model: str
    out_dir: str = "unsquash_train_out"
    # data
    dataset: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: str | None = "sample-10BT"
    split: str = "train"
    text_column: str = "text"
    text_file: str | None = None
    seq_len: int = 1024
    batch_size: int = 8
    grad_accum: int = 4
    # optimization
    steps: int = 2000
    lr: float = 3e-5
    weight_decay: float = 0.01
    lr_warmup_steps: int = 100
    grad_clip: float = 1.0
    dtype: str = "float32"
    autocast_bf16: bool = False
    device: str | None = None
    seed: int = 0
    # prior
    prior_k: float | None = None  # default: num_hidden_layers
    prior_warmup_steps: int = 500
    # bookkeeping
    log_every: int = 10
    eval_every: int = 200
    eval_batches: int = 8
    save_every: int = 500
    trust_remote_code: bool = False


class PriorMask:
    """Cached components of the 4D bias so per-step lambda changes are a
    single fused multiply-add, not a recompute."""

    def __init__(self, seq_len: int, k: float, dtype: torch.dtype, device):
        full = prior_attention_bias(seq_len, k, lam=1.0, dtype=torch.float32,
                                    device=device)
        idx = torch.arange(seq_len, device=device)
        causal = idx[:, None] >= idx[None, :]
        self._logc = torch.where(causal, full[0, 0].to(torch.float32), 0.0)
        self._neg_inf = torch.where(
            causal, 0.0, torch.tensor(torch.finfo(dtype).min, device=device)
        )
        self._dtype = dtype

    def at(self, lam: float) -> torch.Tensor:
        return (lam * self._logc + self._neg_inf).to(self._dtype)[None, None]


@torch.no_grad()
def _eval_loss(model, batches: list[torch.Tensor], mask: torch.Tensor, device) -> float:
    model.eval()
    total = 0.0
    for batch in batches:
        batch = batch.to(device)
        out = model(input_ids=batch, attention_mask=mask, labels=batch,
                    use_cache=False)
        total += float(out.loss)
    model.train()
    return total / max(1, len(batches))


@torch.no_grad()
def _sink_mass(model, probe: torch.Tensor, mask: torch.Tensor, device) -> float:
    """Mean attention probability on position 0, averaged over layers, heads,
    and (non-trivial) query positions."""
    model.eval()
    out = model(input_ids=probe.to(device), attention_mask=mask,
                output_attentions=True, use_cache=False)
    model.train()
    sink = torch.stack([a[..., 1:, 0] for a in out.attentions])
    return float(sink.mean())


def train(settings: TrainSettings) -> str:
    """Run continued pretraining; returns the final checkpoint directory."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from unsquash.train.data import packed_batches

    torch.manual_seed(settings.seed)
    device = settings.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, settings.dtype)

    logger.info("Loading %s (%s) on %s", settings.model, settings.dtype, device)
    model = AutoModelForCausalLM.from_pretrained(
        settings.model,
        torch_dtype=dtype,
        trust_remote_code=settings.trust_remote_code,
        # eager attention keeps 4D float masks and output_attentions
        # (sink monitoring) uniformly supported across versions.
        attn_implementation="eager",
    ).to(device)
    model.gradient_checkpointing_disable()
    model.train()
    tokenizer = AutoTokenizer.from_pretrained(
        settings.model, trust_remote_code=settings.trust_remote_code
    )

    k = settings.prior_k or float(model.config.num_hidden_layers)
    prior_mask = PriorMask(settings.seq_len, k, dtype, device)
    logger.info("Prior: k=%s, warmup=%d steps", k, settings.prior_warmup_steps)

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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.lr,
        weight_decay=settings.weight_decay,
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
    with open(os.path.join(settings.out_dir, "train_settings.json"), "w") as fd:
        json.dump(asdict(settings), fd, indent=2)

    def save_checkpoint(name: str, lam: float) -> str:
        path = os.path.join(settings.out_dir, name)
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)
        PriorConfig(k=k, lam=lam).save(path)
        logger.info("Saved checkpoint %s (prior lambda=%.3f)", path, lam)
        return path

    log_path = os.path.join(settings.out_dir, "train_log.jsonl")
    log_fd = open(log_path, "a")
    running_loss, tokens_seen, t0 = 0.0, 0, time.time()

    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if settings.autocast_bf16 and device == "cuda"
        else None
    )

    for step in range(settings.steps):
        lam = linear_anneal(step, settings.prior_warmup_steps)
        mask = prior_mask.at(lam)

        optimizer.zero_grad(set_to_none=True)
        micro_loss = 0.0
        for _ in range(settings.grad_accum):
            batch = next(batches).to(device)
            if autocast is not None:
                with autocast:
                    out = model(input_ids=batch, attention_mask=mask,
                                labels=batch, use_cache=False)
            else:
                out = model(input_ids=batch, attention_mask=mask,
                            labels=batch, use_cache=False)
            loss = out.loss / settings.grad_accum
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
                "lambda": lam,
                "lr": scheduler.get_last_lr()[0],
                "tokens_seen": tokens_seen,
                "tokens_per_s": tokens_seen / max(elapsed, 1e-9),
            }
            logger.info(
                "step %d  loss %.4f  lambda %.3f  lr %.2e  %.0f tok/s",
                entry["step"], entry["loss"], lam, entry["lr"],
                entry["tokens_per_s"],
            )
            log_fd.write(json.dumps(entry) + "\n")
            log_fd.flush()
            running_loss = 0.0

        if (step + 1) % settings.eval_every == 0:
            eval_loss = _eval_loss(model, eval_batches, mask, device)
            sink = _sink_mass(model, probe, mask, device)
            entry = {
                "step": step + 1,
                "eval_loss": eval_loss,
                "eval_ppl": math.exp(min(eval_loss, 20.0)),
                "lambda": lam,
                "sink_mass": sink,
            }
            logger.info(
                "step %d  eval_loss %.4f (ppl %.2f)  sink_mass %.4f",
                entry["step"], eval_loss, entry["eval_ppl"], sink,
            )
            log_fd.write(json.dumps(entry) + "\n")
            log_fd.flush()

        if (step + 1) % settings.save_every == 0 and (step + 1) < settings.steps:
            save_checkpoint(f"step-{step + 1}", lam)

    final_lam = linear_anneal(settings.steps, settings.prior_warmup_steps)
    path = save_checkpoint("final", final_lam)
    log_fd.close()
    return path
