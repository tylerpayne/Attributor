"""The unsquash prior as an additive attention-logit bias.

Adding ``lambda * ln(c_{i-j})`` to every pre-softmax attention logit is the
in-model form of the unsquash correction: a fixed, content-independent
log-distance penalty (a logarithmic cousin of ALiBi's linear one). A model
trained with this prior has a structurally uniform rollout null, so vanilla
rollout attribution is de-biased by construction.

The bias is delivered as a ready-made 4D float attention mask
``[batch=1, 1, q_len, kv_len]``. Modern ``transformers`` (>= 4.37) accepts 4D
float masks and uses them verbatim, skipping its own causal-mask construction,
so the causal mask (dtype-min above the diagonal) is baked in here.

Notes / assumptions:
- Square masks only (full-sequence forward passes: teacher-forced evaluation
  and packed-block training). Not for incremental decoding with a KV cache.
- No padding: sequences are assumed dense. Padded batches would need the
  padding mask merged in.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict

import torch

from unsquash.coefficients import log_unsquash_coefficients

PRIOR_FILENAME = "unsquash_prior.json"


def prior_attention_bias(
    n: int,
    k: float,
    *,
    lam: float = 1.0,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """``[1, 1, n, n]`` additive attention bias: ``lam * ln(c_{i-j})`` on and
    below the diagonal, dtype-min above it (causal masking included)."""
    logc = log_unsquash_coefficients(n, k, device=device).to(torch.float32)
    idx = torch.arange(n, device=device)
    m = idx[:, None] - idx[None, :]
    bias = lam * logc[m.clamp(min=0)]
    neg_inf = torch.tensor(torch.finfo(dtype).min, dtype=torch.float32, device=device)
    bias = torch.where(m >= 0, bias, neg_inf)
    return bias.to(dtype)[None, None]


def linear_anneal(step: int, warmup_steps: int) -> float:
    """lambda ramps linearly 0 -> 1 over ``warmup_steps``, then stays at 1."""
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, step / warmup_steps)


@dataclass
class PriorConfig:
    """The prior a checkpoint was trained with (and should be run with)."""

    k: float
    lam: float = 1.0

    def save(self, directory: str | os.PathLike) -> str:
        path = os.path.join(directory, PRIOR_FILENAME)
        with open(path, "w") as fd:
            json.dump(asdict(self), fd, indent=2)
        return path

    @classmethod
    def load(cls, directory: str | os.PathLike) -> "PriorConfig | None":
        """Load the prior recorded next to a checkpoint, if any."""
        path = os.path.join(str(directory), PRIOR_FILENAME)
        if not os.path.exists(path):
            return None
        with open(path) as fd:
            data = json.load(fd)
        return cls(k=float(data["k"]), lam=float(data.get("lam", 1.0)))
