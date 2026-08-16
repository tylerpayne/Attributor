"""ALiBi: the linear-distance attention bias (Press et al., 2022).

The pretraining control for the unsquash prior. Both are fixed,
content-independent, additive penalties on pre-softmax logits that depend only
on the query-key distance ``d = i - j``; they differ in shape. ALiBi subtracts
``m_h * d`` (linear, per-head slope ``m_h``), the unsquash prior subtracts
``lam * |ln c_d|`` (asymptotically logarithmic, shared across heads). Post
softmax that is a geometric distance kernel ``exp(-m_h d)`` versus a power law
``d^(1/k - 1)`` — ALiBi effectively windows far context out, the prior only
discounts it. Training an ALiBi run on the same data/seed isolates the bias
*shape* as the treatment.

As in the prior model, RoPE is kept: the existing prior run is RoPE + log
bias, so the matched control is RoPE + linear bias (pure no-RoPE ALiBi would
confound bias shape with position-encoding removal).

Slopes follow the reference recipe from the ALiBi paper: for ``H`` a power of
two, the geometric sequence starting at ``2^(-8/H)``; otherwise the closest
power of two's slopes plus every other slope of the ``2H`` sequence. For the
SmolLM2 shape (9 heads): ``1/2 .. 1/256`` plus ``2^(-1/2)``.
"""

from __future__ import annotations

import math

import torch


def alibi_slopes(
    num_heads: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Per-head slopes ``[num_heads]``, reference ALiBi recipe."""
    if num_heads < 1:
        raise ValueError(f"num_heads must be >= 1, got {num_heads}")

    def power_of_two(n: int) -> list[float]:
        start = 2.0 ** (-(2.0 ** -(math.log2(n) - 3)))
        return [start ** (i + 1) for i in range(n)]

    if math.log2(num_heads).is_integer():
        slopes = power_of_two(num_heads)
    else:
        closest = 2 ** math.floor(math.log2(num_heads))
        slopes = (
            power_of_two(closest)
            + power_of_two(2 * closest)[0::2][: num_heads - closest]
        )
    return torch.tensor(slopes, dtype=dtype, device=device)


def alibi_attention_bias(
    n: int,
    num_heads: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """``[1, num_heads, n, n]`` additive attention bias: ``-m_h * (i - j)`` on
    and below the diagonal, dtype-min above it (causal masking included).

    The per-head analogue of ``unsquash.prior.unsquashed_attention_bias``; the
    same 4D-float-mask contract, so it drops into SDPA, the HF eager path,
    and rollout capture unchanged.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    slopes = alibi_slopes(num_heads, dtype=torch.float32, device=device)
    idx = torch.arange(n, device=device)
    m = idx[:, None] - idx[None, :]
    bias = -slopes[:, None, None] * m.clamp(min=0).to(torch.float32)
    neg_inf = torch.tensor(torch.finfo(dtype).min, dtype=torch.float32, device=device)
    bias = torch.where(m >= 0, bias, neg_inf)
    return bias.to(dtype)[None]
