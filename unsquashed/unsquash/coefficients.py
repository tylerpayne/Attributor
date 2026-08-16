"""Unsquash coefficients: the k-th convolution root of the all-ones sequence.

The coefficients are

    c_m = Gamma(m + 1/k) / (Gamma(m + 1) * Gamma(1/k)),

the Taylor coefficients of (1 - x)^(-1/k). The lower-triangular Toeplitz matrix
built from them is the k-th matrix root of the lower-triangular all-ones matrix
L (exactly, for any finite size, since lower-triangular Toeplitz matrices
multiply like polynomials mod x^n).

Interpretation: uniform causal attention acts as a discrete integrator (its
unnormalized pattern is L). A stack of k such layers integrates k times, piling
attribution mass onto early tokens like m^(k-1). Hadamard-multiplying each
layer's attention by c_{i-j} turns each layer into a fractional integrator of
order 1/k, so the k-layer composition integrates exactly once and the null
model's rollout attribution comes out uniform.

Everything here is computed with the stable recurrence

    c_0 = 1,   c_m = c_{m-1} * (m - 1 + 1/k) / m,

which matches the lgamma formulation to machine precision and needs no special
functions.
"""

from __future__ import annotations

import torch


def _validate(n: int, k: float) -> None:
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")


def unsquash_coefficients(
    n: int,
    k: float,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """The first ``n`` coefficients ``c_0 .. c_{n-1}`` of ``(1 - x)^(-1/k)``.

    ``c_m`` is the factor to apply at lag ``m = i - j`` (query position i,
    key position j).
    """
    _validate(n, k)
    m = torch.arange(n, dtype=dtype, device=device)
    ratios = (m - 1 + 1.0 / k) / m.clamp(min=1)
    ratios[0] = 1.0
    return ratios.cumprod(0)


def log_unsquash_coefficients(
    n: int,
    k: float,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """``ln(c_m)`` for ``m = 0 .. n-1``, computed in log space.

    This is the additive pre-softmax form of the prior: adding
    ``lambda * ln(c_{i-j})`` to attention logits is equivalent to multiplying
    post-softmax attention by ``c_{i-j}^lambda`` and renormalizing rows.
    """
    _validate(n, k)
    m = torch.arange(n, dtype=dtype, device=device)
    steps = torch.zeros_like(m)
    if n > 1:
        steps[1:] = torch.log(m[1:] - 1 + 1.0 / k) - torch.log(m[1:])
    return steps.cumsum(0)


def unsquash_factor(
    n: int,
    k: float,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """The ``[n, n]`` factor matrix ``F`` with ``F[i, j] = c_{i-j}`` (0 above
    the diagonal). ``F`` is the k-th matrix root of the lower-triangular ones
    matrix: ``torch.linalg.matrix_power(F, k) == tril(ones)`` for integer k.
    """
    c = unsquash_coefficients(n, k, dtype=dtype, device=device)
    idx = torch.arange(n, device=device)
    m = idx[:, None] - idx[None, :]
    zero = torch.zeros((), dtype=dtype, device=device)
    return torch.where(m >= 0, c[m.clamp(min=0)], zero)
