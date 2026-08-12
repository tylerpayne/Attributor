import math

import pytest
import torch

from unsquash.coefficients import (
    log_unsquash_coefficients,
    unsquash_coefficients,
    unsquash_factor,
)


def lgamma_coefficient(m: int, k: float) -> float:
    """The Triton kernel's formulation: Gamma(m+1/k)/(Gamma(m+1)Gamma(1/k))."""
    if m == 0:
        return 1.0
    return math.exp(
        math.lgamma(m + 1.0 / k) - math.lgamma(m + 1.0) - math.lgamma(1.0 / k)
    )


@pytest.mark.parametrize("k", [1.0, 2.0, 2.5, 8.0, 30.0])
def test_recurrence_matches_lgamma(k):
    n = 200
    c = unsquash_coefficients(n, k)
    expected = torch.tensor([lgamma_coefficient(m, k) for m in range(n)],
                            dtype=torch.float64)
    assert torch.allclose(c, expected, rtol=1e-12, atol=1e-300)


@pytest.mark.parametrize("k", [2, 3, 8])
def test_factor_is_kth_root_of_ones(k):
    n = 64
    F = unsquash_factor(n, k)
    L = torch.tril(torch.ones(n, n, dtype=torch.float64))
    assert torch.allclose(torch.linalg.matrix_power(F, k), L, atol=1e-10)


def test_factor_strictly_causal():
    F = unsquash_factor(16, 4)
    assert torch.all(torch.triu(F, diagonal=1) == 0)
    assert torch.all(torch.diagonal(F) == 1)


def test_log_coefficients_consistent():
    n, k = 100, 6
    logc = log_unsquash_coefficients(n, k)
    c = unsquash_coefficients(n, k)
    assert torch.allclose(logc, c.log(), rtol=1e-10)
    assert logc[0] == 0.0


def test_coefficients_positive_and_decaying():
    c = unsquash_coefficients(50, 8)
    assert torch.all(c > 0)
    assert torch.all(c[1:] <= c[:-1])  # decreasing for k > 1


def test_k_equals_one_is_identity_correction():
    # k=1: (1-x)^-1 is the ones sequence itself -> no correction.
    c = unsquash_coefficients(20, 1.0)
    assert torch.allclose(c, torch.ones(20, dtype=torch.float64))


def test_validation():
    with pytest.raises(ValueError):
        unsquash_coefficients(0, 2)
    with pytest.raises(ValueError):
        unsquash_coefficients(10, 0)
