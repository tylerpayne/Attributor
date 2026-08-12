import torch

from unsquash.coefficients import unsquash_coefficients
from unsquash.prior import PriorConfig, linear_anneal, prior_attention_bias


def test_softmax_of_bias_reproduces_coefficients():
    """A head with flat logits + the prior should attend proportionally to c_m."""
    n, k = 32, 8
    bias = prior_attention_bias(n, k, dtype=torch.float32)[0, 0]
    probs = torch.softmax(bias.to(torch.float64), dim=-1)

    c = unsquash_coefficients(n, k)
    i = n - 1  # last row sees all lags
    expected = c.flip(0) / c.sum()
    assert torch.allclose(probs[i], expected, rtol=1e-4)


def test_bias_is_causal():
    n = 16
    bias = prior_attention_bias(n, 4, dtype=torch.float32)[0, 0]
    probs = torch.softmax(bias, dim=-1)
    assert torch.all(torch.triu(probs, diagonal=1) < 1e-8)


def test_lambda_scales_bias():
    n, k = 16, 4
    half = prior_attention_bias(n, k, lam=0.5)[0, 0]
    full = prior_attention_bias(n, k, lam=1.0)[0, 0]
    tril = torch.tril_indices(n, n)
    assert torch.allclose(half[tril[0], tril[1]] * 2, full[tril[0], tril[1]],
                          rtol=1e-5)


def test_lambda_zero_is_plain_causal_mask():
    bias = prior_attention_bias(8, 4, lam=0.0)[0, 0]
    assert torch.all(torch.tril(bias) == 0)


def test_linear_anneal():
    assert linear_anneal(0, 100) == 0.0
    assert linear_anneal(50, 100) == 0.5
    assert linear_anneal(100, 100) == 1.0
    assert linear_anneal(500, 100) == 1.0
    assert linear_anneal(0, 0) == 1.0  # no warmup -> prior fully on


def test_prior_config_roundtrip(tmp_path):
    PriorConfig(k=30.0, lam=0.75).save(tmp_path)
    loaded = PriorConfig.load(tmp_path)
    assert loaded == PriorConfig(k=30.0, lam=0.75)
    assert PriorConfig.load(tmp_path / "nope") is None
