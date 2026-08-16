"""ALiBi control: slope recipe, bias tensor, model parity, and the
checkpoint round-trip (save with sidecar -> from_pretrained -> same logits),
which is the path the long-context ladder relies on."""

import math

import pytest
import torch

from unsquash.alibi import alibi_attention_bias, alibi_slopes
from unsquash.prior import PriorConfig
from unsquash.pretrain.model import PriorLlama, ModelSpec

TINY = dict(
    vocab_size=64,
    hidden_size=32,
    intermediate_size=64,
    num_layers=2,
    num_heads=4,
    num_kv_heads=2,
    max_seq_len=16,
    rope_theta=10000.0,
)


def tiny_alibi_model(seed=0):
    torch.manual_seed(seed)
    return PriorLlama(ModelSpec(unsquashed_k=None, alibi=True, **TINY)).eval()


def tiny_ids(seed=1):
    torch.manual_seed(seed)
    return torch.randint(0, TINY["vocab_size"], (2, TINY["max_seq_len"]))


def test_slopes_power_of_two():
    # Reference recipe: for 8 heads, 2^-1 .. 2^-8.
    slopes = alibi_slopes(8)
    expected = torch.tensor([2.0 ** -(i + 1) for i in range(8)])
    torch.testing.assert_close(slopes, expected)


def test_slopes_nine_heads():
    # SmolLM2's 9 heads: the 8-head slopes plus the first of the interleaved
    # 16-head sequence, 2^-0.5.
    slopes = alibi_slopes(9)
    torch.testing.assert_close(slopes[:8], alibi_slopes(8))
    assert slopes[8].item() == pytest.approx(2.0 ** -0.5)


def test_slopes_all_positive_and_distinct():
    for h in (1, 2, 3, 4, 6, 8, 9, 12, 16):
        slopes = alibi_slopes(h)
        assert (slopes > 0).all()
        assert len(set(slopes.tolist())) == h


def test_bias_values_and_mask():
    n, heads = 8, 4
    bias = alibi_attention_bias(n, heads)
    assert bias.shape == (1, heads, n, n)
    slopes = alibi_slopes(heads)
    for h in range(heads):
        for i in range(n):
            for j in range(n):
                if j > i:
                    assert bias[0, h, i, j] == torch.finfo(torch.float32).min
                else:
                    assert bias[0, h, i, j].item() == pytest.approx(
                        -slopes[h].item() * (i - j)
                    )


def test_bias_head_shared_at_zero_distance():
    bias = alibi_attention_bias(6, 3)
    assert (bias[0, :, range(6), range(6)] == 0).all()


def test_alibi_and_prior_mutually_exclusive():
    with pytest.raises(ValueError):
        PriorLlama(ModelSpec(unsquashed_k=2.0, alibi=True, **TINY))


def test_sdpa_matches_eager_bias_path():
    model = tiny_alibi_model()
    ids = tiny_ids()
    with torch.no_grad():
        sdpa_logits = model(ids)
    eager_logits = model.eager_logits(ids)
    torch.testing.assert_close(sdpa_logits, eager_logits, atol=1e-4, rtol=1e-4)


def test_alibi_changes_the_function():
    # Same weights, bias on vs off: outputs must differ (the bias is real).
    torch.manual_seed(0)
    with_bias = PriorLlama(ModelSpec(unsquashed_k=None, alibi=True, **TINY)).eval()
    torch.manual_seed(0)
    without = PriorLlama(ModelSpec(unsquashed_k=None, alibi=False, **TINY)).eval()
    ids = tiny_ids()
    with torch.no_grad():
        assert not torch.allclose(with_bias(ids), without(ids))


def test_sink_mass_runs_under_alibi():
    model = tiny_alibi_model()
    mass = model.sink_mass(tiny_ids()[:1])
    assert 0.0 <= mass <= 1.0


def test_prior_config_alibi_roundtrip(tmp_path):
    cfg = PriorConfig(k=0.0, kind="alibi", num_heads=9)
    cfg.save(tmp_path)
    loaded = PriorConfig.load(tmp_path)
    assert loaded.kind == "alibi" and loaded.num_heads == 9


def test_prior_config_legacy_file_defaults_to_unsquashed(tmp_path):
    (tmp_path / "unsquash_prior.json").write_text('{"k": 30.0, "lam": 1.0}')
    loaded = PriorConfig.load(tmp_path)
    assert loaded.kind == "unsquashed" and loaded.k == 30.0


def test_prior_config_attention_bias_dispatch():
    alibi_cfg = PriorConfig(k=0.0, kind="alibi", num_heads=4)
    torch.testing.assert_close(
        alibi_cfg.attention_bias(8), alibi_attention_bias(8, 4)
    )
    prior_cfg = PriorConfig(k=2.0, lam=1.0)
    bias = prior_cfg.attention_bias(8)
    assert bias.shape == (1, 1, 8, 8)


@pytest.mark.parametrize("kind", ["alibi", "unsquashed", "none"])
def test_from_pretrained_roundtrip(tmp_path, kind):
    if kind == "alibi":
        model = tiny_alibi_model()
        sidecar = PriorConfig(k=0.0, kind="alibi", num_heads=TINY["num_heads"])
    elif kind == "unsquashed":
        torch.manual_seed(0)
        model = PriorLlama(ModelSpec(unsquashed_k=2.0, **TINY)).eval()
        sidecar = PriorConfig(k=2.0, lam=1.0)
    else:
        torch.manual_seed(0)
        model = PriorLlama(ModelSpec(unsquashed_k=None, **TINY)).eval()
        sidecar = None

    path = tmp_path / kind
    model.to_hf().save_pretrained(path)
    if sidecar is not None:
        sidecar.save(path)

    loaded = PriorLlama.from_pretrained(str(path))
    assert loaded.spec.alibi == (kind == "alibi")
    assert (loaded.spec.unsquashed_k is not None) == (kind == "unsquashed")
    ids = tiny_ids()
    with torch.no_grad():
        torch.testing.assert_close(loaded(ids), model(ids), atol=1e-5, rtol=1e-5)


def test_from_pretrained_extends_buffers(tmp_path):
    model = tiny_alibi_model()
    path = tmp_path / "ckpt"
    model.to_hf().save_pretrained(path)
    PriorConfig(k=0.0, kind="alibi", num_heads=TINY["num_heads"]).save(path)
    longer = PriorLlama.from_pretrained(str(path), max_seq_len=64)
    assert longer.rope_cos.shape[0] == 64
    ids = torch.randint(0, TINY["vocab_size"], (1, 48))
    with torch.no_grad():
        assert longer(ids).shape == (1, 48, TINY["vocab_size"])
