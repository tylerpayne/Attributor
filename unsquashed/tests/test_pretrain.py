"""Parity tests for the FlexAttention pretraining model.

The load-bearing invariant: the flex path (score_mod prior) and the eager
path (additive 4D bias) are the same function, and the HF export of a
FlexLlama is the same function again under the eval harness's mask. If these
hold at tiny scale in fp32 they hold at 135M, because every component is
shape-generic.
"""

import pytest
import torch

from unsquash.pretrain.model import FlexLlama, ModelSpec
from unsquash.prior import prior_attention_bias

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


def tiny_model(prior_k=2.0, seed=0):
    torch.manual_seed(seed)
    return FlexLlama(ModelSpec(prior_k=prior_k, **TINY)).eval()


def tiny_ids(seed=1):
    torch.manual_seed(seed)
    return torch.randint(0, TINY["vocab_size"], (2, TINY["max_seq_len"]))


def test_flex_matches_eager_bias_path():
    model = tiny_model()
    ids = tiny_ids()
    with torch.no_grad():
        flex_logits = model(ids)
    eager_logits = model.eager_logits(ids)
    torch.testing.assert_close(flex_logits, eager_logits, atol=1e-4, rtol=1e-4)


def test_hf_export_matches_under_prior_mask():
    model = tiny_model()
    hf = model.to_hf()
    hf.eval()
    ids = tiny_ids()
    mask = prior_attention_bias(ids.shape[1], k=2.0, lam=1.0)
    with torch.no_grad():
        ours = model(ids)
        theirs = hf(input_ids=ids, attention_mask=mask, use_cache=False).logits
    torch.testing.assert_close(ours, theirs, atol=1e-4, rtol=1e-4)


def test_hf_export_no_prior_matches_plain_causal():
    model = tiny_model(prior_k=None)
    hf = model.to_hf()
    hf.eval()
    ids = tiny_ids()
    with torch.no_grad():
        ours = model(ids)
        theirs = hf(input_ids=ids, use_cache=False).logits
    torch.testing.assert_close(ours, theirs, atol=1e-4, rtol=1e-4)


def test_export_ties_embeddings():
    hf = tiny_model().to_hf()
    assert hf.lm_head.weight.data_ptr() == hf.model.embed_tokens.weight.data_ptr()


def test_probs_are_row_normalized_and_causal():
    model = tiny_model()
    ids = tiny_ids()
    s = ids.shape[1]
    cos = model.rope_cos[None, None, :s]
    sin = model.rope_sin[None, None, :s]
    bias = model._bias(s, ids.device)
    x = model.embed_tokens(ids)
    probs, _ = model.layers[0].self_attn.probs(
        model.layers[0].input_layernorm(x), cos, sin, bias
    )
    torch.testing.assert_close(
        probs.sum(-1), torch.ones_like(probs.sum(-1)), atol=1e-5, rtol=1e-5
    )
    assert float(probs.detach().triu(1).abs().max()) == 0.0


def test_sink_mass_in_unit_interval():
    model = tiny_model()
    sink = model.sink_mass(tiny_ids()[:1])
    assert 0.0 <= sink <= 1.0


def test_loss_decreases_on_overfit_batch():
    model = tiny_model()
    model.train()
    ids = tiny_ids()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = None
    for _ in range(20):
        try:
            loss = model(ids, labels=ids)
            opt.zero_grad()
            loss.backward()
        except NotImplementedError:
            pytest.skip("flex_attention has no CPU backward; needs CUDA")
        opt.step()
        if first is None:
            first = float(loss)
    assert float(loss) < first


def test_spec_from_hf_reads_nested_rope():
    class FakeConfig:
        vocab_size = 64
        hidden_size = 32
        intermediate_size = 64
        num_hidden_layers = 2
        num_attention_heads = 4
        num_key_value_heads = 2
        rms_norm_eps = 1e-5
        initializer_range = 0.02
        rope_theta = None
        rope_parameters = {"rope_theta": 100000.0}

    import unsquash.pretrain.model as m

    orig = m.ModelSpec.from_hf.__func__
    try:
        import transformers

        real = transformers.AutoConfig.from_pretrained
        transformers.AutoConfig.from_pretrained = staticmethod(
            lambda name: FakeConfig()
        )
        spec = ModelSpec.from_hf("fake")
        assert spec.rope_theta == 100000.0
    finally:
        transformers.AutoConfig.from_pretrained = real
