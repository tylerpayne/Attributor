import torch

from unsquash.rollout import RolloutAttributor


def uniform_attention_layers(n_layers, n_heads, n):
    """The null model: every head attends uniformly over past + self."""
    A = torch.tril(torch.ones(n, n, dtype=torch.float32))
    A = A / A.sum(dim=-1, keepdim=True)
    return [A.expand(1, n_heads, n, n).clone() for _ in range(n_layers)]


def flatness(Y, row=-1):
    r = Y[row]
    r = r[r > 0]
    return float(r.max() / r.min())


def make_attributor():
    # attribute() never touches the model/tokenizer; capture() does.
    return RolloutAttributor(None, None, head_weighting="uniform")


def test_unsquashed_flattens_the_null_model():
    attributor = make_attributor()
    attentions = uniform_attention_layers(8, 2, 64)

    vanilla = attributor.attribute(attentions, "rollout", residual=False)
    unsquashed = attributor.attribute(attentions, "unsquashed")

    # Vanilla rollout piles mass on early tokens like m^(k-1); the unsquash
    # correction removes the structural bias almost entirely.
    assert flatness(vanilla) > 1e6
    assert flatness(unsquashed) < 10


def test_residual_defaults():
    attributor = make_attributor()
    attentions = uniform_attention_layers(4, 2, 16)
    # rollout defaults to residual=True, unsquashed to residual=False;
    # overriding must change the result.
    r_default = attributor.attribute(attentions, "rollout")
    r_off = attributor.attribute(attentions, "rollout", residual=False)
    assert not torch.allclose(r_default, r_off)
    u_default = attributor.attribute(attentions, "unsquashed")
    u_off = attributor.attribute(attentions, "unsquashed", residual=False)
    assert torch.allclose(u_default, u_off)


def test_rows_are_distributions_and_shifted():
    attributor = make_attributor()
    attentions = uniform_attention_layers(4, 2, 16)
    for method in ("attention_sum", "rollout", "unsquashed"):
        Y = attributor.attribute(attentions, method)
        assert Y.shape == (16, 16)
        assert torch.all(Y[0] == 0)  # first token has no source
        sums = Y[1:].sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-8)
        # Token t is generated from stream t-1: no self- or future-attribution.
        assert torch.all(torch.triu(Y, diagonal=1)[1:, 2:] == 0)


def test_head_weights_change_attribution():
    n, n_heads = 12, 2
    torch.manual_seed(0)
    # Two very different heads: local vs long-range.
    local = torch.eye(n) + 0.01
    longrange = torch.ones(n, n)
    heads = torch.stack([torch.tril(local), torch.tril(longrange)])
    heads = heads / heads.sum(dim=-1, keepdim=True)
    attentions = [heads.unsqueeze(0)] * 3

    uniform = RolloutAttributor(None, None, head_weighting="uniform")
    Y_uniform = uniform.attribute(attentions, "rollout")

    weighted = RolloutAttributor(None, None, head_weighting="uniform")
    weighted._head_weights = [
        torch.tensor([[[0.9]], [[0.1]]]) for _ in range(3)
    ]
    Y_weighted = weighted.attribute(attentions, "rollout")
    assert not torch.allclose(Y_uniform, Y_weighted)


def test_o_proj_head_weights_correct_shape_and_axis(tiny_model):
    from unsquash.heads import o_proj_head_weights

    weights = o_proj_head_weights(tiny_model)
    assert weights is not None
    n_layers = tiny_model.config.num_hidden_layers
    n_heads = tiny_model.config.num_attention_heads
    assert len(weights) == n_layers
    for w in weights:
        assert w.shape == (n_heads, 1, 1)
        assert torch.isclose(w.sum(), torch.tensor(1.0))
        assert torch.all(w > 0)

    # Axis check: scaling ONE head's column block of W_O must change only
    # that head's weight share, and increase it.
    layer0 = tiny_model.model.layers[0].self_attn.o_proj
    head_dim = layer0.weight.shape[1] // n_heads
    with torch.no_grad():
        layer0.weight[:, :head_dim] *= 10
    scaled = o_proj_head_weights(tiny_model)
    assert scaled[0][0] > weights[0][0]
    with torch.no_grad():  # restore
        layer0.weight[:, :head_dim] /= 10
