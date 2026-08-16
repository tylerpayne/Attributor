"""Per-head weights for mixing attention heads into one matrix per layer.

Each attention head h contributes to the residual stream through its slice of
the output projection W_O. That slice is a *column* block: W_O has shape
``[hidden_out, n_heads * head_dim]`` and head h owns columns
``h*head_dim : (h+1)*head_dim``. The Frobenius norm of that block is a cheap,
static proxy for how loudly the head speaks into the residual stream.

(The original Attributor reshaped the *rows* of W_O into "heads", which slices
the output dimension instead of the per-head input blocks; the weights it
produced were norms of arbitrary row blocks. This module is the corrected
version.)
"""

from __future__ import annotations

import torch


def _decoder_layers(model):
    for path in ("model.layers", "transformer.h", "gpt_neox.layers"):
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        return list(obj)
    return None


def _o_proj(layer):
    for attn_attr in ("self_attn", "attn", "attention"):
        attn = getattr(layer, attn_attr, None)
        if attn is None:
            continue
        for proj_attr in ("o_proj", "out_proj", "dense", "c_proj"):
            proj = getattr(attn, proj_attr, None)
            if proj is not None and hasattr(proj, "weight"):
                return proj
    return None


def o_proj_head_weights(model) -> list[torch.Tensor] | None:
    """Per-layer ``[n_heads, 1, 1]`` head weights (each layer's sum to 1),
    from the Frobenius norm of each head's column block of W_O.

    Returns None when the architecture isn't recognized; callers should fall
    back to uniform head weights.
    """
    layers = _decoder_layers(model)
    if not layers:
        return None
    n_heads = getattr(model.config, "num_attention_heads", None)
    if not n_heads:
        return None

    weights = []
    for layer in layers:
        proj = _o_proj(layer)
        if proj is None:
            return None
        W = proj.weight.detach()  # [hidden_out, n_heads * head_dim]
        if W.shape[1] % n_heads != 0:
            return None
        head_dim = W.shape[1] // n_heads
        # Split columns into per-head blocks: [hidden_out, n_heads, head_dim]
        blocks = W.reshape(W.shape[0], n_heads, head_dim).permute(1, 0, 2)
        norms = torch.linalg.matrix_norm(blocks.to(torch.float32))
        norms = norms / norms.sum()
        weights.append(norms.reshape(n_heads, 1, 1))
    return weights


def uniform_head_weights(n_layers: int, n_heads: int, device=None) -> list[torch.Tensor]:
    w = torch.full((n_heads, 1, 1), 1.0 / n_heads, dtype=torch.float32, device=device)
    return [w] * n_layers
