"""Attention-attribution methods over a single captured forward pass.

Three methods, all computed from the same attention tensors so they can be
A/B'd without re-running the model:

- ``attention_sum``: raw attention summed over layers and heads (the weakest
  baseline; no notion of composition).
- ``rollout``: classic attention rollout (Abnar & Zuidema, 2020) — per layer,
  head-mixed attention plus residual, composed across layers.
- ``unsquashed``: rollout with each layer's attention Hadamard-multiplied by
  the unsquash factor c_{i-j} (k = number of layers) before composing. This
  removes the structural early-token pile-up that vanilla rollout fabricates:
  under the uniform-attention null, the composition of corrected layers is
  exactly the all-ones causal matrix, i.e. uniform attribution.

Attribution convention: the returned matrix ``Y`` is ``[n, n]`` with ``Y[t, j]``
the amount source position ``j`` contributed to *generating* the token at
position ``t``. (The token at position t is predicted from the residual stream
at position t-1, so rows are shifted by one relative to the raw rollout; row 0
is zero.)
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch

from unsquash.coefficients import unsquash_factor
from unsquash.heads import o_proj_head_weights, uniform_head_weights
from unsquash.prior import PriorConfig

METHODS = ("attention_sum", "rollout", "unsquashed")

# Method defaults for the residual term in the composition. Classic rollout
# includes it; the unsquash correction is derived for the attention-only
# composition (with the residual, the null changes and the same coefficients
# overcorrect).
_RESIDUAL_DEFAULT = {"rollout": True, "unsquashed": False}


def _row_normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.sum(dim=-1, keepdim=True).clamp(min=torch.finfo(x.dtype).tiny)


class RolloutAttributor:
    """Captures attentions for a token sequence and computes attributions.

    Parameters
    ----------
    model, tokenizer:
        A Hugging Face causal LM (loaded with ``attn_implementation="eager"``
        so ``output_attentions=True`` returns real attention probabilities)
        and its tokenizer.
    head_weighting:
        ``"o_proj_norm"`` weights each head by the Frobenius norm of its
        column block of W_O (falls back to uniform with a warning if the
        architecture isn't recognized); ``"uniform"`` averages heads.
    prior:
        Optional :class:`PriorConfig`. When set, the log-distance prior bias
        is applied *during capture* (for models retrofitted with
        ``unsquash.train``); attribution methods then see the model's real,
        prior-included attention distributions.
    """

    def __init__(
        self,
        model,
        tokenizer,
        *,
        head_weighting: str = "o_proj_norm",
        prior: PriorConfig | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.prior = prior
        self._head_weights = None
        if head_weighting == "o_proj_norm":
            self._head_weights = o_proj_head_weights(model)
            if self._head_weights is None:
                import warnings

                warnings.warn(
                    "Could not locate per-head output projections for "
                    f"{type(model).__name__}; falling back to uniform head weights."
                )
        elif head_weighting != "uniform":
            raise ValueError(f"Unknown head_weighting: {head_weighting!r}")

    @torch.no_grad()
    def capture(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """One forward pass; returns per-layer attention tensors
        ``[1, n_heads, n, n]``."""
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.shape[0] != 1:
            raise ValueError("Only batch size 1 is supported")
        input_ids = input_ids.to(self.model.device)

        attention_mask = None
        if self.prior is not None:
            # Dispatches on the recorded kind: unsquash log-distance prior or
            # the ALiBi linear-distance control, either way applied during
            # capture so attribution sees the model's real distributions.
            attention_mask = self.prior.attention_bias(
                input_ids.shape[1],
                dtype=self.model.dtype,
                device=self.model.device,
            )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            use_cache=False,
        )
        return outputs.attentions

    def _mixed_layer(self, attention: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Head-mix one layer's ``[1, n_heads, n, n]`` attention into a
        row-stochastic ``[n, n]`` float64 matrix."""
        A = attention.squeeze(0).to(torch.float64)
        if self._head_weights is not None:
            w = self._head_weights[layer_idx].to(A.device, torch.float64)
            A = (A * w).sum(dim=0)
        else:
            A = A.mean(dim=0)
        return _row_normalize(A)

    def attribute(
        self,
        attentions: Sequence[torch.Tensor],
        method: str = "unsquashed",
        *,
        residual: bool | None = None,
        k: float | None = None,
    ) -> torch.Tensor:
        """Compute the ``[n, n]`` attribution matrix from captured attentions."""
        if method not in METHODS:
            raise ValueError(f"Unknown method {method!r}; expected one of {METHODS}")
        n_layers = len(attentions)
        n = attentions[0].shape[-1]
        device = attentions[0].device

        if method == "attention_sum":
            Y = torch.zeros((n, n), dtype=torch.float64, device=device)
            for i, attn in enumerate(attentions):
                Y += self._mixed_layer(attn, i)
            Y = _row_normalize(Y)
        else:
            if residual is None:
                residual = _RESIDUAL_DEFAULT[method]
            F = None
            if method == "unsquashed":
                F = unsquash_factor(n, k or n_layers, device=device)
            Y = torch.eye(n, dtype=torch.float64, device=device)
            for i, attn in enumerate(attentions):
                A = self._mixed_layer(attn, i)
                if F is not None:
                    A = _row_normalize(A * F)
                Y_next = A @ Y
                if residual:
                    Y_next = Y_next + Y
                Y = _row_normalize(Y_next)

        # Token at position t is generated from the stream at position t-1.
        Y = torch.roll(Y, 1, 0)
        Y[0, :] = 0
        return Y

    @torch.no_grad()
    def __call__(
        self,
        input_ids: torch.Tensor,
        methods: Iterable[str] = ("unsquashed",),
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Capture once, attribute with each requested method."""
        attentions = self.capture(input_ids)
        return {m: self.attribute(attentions, m, **kwargs) for m in methods}
