"""A minimal Llama (SmolLM2-shaped) with the unsquash prior as an SDPA
additive mask.

The prior is content-independent, so it enters training as one precomputed
``[seq, seq]`` bias tensor (``lam * ln(c_{q-kv})`` below the diagonal, -inf
above) handed to ``scaled_dot_product_attention`` — the masked memory-
efficient kernel adds it tile-by-tile without materializing scores. On an
H100 at 135M shapes this benchmarked at 4.1ms/layer fwd+bwd vs 25.5ms for
FlexAttention with an equivalent ``score_mod`` (the in-kernel table gather is
a ~12x kernel slowdown) and 2.1ms for plain causal, which the no-prior
control uses via ``is_causal=True``.

The mask is cast to the query dtype (SDPA requires it), so bf16 training
quantizes adjacent-lag bias differences into plateaus past lag ~40 (<0.4%
relative steps); evaluation always re-applies the exact fp32 prior.

Module names deliberately mirror ``LlamaForCausalLM``'s ``model.*`` subtree
(embed_tokens, layers.N.self_attn.q_proj, ...), so ``to_hf()`` is a pure
key-prefix rename and checkpoints saved through it load in the existing eval
harness (eager attention + 4D prior mask) with no conversion.

The eager path (``sink_mass``, and ``attention_probs`` in tests) recomputes
attention with the same additive bias; parity between the paths is what
tests/test_pretrain.py pins down.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from unsquash.coefficients import log_unsquash_coefficients


@dataclass
class ModelSpec:
    vocab_size: int = 49152
    hidden_size: int = 576
    intermediate_size: int = 1536
    num_layers: int = 30
    num_heads: int = 9
    num_kv_heads: int = 3
    rope_theta: float = 100000.0
    rms_norm_eps: float = 1e-5
    initializer_range: float = 0.041666666666666664
    max_seq_len: int = 2048
    # None disables the prior (the no-prior control); k defaults to num_layers
    # at build time when prior enabled.
    prior_k: float | None = 30.0
    prior_lam: float = 1.0

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @classmethod
    def from_hf(cls, model_name: str, **overrides) -> "ModelSpec":
        """Spec matching a HF Llama config (e.g. HuggingFaceTB/SmolLM2-135M)."""
        from transformers import AutoConfig

        c = AutoConfig.from_pretrained(model_name)
        rope = getattr(c, "rope_theta", None)
        if rope is None:
            rope = (getattr(c, "rope_parameters", None) or {}).get(
                "rope_theta", 10000.0
            )
        spec = cls(
            vocab_size=c.vocab_size,
            hidden_size=c.hidden_size,
            intermediate_size=c.intermediate_size,
            num_layers=c.num_hidden_layers,
            num_heads=c.num_attention_heads,
            num_kv_heads=c.num_key_value_heads,
            rope_theta=float(rope),
            rms_norm_eps=c.rms_norm_eps,
            initializer_range=c.initializer_range,
        )
        for key, value in overrides.items():
            setattr(spec, key, value)
        return spec


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class _Attention(nn.Module):
    def __init__(self, spec: ModelSpec):
        super().__init__()
        d, hd = spec.hidden_size, spec.head_dim
        self.num_heads, self.num_kv_heads, self.head_dim = (
            spec.num_heads, spec.num_kv_heads, hd,
        )
        self.q_proj = nn.Linear(d, spec.num_heads * hd, bias=False)
        self.k_proj = nn.Linear(d, spec.num_kv_heads * hd, bias=False)
        self.v_proj = nn.Linear(d, spec.num_kv_heads * hd, bias=False)
        self.o_proj = nn.Linear(spec.num_heads * hd, d, bias=False)

    def _qkv(self, x, cos, sin):
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = cos.to(q.dtype), sin.to(q.dtype)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        return q, k, v

    def forward(self, x, cos, sin, bias):
        q, k, v = self._qkv(x, cos, sin)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, is_causal=bias is None, enable_gqa=True
        )
        b, s = x.shape[0], x.shape[1]
        return self.o_proj(out.transpose(1, 2).reshape(b, s, -1))

    def probs(self, x, cos, sin, bias):
        """Eager attention probabilities [b, heads, s, s] (fp32 softmax) under
        the same additive bias; the parity/probe path."""
        q, k, v = self._qkv(x, cos, sin)
        rep = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
        scores = scores.to(torch.float32) + bias
        probs = torch.softmax(scores, dim=-1)
        out = probs.to(v.dtype) @ v
        b, s = x.shape[0], x.shape[1]
        return probs, self.o_proj(out.transpose(1, 2).reshape(b, s, -1))


class _MLP(nn.Module):
    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.gate_proj = nn.Linear(spec.hidden_size, spec.intermediate_size, bias=False)
        self.up_proj = nn.Linear(spec.hidden_size, spec.intermediate_size, bias=False)
        self.down_proj = nn.Linear(spec.intermediate_size, spec.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _Block(nn.Module):
    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.self_attn = _Attention(spec)
        self.mlp = _MLP(spec)
        self.input_layernorm = nn.RMSNorm(spec.hidden_size, eps=spec.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            spec.hidden_size, eps=spec.rms_norm_eps
        )


class PriorLlama(nn.Module):
    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.spec = spec
        self.embed_tokens = nn.Embedding(spec.vocab_size, spec.hidden_size)
        self.layers = nn.ModuleList(_Block(spec) for _ in range(spec.num_layers))
        self.norm = nn.RMSNorm(spec.hidden_size, eps=spec.rms_norm_eps)

        hd = spec.head_dim
        inv_freq = 1.0 / (
            spec.rope_theta ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd)
        )
        pos = torch.arange(spec.max_seq_len, dtype=torch.float32)
        freqs = torch.outer(pos, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("rope_cos", emb.cos(), persistent=False)
        self.register_buffer("rope_sin", emb.sin(), persistent=False)

        if spec.prior_k is not None:
            logc = log_unsquash_coefficients(spec.max_seq_len, spec.prior_k).to(
                torch.float32
            )
        else:
            logc = torch.zeros(spec.max_seq_len)
        self.register_buffer("logc", logc, persistent=False)

        self._bias_cache: dict = {}
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # mirrors HF Llama init so the from-scratch recipe matches SmolLM2's
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.spec.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    # -- attention plumbing ---------------------------------------------------

    def _sdpa_bias(self, seq_len: int, device, dtype) -> torch.Tensor | None:
        """The prior as a cached SDPA ``attn_mask`` in the query dtype, or
        None (plain ``is_causal``) when the prior is disabled."""
        if self.spec.prior_k is None:
            return None
        key = (seq_len, str(device), dtype)
        if key not in self._bias_cache:
            self._bias_cache[key] = self._bias(seq_len, device).to(dtype)
        return self._bias_cache[key]

    def _bias(self, seq_len: int, device) -> torch.Tensor:
        """The eager-path equivalent of block_mask + score_mod: lam*ln(c) on
        and below the diagonal, -inf above."""
        idx = torch.arange(seq_len, device=device)
        m = idx[:, None] - idx[None, :]
        bias = self.spec.prior_lam * self.logc[m.clamp(min=0)]
        return torch.where(m >= 0, bias, torch.tensor(float("-inf"), device=device))

    # -- forward paths --------------------------------------------------------

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        s = input_ids.shape[1]
        cos = self.rope_cos[None, None, :s]
        sin = self.rope_sin[None, None, :s]
        dtype = (
            torch.get_autocast_dtype("cuda")
            if torch.is_autocast_enabled("cuda")
            else self.embed_tokens.weight.dtype
        )
        bias = self._sdpa_bias(s, input_ids.device, dtype)

        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = x + layer.self_attn(
                layer.input_layernorm(x), cos, sin, bias
            )
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        x = self.norm(x)
        logits = F.linear(x, self.embed_tokens.weight)  # tied lm_head

        if labels is None:
            return logits
        loss = F.cross_entropy(
            logits[:, :-1].float().flatten(0, 1), labels[:, 1:].flatten()
        )
        return loss

    @torch.no_grad()
    def eager_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass through the eager/bias path (parity testing)."""
        probs_unused, logits = self._eager_forward(input_ids, want_probs=False)
        return logits

    def _eager_forward(self, input_ids, want_probs: bool):
        s = input_ids.shape[1]
        cos = self.rope_cos[None, None, :s]
        sin = self.rope_sin[None, None, :s]
        bias = self._bias(s, input_ids.device)

        all_probs = [] if want_probs else None
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            probs, attn_out = layer.self_attn.probs(
                layer.input_layernorm(x), cos, sin, bias
            )
            if want_probs:
                all_probs.append(probs)
            x = x + attn_out
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        x = self.norm(x)
        logits = F.linear(x, self.embed_tokens.weight)
        return all_probs, logits

    @torch.no_grad()
    def sink_mass(self, input_ids: torch.Tensor) -> float:
        """Mean attention on position 0 over layers/heads/queries>0 (the same
        probe as the retrofit trainer), via the eager path."""
        total, count = 0.0, 0
        s = input_ids.shape[1]
        cos = self.rope_cos[None, None, :s]
        sin = self.rope_sin[None, None, :s]
        bias = self._bias(s, input_ids.device)
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            probs, attn_out = layer.self_attn.probs(
                layer.input_layernorm(x), cos, sin, bias
            )
            total += float(probs[..., 1:, 0].mean())
            count += 1
            x = x + attn_out
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        return total / max(1, count)

    # -- export ---------------------------------------------------------------

    def to_hf(self, hf_config=None):
        """An equivalent ``LlamaForCausalLM`` (for save_pretrained; loads in
        the eval harness with the prior delivered as a 4D mask)."""
        from transformers import LlamaConfig, LlamaForCausalLM

        spec = self.spec
        if hf_config is None:
            hf_config = LlamaConfig(
                vocab_size=spec.vocab_size,
                hidden_size=spec.hidden_size,
                intermediate_size=spec.intermediate_size,
                num_hidden_layers=spec.num_layers,
                num_attention_heads=spec.num_heads,
                num_key_value_heads=spec.num_kv_heads,
                max_position_embeddings=max(spec.max_seq_len, 2048),
                rope_theta=spec.rope_theta,
                rms_norm_eps=spec.rms_norm_eps,
                initializer_range=spec.initializer_range,
                tie_word_embeddings=True,
                attention_bias=False,
                mlp_bias=False,
            )
        hf = LlamaForCausalLM(hf_config)
        state = {f"model.{k}": v for k, v in self.state_dict().items()}
        missing, unexpected = hf.load_state_dict(state, strict=False)
        missing = [k for k in missing if k != "lm_head.weight"]  # tied
        if missing or unexpected:
            raise RuntimeError(f"HF export mismatch: {missing=} {unexpected=}")
        hf.tie_weights()
        return hf
