"""Unsquashed attention rollout: de-biased attention attribution.

- ``unsquash.coefficients`` — the k-th convolution root of the ones sequence
- ``unsquash.rollout`` — attribution methods (attention_sum / rollout / unsquashed)
- ``unsquash.prior`` — the log-distance prior as an attention-logit bias
- ``unsquash.alibi`` — the ALiBi linear-distance bias (pretraining control)
- ``unsquash.eval`` — HotpotQA A/B evaluation harness (``python -m unsquash.eval``)
- ``unsquash.train`` — continued-pretraining retrofit (``python -m unsquash.train``)
- ``unsquash.ladder`` — long-context benchmark ladder (``python -m unsquash.ladder``)
"""

from unsquash.alibi import alibi_attention_bias, alibi_slopes
from unsquash.coefficients import (
    log_unsquash_coefficients,
    unsquash_coefficients,
    unsquash_factor,
)
from unsquash.prior import PriorConfig, unsquashed_attention_bias
from unsquash.rollout import METHODS, RolloutAttributor

__all__ = [
    "METHODS",
    "PriorConfig",
    "RolloutAttributor",
    "alibi_attention_bias",
    "alibi_slopes",
    "log_unsquash_coefficients",
    "unsquashed_attention_bias",
    "unsquash_coefficients",
    "unsquash_factor",
]
