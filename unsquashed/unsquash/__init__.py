"""Unsquashed attention rollout: de-biased attention attribution.

- ``unsquash.coefficients`` — the k-th convolution root of the ones sequence
- ``unsquash.rollout`` — attribution methods (attention_sum / rollout / unsquashed)
- ``unsquash.prior`` — the log-distance prior as an attention-logit bias
- ``unsquash.eval`` — HotpotQA A/B evaluation harness (``python -m unsquash.eval``)
- ``unsquash.train`` — continued-pretraining retrofit (``python -m unsquash.train``)
"""

from unsquash.coefficients import (
    log_unsquash_coefficients,
    unsquash_coefficients,
    unsquash_factor,
)
from unsquash.prior import PriorConfig, prior_attention_bias
from unsquash.rollout import METHODS, RolloutAttributor

__all__ = [
    "METHODS",
    "PriorConfig",
    "RolloutAttributor",
    "log_unsquash_coefficients",
    "prior_attention_bias",
    "unsquash_coefficients",
    "unsquash_factor",
]
