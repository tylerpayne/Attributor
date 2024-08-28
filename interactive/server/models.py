from typing import Literal
from pydantic import BaseModel

class Flow(BaseModel):
    inputs: list[list[float]]
    """The inputs to this flow component. A length N sequence of D-dimensional vectors"""
    flow: list[list[float]]
    """How this component exchanges information from outputs to inputs. A N x N matrix."""
    outputs: list[list[float]]
    """The outputs of this flow component after applying the flow matrix to the input. A length N sequence of D-dimensional vectors"""

class AttentionHead(Flow):
    pass

class Layer(Flow):
    attention_heads: list[AttentionHead]
    """The k attention heads in this layer."""
    attention_head_weights: list[float]
    """The weight of each attention head"""
    pre_residual: list[list[float]]
    """The length N sequence of D dimensional vectors after composing the K attention heads."""

class Model(Flow):
    layers: list[Layer]
    """The L layers of this model."""

class ModelParms(BaseModel):
    tokens: list[str]
    nr_layers: int
    nr_attention_heads: int

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ConfigParams(BaseModel):
    model: str
    device_map: str
    dtype: str
    max_context_tokens: int