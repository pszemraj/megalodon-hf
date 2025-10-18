# coding: utf-8
"""
megalodon: pure-PyTorch Megalodon (decoder-only) with EMA + chunked attention
"""

from .configuration_megalodon import MegalodonConfig, MegalodonDefaults
from .modeling_megalodon import (
    AttentionCache,
    LayerCache,
    MegalodonForCausalLM,
    MegalodonModel,
)

__all__ = [
    "MegalodonConfig",
    "MegalodonDefaults",
    "MegalodonModel",
    "MegalodonForCausalLM",
    "AttentionCache",
    "LayerCache",
]

__version__ = "0.1.0"
