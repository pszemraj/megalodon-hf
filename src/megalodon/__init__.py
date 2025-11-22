# coding: utf-8
"""
megalodon: pure-PyTorch Megalodon (decoder-only) with EMA + chunked attention
"""

from typing import Optional

import torch

from .configuration_megalodon import MegalodonConfig, MegalodonDefaults
from .modeling_megalodon import (
    AttentionCache,
    LayerCache,
    MegalodonForCausalLM,
    MegalodonModel,
)


def configure_precision(
    *,
    allow_tf32: Optional[bool] = True,
    allow_bf16_reduced_precision_reduction: Optional[bool] = None,
) -> None:
    """Set recommended backend precision toggles for Megalodon workloads.

    :param allow_tf32: Whether TF32 matmuls are permitted (defaults to ``True``).
    :param allow_bf16_reduced_precision_reduction: Toggle cuBLAS reduced-precision
        reductions for BF16 matmuls. ``None`` leaves the PyTorch default in place.
    """
    if torch.cuda.is_available():
        if allow_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
            torch.backends.cudnn.allow_tf32 = allow_tf32
        if allow_bf16_reduced_precision_reduction is not None:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
                allow_bf16_reduced_precision_reduction
            )


__all__ = [
    "MegalodonConfig",
    "MegalodonDefaults",
    "MegalodonModel",
    "MegalodonForCausalLM",
    "AttentionCache",
    "LayerCache",
    "configure_precision",
]

__version__ = "0.1.0"
