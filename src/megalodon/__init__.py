# coding=utf-8
# Copyright 2025 Peter Szemraj.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""megalodon: pure-PyTorch Megalodon (decoder-only) with EMA + chunked attention."""

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

    :param Optional[bool] allow_tf32: Whether TF32 matmuls are permitted (defaults to ``True``).
    :param Optional[bool] allow_bf16_reduced_precision_reduction: Toggle cuBLAS reduced-precision reductions for BF16 matmuls, defaults to ``None`` (leaves the PyTorch default in place).
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

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.0.0+unknown"
