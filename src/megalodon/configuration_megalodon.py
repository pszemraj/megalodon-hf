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
"""Configuration for the decoder-only Megalodon model.

Clean, Torch-first configuration mirrors the knobs used by the original
implementation while remaining free of CUDA-specific requirements.
Use together with ``modeling_megalodon.py``.

Defaults mirror the 200M reference setup noted in ``README.md``; call
``MegalodonConfig.from_7b_setup()`` to reproduce the paper's 7B recipe.

References:
    Paper: https://arxiv.org/abs/2404.08801
    Original Megalodon repo: https://github.com/XuezheMax/megalodon

Example:
    >>> from megalodon import MegalodonConfig
    >>> cfg = MegalodonConfig(vocab_size=50_000, model_dim=768, num_layers=24, num_heads=8)
    >>> cfg.model_type
    'megalodon'
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


@dataclass
class MegalodonDefaults:
    """Default Megalodon hyperparameters mirroring the original 200M architecture.

    :cvar vocab_size: Token vocabulary size used for embeddings.
    :cvar model_dim: Transformer hidden width ``D``.
    :cvar num_layers: Number of decoder blocks stacked in the network.
    :cvar num_heads: Count of attention heads per block.
    :cvar z_dim: Shared query/key projection width (divisible by ``num_heads``).
    :cvar value_dim: Value projection width (divisible by ``num_heads``).
    :cvar ffn_hidden_dim: Hidden dimension inside the feed-forward network.
    :cvar cema_ndim: Complex exponential moving average components per channel.
    :cvar chunk_size: Streaming attention chunk length.
    :cvar max_cache_len: Maximum number of tokens to retain in the streaming KV cache.
      ``None`` means "auto" and defaults to ``chunk_size`` unless ``cache_unbounded=True``.
    :cvar cache_unbounded: Disable KV cache clamping regardless of ``max_cache_len`` (use with caution; memory grows linearly with tokens).
    :cvar norm_num_groups: Group count for TimestepNorm.
    :cvar dropout: Dropout probability applied to residual outputs.
    :cvar attention_dropout: Dropout applied to attention logits.
    :cvar hidden_dropout: Dropout applied within FFN and EMA branches.
    :cvar swiglu: Flag indicating whether to use a SwiGLU FFN.
    :cvar rescale_nffn: Flag enabling residual rescaling in the FFN.
    :cvar scale_emb: Flag controlling ``sqrt(model_dim)`` embedding scaling.
    :cvar share_emb: Whether to share input embeddings with the LM head.
    :cvar efficient_attn: Optional identifier for custom attention backends.
    :cvar norm_affine: Whether normalization layers include affine parameters.
    :cvar norm_eps: Epsilon used by normalization layers.
    :cvar init_mode: Weight initialisation scheme, matching :class:`InitMode`.
    :cvar max_positions: Maximum rotary cache length.
    :cvar rope_base: Optional RoPE base (``None`` falls back to ``10_000``; use ``100_000`` via :meth:`from_7b_setup` for the paper setup).
    :cvar output_size: LM head width override (``-1`` ties to ``vocab_size``).
    :cvar pad_token_id: Padding token identifier.
    :cvar bos_token_id: Beginning-of-sequence token identifier.
    :cvar eos_token_id: End-of-sequence token identifier.
    :cvar gradient_checkpointing: Flag toggling block-level checkpointing.
    """

    vocab_size: int = 32_000  # ex: load from unsloth/llama-2-7b-chat
    model_dim: int = 1024
    num_layers: int = 12
    num_heads: int = 1
    z_dim: int = 256
    value_dim: int = 2048
    ffn_hidden_dim: int = 2560
    cema_ndim: int = 16
    chunk_size: int = 2048
    # ``None`` is interpreted as "auto" and defaults to ``chunk_size`` in MegalodonConfig.
    max_cache_len: Optional[int] = None
    cache_unbounded: bool = False
    norm_num_groups: int = 32
    dropout: float = 0.0
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    swiglu: bool = False
    rescale_nffn: bool = False
    scale_emb: bool = False
    share_emb: bool = False
    efficient_attn: Optional[str] = None
    norm_affine: bool = True
    norm_eps: float = 1e-5
    init_mode: InitMode = "he"
    max_positions: int = 1_000_000
    rope_base: Optional[float] = None
    output_size: int = -1
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    gradient_checkpointing: bool = False


class MegalodonConfig(PretrainedConfig):
    """Configuration container for decoder-only Megalodon models.

    Defaults target the 200M reference variant described in ``README.md``; use
    :meth:`from_7b_setup` to jump to the paper's 7B hyper-parameters while
    keeping the same API surface.

    :ivar vocab_size: Token vocabulary size expected by the decoder.
    :ivar model_dim: Transformer hidden width ``D``.
    :ivar num_layers: Number of decoder blocks stacked in the network.
    :ivar num_heads: Count of attention heads per block.
    :ivar num_attention_heads: Alias maintained for Hugging Face compatibility.
    :ivar z_dim: Shared query/key projection width (divisible by ``num_heads``).
    :ivar value_dim: Value projection width (divisible by ``num_heads``).
    :ivar ffn_hidden_dim: Hidden dimension inside the feed-forward network.
    :ivar cema_ndim: Complex EMA components per channel.
    :ivar chunk_size: Streaming attention chunk length.
    :ivar max_cache_len: KV cache horizon for streaming decode (defaults to ``chunk_size``).
    :ivar cache_unbounded: Disable KV cache clamping (opt-in extension).
    :ivar norm_num_groups: Group count for TimestepNorm.
    :ivar dropout: Dropout probability applied to residual outputs.
    :ivar attention_dropout: Dropout applied to attention logits.
    :ivar hidden_dropout: Dropout applied within FFN and EMA branches.
    :ivar swiglu: Flag indicating whether to use a SwiGLU FFN.
    :ivar rescale_nffn: Flag enabling residual rescaling in the FFN.
    :ivar scale_emb: Flag controlling ``sqrt(model_dim)`` embedding scaling.
    :ivar share_emb: Whether to share input embeddings with the LM head.
    :ivar efficient_attn: Optional identifier for custom attention backends.
    :ivar norm_affine: Whether normalization layers include affine parameters.
    :ivar norm_eps: Epsilon used by normalization layers.
    :ivar init_mode: Weight initialisation scheme, matching :class:`InitMode`.
    :ivar max_positions: Maximum rotary cache length.
    :ivar rope_base: Optional base frequency for rotary embeddings (``None`` uses the modeling default).
    :ivar output_size: LM head width override (``-1`` ties to ``vocab_size``).
    :ivar gradient_checkpointing: Flag toggling block-level checkpointing.
    :ivar is_decoder: Whether the module should behave as a decoder during generation.
    :ivar use_cache: Cache flag required by Hugging Face generation APIs.
    """

    model_type = "megalodon"

    def __init__(
        self,
        vocab_size: int = MegalodonDefaults.vocab_size,
        model_dim: int = MegalodonDefaults.model_dim,
        num_layers: int = MegalodonDefaults.num_layers,
        num_heads: int = MegalodonDefaults.num_heads,
        z_dim: int = MegalodonDefaults.z_dim,
        value_dim: int = MegalodonDefaults.value_dim,
        ffn_hidden_dim: int = MegalodonDefaults.ffn_hidden_dim,
        cema_ndim: int = MegalodonDefaults.cema_ndim,
        chunk_size: int = MegalodonDefaults.chunk_size,
        max_cache_len: Optional[int] = MegalodonDefaults.max_cache_len,
        cache_unbounded: bool = MegalodonDefaults.cache_unbounded,
        norm_num_groups: int = MegalodonDefaults.norm_num_groups,
        dropout: float = MegalodonDefaults.dropout,
        attention_dropout: float = MegalodonDefaults.attention_dropout,
        hidden_dropout: float = MegalodonDefaults.hidden_dropout,
        swiglu: bool = MegalodonDefaults.swiglu,
        rescale_nffn: bool = MegalodonDefaults.rescale_nffn,
        scale_emb: bool = MegalodonDefaults.scale_emb,
        share_emb: bool = MegalodonDefaults.share_emb,
        efficient_attn: Optional[str] = MegalodonDefaults.efficient_attn,
        norm_affine: bool = MegalodonDefaults.norm_affine,
        norm_eps: float = MegalodonDefaults.norm_eps,
        init_mode: InitMode = MegalodonDefaults.init_mode,
        max_positions: int = MegalodonDefaults.max_positions,
        rope_base: Optional[float] = MegalodonDefaults.rope_base,
        output_size: int = MegalodonDefaults.output_size,
        pad_token_id: int = MegalodonDefaults.pad_token_id,
        bos_token_id: int = MegalodonDefaults.bos_token_id,
        eos_token_id: int = MegalodonDefaults.eos_token_id,
        gradient_checkpointing: bool = MegalodonDefaults.gradient_checkpointing,
        **kwargs: Any,
    ) -> None:
        """Populate the Megalodon configuration with decoder hyper-parameters.

        :param vocab_size: Size of the tokenizer vocabulary.
        :type vocab_size: int
        :param model_dim: Transformer hidden size ``D``.
        :type model_dim: int
        :param num_layers: Number of decoder blocks.
        :type num_layers: int
        :param num_heads: Number of attention heads.
        :type num_heads: int
        :param z_dim: Shared Q/K projection width (must divide ``num_heads``).
        :type z_dim: int
        :param value_dim: Value projection width (must divide ``num_heads``).
        :type value_dim: int
        :param ffn_hidden_dim: Hidden dimension inside the feed-forward network.
        :type ffn_hidden_dim: int
        :param cema_ndim: Number of complex EMA channels per hidden unit.
        :type cema_ndim: int
        :param chunk_size: Maximum chunk processed by streaming self-attention.
        :type chunk_size: int
        :param max_cache_len: Maximum KV length retained during streaming decode.
          If ``None``, defaults to ``chunk_size`` unless ``cache_unbounded=True``.
        :type max_cache_len: Optional[int]
        :param cache_unbounded: Disable KV cache clamping regardless of ``max_cache_len`` (VRAM grows linearly with tokens).
        :type cache_unbounded: bool
        :param norm_num_groups: Groups used by timestep normalization.
        :type norm_num_groups: int
        :param dropout: Dropout applied to residual outputs.
        :type dropout: float
        :param attention_dropout: Dropout applied to attention probabilities.
        :type attention_dropout: float
        :param hidden_dropout: Dropout applied to intermediate projections.
        :type hidden_dropout: float
        :param swiglu: Whether to use a SwiGLU feed-forward block.
        :type swiglu: bool
        :param rescale_nffn: Enable layer-wise residual rescaling in the FFN.
        :type rescale_nffn: bool
        :param scale_emb: Multiply input embeddings by ``sqrt(model_dim)``.
        :type scale_emb: bool
        :param share_emb: Maintain compatibility with configs that toggle weight tying.
        :type share_emb: bool
        :param efficient_attn: Placeholder for upstream efficient kernels (unused).
        :type efficient_attn: Optional[str]
        :param norm_affine: Include affine parameters in normalization layers.
        :type norm_affine: bool
        :param norm_eps: Epsilon used by timestep and RMS norms.
        :type norm_eps: float
        :param init_mode: Scheme used to initialize linear layers.
        :type init_mode: InitMode
        :param max_positions: Maximum number of rotary positions cached.
        :type max_positions: int
        :param rope_base: RoPE base (``None`` => ``10_000`` default; ``100_000`` via :meth:`from_7b_setup`).
        :type rope_base: Optional[float]
        :param output_size: Optional LM head size override (``-1`` ties to vocab).
        :type output_size: int
        :param pad_token_id: Padding token id.
        :type pad_token_id: int
        :param bos_token_id: Beginning-of-sequence token id.
        :type bos_token_id: int
        :param eos_token_id: End-of-sequence token id.
        :type eos_token_id: int
        :param gradient_checkpointing: Enable block-level gradient checkpointing.
        :type gradient_checkpointing: bool
        :raises ValueError: If ``layerwise_ckpt`` is supplied (removed; use ``gradient_checkpointing``).
        :raises ValueError: If ``z_dim`` or ``value_dim`` are not divisible by ``num_heads``.
        :raises ValueError: If ``norm_num_groups`` does not divide ``model_dim``.
        :raises ValueError: If ``norm_eps`` is not strictly positive.
        """
        if "layerwise_ckpt" in kwargs:
            raise ValueError(
                "`layerwise_ckpt` has been removed; use `gradient_checkpointing` instead."
            )
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

        # Core dims & architecture
        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_attention_heads = num_heads  # HF compatibility
        self.z_dim = z_dim
        self.value_dim = value_dim
        self.ffn_hidden_dim = ffn_hidden_dim
        self.cema_ndim = cema_ndim

        # Streaming / chunked attention
        self.chunk_size = chunk_size
        self.cache_unbounded = bool(cache_unbounded)
        self.max_cache_len = (
            None
            if self.cache_unbounded
            else chunk_size
            if max_cache_len is None
            else max_cache_len
        )
        self.max_positions = max_positions
        self.rope_base = rope_base
        self.efficient_attn = efficient_attn

        # Normalization
        self.norm_num_groups = norm_num_groups
        self.norm_affine = norm_affine
        self.norm_eps = norm_eps

        # Dropouts
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout

        # FFN / residual tweaks
        self.swiglu = swiglu
        self.rescale_nffn = rescale_nffn

        # Embedding behavior & tying
        self.scale_emb = scale_emb
        self.share_emb = share_emb  # present for parity; tying done in modeling

        # Initialization mode
        self.init_mode = init_mode

        # Output projection
        self.output_size = output_size

        # Decoder-only flags
        self.is_decoder = True
        self.use_cache = True

        # Training memory
        self.gradient_checkpointing = gradient_checkpointing

        # Sanity checks (mirror modeling expectations)
        # Positivity checks for core dimensions
        if self.vocab_size <= 0:
            raise ValueError(f"`vocab_size` must be positive, got {self.vocab_size}.")
        if self.model_dim <= 0:
            raise ValueError(f"`model_dim` must be positive, got {self.model_dim}.")
        if self.num_layers <= 0:
            raise ValueError(f"`num_layers` must be positive, got {self.num_layers}.")
        if self.num_heads <= 0:
            raise ValueError(f"`num_heads` must be positive, got {self.num_heads}.")
        if self.z_dim <= 0:
            raise ValueError(f"`z_dim` must be positive, got {self.z_dim}.")
        if self.value_dim <= 0:
            raise ValueError(f"`value_dim` must be positive, got {self.value_dim}.")
        if self.ffn_hidden_dim <= 0:
            raise ValueError(
                f"`ffn_hidden_dim` must be positive, got {self.ffn_hidden_dim}."
            )
        if self.cema_ndim <= 0:
            raise ValueError(f"`cema_ndim` must be positive, got {self.cema_ndim}.")
        if self.chunk_size <= 0:
            raise ValueError(f"`chunk_size` must be positive, got {self.chunk_size}.")
        if self.max_positions <= 0:
            raise ValueError(
                f"`max_positions` must be positive, got {self.max_positions}."
            )
        if self.rope_base is not None and self.rope_base <= 0:
            raise ValueError(
                f"`rope_base` must be positive when provided, got {self.rope_base}."
            )

        # Divisibility checks
        if self.z_dim % self.num_heads != 0:
            raise ValueError(
                f"`z_dim` ({self.z_dim}) must be divisible by `num_heads` ({self.num_heads})."
            )
        if self.value_dim % self.num_heads != 0:
            raise ValueError(
                f"`value_dim` ({self.value_dim}) must be divisible by `num_heads` ({self.num_heads})."
            )
        if self.max_cache_len is not None and self.max_cache_len <= 0:
            raise ValueError("`max_cache_len` must be positive when provided.")
        if self.model_dim % self.norm_num_groups != 0:
            raise ValueError(
                f"`norm_num_groups` ({self.norm_num_groups}) must divide `model_dim` ({self.model_dim})."
            )
        if self.norm_eps <= 0.0:
            raise ValueError("`norm_eps` must be positive.")

        # Dropout bounds [0, 1]
        for name, value in [
            ("dropout", self.dropout),
            ("attention_dropout", self.attention_dropout),
            ("hidden_dropout", self.hidden_dropout),
        ]:
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"`{name}` must be in [0, 1], got {value}.")

    @staticmethod
    def from_7b_setup() -> "MegalodonConfig":
        """Return a configuration mirroring the 7B setup from the paper.

        Uses the paper's larger rotary base (100k) for extended context support.
        """
        return MegalodonConfig(
            vocab_size=32_000,
            model_dim=4_096,
            num_layers=32,
            num_heads=4,
            z_dim=1_024,
            value_dim=8_192,
            ffn_hidden_dim=11_264,
            cema_ndim=16,
            chunk_size=4_096,
            norm_num_groups=64,
            rope_base=100_000.0,
            swiglu=True,
            rescale_nffn=False,
        )


__all__ = ["MegalodonConfig", "MegalodonDefaults", "InitMode"]
InitMode = Literal["gaussian", "xavier", "he", "bert", "none"]
