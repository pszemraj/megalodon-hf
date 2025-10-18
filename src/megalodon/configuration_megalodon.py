# coding=utf-8
"""
configuration_megalodon.py

Clean, Torch-first configuration for the **decoder-only** Megalodon model.
This mirrors the knobs used by the original implementation while remaining
free of CUDA-specific requirements. Use together with `modeling_megalodon.py`.

Example
-------
>>> from configuration_megalodon import MegalodonConfig
>>> cfg = MegalodonConfig(vocab_size=50_000, model_dim=768, num_layers=24, num_heads=8)
>>> cfg.model_type
'megalodon'
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


@dataclass
class MegalodonDefaults:
    """
    Default configuration parameters based off of original 200M arch.
    https://github.com/XuezheMax/megalodon/blob/cff8ba5f607a2176bbd0166afc09842984433f93/megalodon/model/mega.py#L275
    """

    vocab_size: int = 50_257
    model_dim: int = 1024
    num_layers: int = 12
    num_heads: int = 1
    z_dim: int = 256
    value_dim: int = 2048
    ffn_hidden_dim: int = 2560
    cema_ndim: int = 16
    chunk_size: int = 2048
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
    rope_base: float = 10_000.0
    output_size: int = -1
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    gradient_checkpointing: bool = False


class MegalodonConfig(PretrainedConfig):
    """
    Configuration for a decoder-only Megalodon model (causal language model).

    Parameters
    ----------
    vocab_size:
        Vocabulary size for token embeddings.
    model_dim:
        Hidden size ``D``.
    num_layers:
        Number of decoder blocks.
    num_heads:
        Number of inner-attention heads ``H``.
    z_dim:
        Shared Q/K representation size ``Z``. Must be divisible by ``num_heads``.
    value_dim:
        Value / inner-attention output size ``E``. Must be divisible by ``num_heads``.
    ffn_hidden_dim:
        Hidden size for the FFN.
    cema_ndim:
        Number of complex EMA components per channel.
    chunk_size:
        Chunk length for block-diagonal causal attention. Enables unlimited context via streaming.
    norm_num_groups:
        Number of feature groups in TimestepNorm. Must divide ``model_dim``.
    dropout, attention_dropout, hidden_dropout:
        Dropout probabilities at various sites (see modeling docs).
    dropout, attention_dropout, hidden_dropout:
        Dropout probabilities at various sites (see modeling docs).
    hidden_dropout:
        Dropout applied to intermediate projections (EMA output, FFN hidden/output).
    attention_dropout:
        Dropout applied to the attention weights.
    swiglu:
        If True, use SwiGLU FFN variant.
    rescale_nffn:
        If True, apply small residual rescaling on FFN outputs (stabilization trick).
    scale_emb:
        If True, scale token embeddings by ``sqrt(model_dim)``.
    share_emb:
        Kept for parity; LM head is tied to embeddings in code regardless.
    efficient_attn:
        Placeholder for upstream efficient attention kernels. Not implemented in this port.
    norm_affine:
        Whether to use affine parameters (scale/bias) in the RMS/Timestep norms.
    norm_eps:
        Numerical epsilon for normalization layers.
    init_mode:
        Init scheme for linear layers. ``InitMode`` literal covering {"gaussian","xavier","he","bert","none"}.
    max_positions, rope_base:
        Limits and base for rotary embedding cache.
    max_positions, rope_base:
        Limits and base for rotary embedding cache.
    pad_token_id, bos_token_id, eos_token_id:
        Special token ids.
    pad_token_id, bos_token_id, eos_token_id:
        Special token ids.
    pad_token_id, bos_token_id, eos_token_id:
        Special token ids.
    output_size:
        Optional override for LM head output dimensionality. ``-1`` ties to ``vocab_size``.
    gradient_checkpointing:
        If True, use checkpointing over blocks during training to reduce memory.

    Attributes
    ----------
    vocab_size : int
        Vocabulary size the decoder expects.
    model_dim : int
        Transformer hidden dimension ``D``.
    num_layers : int
        Number of decoder blocks.
    num_heads : int
        Number of attention heads.
    num_attention_heads : int
        Alias used for Hugging Face compatibility.
    z_dim : int
        Shared Q/K projection width.
    value_dim : int
        Value projection width.
    ffn_hidden_dim : int
        Hidden size of the feed-forward network.
    cema_ndim : int
        Number of complex EMA channels.
    chunk_size : int
        Chunk length processed by streaming attention.
    norm_num_groups : int
        Groups used by timestep normalization.
    dropout : float
        Dropout probability on residual outputs.
    attention_dropout : float
        Dropout probability on attention weights.
    hidden_dropout : float
        Dropout probability inside the FFN and EMA branches.
    swiglu : bool
        Whether the FFN uses a SwiGLU variant.
    rescale_nffn : bool
        Whether to apply residual rescaling in the FFN.
    scale_emb : bool
        Whether to scale input embeddings by ``sqrt(model_dim)``.
    share_emb : bool
        Compatibility flag for weight tying semantics.
    efficient_attn : Optional[str]
        Placeholder for alternative attention kernels (unused).
    norm_affine : bool
        Whether normalization layers include affine parameters.
    norm_eps : float
        Epsilon applied in normalization layers.
    init_mode : InitMode
        Initialisation scheme applied to linear weights.
    max_positions : int
        Maximum number of rotary positions cached.
    rope_base : float
        Base used for rotary angular frequencies.
    output_size : int
        Output dimension of the LM head (``-1`` ties to ``vocab_size``).
    gradient_checkpointing : bool
        Whether blocks use gradient checkpointing by default.
    use_cache : bool
        Hugging Face flag controlling cache returns.
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
        rope_base: float = MegalodonDefaults.rope_base,
        output_size: int = MegalodonDefaults.output_size,
        pad_token_id: int = MegalodonDefaults.pad_token_id,
        bos_token_id: int = MegalodonDefaults.bos_token_id,
        eos_token_id: int = MegalodonDefaults.eos_token_id,
        gradient_checkpointing: bool = MegalodonDefaults.gradient_checkpointing,
        **kwargs,
    ):
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
        :param rope_base: Base frequency for rotary embeddings.
        :type rope_base: float
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
        :raises ValueError: If ``z_dim`` or ``value_dim`` are not divisible by ``num_heads``.
        :raises ValueError: If ``norm_num_groups`` does not divide ``model_dim``.
        :raises ValueError: If ``norm_eps`` is not strictly positive.
        """
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
        if self.z_dim % self.num_heads != 0:
            raise ValueError(
                f"`z_dim` ({self.z_dim}) must be divisible by `num_heads` ({self.num_heads})."
            )
        if self.value_dim % self.num_heads != 0:
            raise ValueError(
                f"`value_dim` ({self.value_dim}) must be divisible by `num_heads` ({self.num_heads})."
            )
        if self.model_dim % self.norm_num_groups != 0:
            raise ValueError(
                f"`norm_num_groups` ({self.norm_num_groups}) must divide `model_dim` ({self.model_dim})."
            )
        if self.norm_eps <= 0.0:
            raise ValueError("`norm_eps` must be positive.")


__all__ = ["MegalodonConfig", "MegalodonDefaults"]
InitMode = Literal["gaussian", "xavier", "he", "bert", "none"]
