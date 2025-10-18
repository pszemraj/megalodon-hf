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

try:
    from transformers.configuration_utils import PretrainedConfig as _HFPretrainedConfig
except Exception:
    try:  # older transformers releases expose the camel-cased variant
        from transformers.configuration_utils import (
            PreTrainedConfig as _HFPretrainedConfig,
        )
    except Exception:  # keep importable without transformers installed
        _HFPretrainedConfig = None

if _HFPretrainedConfig is not None:
    from transformers.utils import logging

    _HAS_HF = True
else:
    _HAS_HF = False

    class _HFPretrainedConfig:  # type: ignore

        model_type: str = "megalodon"

        def __init__(self, **kwargs) -> None:
            """Populate attributes dynamically so the shim mimics HF's config.

            Parameters
            ----------
            **kwargs
                _description_
            """
            for k, v in kwargs.items():
                setattr(self, k, v)

    class _DummyLogger:
        """
        _summary_
        """

        def get_logger(self, name):
            """Return self to emulate the HF logging API without side effects.

            Parameters
            ----------
            name : _type_
                _description_

            Returns
            -------
            _type_
                _description_
            """
            return self

        def info(self, *a, **k):
            """No-op 'info' logger hook for environments without HF logging.

            Parameters
            ----------
            *a
                _description_
            **k
                _description_
            """
            pass

        def warning(self, *a, **k):
            """No-op 'warning' logger hook for environments without HF logging.

            Parameters
            ----------
            *a
                _description_
            **k
                _description_
            """
            pass

        def debug(self, *a, **k):
            """No-op 'debug' logger hook for environments without HF logging.

            Parameters
            ----------
            *a
                _description_
            **k
                _description_
            """
            pass

    logging = _DummyLogger()  # type: ignore

logger = logging.get_logger(__name__) if _HAS_HF else logging


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


class MegalodonConfig(_HFPretrainedConfig):
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
    attention_dropout : _type_
        _description_
    cema_ndim : _type_
        _description_
    chunk_size : _type_
        _description_
    dropout : _type_
        _description_
    efficient_attn : _type_
        _description_
    ffn_hidden_dim : _type_
        _description_
    gradient_checkpointing : _type_
        _description_
    hidden_dropout : _type_
        _description_
    init_mode : _type_
        _description_
    is_decoder : bool
        _description_
    max_positions : _type_
        _description_
    model_dim : _type_
        _description_
    model_type : str
        _description_
    norm_affine : _type_
        _description_
    norm_eps : _type_
        _description_
    norm_num_groups : _type_
        _description_
    num_attention_heads : _type_
        _description_
    num_heads : _type_
        _description_
    num_layers : _type_
        _description_
    output_size : _type_
        _description_
    rescale_nffn : _type_
        _description_
    rope_base : _type_
        _description_
    scale_emb : _type_
        _description_
    share_emb : _type_
        _description_
    swiglu : _type_
        _description_
    use_cache : bool
        _description_
    value_dim : _type_
        _description_
    vocab_size : _type_
        _description_
    z_dim : _type_
        _description_
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
        """
        _summary_

        :param int vocab_size: _description_, defaults to MegalodonDefaults.vocab_size
        :param int model_dim: _description_, defaults to MegalodonDefaults.model_dim
        :param int num_layers: _description_, defaults to MegalodonDefaults.num_layers
        :param int num_heads: _description_, defaults to MegalodonDefaults.num_heads
        :param int z_dim: _description_, defaults to MegalodonDefaults.z_dim
        :param int value_dim: _description_, defaults to MegalodonDefaults.value_dim
        :param int ffn_hidden_dim: _description_, defaults to MegalodonDefaults.ffn_hidden_dim
        :param int cema_ndim: _description_, defaults to MegalodonDefaults.cema_ndim
        :param int chunk_size: _description_, defaults to MegalodonDefaults.chunk_size
        :param int norm_num_groups: _description_, defaults to MegalodonDefaults.norm_num_groups
        :param float dropout: _description_, defaults to MegalodonDefaults.dropout
        :param float attention_dropout: _description_, defaults to MegalodonDefaults.attention_dropout
        :param float hidden_dropout: _description_, defaults to MegalodonDefaults.hidden_dropout
        :param bool swiglu: _description_, defaults to MegalodonDefaults.swiglu
        :param bool rescale_nffn: _description_, defaults to MegalodonDefaults.rescale_nffn
        :param bool scale_emb: _description_, defaults to MegalodonDefaults.scale_emb
        :param bool share_emb: _description_, defaults to MegalodonDefaults.share_emb
        :param Optional[str] efficient_attn: _description_, defaults to MegalodonDefaults.efficient_attn
        :param bool norm_affine: _description_, defaults to MegalodonDefaults.norm_affine
        :param float norm_eps: _description_, defaults to MegalodonDefaults.norm_eps
        :param InitMode init_mode: _description_, defaults to MegalodonDefaults.init_mode
        :param int max_positions: _description_, defaults to MegalodonDefaults.max_positions
        :param float rope_base: _description_, defaults to MegalodonDefaults.rope_base
        :param int output_size: _description_, defaults to MegalodonDefaults.output_size
        :param int pad_token_id: _description_, defaults to MegalodonDefaults.pad_token_id
        :param int bos_token_id: _description_, defaults to MegalodonDefaults.bos_token_id
        :param int eos_token_id: _description_, defaults to MegalodonDefaults.eos_token_id
        :param bool gradient_checkpointing: _description_, defaults to MegalodonDefaults.gradient_checkpointing
        :raises ValueError: _description_
        :raises ValueError: _description_
        :raises ValueError: _description_
        :raises ValueError: _description_
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
