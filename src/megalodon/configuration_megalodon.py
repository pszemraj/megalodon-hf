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

        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class _DummyLogger:
        def get_logger(self, name):
            return self

        def info(self, *a, **k):
            pass

        def warning(self, *a, **k):
            pass

        def debug(self, *a, **k):
            pass

    logging = _DummyLogger()  # type: ignore

logger = logging.get_logger(__name__) if _HAS_HF else logging


@dataclass
class MegalodonDefaults:
    """Reasonable defaults for medium-scale training."""

    vocab_size: int = 50_257
    model_dim: int = 768
    num_layers: int = 24
    num_heads: int = 8
    z_dim: int = 512
    value_dim: int = 1024
    ffn_hidden_dim: int = 3072
    cema_ndim: int = 16
    chunk_size: int = 2048
    norm_num_groups: int = 64
    dropout: float = 0.0
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    swiglu: bool = False
    rescale_nffn: bool = False
    scale_emb: bool = False
    share_emb: bool = False
    init_mode: str = "gaussian"  # {"gaussian","xavier","he","bert","none"}
    max_positions: int = 1_000_000
    rope_base: float = 10_000.0
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    gradient_checkpointing: bool = False


class MegalodonConfig(_HFPretrainedConfig):
    r"""
    Configuration for a **decoder-only** Megalodon model (causal language model).

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
    swiglu:
        If True, use SwiGLU FFN variant.
    rescale_nffn:
        If True, apply small residual rescaling on FFN outputs (stabilization trick).
    scale_emb:
        If True, scale token embeddings by ``sqrt(model_dim)``.
    share_emb:
        Kept for parity; LM head is tied to embeddings in code regardless.
    init_mode:
        Init scheme for linear layers. One of {"gaussian","xavier","he","bert","none"}.
    max_positions, rope_base:
        Limits and base for rotary embedding cache.
    pad_token_id, bos_token_id, eos_token_id:
        Special token ids.
    gradient_checkpointing:
        If True, use checkpointing over blocks during training to reduce memory.
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
        init_mode: str = MegalodonDefaults.init_mode,
        max_positions: int = MegalodonDefaults.max_positions,
        rope_base: float = MegalodonDefaults.rope_base,
        pad_token_id: int = MegalodonDefaults.pad_token_id,
        bos_token_id: int = MegalodonDefaults.bos_token_id,
        eos_token_id: int = MegalodonDefaults.eos_token_id,
        gradient_checkpointing: bool = MegalodonDefaults.gradient_checkpointing,
        **kwargs,
    ):
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

        # Normalization
        self.norm_num_groups = norm_num_groups

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


__all__ = ["MegalodonConfig", "MegalodonDefaults"]
