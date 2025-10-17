# coding=utf-8
# configuration_megalodon.py
# Hugging Face-style configuration for the Megalodon decoder-only model.

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, OrderedDict as _OrderedDictType

try:
    from transformers.configuration_utils import PreTrainedConfig
    from transformers.onnx import OnnxConfig
    from transformers.utils import logging
    _HAS_HF = True
except Exception:  # minimal fallback so file stays importable without transformers installed
    _HAS_HF = False
    class PreTrainedConfig:  # type: ignore
        model_type: str = "megalodon"
        def __init__(self, **kwargs):
            for k, v in kwargs.items(): setattr(self, k, v)
    class OnnxConfig:  # type: ignore
        @property
        def inputs(self):
            return {"input_ids": {0: "batch", 1: "sequence"}, "attention_mask": {0: "batch", 1: "sequence"}}
    class _DummyLogger:
        def get_logger(self, name): return self
        def info(self, *a, **k): pass
        def warning(self, *a, **k): pass
        def debug(self, *a, **k): pass
    logging = _DummyLogger()  # type: ignore

logger = logging.get_logger(__name__) if _HAS_HF else logging

@dataclass
class MegalodonDefaults:
    vocab_size: int = 50257
    model_dim: int = 1024
    num_layers: int = 24
    num_heads: int = 8
    z_dim: int = 512
    value_dim: int = 2048
    ffn_hidden_dim: int = 4096
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
    init_mode: str = "gaussian"      # {"gaussian","xavier","he","bert","none"}
    max_positions: int = 1_000_000
    rope_base: float = 10000.0
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

class MegalodonConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a Megalodon model.
    It mirrors the official Megalodon settings while being minimal and self-contained
    (no fused kernels required). The model is **decoder-only** and targets **causal** LM.

    Args:
        vocab_size (int, defaults to 50257):
            Vocabulary size for the token embeddings.
        model_dim (int, defaults to 1024):
            Hidden size of the model.
        num_layers (int, defaults to 24):
            Number of decoder blocks.
        num_heads (int, defaults to 8):
            Number of attention heads for the inner (chunked) attention.
        z_dim (int, defaults to 512):
            Dimension of the shared Q/K representation (`z`); must be divisible by `num_heads`.
        value_dim (int, defaults to 2048):
            Dimension of the value projection (and inner attention output); must be divisible by `num_heads`.
        ffn_hidden_dim (int, defaults to 4096):
            Hidden size for the feed-forward network.
        cema_ndim (int, defaults to 16):
            Number of complex EMA components per hidden channel.
        chunk_size (int, defaults to 2048):
            Chunk length for block-diagonal causal attention (unlimited context via streaming).
        norm_num_groups (int, defaults to 64):
            Number of feature groups for TimestepNorm; must divide `model_dim`.
        dropout (float, defaults to 0.0):
            Dropout applied to residual outputs.
        attention_dropout (float, defaults to 0.0):
            Dropout applied to attention probabilities.
        hidden_dropout (float, defaults to 0.0):
            Dropout applied inside FFN / projection layers.
        swiglu (bool, defaults to False):
            If True, use SwiGLU FFN; otherwise use standard (SiLU) FFN.
        rescale_nffn (bool, defaults to False):
            If True, apply per-layer residual rescaling on the FFN output.
        scale_emb (bool, defaults to False):
            If True, scale token embeddings by sqrt(model_dim).
        share_emb (bool, defaults to False):
            Kept for parity; the reference model ties LM head <-> embeddings in code.
        init_mode (str, defaults to "gaussian"):
            Weight init scheme for linear layers. One of {"gaussian","xavier","he","bert","none"}.
        max_positions (int, defaults to 1_000_000):
            Upper bound for rotary cache and sequence metadata.
        rope_base (float, defaults to 10000.0):
            Base for rotary frequency schedule.
        pad_token_id (int, defaults to 0), bos_token_id (int, defaults to 1), eos_token_id (int, defaults to 2):
            Special token ids.

    Note:
        Megalodon is decoder-only (`is_decoder=True`) and supports caching (`use_cache=True`).
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
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )

        # Core dims & architecture
        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_attention_heads = num_heads  # for HF compatibility
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

        # Sanity checks (mirror modeling expectations)
        if self.z_dim % self.num_heads != 0:
            raise ValueError(f"`z_dim` ({self.z_dim}) must be divisible by `num_heads` ({self.num_heads}).")
        if self.value_dim % self.num_heads != 0:
            raise ValueError(f"`value_dim` ({self.value_dim}) must be divisible by `num_heads` ({self.num_heads}).")
        if self.model_dim % self.norm_num_groups != 0:
            raise ValueError(f"`norm_num_groups` ({self.norm_num_groups}) must divide `model_dim` ({self.model_dim}).")

class MegalodonOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # For causal LM export; keep it simple with two dynamic axes.
        return {
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
        }

__all__ = ["MegalodonConfig", "MegalodonOnnxConfig"]
