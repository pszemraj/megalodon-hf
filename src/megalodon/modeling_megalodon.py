# coding=utf-8
"""
modeling_megalodon.py

A PyTorch/transformers Megalodon (decoder-only) implementation with:
  * Complex EMA long-memory (no custom kernels; FFT-based conv)
  * TimestepNorm (streaming group-wise norm, carries state across chunks)
  * Chunked, causal inner attention with Rotary Embeddings and caching
  * Normalized FFN (SwiGLU optional)
  * HF-compatible classes (Config + ForCausalLM) without relying on fused ops

Defaults target the 200M reference variant; use ``MegalodonConfig.from_7b_setup()``
to mirror the paper's 7B hyper-parameters without changing APIs.

Details:
  - Explicit shapes in docstrings
  - Minimal dtype casts for numerical stability (FFT in fp32, return to input dtype)
  - Deterministic cache semantics (sliding window up to ``max_cache_len``)

References:
Paper: https://arxiv.org/abs/2404.08801
Original Megalodon repo: https://github.com/XuezheMax/megalodon
"""

from __future__ import annotations

import math
import contextlib
import warnings
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from .configuration_megalodon import InitMode, MegalodonConfig

# -----------------------------------------------------------------------------
# Utilities / inits
# -----------------------------------------------------------------------------


InitFn = Callable[[Tensor], Tensor]


def get_init_fn(mode: InitMode, dim: Optional[int] = None) -> InitFn:
    """Return a callable that applies the requested parameter initialisation.

    :param mode: Initialisation scheme matching :class:`InitMode`.
    :type mode: InitMode
    :param dim: Optional feature dimension used to scale the ``gaussian`` scheme.
    :type dim: Optional[int]
    :returns: Callable that initialises parameter tensors in-place.
    :rtype: InitFn
    :raises ValueError: If an unknown ``mode`` is supplied.
    """
    if mode == "none":
        return lambda w: w
    if mode == "bert":
        std = 0.02
        return lambda w: nn.init.normal_(w, mean=0.0, std=std)
    if mode == "he":
        return lambda w: nn.init.kaiming_normal_(w, a=0.0, nonlinearity="relu")
    if mode == "gaussian":
        std = 1.0 if dim is None else 1.0 / math.sqrt(dim)
        a, b = -3 * std, 3 * std
        return lambda w: nn.init.trunc_normal_(w, mean=0.0, std=std, a=a, b=b)
    if mode == "xavier":
        return lambda w: nn.init.xavier_uniform_(w)
    raise ValueError(f"Unknown init mode: {mode}")


# -----------------------------------------------------------------------------
# Norm layers
# -----------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root-mean-square normalization with an optional affine scale (+1 reparameterization)."""

    def __init__(self, dim: int, eps: float = 1e-6, affine: bool = True):
        """Construct an RMS norm over ``dim`` features.

        :param dim: Hidden dimensionality to normalize over.
        :type dim: int
        :param eps: Small constant added before the reciprocal square root.
        :type eps: float
        :param affine: Whether to learn per-feature scale parameters.
        :type affine: bool
        """
        super().__init__()
        self.eps = eps
        if affine:
            self.gamma = nn.Parameter(torch.zeros(dim))
        else:
            self.register_buffer("gamma", torch.zeros(dim), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize ``x`` using RMS statistics and optional affine weights.

        :param x: Input tensor of shape ``(batch, length, dim)``.
        :type x: torch.Tensor
        :returns: Normalized tensor with the same shape as ``x``.
        :rtype: torch.Tensor
        """
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        y = x * rms
        if self.gamma is not None:
            y = y * (self.gamma + 1.0)
        return y


# -----------------------------------------------------------------------------
# Rotary positional embedding
# -----------------------------------------------------------------------------


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding helper for head-wise Q/K rotation.

    Computes cos/sin on the fly to avoid large precomputed tables.
    """

    def __init__(
        self, dim: int, max_positions: int = 1_000_000, base: float = 10_000.0
    ):
        """Prepare rotary frequencies for a ``dim``-dimensional head space.

        :param dim: Per-head dimensionality (must be even).
        :type dim: int
        :param max_positions: Retained for API compatibility (unused).
        :type max_positions: int
        :param base: Exponential base controlling angular step size.
        :type base: float
        :raises ValueError: If ``dim`` is not an even number.
        """
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RotaryEmbedding expects even head dimension.")
        self.dim = dim
        self.base = base
        self.max_positions = max_positions
        half = dim // 2
        # Store per-dimension frequencies; angles computed per-call.
        inv_freq = torch.exp(
            torch.arange(half, dtype=torch.float32) * -(math.log(base) / half)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _get_cis(
        self, start: int, length: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[Tensor, Tensor]:
        """Return cosine and sine tables starting at ``start`` for ``length`` steps.

        :param start: Offset into the cached rotary angles.
        :type start: int
        :param length: Number of consecutive positions to materialize.
        :type length: int
        :param device: Device to place the resulting tensors on.
        :type device: torch.device
        :param dtype: Target dtype for the cosine/sine tables.
        :type dtype: torch.dtype
        :returns: Cosine and sine tensors of shape ``(length, dim/2)``.
        :rtype: Tuple[Tensor, Tensor]
        """
        positions = torch.arange(
            start, start + length, dtype=torch.float32, device=device
        )
        angles = positions.unsqueeze(1) * self.inv_freq.unsqueeze(0)
        return torch.cos(angles).to(dtype), torch.sin(angles).to(dtype)

    @staticmethod
    def _pair_to_complex(x: torch.Tensor) -> torch.Tensor:
        """Interpret the last dimension as stacked real/imag pairs."""
        a, b = x.chunk(2, dim=-1)
        return torch.complex(a, b)

    @staticmethod
    def _complex_to_pair(x: torch.Tensor) -> torch.Tensor:
        """Flatten a complex tensor into concatenated real/imag components."""
        return torch.cat([x.real, x.imag], dim=-1)

    def forward(
        self, q: Tensor, k: Tensor, start_index: int = 0
    ) -> Tuple[Tensor, Tensor]:
        """Rotate per-head ``q`` and ``k`` vectors starting at ``start_index``.

        :param q: Query tensor shaped ``(batch, time, heads, dim)``.
        :type q: Tensor
        :param k: Key tensor shaped ``(batch, time, heads, dim)``.
        :type k: Tensor
        :param start_index: Absolute position offset for the rotary phase.
        :type start_index: int
        :returns: Tuple of rotated ``(q, k)`` tensors.
        :rtype: Tuple[Tensor, Tensor]
        """
        B, T, H, Dh = q.shape
        cos, sin = self._get_cis(start_index, T, q.device, q.dtype)  # (T, Dh/2)
        cos = cos.unsqueeze(0).unsqueeze(2)  # (1, T, 1, Dh/2)
        sin = sin.unsqueeze(0).unsqueeze(2)  # (1, T, 1, Dh/2)
        q1, q2 = q.chunk(2, dim=-1)
        k1, k2 = k.chunk(2, dim=-1)
        # (a + ib) * (cos + i sin) => rotate pairs
        q_rot = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
        k_rot = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
        return q_rot, k_rot


# -----------------------------------------------------------------------------
# TimestepNorm (streaming, per-group Welford stats)
# -----------------------------------------------------------------------------


class TimestepNorm(nn.Module):
    """Streaming group-wise normalization across time with optional state."""

    # Note: the reference implementation fuses this Welford update into a custom
    # CUDA kernel for performance. This pure PyTorch version keeps the same
    # numerics while trading speed for readability and portability.

    def __init__(
        self,
        num_features: int,
        num_groups: int,
        eps: float = 1e-5,
        affine: bool = True,
    ):
        """Instantiate streaming normalization with optional affine parameters.

        :param num_features: Total number of feature channels ``D``.
        :type num_features: int
        :param num_groups: Number of groups the features are split into.
        :type num_groups: int
        :param eps: Numerical epsilon applied to the variance accumulator.
        :type eps: float
        :param affine: Whether to learn per-feature scale and bias.
        :type affine: bool
        :raises ValueError: If ``num_features`` is not divisible by ``num_groups``.
        """
        super().__init__()
        if num_features % num_groups != 0:
            raise ValueError("num_features must be divisible by num_groups")
        self.num_features = num_features
        self.num_groups = num_groups
        self.group_size = num_features // num_groups
        self.eps = eps
        if affine:
            self.weight = nn.Parameter(torch.zeros(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_buffer("weight", torch.zeros(num_features), persistent=False)
            self.register_buffer("bias", torch.zeros(num_features), persistent=False)

    def forward(
        self,
        x: Tensor,
        prev_count: Optional[Tensor] = None,
        prev_mean: Optional[Tensor] = None,
        prev_var: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Normalize ``x`` while carrying forward streaming statistics.

        :param x: Input tensor of shape ``(batch, length, dim)``.
        :type x: Tensor
        :param prev_count: Running token counts per example, if available.
        :type prev_count: Optional[Tensor]
        :param prev_mean: Running mean per group from the previous chunk.
        :type prev_mean: Optional[Tensor]
        :param prev_var: Running variance estimator per group.
        :type prev_var: Optional[Tensor]
        :param padding_mask: Boolean mask where ``1`` marks valid tokens.
        :type padding_mask: Optional[Tensor]
        :returns: Normalized tensor and updated ``count``, ``mean``, ``var``.
        :rtype: Tuple[Tensor, Tensor, Tensor, Tensor]
        """
        B, L, D = x.shape
        G, gs = self.num_groups, self.group_size
        device, dtype = x.device, x.dtype

        if dtype not in (torch.float32, torch.bfloat16):
            raise ValueError(
                f"Megalodon requires float32 or bfloat16 inputs for TimestepNorm, got {dtype}. "
                "float16 is not supported due to numerical stability issues."
            )

        if prev_count is None:
            prev_count = torch.zeros(B, dtype=torch.long, device=device)
        if prev_mean is None:
            prev_mean = torch.zeros(B, G, dtype=torch.float32, device=device)
        if prev_var is None:
            prev_var = torch.ones(B, G, dtype=torch.float32, device=device)

        if padding_mask is None:
            padding_mask = torch.ones(B, L, dtype=torch.bool, device=device)

        if L == 0:
            return (
                x,
                prev_count,
                prev_mean.to(dtype),
                prev_var.to(dtype),
            )

        stats_dtype = torch.float32 if dtype == torch.bfloat16 else dtype

        x_groups = x.view(B, L, G, gs).to(stats_dtype)
        prev_mean_f = prev_mean.to(stats_dtype)
        prev_var_f = prev_var.to(stats_dtype)
        prev_count_f = prev_count.to(stats_dtype)

        valid = padding_mask.to(stats_dtype)
        valid_exp = valid.unsqueeze(-1)

        # Running mean via cumulative sums of per-group averages
        group_means = x_groups.mean(dim=-1)
        prev_sum = prev_mean_f * prev_count_f.unsqueeze(-1)
        cumsum_means = torch.cumsum(group_means * valid_exp, dim=1)
        sum_t = prev_sum.unsqueeze(1) + cumsum_means

        count_t = prev_count_f.unsqueeze(1) + torch.cumsum(valid, dim=1)
        count_clamped = torch.clamp(count_t, min=1.0)

        mean_t = torch.where(
            count_t.unsqueeze(-1) > 0.0,
            sum_t / count_clamped.unsqueeze(-1),
            prev_mean_f.unsqueeze(1),
        )

        mean_prev = torch.cat([prev_mean_f.unsqueeze(1), mean_t[:, :-1, :]], dim=1)
        delta = group_means - mean_prev
        delta2 = group_means - mean_t

        prev_count_clamped = torch.clamp(prev_count_f, min=1.0)
        prev_m2 = prev_var_f * prev_count_clamped.unsqueeze(-1)
        delta_term = delta * delta2 * valid_exp
        m2_t = prev_m2.unsqueeze(1) + torch.cumsum(delta_term, dim=1)

        var_t = torch.where(
            count_t.unsqueeze(-1) > 0.0,
            m2_t / count_clamped.unsqueeze(-1),
            prev_var_f.unsqueeze(1),
        )
        var_t = var_t.clamp_min(0.0)

        mean_b = mean_t.unsqueeze(-1)
        var_b = var_t.unsqueeze(-1)
        x_hat = (x_groups - mean_b) * torch.rsqrt(var_b + self.eps)

        scale = (self.weight + 1.0).view(1, 1, G, gs).to(stats_dtype)
        bias = self.bias.view(1, 1, G, gs).to(stats_dtype)
        y = (x_hat * scale + bias).reshape(B, L, D).to(dtype)

        new_count = prev_count + padding_mask.to(prev_count.dtype).sum(dim=1)
        mean_out = mean_t[:, -1, :].to(dtype)
        var_out = var_t[:, -1, :].to(dtype)

        return y, new_count, mean_out, var_out


# -----------------------------------------------------------------------------
# Complex EMA (sequential recurrence with optional FFT fast path)
# -----------------------------------------------------------------------------


class ComplexEMA(nn.Module):
    """Complex exponential moving average (CEMA) with automatic FFT/sequential dispatch.

    The module mirrors the reference implementation's diagonal SSM by learning
    the original alpha/delta/theta parameterization with evenly spaced phases.
    Implements Megalodon's long-range memory component via complex-valued EMA:
    - FFT convolution ``O(L log L)`` when no cache state is requested (training)
    - Sequential recurrence ``O(L)`` when streaming cache is required (inference)

    The implementation switches between both modes depending on whether a cached
    state is provided or the caller requests the final EMA state.
    """

    def __init__(self, embed_dim: int, ndim: int):
        """Store learnable EMA parameters and prepare FFT helpers.

        :param embed_dim: Hidden dimension ``D`` of the input tensor.
        :type embed_dim: int
        :param ndim: Number of EMA orders tracked per hidden unit.
        :type ndim: int
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.ndim = ndim
        self.scale = math.sqrt(1.0 / float(ndim))

        # Parameters (compatible with reference names used in tests)
        self.p_logit = nn.Parameter(torch.zeros(embed_dim, ndim))
        self.log_q = nn.Parameter(torch.zeros(embed_dim, ndim, dtype=torch.complex64))
        self.gamma = nn.Parameter(torch.zeros(embed_dim, ndim, dtype=torch.complex64))
        self.omega = nn.Parameter(torch.zeros(embed_dim))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize EMA parameters following the reference distribution."""
        device = self.p_logit.device
        dtype = torch.float32
        with torch.no_grad():
            # p_logit
            nn.init.normal_(self.p_logit, mean=0.0, std=0.2)

            # log_q initialised from reference-style theta/delta sampling
            delta = torch.sigmoid(
                torch.normal(
                    mean=0.0, std=0.2, size=(self.embed_dim, self.ndim), device=device
                )
            )
            freqs = math.log(self.embed_dim) / float(self.embed_dim)
            freqs = torch.exp(
                torch.arange(1, self.embed_dim + 1, device=device, dtype=dtype) * -freqs
            )
            freqs = freqs[torch.randperm(self.embed_dim, device=device)]
            freqs = freqs.unsqueeze(-1)
            theta = freqs * (2.0 * math.pi / float(self.ndim))
            wave = torch.arange(1, self.ndim + 1, device=device, dtype=dtype).view(
                1, self.ndim
            )
            phi = theta * wave
            p = torch.sigmoid(self.p_logit.float())
            radius = 1.0 - p * delta
            q_init = torch.polar(radius, phi)  # (D, N) complex64
            self.log_q.copy_(torch.log(q_init.to(self.log_q.dtype)))

            # gamma: complex with zero imaginary part
            gamma_real = torch.normal(
                mean=0.0, std=1.0, size=(self.embed_dim, self.ndim), device=device
            )
            self.gamma.copy_(torch.complex(gamma_real, torch.zeros_like(gamma_real)))

            # omega: truncated normal
            nn.init.trunc_normal_(self.omega, mean=0.0, std=0.25, a=-1.0, b=1.0)

    @staticmethod
    def _real_of_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Return the real component of ``a * b`` efficiently."""
        return a.real * b.real - a.imag * b.imag

    def _coeffs(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return EMA coefficients with decaying eigenvalues inside the unit circle."""
        p = torch.sigmoid(self.p_logit.float())  # (D, N)
        q = torch.exp(self.log_q).to(torch.complex64)  # (D, N)
        gamma = self.gamma.to(torch.complex64) * self.scale  # (D, N)
        return p, q, gamma

    def _forward_sequential(
        self, x: torch.Tensor, hx: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the EMA recurrence sequentially, optionally using cached state.

        :param x: Input activations shaped ``(batch, dim, length)``.
        :type x: torch.Tensor
        :param hx: Optional previous complex EMA state.
        :type hx: Optional[torch.Tensor]
        :returns: Tuple of real outputs ``(batch, dim, length)`` and final complex state.
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        B, D, L = x.shape
        autocast_ctx = (
            torch.amp.autocast("cuda", enabled=False)
            if x.is_cuda
            else contextlib.nullcontext()
        )
        with autocast_ctx:
            p, q, gamma = self._coeffs()
            p_complex = p.to(torch.complex64)  # (D, N)
            q = q.to(torch.complex64)  # (D, N)
            gamma = gamma.to(torch.complex64)  # (D, N)
            x_c = x.to(torch.complex64)

        if hx is not None:
            hx_c = hx if hx.dtype.is_complex else torch.complex(hx[..., 0], hx[..., 1])
            h = hx_c.to(torch.complex64)
        else:
            h = torch.zeros(B, D, self.ndim, dtype=torch.complex64, device=x.device)

        out_r = torch.empty(B, D, L, dtype=torch.float32, device=x.device)
        p_b = p_complex.unsqueeze(0)  # (1, D, N)
        q_b = q.unsqueeze(0)  # (1, D, N)
        gamma_b = gamma.unsqueeze(0)  # (1, D, N)

        for t in range(L):
            xt = x_c[:, :, t].unsqueeze(-1)  # (B, D, 1)
            h = q_b * h + p_b * xt
            out_r[:, :, t] = self._real_of_product(h, gamma_b).sum(dim=-1)

        y = out_r.to(dtype=x.dtype)
        return y, h

    def _forward_fft(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """FFT-based convolution for training when no streaming state is required.

        Uses ``O(L log L)`` FFT convolution when cache state is not needed. For very
        long sequences (``L > 16_384``), consider forcing the sequential path if NaNs
        or Infs are observed.
        """
        B, D, L = x.shape
        if L == 0:
            return x.new_zeros(B, D, L), None

        if L > 16_384:
            dynamo = getattr(torch, "_dynamo", None)
            if not (dynamo is not None and dynamo.is_compiling()):
                warnings.warn(
                    f"FFT path with sequence length {L} exceeds reference implementation "
                    f"constraint (16,384). Consider chunking if numerical issues occur.",
                    UserWarning,
                    stacklevel=2,
                )

        autocast_ctx = (
            torch.amp.autocast("cuda", enabled=False)
            if x.is_cuda
            else contextlib.nullcontext()
        )
        with autocast_ctx:
            p, q, gamma = self._coeffs()
            p_fp32 = p.to(torch.float32)  # (D, N)
            q_fp32 = q.to(torch.complex64)  # (D, N)
            gamma_fp32 = gamma.to(torch.complex64)  # (D, N)
            if L == 1:
                q_pows = torch.ones(
                    D,
                    self.ndim,
                    1,
                    dtype=torch.complex64,
                    device=x.device,
                )
            else:
                q_repeat = q_fp32.unsqueeze(-1).expand(-1, -1, L - 1)
                q_cum = torch.cumprod(q_repeat, dim=-1)
                ones = torch.ones(
                    D,
                    self.ndim,
                    1,
                    dtype=torch.complex64,
                    device=x.device,
                )
                q_pows = torch.cat((ones, q_cum), dim=-1)

            kernel = (gamma_fp32.unsqueeze(-1) * p_fp32.unsqueeze(-1) * q_pows).sum(
                dim=1
            )

            fft_len = 1 << (int(2 * L - 1).bit_length())
            x_fp32 = x.to(torch.float32)
            x_complex = torch.complex(x_fp32, torch.zeros_like(x_fp32))

            X = torch.fft.fft(x_complex, n=fft_len, dim=-1)
            K = torch.fft.fft(kernel, n=fft_len, dim=-1)
            Y = X * K.unsqueeze(0)
            y_complex = torch.fft.ifft(Y, n=fft_len, dim=-1)[..., :L]
            y = y_complex.real.to(dtype=x.dtype)
            return y, None

    def forward(
        self,
        x: Tensor,  # (B, D, L)
        hx: Optional[Tensor] = None,  # (B, D, N) complex or last dim 2
        compute_last_state: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply the EMA block and optionally return the final complex state.

        :param x: Input tensor shaped ``(batch, dim, length)``.
        :type x: Tensor
        :param hx: Optional initial EMA state for streaming inference.
        :type hx: Optional[Tensor]
        :param compute_last_state: Whether to return the final complex EMA state.
        :type compute_last_state: bool
        :returns: Tuple of real-valued outputs and optional final complex state.
        :rtype: Tuple[torch.Tensor, Optional[torch.Tensor]]
        """
        residual = x * self.omega.view(1, -1, 1).to(x)
        use_fft = hx is None and not compute_last_state
        if use_fft:
            y_fft, _ = self._forward_fft(x)
            y = y_fft + residual
            return y, None

        y_seq, h_last = self._forward_sequential(x, hx)
        y = y_seq + residual
        return y, (h_last if compute_last_state else None)


# -----------------------------------------------------------------------------
# Inner (chunked) attention
# -----------------------------------------------------------------------------


@dataclass
class AttentionCache:
    """Cached keys/values for streaming attention."""

    k: Tensor  # (B, Lc, H, Dh)
    v: Tensor  # (B, Lc, H, Dv)
    count: int  # total tokens seen (for absolute position indexing)

    @property
    def length(self) -> int:
        """Number of cached time steps retained."""
        return self.k.size(1)

    @property
    def start_index(self) -> int:
        """Absolute position of the first cached token in this cache."""
        return self.count - self.length


def _clamp_attn_cache(
    cache: Optional[AttentionCache], limit: int
) -> Optional[AttentionCache]:
    """Clamp an attention cache to the most recent ``limit`` tokens."""
    if cache is None:
        return None
    if cache.length <= limit:
        return cache
    return AttentionCache(
        k=cache.k[:, -limit:], v=cache.v[:, -limit:], count=cache.count
    )


def _clamp_layer_cache(
    cache: Optional["LayerCache"], limit: int
) -> Optional["LayerCache"]:
    """Clamp a full LayerCache (attn only) to a fixed window."""
    if cache is None:
        return None
    attn = _clamp_attn_cache(cache.attn, limit)
    position = attn.count if attn is not None else cache.position
    return LayerCache(
        attn=attn,
        norm=cache.norm,
        ema=cache.ema,
        position=position,
    )


@dataclass
class NormState:
    """Running statistics for TimestepNorm."""

    count: Tensor
    mean: Tensor
    var: Tensor

    @staticmethod
    def from_legacy(state) -> Optional["NormState"]:
        """Convert legacy 3-tuples or return existing NormState."""
        if state is None or isinstance(state, NormState):
            return state
        if isinstance(state, (list, tuple)) and len(state) == 3:
            return NormState(state[0], state[1], state[2])
        raise TypeError(f"Unsupported norm cache format: {type(state)!r}")


@dataclass
class LayerCache:
    """Streaming cache combining attention, normalization, and EMA state."""

    attn: Optional[AttentionCache] = None
    norm: Optional[NormState] = None
    ema: Optional[Tensor] = None
    position: int = 0  # absolute token position for RoPE

    @staticmethod
    def from_legacy(cache) -> Optional["LayerCache"]:
        """Convert legacy tuple caches into a LayerCache instance."""
        if cache is None or isinstance(cache, LayerCache):
            return cache
        if not isinstance(cache, (list, tuple)):
            raise TypeError(f"Unsupported cache format: {type(cache)!r}")
        length = len(cache)
        attn = cache[0] if length > 0 else None
        norm_state = (
            NormState.from_legacy(cache[1])
            if length > 1 and cache[1] is not None
            else None
        )
        ema = cache[2] if length > 2 else None
        position = attn.count if isinstance(attn, AttentionCache) else 0
        return LayerCache(attn=attn, norm=norm_state, ema=ema, position=position)


class ChunkedSelfAttention(nn.Module):
    """Scaled dot-product attention with chunking, RoPE, and caching support.

    :ivar num_heads: Number of attention heads ``H``.
    :ivar head_dim: Per-head dimensionality for queries and keys ``Dh``.
    :ivar value_head_dim: Per-head dimensionality for values ``Dv``.
    :ivar chunk_size: Maximum chunk processed in a single attention block.
    :ivar rope: Rotary embedding helper applied to queries and keys.
    :ivar attention_dropout: Dropout probability applied to attention weights.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        value_head_dim: int,
        chunk_size: int,
        rope_base: Optional[float],
        attention_dropout: float,
    ):
        """Initialise chunked attention with rotary embeddings and caching.

        :param num_heads: Number of attention heads ``H``.
        :type num_heads: int
        :param head_dim: Per-head dimensionality for queries and keys ``Dh``.
        :type head_dim: int
        :param value_head_dim: Per-head dimensionality for values ``Dv``.
        :type value_head_dim: int
        :param chunk_size: Maximum chunk processed in a single attention block.
        :type chunk_size: int
        :param rope_base: Base used for rotary positional embeddings (defaults to ``10_000`` when ``None``).
        :type rope_base: Optional[float]
        :param attention_dropout: Dropout probability applied to the attention map.
        :type attention_dropout: float
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.value_head_dim = value_head_dim
        self.chunk_size = chunk_size
        base = 10_000.0 if rope_base is None else rope_base
        self.rope = RotaryEmbedding(head_dim, base=base)
        self.attention_dropout = attention_dropout
        self._sdpa_available = hasattr(F, "scaled_dot_product_attention")

    @staticmethod
    def _causal_mask(Lq: int, Lk: int, device, dtype, offset: int = 0):
        """Return an upper-triangular causal mask with an optional time offset.

        :param Lq: Query sequence length.
        :type Lq: int
        :param Lk: Key sequence length.
        :type Lk: int
        :param device: Device where the mask will be allocated.
        :type device: torch.device
        :param dtype: Desired dtype for the mask tensor.
        :type dtype: torch.dtype
        :param offset: Additional prefix length already seen in the cache.
        :type offset: int
        :returns: Tensor of shape ``(Lq, Lk)`` with ``0`` on allowed positions and ``-inf`` elsewhere.
        :rtype: torch.Tensor
        """
        m = torch.full((Lq, Lk), float("-inf"), device=device, dtype=dtype)
        i = torch.arange(Lq, device=device).unsqueeze(1) + offset
        j = torch.arange(Lk, device=device).unsqueeze(0)
        m[j <= i] = 0.0
        return m

    def _dropkey_additive_mask(
        self,
        shape: Tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
        training: bool,
        valid_mask: Optional[Tensor] = None,
        keep_cols: Optional[Tensor] = None,
    ) -> Optional[Tensor]:
        """Deprecated DropKey helper (kept for API parity)."""
        return None

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        start_index: int,
        cache: Optional[AttentionCache],
        attn_mask: Optional[Tensor],
        training: bool,
        max_cache_len: Optional[int] = None,
        return_cache: bool = False,
        return_position: bool = False,
    ) -> (
        Tuple[Tensor, Optional[AttentionCache]]
        | Tuple[Tensor, Optional[AttentionCache], int]
    ):
        """Compute chunked self-attention and return the result plus cache.

        :param q: Query tensor shaped ``(batch, length, heads, dim_q)``.
        :type q: Tensor
        :param k: Key tensor shaped ``(batch, length, heads, dim_q)``.
        :type k: Tensor
        :param v: Value tensor shaped ``(batch, length, heads, dim_v)``.
        :type v: Tensor
        :param start_index: Absolute offset for rotary embeddings.
        :type start_index: int
        :param cache: Optional cached keys/values from previous chunks.
        :type cache: Optional[AttentionCache]
        :param attn_mask: Optional attention mask with ones for valid tokens.
        :type attn_mask: Optional[Tensor]
        :param training: Flag controlling dropout usage.
        :type training: bool
        :param max_cache_len: Maximum KV tokens to retain in the cache (defaults to ``chunk_size``).
        :type max_cache_len: Optional[int]
        :param return_cache: Whether to produce an updated cache (used to trigger streaming chunk processing when ``L > chunk_size``).
        :type return_cache: bool
        :param return_position: Whether to also return the new absolute position (for streaming cache).
        :type return_position: bool
        :returns: Tuple of attention output and updated cache; includes the new absolute position when ``return_position`` is True.
        :rtype: Tuple[Tensor, Optional[AttentionCache]] or Tuple[Tensor, Optional[AttentionCache], int]
        :raises AssertionError: If ``length`` is not divisible by ``chunk_size`` when processing multi-chunk training batches.
        """
        B, L, H, Dh = q.shape
        Dv = v.size(-1)
        device = q.device
        max_cache_len = self.chunk_size if max_cache_len is None else max_cache_len
        cache = _clamp_attn_cache(cache, max_cache_len)

        def attend_single_chunk(
            q_blk: Tensor,
            k_blk: Tensor,
            v_blk: Tensor,
            start_pos: int,
            cache_blk: Optional[AttentionCache],
            mask_blk: Optional[Tensor],
        ) -> Tuple[Tensor, Optional[AttentionCache], int]:
            """Attend over a single chunk (L <= chunk_size) with optional cache."""
            keep_limit = (
                max_cache_len
                if return_cache
                else (cache_blk.length if cache_blk is not None else 0) + k_blk.size(1)
            )
            # Clamp incoming cache only when streaming.
            if cache_blk is not None and cache_blk.length > keep_limit:
                cache_blk = AttentionCache(
                    k=cache_blk.k[:, -keep_limit:],
                    v=cache_blk.v[:, -keep_limit:],
                    count=cache_blk.count,
                )
            B_, L_, H_, Dh_ = q_blk.shape
            # Rotate only the new block, then concatenate with already-rotated cache.
            q_blk, k_blk = self.rope(q_blk, k_blk, start_index=start_pos)
            if cache_blk is not None:
                k_cat = torch.cat([cache_blk.k, k_blk], dim=1)
                v_cat = torch.cat([cache_blk.v, v_blk], dim=1)
                prefix_start = cache_blk.start_index
            else:
                k_cat = k_blk
                v_cat = v_blk
                prefix_start = start_pos
            keep = min(keep_limit, k_cat.size(1))
            if k_cat.size(1) > keep:
                dropped = k_cat.size(1) - keep
                prefix_start = prefix_start + dropped
                k_cat = k_cat[:, -keep:]
                v_cat = v_cat[:, -keep:]
            Lk_blk = k_cat.size(1)
            q_ = q_blk.transpose(1, 2)  # (B,H,L,Dh)
            k_ = k_cat.transpose(1, 2)  # (B,H,Lk,Dh)
            v_ = v_cat.transpose(1, 2)  # (B,H,Lk,Dv)

            base_mask = None
            prefix_len_blk = max(0, Lk_blk - L_)  # trimmed cache length
            if (
                prefix_len_blk > 0
                or mask_blk is not None
                or start_pos != prefix_len_blk
            ):
                key_positions = prefix_start + torch.arange(
                    Lk_blk, device=device
                )  # absolute positions of keys
                query_positions = start_pos + torch.arange(L_, device=device)
                causal = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
                base_mask = torch.where(
                    causal,
                    torch.zeros_like(causal, dtype=torch.float32),
                    float("-inf"),
                )
                base_mask = base_mask.view(1, L_, Lk_blk).unsqueeze(1)  # (1,1,L,Lk)
                if mask_blk is not None:
                    if prefix_len_blk > 0:
                        prefix_mask = mask_blk.new_ones(B_, prefix_len_blk)
                        mask_tokens = torch.cat([prefix_mask, mask_blk], dim=1)
                    else:
                        mask_tokens = mask_blk
                    if mask_tokens.size(1) != Lk_blk:
                        if mask_tokens.size(1) > Lk_blk:
                            mask_tokens = mask_tokens[:, :Lk_blk]
                        else:
                            pad_len = Lk_blk - mask_tokens.size(1)
                            mask_tokens = F.pad(mask_tokens, (0, pad_len), value=1)
                    base_mask = base_mask.masked_fill(
                        (mask_tokens == 0).view(B_, 1, 1, Lk_blk), float("-inf")
                    )
                base_mask = base_mask.expand(B_, H_, L_, Lk_blk)

            use_sdpa_blk = self._sdpa_available

            if use_sdpa_blk:
                is_causal = prefix_len_blk == 0 and mask_blk is None
                attn_blk = F.scaled_dot_product_attention(
                    q_,
                    k_,
                    v_,
                    attn_mask=base_mask,
                    dropout_p=self.attention_dropout if training else 0.0,
                    is_causal=is_causal,
                )
                out_blk = attn_blk
            else:
                scores = torch.matmul(q_, k_.transpose(-2, -1)) / math.sqrt(Dh_)
                scores = scores.float()
                scores = scores + self._causal_mask(
                    L_, Lk_blk, device, torch.float32, offset=prefix_len_blk
                )

                if mask_blk is not None:
                    if prefix_len_blk > 0:
                        prefix_mask = mask_blk.new_ones(B_, prefix_len_blk)
                        mask = torch.cat([prefix_mask, mask_blk], dim=1)
                    else:
                        mask = mask_blk
                    pad = (mask.to(torch.float32) - 1.0) * 1e9
                    scores = scores + pad.unsqueeze(1).unsqueeze(2)

                attn = torch.softmax(scores, dim=-1).to(q_)
                attn = F.dropout(attn, p=self.attention_dropout, training=training)
                out_blk = torch.matmul(attn, v_)

            out_blk = out_blk.transpose(1, 2)  # (B,L,H,Dv)

            total = (cache_blk.count if cache_blk is not None else start_pos) + L_
            new_cache_blk = AttentionCache(
                k=k_cat[:, -keep:], v=v_cat[:, -keep:], count=total
            )

            out_blk = out_blk.reshape(B_, L_, H_ * Dv)
            return out_blk, new_cache_blk, total

        streaming_mode = return_cache or (cache is not None)
        # If cache is provided, handle streaming in a single block against cached KV.
        if cache is not None:
            out, new_cache, new_pos = attend_single_chunk(
                q, k, v, cache.count, cache, attn_mask
            )
            result_cache = new_cache if return_cache else None
            if return_position:
                return out, result_cache, new_pos
            return out, result_cache

        if streaming_mode and L > self.chunk_size:
            outs: List[Tensor] = []
            cur_cache = None
            cur_pos = start_index
            offset = 0
            while offset < L:
                end = min(offset + self.chunk_size, L)
                mask_chunk = None if attn_mask is None else attn_mask[:, offset:end]
                q_blk = q[:, offset:end]
                k_blk = k[:, offset:end]
                v_blk = v[:, offset:end]
                out_chunk, cur_cache, cur_pos = attend_single_chunk(
                    q_blk, k_blk, v_blk, cur_pos, cur_cache, mask_chunk
                )
                outs.append(out_chunk)
                offset = end
            result_cache = cur_cache if return_cache else None
            if return_position:
                return torch.cat(outs, dim=1), result_cache, cur_pos
            return torch.cat(outs, dim=1), result_cache

        # Cache handling: prefix context (no cache in this branch)

        orig_L = L
        pad_len = 0
        if L > self.chunk_size and (L % self.chunk_size != 0):
            pad_len = self.chunk_size - (L % self.chunk_size)
            q = F.pad(q, (0, 0, 0, 0, 0, pad_len))
            k = F.pad(k, (0, 0, 0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, 0, 0, pad_len))
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, pad_len), value=0)
            L = q.size(1)

        # Single-block path (non-streaming or already chunked)
        if L <= self.chunk_size:
            out, new_cache, new_pos = attend_single_chunk(
                q, k, v, start_index, cache, attn_mask
            )
            result_cache = new_cache if return_cache else None
            if return_position:
                return out, result_cache, new_pos
            return out, result_cache

        # Multi-chunk: block-diagonal causal attention
        assert L % self.chunk_size == 0, (
            "For training, sequence length must be divisible by chunk_size"
        )
        nc = L // self.chunk_size
        q_chunks = q.view(B, nc, self.chunk_size, H, Dh)
        k_chunks = k[:, -L:].view(B, nc, self.chunk_size, H, Dh)
        v_chunks = v[:, -L:].view(B, nc, self.chunk_size, H, Dv)

        outs = []
        mask_block = self._causal_mask(
            self.chunk_size, self.chunk_size, device, torch.float32
        )
        if attn_mask is not None:
            attn_mask = attn_mask.view(B, nc, self.chunk_size)

        for i in range(nc):
            q_i = q_chunks[:, i].transpose(1, 2)  # (B,H,C,Dh)
            k_i = k_chunks[:, i].transpose(1, 2)
            v_i = v_chunks[:, i].transpose(1, 2)

            chunk_len = q_i.size(-2)
            mask_chunk = None
            mask_i = None
            if attn_mask is not None:
                mask_i = attn_mask[:, i]
                mask_chunk = torch.zeros(
                    B, 1, chunk_len, chunk_len, device=device, dtype=torch.float32
                )
                mask_chunk = mask_chunk.masked_fill(
                    (mask_i == 0).view(B, 1, 1, chunk_len), float("-inf")
                )
                mask_chunk = mask_chunk.expand(B, H, chunk_len, chunk_len)

            use_sdpa_chunk = self._sdpa_available and attn_mask is None

            if use_sdpa_chunk:
                attn_chunk = F.scaled_dot_product_attention(
                    q_i,
                    k_i,
                    v_i,
                    attn_mask=mask_chunk,
                    dropout_p=self.attention_dropout if training else 0.0,
                    is_causal=True,
                )
                out_i = attn_chunk.transpose(1, 2)
            else:
                scores = torch.matmul(q_i, k_i.transpose(-2, -1)) / math.sqrt(Dh)
                scores = scores.float()
                scores = scores + mask_block

                if attn_mask is not None:
                    pad = (mask_i.to(torch.float32) - 1.0) * 1e9
                    scores = scores + pad.unsqueeze(1).unsqueeze(2)

                attn = torch.softmax(scores, dim=-1).to(q_i)
                attn = F.dropout(attn, p=self.attention_dropout, training=training)
                out_i = torch.matmul(attn, v_i).transpose(1, 2)

            outs.append(out_i)

        out = torch.cat(outs, dim=1).reshape(B, L, H * Dv)
        if pad_len:
            out = out[:, :orig_L, :]
        final_pos = start_index + orig_L
        if return_position:
            return out, None, final_pos
        return out, None


# -----------------------------------------------------------------------------
# Megalodon Attention block (EMA → gates → chunked attention)
# -----------------------------------------------------------------------------


class MegalodonAttention(nn.Module):
    """EMA + gated chunked attention block matching the Megalodon design."""

    def __init__(self, cfg: MegalodonConfig):
        """Instantiate projections, norms, and rotary helpers for one block.

        :param cfg: Megalodon configuration providing dimensionality and flags.
        :type cfg: MegalodonConfig
        :raises ValueError: If ``cfg.efficient_attn`` requests unsupported kernels.
        """
        super().__init__()
        D, H = cfg.model_dim, cfg.num_heads
        Z, E = cfg.z_dim, cfg.value_dim
        self.cfg = cfg
        self.H = H
        self.z_head = Z // H
        self.v_head = E // H
        if cfg.efficient_attn is not None:
            raise ValueError(
                "MegalodonAttention currently does not support `efficient_attn` kernels."
            )

        # Normalizations
        self.timenorm = TimestepNorm(
            D, cfg.norm_num_groups, eps=cfg.norm_eps, affine=cfg.norm_affine
        )
        self.rmsnorm = RMSNorm(D, eps=cfg.norm_eps, affine=cfg.norm_affine)

        # Long-memory EMA
        self.cema = ComplexEMA(D, cfg.cema_ndim)

        # Projections
        init = get_init_fn(cfg.init_mode)
        self.wz = nn.Linear(D, Z)
        self.wv = nn.Linear(D, E)
        self.wr = nn.Linear(D, E)
        self.wh1 = nn.Linear(D, D)
        self.wh2 = nn.Linear(E, D)
        for lin in (self.wz, self.wv, self.wr, self.wh1, self.wh2):
            init(lin.weight)
            nn.init.zeros_(lin.bias)

        # Per-dim affine for Q/K from shared Z
        self.gamma = nn.Parameter(torch.zeros(2, Z))
        self.beta = nn.Parameter(torch.zeros(2, Z))

        # Inner attention
        self.inner = ChunkedSelfAttention(
            H,
            self.z_head,
            self.v_head,
            cfg.chunk_size,
            cfg.rope_base,
            cfg.attention_dropout,
        )

        self.max_cache_len = cfg.max_cache_len
        self.dropout = cfg.dropout
        self.hidden_dropout = cfg.hidden_dropout
        self.norm_eps = cfg.norm_eps

    def _split_heads(self, x: Tensor, head_dim: int) -> Tensor:
        """Reshape a ``(B, L, H*Dh)`` tensor into ``(B, L, H, Dh)``."""
        B, L, T = x.shape
        return x.view(B, L, self.H, head_dim)

    def _merge_heads(self, x: Tensor) -> Tensor:
        """Flatten ``(B, L, H, Dh)`` back into ``(B, L, H*Dh)``."""
        B, L, H, Dh = x.shape
        return x.reshape(B, L, H * Dh)

    def forward(
        self,
        x: Tensor,
        cache: Optional[LayerCache] = None,
        attn_mask: Optional[Tensor] = None,
        return_cache: bool = False,
        max_cache_len: Optional[int] = None,
    ) -> Tuple[
        Tensor,
        Optional[LayerCache],
    ]:
        """Run the Megalodon attention block and return outputs plus cache.

        :param x: Input activations shaped ``(batch, length, dim)``.
        :type x: Tensor
        :param cache: Optional previous :class:`LayerCache` for streaming.
        :type cache: Optional[LayerCache]
        :param attn_mask: Optional attention mask with ones for valid tokens.
        :type attn_mask: Optional[Tensor]
        :param return_cache: Whether to detach and return updated cache state.
        :type return_cache: bool
        :param max_cache_len: Override for the per-layer cache horizon (defaults to config value).
        :type max_cache_len: Optional[int]
        :returns: Tuple containing the updated activations and optional cache.
        :rtype: Tuple[Tensor, Optional[LayerCache]]
        """
        B, L, D = x.shape
        residual = x

        # Unpack caches
        cache = LayerCache.from_legacy(cache)
        if cache is not None:
            attn_cache = cache.attn
            position = cache.position
            if cache.norm is not None:
                prev_count = cache.norm.count
                prev_mean = cache.norm.mean
                prev_var = cache.norm.var
            else:
                prev_count = prev_mean = prev_var = None
            hx = cache.ema
        else:
            attn_cache = None
            position = 0
            prev_count = prev_mean = prev_var = None
            hx = None

        # 1) TimestepNorm (streaming)
        x_tn, new_count, new_mean, new_var = self.timenorm(
            x, prev_count, prev_mean, prev_var, attn_mask
        )

        # 2) Complex EMA over channels (B,D,L)
        need_last_state = return_cache or (hx is not None)
        y_cema, h_last = self.cema(
            x_tn.transpose(1, 2), hx=hx, compute_last_state=need_last_state
        )
        y_cema = y_cema.transpose(1, 2)

        # 3) RMSNorm + dropout
        mx = F.dropout(
            self.rmsnorm(y_cema), p=self.hidden_dropout, training=self.training
        )

        # 4) Shared Z, per-head L2 normalise, then affine to Q/K
        z = self.wz(mx)  # (B, L, Z)
        z = self._split_heads(z, self.z_head)  # (B, L, H, z_head)
        # Per-head L2 normalisation
        l2 = z.float().pow(2).sum(dim=-1, keepdim=True).sqrt()
        z = z / l2.clamp_min(self.norm_eps).to(z.dtype)

        gamma = self.gamma.view(2, self.H, self.z_head).unsqueeze(1).unsqueeze(1)
        beta = self.beta.view(2, self.H, self.z_head).unsqueeze(1).unsqueeze(1)
        scale = (gamma + 1.0) / math.sqrt(self.z_head)
        z_aff = z.unsqueeze(0) * scale + beta
        q, k = torch.unbind(z_aff, dim=0)  # (B, L, H, z_head) each

        # 5) Values and residual gate
        v = F.silu(self.wv(x_tn)).view(B, L, self.H, self.v_head)  # (B,L,H,v_head)
        r = F.silu(self.wr(mx))  # (B,L,E)

        # 6) Inner attention
        start_index = position
        cache_limit = self.max_cache_len if max_cache_len is None else max_cache_len
        attn_cache = _clamp_attn_cache(attn_cache, cache_limit)
        out, new_attn, new_pos = self.inner(
            q,
            k,
            v,
            start_index=start_index,
            cache=attn_cache,
            attn_mask=attn_mask,
            training=self.training,
            max_cache_len=cache_limit,
            return_cache=return_cache,
            return_position=True,
        )

        # 7) Gate and project (+ hidden dropout on gated attention)
        out = F.dropout(out * r, p=self.hidden_dropout, training=self.training)
        h = self.wh1(mx) + self.wh2(out)
        h = F.dropout(h, p=self.dropout, training=self.training)
        y = h + residual

        if not return_cache:
            return y, None

        ema_next = h_last.detach() if h_last is not None else None
        norm_state = NormState(
            count=new_count.detach(), mean=new_mean.detach(), var=new_var.detach()
        )
        # Detach attention cache tensors to avoid holding autograd graphs
        if new_attn is not None:
            new_attn = AttentionCache(
                k=new_attn.k.detach(),
                v=new_attn.v.detach(),
                count=new_attn.count,
            )
            new_attn = _clamp_attn_cache(new_attn, cache_limit)

        new_cache = LayerCache(
            attn=new_attn,
            norm=norm_state,
            ema=ema_next,
            position=new_pos,
        )
        return y, new_cache


# -----------------------------------------------------------------------------
# FFN
# -----------------------------------------------------------------------------


class NormalizedFFN(nn.Module):
    """SwiGLU FFN with RMSNorm pre/post and residual rescale."""

    def __init__(self, cfg: MegalodonConfig, layer_id: int):
        """Build FFN projections and optional residual rescale parameters.

        :param cfg: Megalodon configuration supplying dimensions and flags.
        :type cfg: MegalodonConfig
        :param layer_id: Index used for layer-wise residual scaling.
        :type layer_id: int
        """
        super().__init__()
        D, H = cfg.model_dim, cfg.ffn_hidden_dim
        self.norm = nn.LayerNorm(
            D, eps=cfg.norm_eps, elementwise_affine=cfg.norm_affine
        )
        self.swiglu = cfg.swiglu
        self.alpha = (0.1 * (0.5**layer_id)) if cfg.rescale_nffn else None

        if self.swiglu:
            self.fc1 = nn.Linear(D, H)
            self.fc3 = nn.Linear(D, H)
            self.fc2 = nn.Linear(H, D)
        else:
            self.fc1 = nn.Linear(D, H)
            self.fc2 = nn.Linear(H, D)

        for lin in (self.fc1, self.fc2) + ((self.fc3,) if self.swiglu else ()):
            get_init_fn(cfg.init_mode)(lin.weight)
            nn.init.zeros_(lin.bias)

        self.hidden_dropout = cfg.hidden_dropout
        self.dropout = cfg.dropout

    def _rescale(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer-specific residual scaling when enabled."""
        return x if self.alpha is None else (self.alpha * x)

    def forward(self, x: Tensor, residual_base: Optional[Tensor] = None) -> Tensor:
        """Run the normalized feed-forward block with optional SwiGLU."""
        residual = x if residual_base is None else residual_base
        x = self.norm(x)
        if self.swiglu:
            hidden = F.silu(self.fc1(x)) * self.fc3(x)
        else:
            hidden = F.silu(self.fc1(x))
        hidden = F.dropout(hidden, p=self.hidden_dropout, training=self.training)
        out = self.fc2(hidden)
        out = F.dropout(out, p=self.dropout, training=self.training)
        return residual + self._rescale(out)


# -----------------------------------------------------------------------------
# Transformer block
# -----------------------------------------------------------------------------


class MegalodonBlock(nn.Module):
    """A single decoder block: Attention + FFN."""

    def __init__(self, cfg: MegalodonConfig, layer_id: int):
        """Pair attention and FFN submodules for one transformer block.

        :param cfg: Megalodon configuration for dimensions and dropout.
        :type cfg: MegalodonConfig
        :param layer_id: Index of the block in the stack (used for rescaling).
        :type layer_id: int
        """
        super().__init__()
        self.attn = MegalodonAttention(cfg)
        self.ffn = NormalizedFFN(cfg, layer_id)

    def forward(
        self,
        x: Tensor,
        cache: Optional[LayerCache] = None,
        attn_mask: Optional[Tensor] = None,
        return_cache: bool = False,
        max_cache_len: Optional[int] = None,
    ) -> Tuple[Tensor, Optional[LayerCache]]:
        """Apply attention + FFN returning updated states and cache.

        :param x: Input activations shaped ``(batch, length, dim)``.
        :type x: Tensor
        :param cache: Optional streaming cache for this block.
        :type cache: Optional[LayerCache]
        :param attn_mask: Optional mask with ones for valid tokens.
        :type attn_mask: Optional[Tensor]
        :param return_cache: Whether to detach and return updated cache state.
        :type return_cache: bool
        :param max_cache_len: Optional override for the attention cache horizon.
        :type max_cache_len: Optional[int]
        :returns: Tuple of updated hidden states and optional cache.
        :rtype: Tuple[Tensor, Optional[LayerCache]]
        """
        cache = LayerCache.from_legacy(cache)
        residual_base = x
        x, cache = self.attn(
            x,
            cache=cache,
            attn_mask=attn_mask,
            return_cache=return_cache,
            max_cache_len=max_cache_len,
        )
        x = self.ffn(x, residual_base=residual_base)
        return x, cache


# -----------------------------------------------------------------------------
# Model + LM head
# -----------------------------------------------------------------------------


class MegalodonModel(PreTrainedModel):
    """Bare Megalodon decoder built from EMA-attention blocks and RMSNorm.

    :ivar config: Megalodon configuration describing model hyperparameters.
    :ivar embed: Token embedding layer mapping ids to hidden states.
    :ivar layers: Stack of :class:`MegalodonBlock` modules.
    :ivar norm: Final RMS normalization applied to decoder outputs.
    :ivar gradient_checkpointing: Flag controlling block-level checkpointing.
    """

    config_class = MegalodonConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ["MegalodonBlock"]

    def __init__(self, config: MegalodonConfig):
        """Construct embeddings, transformer blocks, and final RMSNorm.

        :param config: Megalodon configuration describing the decoder.
        :type config: MegalodonConfig
        """
        super().__init__(config)
        D = config.model_dim
        self.embed = nn.Embedding(config.vocab_size, D, padding_idx=config.pad_token_id)
        if config.init_mode != "none":
            init_emb = get_init_fn(config.init_mode, dim=D)
            init_emb(self.embed.weight)
        self.layers = nn.ModuleList(
            [MegalodonBlock(config, i) for i in range(config.num_layers)]
        )
        self.norm = TimestepNorm(
            D, config.norm_num_groups, eps=config.norm_eps, affine=config.norm_affine
        )
        self.scale = math.sqrt(D) if config.scale_emb else 1.0
        self.gradient_checkpointing = bool(config.gradient_checkpointing)

        self.post_init()

    def get_input_embeddings(self):
        """Return the token embedding layer so callers can reuse/replace it."""
        return self.embed

    def set_input_embeddings(self, value: nn.Embedding):
        """Set the token embedding layer (HF API compatibility)."""
        self.embed = value

    def _gradient_checkpointing_func(self, func, *inputs):
        """Forward wrapper passed to PyTorch checkpoint with new API signature."""
        return torch.utils.checkpoint.checkpoint(func, *inputs, use_reentrant=False)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[Tensor] = None,
        past_key_values: Optional[List[Optional[LayerCache]]] = None,
        use_cache: bool = True,
        output_hidden_states: bool = False,
        output_attentions: bool = False,  # not used; kept for HF parity
        return_dict: Optional[bool] = None,
        max_cache_len: Optional[int] = None,
        enable_training_cache: bool = False,
    ) -> BaseModelOutputWithPast | Tuple[Tensor, ...]:
        """Run embedding lookup and stacked decoder blocks over ``input_ids``.

        :param input_ids: Token ids shaped ``(batch, length)``.
        :type input_ids: torch.LongTensor
        :param attention_mask: Mask with ones for valid tokens.
        :type attention_mask: Optional[Tensor]
        :param past_key_values: Optional list of per-layer :class:`LayerCache` for streaming decoding.
        :type past_key_values: Optional[List[Optional[LayerCache]]]
        :param use_cache: Whether to return updated caches (ignored during training; sequential EMA would be too slow).
        :type use_cache: bool
        :param output_hidden_states: Whether to collect per-layer hidden states.
        :type output_hidden_states: bool
        :param output_attentions: Included for Hugging Face parity (unused).
        :type output_attentions: bool
        :param return_dict: Whether to return a :class:`BaseModelOutputWithPast`.
        :type return_dict: Optional[bool]
        :param max_cache_len: Optional override for the KV cache horizon (defaults to config).
        :type max_cache_len: Optional[int]
        :param enable_training_cache: Opt-in to force cached sequential EMA path during training.
        :type enable_training_cache: bool
        :returns: Decoder outputs following Hugging Face conventions.
        :rtype: BaseModelOutputWithPast or Tuple[Tensor, ...]
        """
        B, L = input_ids.shape
        requested_cache = use_cache
        self.config.gradient_checkpointing = self.gradient_checkpointing
        x = self.embed(input_ids) * self.scale
        cache_limit = (
            max_cache_len if max_cache_len is not None else self.config.max_cache_len
        )

        if x.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError(
                f"Megalodon requires float32 or bfloat16 embeddings, got {x.dtype}. "
                "Call model.to(torch.bfloat16) or model.to(torch.float32)."
            )

        # Training uses FFT EMA path by default (no cache) to avoid slow sequential EMA.
        # Upstream computes the last EMA state via a fused CUDA op; in pure PyTorch,
        # requesting cache implies a sequential recurrence that's much slower.
        # TODO: add a fused/triton sequential CEMA path so cached streaming can
        # match the paper's long-context ambitions (tracked in docs/dev.md).
        use_cache = use_cache and ((not self.training) or enable_training_cache)

        cache_enabled = use_cache or (past_key_values is not None)
        past_final_norm: Optional[NormState] = None
        if past_key_values is None:
            caches = [None] * len(self.layers)
        else:
            pkv_list = list(past_key_values)
            if len(pkv_list) > len(self.layers):
                past_final_norm = NormState.from_legacy(pkv_list.pop())
            caches = [
                _clamp_layer_cache(c, cache_limit) for c in pkv_list[: len(self.layers)]
            ]
            if len(caches) < len(self.layers):
                caches.extend([None] * (len(self.layers) - len(caches)))
        all_hidden = [] if output_hidden_states else None

        for i, layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:

                def custom_forward(y, *, layer=layer):
                    return layer(
                        y,
                        cache=None,
                        attn_mask=attention_mask,
                        return_cache=False,
                        max_cache_len=cache_limit,
                    )[0]

                x = self._gradient_checkpointing_func(custom_forward, x)
                caches[i] = None
            else:
                layer_cache = caches[i] if cache_enabled else None
                x, new_cache = layer(
                    x,
                    cache=layer_cache,
                    attn_mask=attention_mask,
                    return_cache=cache_enabled,
                    max_cache_len=cache_limit,
                )
                caches[i] = new_cache if cache_enabled else None
            if output_hidden_states:
                all_hidden.append(x)

        prev_norm = past_final_norm
        prev_count = prev_norm.count if prev_norm is not None else None
        prev_mean = prev_norm.mean if prev_norm is not None else None
        prev_var = prev_norm.var if prev_norm is not None else None
        x, norm_count, norm_mean, norm_var = self.norm(
            x, prev_count, prev_mean, prev_var, attention_mask
        )
        final_norm_state = NormState(
            count=norm_count.detach(), mean=norm_mean.detach(), var=norm_var.detach()
        )

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        last_hidden = x
        past_key_values = None
        if use_cache:
            pkv_out = list(caches)
            pkv_out.append(final_norm_state)
            past_key_values = tuple(pkv_out)
        hidden_states = tuple(all_hidden) if output_hidden_states else None

        if not return_dict:
            out = (last_hidden,)
            if use_cache:
                out = out + (list(past_key_values),)
            elif requested_cache:
                out = out + (None,)
            if output_hidden_states:
                out = out + (all_hidden,)
            return out

        return BaseModelOutputWithPast(
            last_hidden_state=last_hidden,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=hidden_states,
        )


class MegalodonForCausalLM(PreTrainedModel):
    """Megalodon decoder with tied LM head for causal language modeling."""

    config_class = MegalodonConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ["MegalodonBlock"]

    def __init__(self, config: MegalodonConfig):
        """Wrap a Megalodon decoder with a tied output head.

        :param config: Megalodon configuration describing the decoder.
        :type config: MegalodonConfig
        """
        super().__init__(config)
        self.model = MegalodonModel(config)
        lm_out = (
            config.vocab_size
            if config.output_size in (-1, None)
            else config.output_size
        )
        self.lm_head = nn.Linear(config.model_dim, lm_out, bias=False)
        self._tied_embeddings = lm_out == config.vocab_size
        if not self._tied_embeddings and config.init_mode != "none":
            init_lm = get_init_fn(config.init_mode, dim=lm_out)
            init_lm(self.lm_head.weight)
        if self._tied_embeddings:
            self.tie_weights()
        self.gradient_checkpointing = self.model.gradient_checkpointing

        self.post_init()

    def get_input_embeddings(self):
        """Return the tied input embeddings."""
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Embedding):
        """Replace the shared input embeddings and keep weights tied."""
        self.model.set_input_embeddings(value)
        self.tie_weights()

    def get_output_embeddings(self):
        """Return the LM head (HF API compatibility)."""
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """Replace the LM head (HF API compatibility)."""
        self.lm_head = new_embeddings

    def _tie_weights(self):
        """Tie output logits weights to the input embeddings when allowed."""
        if self._tied_embeddings:
            self.lm_head.weight = self.model.embed.weight

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Optional[LayerCache]]] = None,
        use_cache: bool = True,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_dict: Optional[bool] = None,
        max_cache_len: Optional[int] = None,
        enable_training_cache: bool = False,
    ) -> CausalLMOutputWithPast | Tuple[Tensor, ...]:
        """Run the decoder and LM head, optionally returning loss for labels.

        :param input_ids: Token ids shaped ``(batch, length)``.
        :type input_ids: torch.LongTensor
        :param attention_mask: Mask with ones for tokens to attend to.
        :type attention_mask: Optional[Tensor]
        :param labels: Optional labels for next-token prediction loss.
        :type labels: Optional[torch.LongTensor]
        :param past_key_values: Optional cache list matching :class:`LayerCache` layout from a previous decoding step.
        :type past_key_values: Optional[List[Optional[LayerCache]]]
        :param use_cache: Whether to return updated past key values (ignored during training by the decoder).
        :type use_cache: bool
        :param output_hidden_states: Whether to expose hidden states.
        :type output_hidden_states: bool
        :param output_attentions: Present for HF parity (unused).
        :type output_attentions: bool
        :param return_dict: Whether to return :class:`CausalLMOutputWithPast`.
        :type return_dict: Optional[bool]
        :param max_cache_len: Optional override for the KV cache horizon (defaults to config).
        :type max_cache_len: Optional[int]
        :param enable_training_cache: Opt-in to run cached sequential EMA path during training.
        :type enable_training_cache: bool
        :returns: Language modeling outputs following Hugging Face conventions.
        :rtype: CausalLMOutputWithPast or Tuple[Tensor, ...]
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        model_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
            max_cache_len=max_cache_len,
            enable_training_cache=enable_training_cache,
        )

        if return_dict:
            last_hidden = model_outputs.last_hidden_state
            cache = model_outputs.past_key_values
            hidden_states = model_outputs.hidden_states
        else:
            last_hidden, *rest = model_outputs
            cache = None
            hidden_states = None
            if use_cache and rest:
                cache = rest[0]
                rest = rest[1:]
            if output_hidden_states and rest:
                hidden_states = rest[0]

        logits = self.lm_head(last_hidden)

        loss = None
        if labels is not None:
            # shift for CLM
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        if return_dict:
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=cache if use_cache else None,
                hidden_states=hidden_states,
            )

        out = (logits,)
        if use_cache:
            out = out + (cache,)
        if output_hidden_states:
            out = out + (hidden_states,)
        if loss is not None:
            out = (loss,) + out
        return out


__all__ = [
    "MegalodonConfig",
    "MegalodonModel",
    "MegalodonForCausalLM",
    "AttentionCache",
    "LayerCache",
]
