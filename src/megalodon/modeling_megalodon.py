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
"""PyTorch/Transformers Megalodon (decoder-only) implementation.

Features:
    - Complex EMA long-memory (no custom kernels; FFT-based conv)
    - TimestepNorm (streaming group-wise norm, carries state across chunks)
    - Chunked, causal inner attention with Rotary Embeddings and caching
    - Normalized FFN (SwiGLU optional)
    - HF-compatible classes (Config + ForCausalLM) without relying on fused ops

Defaults target the 200M reference variant; use ``MegalodonConfig.from_7b_setup()``
to mirror the paper's 7B hyper-parameters without changing APIs.

Details:
    - Explicit shapes in docstrings
    - Minimal dtype casts for numerical stability (FFT in fp32, return to input dtype)
    - Deterministic cache semantics (chunk-local by default; optional sliding/unbounded KV)

References:
    Paper: https://arxiv.org/abs/2404.08801
    Original Megalodon repo: https://github.com/XuezheMax/megalodon
"""

from __future__ import annotations

import contextlib
import math
import warnings
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from .configuration_megalodon import InitMode, MegalodonConfig

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

# Minimum variance in TimestepNorm to prevent division instability during early
# training steps when running statistics have not yet stabilized.
VARIANCE_FLOOR = 1e-6

# Sequence length threshold above which the FFT EMA path emits a warning.
# Reference implementation caps FFT convolution at 16,384 tokens; longer
# sequences may exhibit numerical issues without chunking.
FFT_LENGTH_WARN_THRESHOLD = 16_384

# Chunk size for kernel construction in FFT EMA path. Bounds peak memory to
# O(D * N * chunk) instead of O(D * N * L) for long sequences.
FFT_KERNEL_CHUNK = 4096

# -----------------------------------------------------------------------------
# Utilities / inits
# -----------------------------------------------------------------------------


InitFn = Callable[[Tensor], Tensor]


def get_init_fn(mode: InitMode, dim: Optional[int] = None) -> InitFn:
    """Return a callable that applies the requested parameter initialisation.

    :param InitMode mode: Initialisation scheme matching :class:`InitMode`.
    :param Optional[int] dim: Optional feature dimension used to scale the ``gaussian`` scheme, defaults to ``None``.
    :return InitFn: Callable that initialises parameter tensors in-place.
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

        :param int dim: Hidden dimensionality to normalize over.
        :param float eps: Small constant added before the reciprocal square root, defaults to ``1e-6``.
        :param bool affine: Whether to learn per-feature scale parameters, defaults to ``True``.
        """
        super().__init__()
        self.eps = eps
        if affine:
            self.gamma = nn.Parameter(torch.zeros(dim))
        else:
            self.register_buffer("gamma", torch.zeros(dim), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize ``x`` using RMS statistics and optional affine weights.

        :param torch.Tensor x: Input tensor of shape ``(batch, length, dim)``.
        :return torch.Tensor: Normalized tensor with the same shape as ``x``.
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

        :param int dim: Per-head dimensionality (must be even).
        :param int max_positions: Retained for API compatibility (unused), defaults to ``1_000_000``.
        :param float base: Exponential base controlling angular step size, defaults to ``10_000.0``.
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
        self,
        positions: Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[Tensor, Tensor]:
        """Return cosine and sine tables for given positions.

        :param Tensor positions: Position indices, shape ``(length,)`` or ``(batch, length)``.
        :param torch.device device: Device to place the resulting tensors on.
        :param torch.dtype dtype: Target dtype for the cosine/sine tables.
        :return Tuple[Tensor, Tensor]: Cosine and sine tensors matching position shape + ``(dim/2,)``.
        """
        # positions: (L,) or (B, L)
        # inv_freq: (Dh/2,)
        # angles: (L, Dh/2) or (B, L, Dh/2)
        positions_float = positions.to(dtype=torch.float32, device=device)
        if positions_float.dim() == 1:
            angles = positions_float.unsqueeze(1) * self.inv_freq.unsqueeze(0)
        else:
            # (B, L) @ (1, 1, Dh/2) -> (B, L, Dh/2)
            angles = positions_float.unsqueeze(-1) * self.inv_freq.view(1, 1, -1)
        return torch.cos(angles).to(dtype), torch.sin(angles).to(dtype)

    @staticmethod
    def _pair_to_complex(x: torch.Tensor) -> torch.Tensor:
        """Interpret the last dimension as stacked real/imag pairs.

        :param torch.Tensor x: Input tensor with real/imag pairs on the last dimension.
        :return torch.Tensor: Complex tensor formed from paired real/imag components.
        """
        a, b = x.chunk(2, dim=-1)
        return torch.complex(a, b)

    @staticmethod
    def _complex_to_pair(x: torch.Tensor) -> torch.Tensor:
        """Flatten a complex tensor into concatenated real/imag components.

        :param torch.Tensor x: Complex input tensor.
        :return torch.Tensor: Real-valued tensor with real/imag parts concatenated on the last dimension.
        """
        return torch.cat([x.real, x.imag], dim=-1)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        start_index: int = 0,
        position_ids: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Rotate per-head ``q`` and ``k`` vectors.

        :param Tensor q: Query tensor shaped ``(batch, time, heads, dim)``.
        :param Tensor k: Key tensor shaped ``(batch, time, heads, dim)``.
        :param int start_index: Absolute position offset for the rotary phase (used when ``position_ids`` is None), defaults to ``0``.
        :param Optional[Tensor] position_ids: Per-position indices, shape ``(batch, time)``, defaults to ``None``; when provided, overrides ``start_index`` for per-sample position control (e.g., with left-padding).
        :return Tuple[Tensor, Tensor]: Tuple of rotated ``(q, k)`` tensors.
        :raises ValueError: If ``start_index`` is negative when ``position_ids`` is None.
        """
        B, T, H, Dh = q.shape

        if position_ids is not None:
            # Per-batch positions provided
            positions = position_ids  # (B, T)
            cos, sin = self._get_cis(positions, q.device, q.dtype)  # (B, T, Dh/2)
            cos = cos.unsqueeze(2)  # (B, T, 1, Dh/2)
            sin = sin.unsqueeze(2)  # (B, T, 1, Dh/2)
        else:
            # Scalar start_index (backward compatible)
            if start_index < 0:
                raise ValueError(
                    f"start_index must be non-negative for RoPE, got {start_index}."
                )
            positions = torch.arange(
                start_index, start_index + T, dtype=torch.long, device=q.device
            )
            cos, sin = self._get_cis(positions, q.device, q.dtype)  # (T, Dh/2)
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

        :param int num_features: Total number of feature channels ``D``.
        :param int num_groups: Number of groups the features are split into.
        :param float eps: Numerical epsilon applied to the variance accumulator, defaults to ``1e-5``.
        :param bool affine: Whether to learn per-feature scale and bias, defaults to ``True``.
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

        :param Tensor x: Input tensor of shape ``(batch, length, dim)``.
        :param Optional[Tensor] prev_count: Running token counts per example, if available, defaults to ``None``.
        :param Optional[Tensor] prev_mean: Running mean per group from the previous chunk, defaults to ``None``.
        :param Optional[Tensor] prev_var: Running variance estimator per group, defaults to ``None``.
        :param Optional[Tensor] padding_mask: Boolean mask where ``1`` marks valid tokens, defaults to ``None``.
        :return Tuple[Tensor, Tensor, Tensor, Tensor]: Normalized tensor and updated ``count``, ``mean``, ``var``.
        """
        B, L, D = x.shape
        G, gs = self.num_groups, self.group_size
        device, dtype = x.device, x.dtype

        if D != self.num_features:
            raise ValueError(
                f"TimestepNorm expected input with {self.num_features} features, "
                f"got {D}. Input shape: {tuple(x.shape)}"
            )

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
        # NOTE: Reference uses Kahan-compensated sums for better precision on very
        # long sequences. If precision issues arise, consider float64 stats_dtype
        # or implementing a fused Kahan cumsum kernel. See docs/dev.md.
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
        # Floor variance to prevent division instability in early training steps
        var_t = var_t.clamp_min(VARIANCE_FLOOR)

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

    Implements Megalodon's long-range memory component via complex-valued EMA:
    - FFT convolution ``O(L log L)`` when no cache state is requested (training)
    - Sequential recurrence ``O(L)`` when streaming cache is required (inference)

    Coefficients follow the upstream Megalodon parameterization (reference implementation;
    see ``docs/PAPER_DEVIATIONS.md`` for paper-vs-upstream differences):
    - ``p = sigmoid(alpha)`` (real)
    - ``q = (1 - sigmoid(alpha) * sigmoid(delta)) * exp(i * theta_k)``
      with ``theta_k`` built from a learned base angle and evenly spaced wavelets.
    - ``gamma`` projects the complex hidden state back to real output.

    The implementation switches between both modes depending on whether a cached
    state is provided or the caller requests the final EMA state.
    """

    def __init__(self, embed_dim: int, ndim: int):
        """Store learnable EMA parameters and prepare FFT helpers.

        :param int embed_dim: Hidden dimension ``D`` of the input tensor.
        :param int ndim: Number of EMA orders tracked per hidden unit.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.ndim = ndim
        self.scale = math.sqrt(1.0 / float(ndim))

        # Match upstream parameterization: alpha/delta/theta -> (p, q) coefficients.
        # alpha, delta, theta are stored in logit space and passed through sigmoid.
        self.alpha = nn.Parameter(torch.empty(embed_dim, ndim, 1))
        self.delta = nn.Parameter(torch.empty(embed_dim, ndim, 1))
        self.theta = nn.Parameter(torch.empty(embed_dim, 1, 1))
        # Gamma stored as two real fp32 params (real/imag parts) instead of complex64.
        # PyTorch's complex64 loses its imaginary part on bf16 casts; this avoids that.
        self.gamma_real = nn.Parameter(
            torch.zeros(embed_dim, ndim, dtype=torch.float32)
        )
        self.gamma_imag = nn.Parameter(
            torch.zeros(embed_dim, ndim, dtype=torch.float32)
        )
        self.omega = nn.Parameter(torch.zeros(embed_dim))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize EMA parameters following the reference distribution."""
        device = self.alpha.device
        dtype = torch.float32
        with torch.no_grad():
            # alpha & delta (logit space)
            nn.init.normal_(self.alpha, mean=0.0, std=0.2)
            nn.init.normal_(self.delta, mean=0.0, std=0.2)

            # theta (logit space): inverse-sigmoid of a permuted frequency schedule
            freqs = math.log(self.embed_dim) / float(self.embed_dim)
            freqs = torch.exp(
                torch.arange(1, self.embed_dim + 1, device=device, dtype=dtype) * -freqs
            )
            freqs = freqs[torch.randperm(self.embed_dim, device=device)]
            freqs = freqs.to(dtype).view(self.embed_dim, 1, 1)
            freqs = freqs.clamp(min=1e-6, max=1.0 - 1e-6)
            freqs = torch.log(freqs / (1.0 - freqs))
            self.theta.copy_(freqs)

            # gamma: initialize real part with normal, imaginary part with zeros
            nn.init.normal_(self.gamma_real, mean=0.0, std=1.0)
            nn.init.zeros_(self.gamma_imag)

            # omega: truncated normal
            nn.init.trunc_normal_(self.omega, mean=0.0, std=0.25, a=-1.0, b=1.0)

    @staticmethod
    def _real_of_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Return the real component of ``a * b`` efficiently.

        :param torch.Tensor a: First complex tensor.
        :param torch.Tensor b: Second complex tensor.
        :return torch.Tensor: Real component of ``a * b``.
        """
        return a.real * b.real - a.imag * b.imag

    def _apply(self, fn: Callable[[Tensor], Tensor]) -> "ComplexEMA":
        """Override to keep gamma parameters in fp32 regardless of model dtype.

        Complex EMA coefficients require fp32 precision to avoid numerical issues
        during bf16 training. PyTorch's complex64 dtype loses its imaginary part
        when the model is cast to bf16, so we store gamma as two real fp32 tensors
        and force them back to fp32 after any dtype conversion.

        :param Callable[[Tensor], Tensor] fn: The function to apply (e.g., from ``.to(dtype)`` or ``.cuda()``).
        :return ComplexEMA: Self for method chaining.
        """
        super()._apply(fn)
        with torch.no_grad():
            self.gamma_real.data = self.gamma_real.data.float()
            self.gamma_imag.data = self.gamma_imag.data.float()
        return self

    def _coeffs(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return per-channel EMA coefficients ``(p, q, gamma)``.

        ``q`` is complex with magnitude in (0, 1) by construction, ensuring a
        decaying impulse response; the phase is controlled by the learned base
        angle and the fixed wavelet indices.

        :return Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple of EMA coefficients ``(p, q, gamma)``.
        """
        # D x 1 x 1
        theta = torch.sigmoid(self.theta.float()) * (2.0 * math.pi / float(self.ndim))
        # 1 x N x 1
        wavelets = torch.arange(
            1, self.ndim + 1, dtype=theta.dtype, device=theta.device
        ).view(1, self.ndim, 1)
        # D x N x 1
        phi = wavelets * theta

        alpha = torch.sigmoid(self.alpha.float())  # (D, N, 1)
        delta = torch.sigmoid(self.delta.float())  # (D, N, 1)

        p = alpha.squeeze(-1)  # (D, N)
        q = torch.polar((1.0 - alpha * delta).squeeze(-1), phi.squeeze(-1)).to(
            torch.complex64
        )
        # Reconstruct complex gamma from real/imag parts (bf16-safe)
        gamma = (
            torch.complex(self.gamma_real.float(), self.gamma_imag.float()) * self.scale
        )
        return p, q, gamma

    def _forward_sequential(
        self,
        x: torch.Tensor,
        hx: Optional[torch.Tensor],
        last_valid_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the EMA recurrence sequentially, optionally using cached state.

        :param torch.Tensor x: Input activations shaped ``(batch, dim, length)``.
        :param Optional[torch.Tensor] hx: Optional previous complex EMA state.
        :param Optional[torch.Tensor] last_valid_idx: Optional tensor of shape ``(batch,)`` with the last valid (unmasked) index per batch item, defaults to ``None``; when provided, ``h_last`` is taken at that position and ``-1`` preserves the incoming state.
        :return Tuple[torch.Tensor, torch.Tensor]: Tuple of real outputs ``(batch, dim, length)`` and final complex state.
        """
        B, D, L = x.shape
        autocast_ctx = (
            torch.amp.autocast("cuda", enabled=False)
            if x.is_cuda
            else contextlib.nullcontext()
        )
        with autocast_ctx:
            p, q, gamma = self._coeffs()
            x_c = x.to(torch.complex64)

        if hx is not None:
            if not hx.dtype.is_complex and hx.shape[-1] != 2:
                raise ValueError(
                    f"ComplexEMA expected non-complex hx to have shape[-1]=2 (real/imag), "
                    f"got shape {tuple(hx.shape)}."
                )
            hx_c = hx if hx.dtype.is_complex else torch.complex(hx[..., 0], hx[..., 1])
            h = hx_c.to(torch.complex64)
        else:
            h = torch.zeros(B, D, self.ndim, dtype=torch.complex64, device=x.device)

        out_r = torch.empty(B, D, L, dtype=torch.float32, device=x.device)
        p_b = p.unsqueeze(0)  # (1, D, N)
        q_b = q.unsqueeze(0)  # (1, D, N)
        gamma_b = gamma.unsqueeze(0)  # (1, D, N)

        # Track the hidden state at each batch item's last valid position
        # This ensures that masked (padding) positions don't affect the cached state
        h_last_best = h.clone() if last_valid_idx is not None else None

        for t in range(L):
            xt = x_c[:, :, t].unsqueeze(-1)  # (B, D, 1)
            h = q_b * h + p_b * xt
            out_r[:, :, t] = self._real_of_product(h, gamma_b).sum(dim=-1)

            # Update h_last_best for batch items where t is their last valid position
            if h_last_best is not None:
                is_last_valid = last_valid_idx == t  # (B,)
                if is_last_valid.any():
                    is_last_valid_exp = is_last_valid.view(B, 1, 1).expand_as(h)
                    h_last_best = torch.where(is_last_valid_exp, h, h_last_best)

        y = out_r.to(dtype=x.dtype)
        h_out = h_last_best if h_last_best is not None else h
        return y, h_out

    def _forward_fft(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """FFT-based convolution for training when no streaming state is required.

        Uses ``O(L log L)`` FFT convolution when cache state is not needed. For very
        long sequences (``L > 16_384``), consider forcing the sequential path if NaNs
        or Infs are observed.

        :param torch.Tensor x: Input tensor shaped ``(batch, dim, length)``.
        :return Tuple[torch.Tensor, None]: Real-valued outputs and ``None`` for the unused state.
        """
        B, D, L = x.shape
        if L == 0:
            return x.new_zeros(B, D, L), None

        if L > FFT_LENGTH_WARN_THRESHOLD:
            dynamo = getattr(torch, "_dynamo", None)
            if not (dynamo is not None and dynamo.is_compiling()):
                warnings.warn(
                    f"FFT path with sequence length {L} exceeds reference implementation "
                    f"constraint ({FFT_LENGTH_WARN_THRESHOLD:,}). Consider chunking if numerical issues occur.",
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
            p_complex = p.to(torch.complex64)  # (D, N)
            gamma_c = gamma.to(torch.complex64)  # (D, N)

            # Precompute gamma*p for kernel summation
            gp_c = gamma_c * p_complex  # (D, N)

            # Compute kernel in chunks to bound memory to O(D * N * chunk) instead of O(D * N * L)
            # Clamp radius to 1.0 defensively; eigenvalues are stable by construction
            # (|q| = 1 - alpha*delta < 1) but floating-point edge cases could overflow.
            radius = q.abs().to(torch.float32).clamp(max=1.0)  # (D, N)
            phi = q.angle().to(torch.float32)  # (D, N)

            kernel = torch.empty(D, L, dtype=torch.complex64, device=x.device)
            for start in range(0, L, FFT_KERNEL_CHUNK):
                end = min(start + FFT_KERNEL_CHUNK, L)
                j_chunk = torch.arange(start, end, device=x.device, dtype=torch.int64)
                j_chunk_f = j_chunk.float()

                # q^j via magnitude/phase to keep q==0 stable (0^0 -> 1)
                mag_chunk = torch.pow(
                    radius.unsqueeze(-1), j_chunk
                )  # (D, N, chunk_len)
                phase_chunk = torch.exp(1j * phi.unsqueeze(-1) * j_chunk_f).to(
                    torch.complex64
                )  # (D, N, chunk_len)
                q_pows_chunk = mag_chunk.to(torch.complex64) * phase_chunk

                kernel[:, start:end] = (gp_c.unsqueeze(-1) * q_pows_chunk).sum(dim=1)

            fft_len = 1 << (int(2 * L - 1).bit_length())
            x_fp32 = x.to(torch.float32)

            # Since x is real and we only need y.real, only kernel.real contributes.
            # Using rfft/irfft is ~2x faster and uses ~2x less memory than full fft.
            kernel_real = kernel.real  # (D, L)

            X = torch.fft.rfft(
                x_fp32, n=fft_len, dim=-1
            )  # (B, D, fft_len//2+1) complex
            K = torch.fft.rfft(
                kernel_real, n=fft_len, dim=-1
            )  # (D, fft_len//2+1) complex
            Y = X * K.unsqueeze(0)
            y = torch.fft.irfft(Y, n=fft_len, dim=-1)[..., :L].to(dtype=x.dtype)
            return y, None

    def forward(
        self,
        x: Tensor,  # (B, D, L)
        hx: Optional[Tensor] = None,  # (B, D, N) complex or last dim 2
        compute_last_state: bool = False,
        mask: Optional[Tensor] = None,  # (B, L) bool where True marks valid tokens
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply the EMA block and optionally return the final complex state.

        :param Tensor x: Input tensor shaped ``(batch, dim, length)``.
        :param Optional[Tensor] hx: Optional initial EMA state for streaming inference, defaults to ``None``.
        :param bool compute_last_state: Whether to return the final complex EMA state, defaults to ``False``.
        :param Optional[Tensor] mask: Optional boolean mask shaped ``(batch, length)`` where ``True`` marks valid tokens, defaults to ``None``; when provided, masked positions are zeroed **before** the EMA recurrence (and the omega residual) to prevent padding values from entering the EMA state.
        :return Tuple[torch.Tensor, Optional[torch.Tensor]]: Tuple of real-valued outputs and optional final complex state.
        """
        # Optional mask handling: zero masked timesteps before the recurrence so they
        # cannot inject information into the EMA hidden state (important with
        # TimestepNorm, where padded tokens can become non-zero after normalization).
        last_valid_idx: Optional[Tensor] = None
        if mask is not None:
            if mask.dim() != 2:
                raise ValueError(
                    f"ComplexEMA expected `mask` with shape (batch, length), "
                    f"got shape {tuple(mask.shape)}."
                )
            if mask.shape[0] != x.shape[0] or mask.shape[1] != x.shape[-1]:
                raise ValueError(
                    f"ComplexEMA expected `mask` shape (B, L) to match x shape (B, D, L); "
                    f"got mask {tuple(mask.shape)} and x {tuple(x.shape)}."
                )
            mask_bool = mask.to(device=x.device, dtype=torch.bool)
            # Broadcast over the channel dimension: (B, 1, L)
            x = torch.where(mask_bool.unsqueeze(1), x, x.new_zeros(()))
            # Find the actual position of the last True in each row (not count-based).
            # This handles left-padding, internal zeros, and right-padding correctly.
            # For fully-masked sequences, keep a -1 sentinel so the cached state
            # is preserved (no updates).
            L = mask_bool.size(1)
            indices = torch.arange(L, device=mask_bool.device)
            masked_indices = torch.where(mask_bool, indices, -1)
            last_valid_idx = masked_indices.max(dim=-1).values

        # Omega-weighted residual skip (MEGA lineage; present in upstream).
        residual = x * self.omega.view(1, -1, 1).to(x)
        use_fft = hx is None and not compute_last_state
        if use_fft:
            y_fft, _ = self._forward_fft(x)
            y = y_fft + residual
            return y, None

        y_seq, h_last = self._forward_sequential(x, hx, last_valid_idx=last_valid_idx)
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
    count: Tensor  # (B,) total valid tokens seen per batch item (mask-aware positions)
    mask: Optional[Tensor] = None  # (B, Lc) - True for valid, False for padded

    @property
    def length(self) -> int:
        """Number of cached time steps retained.

        :return int: Number of cached time steps.
        """
        return self.k.size(1)

    @property
    def start_index(self) -> Tensor:
        """Absolute position of the first *valid* cached token per batch item.

        :return Tensor: Absolute position tensor shaped ``(batch,)``.
        """
        if self.mask is None:
            valid_len = self.length
        else:
            valid_len = self.mask.to(torch.long).sum(dim=1)
        return self.count - valid_len


def _clamp_attn_cache(
    cache: Optional[AttentionCache], limit: Optional[int]
) -> Optional[AttentionCache]:
    """Clamp an attention cache to the most recent ``limit`` tokens.

    :param Optional[AttentionCache] cache: Existing attention cache to clamp.
    :param Optional[int] limit: Maximum tokens to retain; ``None`` disables clamping.
    :return Optional[AttentionCache]: Clamped cache or ``None`` when no cache is provided.
    """
    if cache is None:
        return None
    if limit is None:
        return cache
    if limit <= 0:
        raise ValueError("Attention cache clamp limit must be positive when provided.")
    if cache.length <= limit:
        return cache
    return AttentionCache(
        k=cache.k[:, -limit:],
        v=cache.v[:, -limit:],
        count=cache.count,
        mask=cache.mask[:, -limit:] if cache.mask is not None else None,
    )


def _clamp_layer_cache(
    cache: Optional["LayerCache"], limit: Optional[int]
) -> Optional["LayerCache"]:
    """Clamp a full LayerCache (attention only) to a fixed window.

    :param Optional[LayerCache] cache: Layer cache containing attention and norm/EMA state.
    :param Optional[int] limit: Maximum tokens to retain in KV; ``None`` disables clamping.
    :return Optional[LayerCache]: Updated LayerCache with clamped attention cache when applicable.
    :raises TypeError: If cache is not a LayerCache instance.
    """
    if cache is None:
        return None
    if not isinstance(cache, LayerCache):
        raise TypeError(
            f"Expected cache to be LayerCache or None, got {type(cache).__name__}"
        )
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


@dataclass
class LayerCache:
    """Streaming cache combining attention, normalization, and EMA state."""

    attn: Optional[AttentionCache] = None
    norm: Optional[NormState] = None
    ema: Optional[Tensor] = None
    position: Optional[Tensor] = (
        None  # (B,) absolute position cursor (valid token count) for RoPE per batch item
    )


class ChunkedSelfAttention(nn.Module):
    """Dot-product attention with chunking, RoPE, and caching support.

    Megalodon uses *normalized attention* (paper Eq. 9), so attention logits are
    not temperature-scaled by ``1/sqrt(d_head)``. This implementation therefore
    computes ``softmax(QK^T) V`` and, when using PyTorch SDPA, passes ``scale=1.0``.

    :ivar int num_heads: Number of attention heads ``H``.
    :ivar int head_dim: Per-head dimensionality for queries and keys ``Dh``.
    :ivar int value_head_dim: Per-head dimensionality for values ``Dv``.
    :ivar int chunk_size: Maximum chunk processed in a single attention block.
    :ivar RotaryEmbedding rope: Rotary embedding helper applied to queries and keys.
    :ivar float attention_dropout: Dropout probability applied to attention weights.
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

        :param int num_heads: Number of attention heads ``H``.
        :param int head_dim: Per-head dimensionality for queries and keys ``Dh``.
        :param int value_head_dim: Per-head dimensionality for values ``Dv``.
        :param int chunk_size: Maximum chunk processed in a single attention block.
        :param Optional[float] rope_base: Base used for rotary positional embeddings (defaults to ``10_000`` when ``None``).
        :param float attention_dropout: Dropout probability applied to the attention map.
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
    def _causal_mask(
        Lq: int, Lk: int, device: torch.device, dtype: torch.dtype, offset: int = 0
    ) -> Tensor:
        """Return an upper-triangular causal mask with an optional time offset.

        :param int Lq: Query sequence length.
        :param int Lk: Key sequence length.
        :param torch.device device: Device where the mask will be allocated.
        :param torch.dtype dtype: Desired dtype for the mask tensor.
        :param int offset: Additional prefix length already seen in the cache, defaults to ``0``.
        :return torch.Tensor: Tensor of shape ``(Lq, Lk)`` with ``0`` on allowed positions and ``-inf`` elsewhere.
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
        """DropKey placeholder (not implemented; kept for API parity).

        :param Tuple[int, ...] shape: Desired mask shape.
        :param torch.device device: Device for the mask.
        :param torch.dtype dtype: Dtype for the mask.
        :param bool training: Whether to apply DropKey (unused).
        :param Optional[Tensor] valid_mask: Optional validity mask, defaults to ``None``.
        :param Optional[Tensor] keep_cols: Optional column keep mask, defaults to ``None``.
        :return Optional[Tensor]: ``None`` (DropKey is not implemented).
        """
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
        cache_unbounded: bool = False,
        position_ids: Optional[Tensor] = None,
    ) -> (
        Tuple[Tensor, Optional[AttentionCache]]
        | Tuple[Tensor, Optional[AttentionCache], Tensor]
    ):
        """Compute chunked self-attention and return the result plus cache.

        :param Tensor q: Query tensor shaped ``(batch, length, heads, dim_q)``.
        :param Tensor k: Key tensor shaped ``(batch, length, heads, dim_q)``.
        :param Tensor v: Value tensor shaped ``(batch, length, heads, dim_v)``.
        :param int start_index: Absolute offset for rotary embeddings (used when ``position_ids`` is None).
        :param Optional[AttentionCache] cache: Optional cached keys/values from previous chunks.
        :param Optional[Tensor] attn_mask: Optional attention mask with ``1`` for valid tokens and ``0`` for positions to ignore. Should be integer dtype (``torch.long`` or ``torch.int``) or boolean; shape ``(batch, length)``. Masked query positions are zeroed in the output for numerical stability.
        :param bool training: Flag controlling dropout usage.
        :param Optional[int] max_cache_len: Maximum KV tokens to retain in the cache, defaults to ``None`` (``None``/``-1`` uses ``chunk_size`` unless ``cache_unbounded=True``; values above ``chunk_size`` enable sliding-window attention).
        :param bool return_cache: Whether to produce an updated cache (used to trigger streaming chunk processing when ``L > chunk_size``), defaults to ``False``.
        :param bool return_position: Whether to also return the new absolute position (for streaming cache), defaults to ``False``.
        :param bool cache_unbounded: Disable KV cache clamping (explicit opt-in), defaults to ``False``.
        :param Optional[Tensor] position_ids: Per-position RoPE indices, shape ``(batch, length)``, defaults to ``None``; when provided, overrides ``start_index`` for per-sample position control.
        :return Tuple[Tensor, Optional[AttentionCache]] or Tuple[Tensor, Optional[AttentionCache], int]: Tuple of attention output and updated cache; includes the new absolute position when ``return_position`` is True.
        :raises AssertionError: If ``length`` is not divisible by ``chunk_size`` when processing multi-chunk training batches.
        """
        B, L, H, Dh = q.shape
        Dv = v.size(-1)
        device = q.device
        if cache is not None and not isinstance(cache, AttentionCache):
            raise TypeError(
                f"Expected cache to be AttentionCache or None, got {type(cache).__name__}"
            )
        if attn_mask is not None:
            attn_mask = attn_mask.to(dtype=torch.bool)
        if cache_unbounded:
            cache_limit = None
        elif max_cache_len is None or max_cache_len == -1:
            cache_limit = self.chunk_size
        else:
            cache_limit = max_cache_len

        # Validate: max_cache_len must be >= chunk_size to preserve causality
        if cache_limit is not None and cache_limit < self.chunk_size:
            raise ValueError(
                f"max_cache_len ({cache_limit}) must be >= chunk_size "
                f"({self.chunk_size}). Smaller values would break causal "
                f"attention within chunks by removing keys that queries need."
            )

        cache = _clamp_attn_cache(cache, cache_limit)
        if L == 0:
            empty_out = q.new_zeros((B, 0, H * Dv))
            result_cache = cache if return_cache else None
            if return_position:
                if cache is not None:
                    cur_pos = cache.count
                else:
                    cur_pos = torch.full(
                        (B,), start_index, dtype=torch.long, device=device
                    )
                return empty_out, result_cache, cur_pos
            return empty_out, result_cache
        if position_ids is None and attn_mask is not None:
            if cache is not None:
                start_pos = cache.count
            else:
                start_pos = torch.full(
                    (B,), start_index, dtype=torch.long, device=device
                )
            mask_long = attn_mask.to(torch.long)
            pos_offsets = mask_long.cumsum(dim=1) - 1
            pos_offsets = pos_offsets.clamp(min=0)
            position_ids = start_pos.unsqueeze(1) + pos_offsets
        faithful_chunk_local = (not cache_unbounded) and (
            cache_limit == self.chunk_size
        )

        def attend_single_chunk(
            q_blk: Tensor,
            k_blk: Tensor,
            v_blk: Tensor,
            start_pos: Tensor,
            cache_blk: Optional[AttentionCache],
            mask_blk: Optional[Tensor],
            position_ids_blk: Optional[Tensor] = None,
        ) -> Tuple[Tensor, Optional[AttentionCache], Tensor]:
            """Attend over a single chunk (L <= chunk_size) with optional cache.

            :param Tensor q_blk: Query block shaped ``(batch, length, heads, dim_q)``.
            :param Tensor k_blk: Key block shaped ``(batch, length, heads, dim_q)``.
            :param Tensor v_blk: Value block shaped ``(batch, length, heads, dim_v)``.
            :param Tensor start_pos: Per-batch starting positions, shape ``(batch,)``.
            :param Optional[AttentionCache] cache_blk: Optional cache for this chunk, defaults to ``None``.
            :param Optional[Tensor] mask_blk: Optional attention mask for this chunk, defaults to ``None``.
            :param Optional[Tensor] position_ids_blk: Optional position ids for this chunk, defaults to ``None``.
            :return Tuple[Tensor, Optional[AttentionCache], Tensor]: Attention outputs, updated cache, and new position cursor.
            """
            if cache_limit is None:
                keep_limit: Optional[int] = None
            else:
                keep_limit = (
                    cache_limit
                    if return_cache
                    else (cache_blk.length if cache_blk is not None else 0)
                    + k_blk.size(1)
                )
            # Clamp incoming cache only when streaming.
            if (
                cache_blk is not None
                and keep_limit is not None
                and cache_blk.length > keep_limit
            ):
                cache_blk = AttentionCache(
                    k=cache_blk.k[:, -keep_limit:],
                    v=cache_blk.v[:, -keep_limit:],
                    count=cache_blk.count,
                    mask=cache_blk.mask[:, -keep_limit:]
                    if cache_blk.mask is not None
                    else None,
                )
            B_, L_, H_, Dh_ = q_blk.shape
            # Build position_ids for RoPE from per-batch start positions
            if position_ids_blk is None:
                offsets = torch.arange(L_, device=device, dtype=torch.long)
                position_ids_blk = start_pos.unsqueeze(1) + offsets.unsqueeze(0)
            # Rotate only the new block, then concatenate with already-rotated cache.
            q_blk, k_blk = self.rope(
                q_blk, k_blk, start_index=0, position_ids=position_ids_blk
            )
            if cache_blk is not None:
                k_cat = torch.cat([cache_blk.k, k_blk], dim=1)
                v_cat = torch.cat([cache_blk.v, v_blk], dim=1)
            else:
                k_cat = k_blk
                v_cat = v_blk
            keep = (
                k_cat.size(1) if keep_limit is None else min(keep_limit, k_cat.size(1))
            )
            if keep_limit is not None and k_cat.size(1) > keep:
                k_cat = k_cat[:, -keep:]
                v_cat = v_cat[:, -keep:]
            Lk_blk = k_cat.size(1)
            q_ = q_blk.transpose(1, 2)  # (B,H,L,Dh)
            k_ = k_cat.transpose(1, 2)  # (B,H,Lk,Dh)
            v_ = v_cat.transpose(1, 2)  # (B,H,Lk,Dv)

            mask_tokens = None
            has_prefix_mask = cache_blk is not None and cache_blk.mask is not None
            has_current_mask = mask_blk is not None
            if has_prefix_mask or has_current_mask:
                if cache_blk is not None and cache_blk.length > 0:
                    if has_prefix_mask:
                        prefix_mask = cache_blk.mask
                    else:
                        prefix_mask = q_blk.new_ones(
                            B_, cache_blk.length, dtype=torch.bool
                        )
                else:
                    prefix_mask = None
                if has_current_mask:
                    current_mask = mask_blk.to(torch.bool)
                else:
                    current_mask = q_blk.new_ones(B_, L_, dtype=torch.bool)
                if prefix_mask is not None:
                    mask_tokens = torch.cat([prefix_mask, current_mask], dim=1)
                else:
                    mask_tokens = current_mask
                if mask_tokens.size(1) > Lk_blk:
                    mask_tokens = mask_tokens[:, -Lk_blk:]
                elif mask_tokens.size(1) < Lk_blk:
                    pad_len = Lk_blk - mask_tokens.size(1)
                    mask_tokens = F.pad(mask_tokens, (pad_len, 0), value=1)

            if cache_blk is not None:
                if cache_blk.mask is None:
                    cache_offset = cache_blk.count - cache_blk.length
                    cache_positions = cache_offset.unsqueeze(1) + torch.arange(
                        cache_blk.length, device=device
                    ).unsqueeze(0)
                else:
                    cache_mask_long = cache_blk.mask.to(torch.long)
                    cache_valid = cache_mask_long.sum(dim=1)
                    cache_offset = cache_blk.count - cache_valid
                    cache_positions = cache_mask_long.cumsum(dim=1) - 1
                    cache_positions = cache_positions.clamp(min=0)
                    cache_positions = cache_offset.unsqueeze(1) + cache_positions
                key_positions = torch.cat(
                    [cache_positions, position_ids_blk.to(torch.long)], dim=1
                )
            else:
                key_positions = position_ids_blk.to(torch.long)
            if key_positions.size(1) > Lk_blk:
                key_positions = key_positions[:, -Lk_blk:]

            base_mask = None
            prefix_len_blk = max(0, Lk_blk - L_)  # trimmed cache length
            # Only build explicit mask when we have prefix cache or padding mask.
            # SDPA is_causal=True handles causality correctly regardless of absolute position.
            if prefix_len_blk > 0 or mask_tokens is not None:
                query_positions = position_ids_blk.to(torch.long)
                # causal: (B, L, Lk) - key_positions <= query_positions
                causal = key_positions.unsqueeze(1) <= query_positions.unsqueeze(2)
                base_mask = torch.where(
                    causal,
                    torch.zeros_like(causal, dtype=q_.dtype),
                    torch.tensor(float("-inf"), dtype=q_.dtype, device=device),
                )
                base_mask = base_mask.unsqueeze(1)  # (B, 1, L, Lk)
                if mask_tokens is not None:
                    base_mask = base_mask.masked_fill(
                        (mask_tokens == 0).view(B_, 1, 1, Lk_blk), float("-inf")
                    )
                # Keep mask as (B_,1,L_,Lk_blk) or (1,1,L_,Lk_blk) - SDPA broadcasts

            # Guard against fully-masked sequences: if any VALID query has no valid keys,
            # softmax would produce NaN (all -inf logits -> 0/0 in softmax)
            # Note: Padded query positions may have no valid keys, but that's fine
            # because their outputs are ignored anyway.
            if base_mask is not None:
                # Check if any row (query position) has all -inf values
                # base_mask: (B_, 1, L_, Lk_blk) - check across last dim
                has_valid_key = (base_mask > float("-inf")).any(dim=-1)  # (B_, 1, L_)

                # Build query validity mask: which query positions need valid keys?
                # Only non-padded (valid) query positions must have at least one valid key
                if mask_blk is not None:
                    query_is_valid = mask_blk.bool()  # (B_, L_)
                else:
                    # No mask means all queries are valid
                    query_is_valid = torch.ones(B_, L_, dtype=torch.bool, device=device)

                # Check: for every valid query, must have at least one valid key
                # has_valid_key: (B_, 1, L_) -> squeeze to (B_, L_)
                # Only check positions where query_is_valid is True
                valid_queries_have_keys = ~query_is_valid | has_valid_key.squeeze(
                    1
                )  # (B_, L_)
                if not valid_queries_have_keys.all():
                    raise ValueError(
                        "Fully-masked sequences detected: some non-padded query positions "
                        "have no valid keys to attend to. This would produce NaN in softmax. "
                        "Ensure at least one valid token exists for each valid query position."
                    )

            if self._sdpa_available:
                is_causal = base_mask is None
                out_blk = F.scaled_dot_product_attention(
                    q_,
                    k_,
                    v_,
                    attn_mask=base_mask,
                    dropout_p=self.attention_dropout if training else 0.0,
                    is_causal=is_causal,
                    scale=1.0,
                )
            else:
                scores = torch.matmul(q_, k_.transpose(-2, -1))
                scores = scores.float()
                if base_mask is not None:
                    scores = scores + base_mask.to(scores.dtype)
                else:
                    scores = scores + self._causal_mask(
                        L_, Lk_blk, device, torch.float32, offset=prefix_len_blk
                    )

                attn = torch.softmax(scores, dim=-1).to(q_)
                attn = F.dropout(attn, p=self.attention_dropout, training=training)
                out_blk = torch.matmul(attn, v_)

            out_blk = out_blk.transpose(1, 2)  # (B,L,H,Dv)

            if mask_blk is not None:
                valid_counts = mask_blk.to(torch.long).sum(dim=1)
            else:
                valid_counts = torch.full((B_,), L_, dtype=torch.long, device=device)
            base_count = cache_blk.count if cache_blk is not None else start_pos
            total = base_count + valid_counts
            new_cache_blk = AttentionCache(
                k=k_cat[:, -keep:], v=v_cat[:, -keep:], count=total, mask=mask_tokens
            )

            out_blk = out_blk.reshape(B_, L_, H_ * Dv)
            if mask_blk is not None:
                out_blk = torch.where(
                    mask_blk.unsqueeze(-1), out_blk, out_blk.new_zeros(())
                )
            return out_blk, new_cache_blk, total

        streaming_mode = return_cache or (cache is not None)
        if streaming_mode:
            outs: List[Tensor] = []
            cur_cache = cache
            # Convert start_index to per-batch tensor if needed
            if cache is not None:
                cur_pos = cache.count  # (B,) tensor
            else:
                cur_pos = torch.full((B,), start_index, dtype=torch.long, device=device)
            offset = 0
            while offset < L:
                remaining = L - offset
                if faithful_chunk_local:
                    # Enforce chunk-local attention by resetting KV at chunk boundaries.
                    # Per-batch positions: pos_in_chunk is (B,) tensor
                    pos_in_chunk = cur_pos % self.chunk_size
                    max_pos_in_chunk = int(pos_in_chunk.max().item())
                    # Check if ALL batches are at chunk boundary (shared reset)
                    all_at_boundary = (pos_in_chunk == 0).all().item()
                    if all_at_boundary:
                        cur_cache = None
                    elif cur_cache is not None:
                        # Trim cache to max position within chunk across all batches
                        if cur_cache.length > max_pos_in_chunk:
                            cur_cache = AttentionCache(
                                k=cur_cache.k[:, -max_pos_in_chunk:],
                                v=cur_cache.v[:, -max_pos_in_chunk:],
                                count=cur_cache.count,
                                mask=cur_cache.mask[:, -max_pos_in_chunk:]
                                if cur_cache.mask is not None
                                else None,
                            )
                        if cur_cache.length > 0:
                            keep_counts = pos_in_chunk.clamp(max=cur_cache.length)
                            idx = torch.arange(
                                cur_cache.length, device=device
                            ).unsqueeze(0)
                            chunk_mask = idx >= (
                                cur_cache.length - keep_counts
                            ).unsqueeze(1)
                            if cur_cache.mask is None:
                                if not chunk_mask.all():
                                    cur_cache = AttentionCache(
                                        k=cur_cache.k,
                                        v=cur_cache.v,
                                        count=cur_cache.count,
                                        mask=chunk_mask,
                                    )
                            else:
                                cur_cache = AttentionCache(
                                    k=cur_cache.k,
                                    v=cur_cache.v,
                                    count=cur_cache.count,
                                    mask=cur_cache.mask & chunk_mask,
                                )
                    # Use maximum position within chunk to determine chunk_len
                    chunk_len = min(remaining, self.chunk_size - max_pos_in_chunk)
                else:
                    chunk_len = (
                        min(remaining, self.chunk_size)
                        if L > self.chunk_size
                        else remaining
                    )
                end = offset + chunk_len
                mask_chunk = None if attn_mask is None else attn_mask[:, offset:end]
                pos_ids_chunk = (
                    None if position_ids is None else position_ids[:, offset:end]
                )
                q_blk = q[:, offset:end]
                k_blk = k[:, offset:end]
                v_blk = v[:, offset:end]
                out_chunk, cur_cache, cur_pos = attend_single_chunk(
                    q_blk, k_blk, v_blk, cur_pos, cur_cache, mask_chunk, pos_ids_chunk
                )
                outs.append(out_chunk)
                offset = end
            out_cat = torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]
            result_cache = cur_cache if return_cache else None
            if return_position:
                return out_cat, result_cache, cur_pos
            return out_cat, result_cache

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
            if position_ids is not None:
                pad_values = position_ids[:, -1:].expand(B, pad_len)
                position_ids = torch.cat([position_ids, pad_values], dim=1)
            L = q.size(1)

        # Single-block path (non-streaming or already chunked)
        if L <= self.chunk_size:
            # Convert start_index to per-batch tensor
            start_pos_tensor = torch.full(
                (B,), start_index, dtype=torch.long, device=device
            )
            out, new_cache, new_pos = attend_single_chunk(
                q, k, v, start_pos_tensor, cache, attn_mask, position_ids
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
        C = self.chunk_size

        if attn_mask is None:
            # === VECTORIZED PATH (fast) ===
            # Apply RoPE once to full sequence with absolute positions
            q_rot, k_rot = self.rope(
                q, k, start_index=start_index, position_ids=position_ids
            )

            # Reshape: (B, L, H, Dh) -> (B*nc, C, H, Dh)
            q_chunks = q_rot.view(B, nc, C, H, Dh).reshape(B * nc, C, H, Dh)
            k_chunks = k_rot.view(B, nc, C, H, Dh).reshape(B * nc, C, H, Dh)
            v_chunks = v.view(B, nc, C, H, Dv).reshape(B * nc, C, H, Dv)

            # Transpose for SDPA: (B*nc, H, C, D)
            q_sdpa = q_chunks.transpose(1, 2)
            k_sdpa = k_chunks.transpose(1, 2)
            v_sdpa = v_chunks.transpose(1, 2)

            # Single SDPA call - each batch item is independent chunk with is_causal=True
            attn_out = F.scaled_dot_product_attention(
                q_sdpa,
                k_sdpa,
                v_sdpa,
                is_causal=True,
                scale=1.0,
                dropout_p=self.attention_dropout if training else 0.0,
            )

            # Reshape back: (B*nc, H, C, Dv) -> (B, L, H*Dv)
            attn_out = attn_out.transpose(1, 2)
            out = attn_out.reshape(B, nc, C, H, Dv).reshape(B, L, H * Dv)
        else:
            # === FALLBACK LOOP (padded batches) ===
            q_chunks = q.view(B, nc, C, H, Dh)
            k_chunks = k[:, -L:].view(B, nc, C, H, Dh)
            v_chunks = v[:, -L:].view(B, nc, C, H, Dv)
            attn_mask_chunks = attn_mask.view(B, nc, C)
            pos_ids_chunks = (
                None if position_ids is None else position_ids.view(B, nc, C)
            )

            # Pre-compute causal mask for manual path
            mask_block = self._causal_mask(C, C, device, torch.float32)

            outs = []
            for i in range(nc):
                start_pos = start_index + i * C
                q_blk = q_chunks[:, i]
                k_blk = k_chunks[:, i]
                pos_ids_blk = None if pos_ids_chunks is None else pos_ids_chunks[:, i]
                q_blk, k_blk = self.rope(
                    q_blk, k_blk, start_index=start_pos, position_ids=pos_ids_blk
                )

                q_i = q_blk.transpose(1, 2)  # (B,H,C,Dh)
                k_i = k_blk.transpose(1, 2)
                v_i = v_chunks[:, i].transpose(1, 2)

                mask_i = attn_mask_chunks[:, i]
                # Combine causal + padding: broadcastable (B, 1, C, C)
                mask_chunk = mask_block.unsqueeze(0).expand(B, -1, -1).unsqueeze(1)
                mask_chunk = mask_chunk.masked_fill(
                    (mask_i == 0).view(B, 1, 1, C), float("-inf")
                )

                if self._sdpa_available:
                    out_i = F.scaled_dot_product_attention(
                        q_i,
                        k_i,
                        v_i,
                        attn_mask=mask_chunk,
                        is_causal=False,
                        scale=1.0,
                        dropout_p=self.attention_dropout if training else 0.0,
                    ).transpose(1, 2)
                else:
                    scores = torch.matmul(q_i, k_i.transpose(-2, -1))
                    scores = scores.float()
                    scores = scores + mask_block
                    pad = (mask_i.to(torch.float32) - 1.0) * 1e9
                    scores = scores + pad.unsqueeze(1).unsqueeze(2)
                    attn = torch.softmax(scores, dim=-1).to(q_i)
                    attn = F.dropout(attn, p=self.attention_dropout, training=training)
                    out_i = torch.matmul(attn, v_i).transpose(1, 2)

                out_i = torch.where(mask_i.view(B, C, 1, 1), out_i, out_i.new_zeros(()))
                outs.append(out_i)

            out = torch.cat(outs, dim=1).reshape(B, L, H * Dv)
        if pad_len:
            out = out[:, :orig_L, :]
        # Convert final_pos to per-batch tensor (mask-aware when provided)
        if attn_mask is None:
            final_pos = torch.full(
                (B,), start_index + orig_L, dtype=torch.long, device=device
            )
        else:
            valid_counts = attn_mask.to(torch.long).sum(dim=1)
            final_pos = (
                torch.full((B,), start_index, dtype=torch.long, device=device)
                + valid_counts
            )
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

        :param MegalodonConfig cfg: Megalodon configuration providing dimensionality and flags.
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

        # Long-memory EMA
        self.cema = ComplexEMA(D, cfg.cema_ndim)
        self.rmsnorm = RMSNorm(D, eps=cfg.norm_eps, affine=cfg.norm_affine)

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
        self.cache_unbounded = cfg.cache_unbounded
        self.dropout = cfg.dropout
        self.hidden_dropout = cfg.hidden_dropout
        self.norm_eps = cfg.norm_eps

    def _split_heads(self, x: Tensor, head_dim: int) -> Tensor:
        """Reshape a ``(B, L, H*Dh)`` tensor into ``(B, L, H, Dh)``.

        :param Tensor x: Input tensor shaped ``(batch, length, heads * head_dim)``.
        :param int head_dim: Per-head dimension ``Dh``.
        :return Tensor: Reshaped tensor ``(batch, length, heads, head_dim)``.
        """
        B, L, T = x.shape
        return x.view(B, L, self.H, head_dim)

    def _merge_heads(self, x: Tensor) -> Tensor:
        """Flatten ``(B, L, H, Dh)`` back into ``(B, L, H*Dh)``.

        :param Tensor x: Input tensor shaped ``(batch, length, heads, head_dim)``.
        :return Tensor: Flattened tensor ``(batch, length, heads * head_dim)``.
        """
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

        :param Tensor x: Input activations shaped ``(batch, length, dim)``.
        :param Optional[LayerCache] cache: Optional previous :class:`LayerCache` for streaming, defaults to ``None``.
        :param Optional[Tensor] attn_mask: Optional attention mask with ones for valid tokens, defaults to ``None``.
        :param bool return_cache: Whether to detach and return updated cache state, defaults to ``False``.
        :param Optional[int] max_cache_len: Override for the per-layer cache horizon, defaults to ``None`` (uses configured value, typically ``chunk_size``); ``-1`` clamps to one chunk; set ``cache_unbounded=True`` in the config to disable clamping.
        :return Tuple[Tensor, Optional[LayerCache]]: Tuple containing the updated activations and optional cache.
        """
        B, L, D = x.shape
        residual = x

        # Validate cache type
        if cache is not None and not isinstance(cache, LayerCache):
            raise TypeError(
                f"Expected cache to be LayerCache or None, got {type(cache).__name__}"
            )
        if cache is not None:
            attn_cache = cache.attn
            # Position is now Optional[Tensor]; when attn_cache exists, its count is used
            # start_index is only needed when attn_cache is None
            if cache.position is not None:
                # Use first element for start_index (backward compat); cache.count handles per-batch
                start_index_int = int(cache.position[0].item())
            else:
                start_index_int = 0
            if cache.norm is not None:
                prev_count = cache.norm.count
                prev_mean = cache.norm.mean
                prev_var = cache.norm.var
            else:
                prev_count = prev_mean = prev_var = None
            hx = cache.ema
        else:
            attn_cache = None
            start_index_int = 0
            prev_count = prev_mean = prev_var = None
            hx = None

        # 1) TimestepNorm (streaming)
        x_tn, new_count, new_mean, new_var = self.timenorm(
            x, prev_count, prev_mean, prev_var, attn_mask
        )

        # 2) Complex EMA over channels (B,D,L)
        need_last_state = return_cache or (hx is not None)
        y_cema, h_last = self.cema(
            x_tn.transpose(1, 2),
            hx=hx,
            compute_last_state=need_last_state,
            mask=attn_mask,
        )
        y_cema = y_cema.transpose(1, 2)

        # 3) RMS-normed CEMA output (mx), optionally regularized
        mx = self.rmsnorm(y_cema)
        mx = F.dropout(mx, p=self.hidden_dropout, training=self.training)

        # 4) Shared Z, per-head RMS normalise, then affine to Q/K
        z = self.wz(mx)  # (B, L, Z)
        z = self._split_heads(z, self.z_head)  # (B, L, H, z_head)
        # Per-head RMS normalisation (matches paper/upstream)
        rms = z.float().pow(2).mean(dim=-1, keepdim=True).add(self.norm_eps).rsqrt()
        z = (z * rms).to(z.dtype)

        gamma = self.gamma.view(2, self.H, self.z_head).unsqueeze(1).unsqueeze(1)
        beta = self.beta.view(2, self.H, self.z_head).unsqueeze(1).unsqueeze(1)
        scale = (gamma + 1.0) / math.sqrt(self.z_head)
        z_aff = z.unsqueeze(0) * scale + beta
        q, k = torch.unbind(z_aff, dim=0)  # (B, L, H, z_head) each

        # 5) Values and residual gate
        v = F.silu(self.wv(x_tn)).view(B, L, self.H, self.v_head)  # (B,L,H,v_head)
        r = F.silu(self.wr(mx))  # (B,L,E)

        # 6) Inner attention
        start_index = start_index_int
        if max_cache_len == -1:
            cache_limit = self.inner.chunk_size
            cache_unbounded = False
        elif max_cache_len is not None:
            cache_limit = max_cache_len
            cache_unbounded = False
        else:
            cache_limit = self.max_cache_len
            cache_unbounded = self.cache_unbounded
            if cache_limit is None and not cache_unbounded:
                cache_limit = self.inner.chunk_size
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
            cache_unbounded=cache_unbounded,
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
                mask=new_attn.mask.detach() if new_attn.mask is not None else None,
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
    """LayerNorm + FFN (optional SwiGLU) with two-hop residual support."""

    def __init__(self, cfg: MegalodonConfig, layer_id: int):
        """Build FFN projections and optional residual rescale parameters.

        :param MegalodonConfig cfg: Megalodon configuration supplying dimensions and flags.
        :param int layer_id: Index used for layer-wise residual scaling.
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
        """Apply layer-specific residual scaling when enabled.

        :param torch.Tensor x: Input tensor to rescale.
        :return torch.Tensor: Rescaled tensor.
        """
        return x if self.alpha is None else (self.alpha * x)

    def forward(self, x: Tensor, residual_base: Optional[Tensor] = None) -> Tensor:
        """Run the normalized feed-forward block with optional SwiGLU.

        When ``residual_base`` is provided, it is used for the residual addition.
        This supports Megalodon's two-hop residual layout where the FFN adds the
        original block input rather than the post-attention activations.

        :param Tensor x: Input activations shaped ``(batch, length, dim)``.
        :param Optional[Tensor] residual_base: Optional residual base for the skip connection, defaults to ``None``.
        :return Tensor: Output activations after the FFN and residual addition.
        """
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

        :param MegalodonConfig cfg: Megalodon configuration for dimensions and dropout.
        :param int layer_id: Index of the block in the stack (used for rescaling).
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

        :param Tensor x: Input activations shaped ``(batch, length, dim)``.
        :param Optional[LayerCache] cache: Optional streaming cache for this block, defaults to ``None``.
        :param Optional[Tensor] attn_mask: Optional mask with ones for valid tokens, defaults to ``None``.
        :param bool return_cache: Whether to detach and return updated cache state, defaults to ``False``.
        :param Optional[int] max_cache_len: Optional override for the attention cache horizon, defaults to ``None`` (uses configured value, typically ``chunk_size``); ``-1`` clamps to one chunk.
        :return Tuple[Tensor, Optional[LayerCache]]: Tuple of updated hidden states and optional cache.
        """
        if cache is not None and not isinstance(cache, LayerCache):
            raise TypeError(
                f"Expected cache to be LayerCache or None, got {type(cache).__name__}"
            )
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
    """Bare Megalodon decoder built from EMA-attention blocks and TimestepNorm.

    :ivar MegalodonConfig config: Megalodon configuration describing model hyperparameters.
    :ivar nn.Embedding embed: Token embedding layer mapping ids to hidden states.
    :ivar nn.ModuleList layers: Stack of :class:`MegalodonBlock` modules.
    :ivar TimestepNorm norm: Final TimestepNorm applied to decoder outputs.
    :ivar bool gradient_checkpointing: Flag controlling block-level checkpointing.
    """

    config_class = MegalodonConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ["MegalodonBlock"]

    def __init__(self, config: MegalodonConfig):
        """Construct embeddings, transformer blocks, and final TimestepNorm.

        :param MegalodonConfig config: Megalodon configuration describing the decoder.
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

    def get_input_embeddings(self) -> nn.Embedding:
        """Return the token embedding layer so callers can reuse/replace it.

        :return nn.Embedding: Token embedding layer.
        """
        return self.embed

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        """Set the token embedding layer (HF API compatibility).

        :param nn.Embedding value: Replacement embedding layer.
        :return None: This method returns ``None``.
        """
        self.embed = value

    def _gradient_checkpointing_func(
        self, func: Callable[..., Tensor], *inputs: Tensor
    ) -> Tensor:
        """Forward wrapper passed to PyTorch checkpoint with new API signature.

        :param Callable[..., Tensor] func: Function to checkpoint.
        :param Tensor inputs: Input tensors forwarded to ``func``.
        :return Tensor: Output tensor from ``func`` under checkpointing.
        """
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

        :param torch.LongTensor input_ids: Token ids shaped ``(batch, length)``.
        :param Optional[Tensor] attention_mask: Mask with ones for valid tokens, defaults to ``None``.
        :param Optional[List[Optional[LayerCache]]] past_key_values: Optional list of per-layer :class:`LayerCache` entries for streaming decoding, optionally followed by the final :class:`NormState`; length must be ``<= num_layers + 1``, defaults to ``None``.
        :param bool use_cache: Whether to return updated caches (ignored during training; sequential EMA would be too slow), defaults to ``True``.
        :param bool output_hidden_states: Whether to collect per-layer hidden states, defaults to ``False``.
        :param bool output_attentions: Included for Hugging Face parity (unused), defaults to ``False``.
        :param Optional[bool] return_dict: Whether to return a :class:`BaseModelOutputWithPast`, defaults to ``None``.
        :param Optional[int] max_cache_len: Optional override for the KV cache horizon, defaults to ``None`` (uses configured value, typically ``chunk_size``); ``-1`` clamps to one chunk; set ``cache_unbounded=True`` in the config to disable clamping.
        :param bool enable_training_cache: Opt-in to force cached sequential EMA path during training, defaults to ``False``.
        :return BaseModelOutputWithPast or Tuple[Tensor, ...]: Decoder outputs following Hugging Face conventions.
        """
        B, L = input_ids.shape
        requested_cache = use_cache

        # Normalize attention mask to bool for consistent semantics
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=torch.bool)
            if past_key_values is not None and attention_mask.size(1) != input_ids.size(
                1
            ):
                if attention_mask.size(1) < input_ids.size(1):
                    raise ValueError(
                        "attention_mask length must match input_ids when using past_key_values."
                    )
                attention_mask = attention_mask[:, -input_ids.size(1) :]
            if attention_mask.all().item():
                attention_mask = None

        x = self.embed(input_ids) * self.scale
        if max_cache_len == -1:
            cache_limit = self.config.chunk_size
        elif max_cache_len is None:
            cache_limit = self.config.max_cache_len
            if cache_limit is None and not self.config.cache_unbounded:
                cache_limit = self.config.chunk_size
        else:
            cache_limit = max_cache_len

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
            # Check if this is an HF-managed cache (e.g., DynamicCache) vs our LayerCache format
            # HF's caches are not compatible with Megalodon's streaming cache, so we skip them
            # and let the model manage its own cache
            try:
                pkv_list = list(past_key_values)
            except (TypeError, AttributeError):
                # Not iterable or not our cache format - ignore and use fresh cache
                caches = [None] * len(self.layers)
                past_key_values = None
                cache_enabled = use_cache
                pkv_list = []

            # Check if the first non-None item is a LayerCache
            if pkv_list:
                first_item = next((x for x in pkv_list if x is not None), None)
                if first_item is not None and not isinstance(
                    first_item, (LayerCache, NormState)
                ):
                    # Incompatible cache type (e.g., HF DynamicCache, tuple, etc.)
                    # Fall back to fresh cache
                    caches = [None] * len(self.layers)
                    pkv_list = []
                else:
                    max_len = len(self.layers) + 1
                    if len(pkv_list) > max_len:
                        raise ValueError(
                            f"past_key_values length ({len(pkv_list)}) exceeds the "
                            f"maximum supported length ({max_len}) for {len(self.layers)} layers."
                        )

            if pkv_list:
                if len(pkv_list) > len(self.layers):
                    past_final_norm = pkv_list.pop()
                    if past_final_norm is not None and not isinstance(
                        past_final_norm, NormState
                    ):
                        raise TypeError(
                            f"Expected final cache entry to be NormState, got {type(past_final_norm).__name__}"
                        )
                caches = [
                    _clamp_layer_cache(c, cache_limit)
                    for c in pkv_list[: len(self.layers)]
                ]
                if len(caches) < len(self.layers):
                    caches.extend([None] * (len(self.layers) - len(caches)))
            else:
                caches = [None] * len(self.layers)
        all_hidden = [] if output_hidden_states else None

        for i, layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:

                def custom_forward(
                    y: Tensor, *, layer: MegalodonBlock = layer
                ) -> Tensor:
                    """Checkpointing wrapper for a single decoder block.

                    :param Tensor y: Input activations for the block.
                    :param MegalodonBlock layer: Decoder block to run, defaults to the enclosing ``layer``.
                    :return Tensor: Output activations from the block.
                    """
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


class MegalodonForCausalLM(PreTrainedModel, GenerationMixin):
    """Megalodon decoder with tied LM head for causal language modeling."""

    config_class = MegalodonConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ["MegalodonBlock"]

    def __init__(self, config: MegalodonConfig):
        """Wrap a Megalodon decoder with a tied output head.

        :param MegalodonConfig config: Megalodon configuration describing the decoder.
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

    def get_input_embeddings(self) -> nn.Embedding:
        """Return the tied input embeddings.

        :return nn.Embedding: Shared input embedding layer.
        """
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        """Replace the shared input embeddings and keep weights tied.

        :param nn.Embedding value: Replacement embedding layer.
        :return None: This method returns ``None``.
        """
        self.model.set_input_embeddings(value)
        self.tie_weights()

    def get_output_embeddings(self) -> nn.Linear:
        """Return the LM head (HF API compatibility).

        :return nn.Linear: Output projection layer.
        """
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        """Replace the LM head (HF API compatibility).

        :param nn.Linear new_embeddings: Replacement LM head.
        :return None: This method returns ``None``.
        """
        self.lm_head = new_embeddings

    def _tie_weights(self) -> None:
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
        ignore_index: int = -100,
    ) -> CausalLMOutputWithPast | Tuple[Tensor, ...]:
        """Run the decoder and LM head, optionally returning loss for labels.

        :param torch.LongTensor input_ids: Token ids shaped ``(batch, length)``.
        :param Optional[Tensor] attention_mask: Mask with ones for tokens to attend to, defaults to ``None``.
        :param Optional[torch.LongTensor] labels: Optional labels for next-token prediction loss, defaults to ``None``; tokens with indices set to ``ignore_index`` (default ``-100``) are ignored, and loss is only computed for labels in ``[0, ..., config.vocab_size)``.
        :param Optional[List[Optional[LayerCache]]] past_key_values: Optional cache list matching the :class:`LayerCache` layout from a previous decoding step, optionally followed by the final :class:`NormState`; length must be ``<= num_layers + 1``, defaults to ``None``.
        :param bool use_cache: Whether to return updated past key values (ignored during training by the decoder), defaults to ``True``.
        :param bool output_hidden_states: Whether to expose hidden states, defaults to ``False``.
        :param bool output_attentions: Present for HF parity (unused), defaults to ``False``.
        :param Optional[bool] return_dict: Whether to return :class:`CausalLMOutputWithPast`, defaults to ``None``.
        :param Optional[int] max_cache_len: Optional override for the KV cache horizon, defaults to ``None`` (uses configured value, typically ``chunk_size``); ``-1`` clamps to one chunk; set ``cache_unbounded=True`` in the config to disable clamping.
        :param bool enable_training_cache: Opt-in to run cached sequential EMA path during training, defaults to ``False``.
        :param int ignore_index: Label value to ignore in cross-entropy loss (default ``-100``).
        :return CausalLMOutputWithPast or Tuple[Tensor, ...]: Language modeling outputs following Hugging Face conventions.
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
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=ignore_index,
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

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[List[Optional[LayerCache]]] = None,
        attention_mask: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> dict:
        """Prepare model inputs for generation.

        When ``past_key_values`` is provided, only the last token of ``input_ids``
        is used (incremental decoding).

        :param torch.LongTensor input_ids: Token ids shaped ``(batch, length)``.
        :param Optional[List[Optional[LayerCache]]] past_key_values: Optional cache from a previous generation step, defaults to ``None``.
        :param Optional[Tensor] attention_mask: Optional attention mask, defaults to ``None``.
        :param dict[str, Any] kwargs: Additional keyword arguments passed through.
        :return dict[str, Any]: Dictionary of model inputs for the forward pass.
        """
        if past_key_values is not None:
            # Incremental decoding: only use the last token
            input_ids = input_ids[:, -1:]
            # Extend attention mask if provided
            if attention_mask is not None:
                attention_mask = attention_mask[:, -1:]

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": kwargs.get("use_cache", True),
        }

    @staticmethod
    def _reorder_cache(
        past_key_values: tuple,
        beam_idx: torch.LongTensor,
    ) -> tuple:
        """Reorder cache for beam search.

        When using beam search, the batch dimension of the cache needs to be
        reordered to match the selected beam indices at each step.

        :param tuple past_key_values: Tuple containing LayerCache objects plus optional final NormState from previous step.
        :param torch.LongTensor beam_idx: Indices selecting which beams to keep, shape ``(batch,)``.
        :return tuple: Reordered cache tuple.
        """
        reordered = []
        for item in past_key_values:
            if item is None:
                reordered.append(None)
                continue

            # Handle NormState (final norm state, not inside LayerCache)
            if isinstance(item, NormState):
                reordered.append(
                    NormState(
                        count=item.count.index_select(0, beam_idx),
                        mean=item.mean.index_select(0, beam_idx),
                        var=item.var.index_select(0, beam_idx),
                    )
                )
                continue

            # Handle LayerCache
            layer_cache = item

            # Reorder attention cache
            new_attn = None
            if layer_cache.attn is not None:
                attn = layer_cache.attn
                new_attn = AttentionCache(
                    k=attn.k.index_select(0, beam_idx),
                    v=attn.v.index_select(0, beam_idx),
                    count=attn.count.index_select(0, beam_idx),
                    mask=attn.mask.index_select(0, beam_idx)
                    if attn.mask is not None
                    else None,
                )

            # Reorder norm state inside LayerCache
            new_norm = None
            if layer_cache.norm is not None:
                norm = layer_cache.norm
                new_norm = NormState(
                    count=norm.count.index_select(0, beam_idx),
                    mean=norm.mean.index_select(0, beam_idx),
                    var=norm.var.index_select(0, beam_idx),
                )

            # Reorder EMA state
            new_ema = None
            if layer_cache.ema is not None:
                new_ema = layer_cache.ema.index_select(0, beam_idx)

            # Reorder position
            new_position = None
            if layer_cache.position is not None:
                new_position = layer_cache.position.index_select(0, beam_idx)

            reordered.append(
                LayerCache(
                    attn=new_attn,
                    norm=new_norm,
                    ema=new_ema,
                    position=new_position,
                )
            )

        return tuple(reordered)


__all__ = [
    "MegalodonConfig",
    "MegalodonModel",
    "MegalodonForCausalLM",
    "AttentionCache",
    "LayerCache",
]
