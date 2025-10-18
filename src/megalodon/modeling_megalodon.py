# coding=utf-8
"""
modeling_megalodon.py

A PyTorch/transformers Megalodon (decoder-only) implementation with:
  * Complex EMA long-memory (no custom kernels; FFT-based conv)
  * TimestepNorm (streaming group-wise norm, carries state across chunks)
  * Chunked, causal inner attention with Rotary Embeddings and caching
  * Normalized FFN (SwiGLU optional)
  * HF-compatible classes (Config + ForCausalLM) without relying on fused ops

Details:
  - Explicit shapes in docstrings
  - Minimal dtype casts for numerical stability (FFT in fp32, return to input dtype)
  - Deterministic cache semantics (remainder modulo chunk boundary)

References:
Paper: https://arxiv.org/abs/2404.08801
Original Megalodon repo: https://github.com/XuezheMax/megalodon
"""

from __future__ import annotations

import math
from typing import Callable, List, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedModel

from .configuration_megalodon import MegalodonConfig

# -----------------------------------------------------------------------------
# Utilities / inits
# -----------------------------------------------------------------------------


InitFn = Callable[[Tensor], Tensor]


def get_init_fn(mode: str, dim: Optional[int] = None) -> InitFn:
    """Return a callable that applies the requested parameter initialisation.

    :param mode: Name of the init scheme (``"none"``, ``"bert"``, ``"he"``, ``"gaussian"``, ``"xavier"``).
    :type mode: str
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
    """Root-mean-square normalization with an optional affine scale."""

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
    """Rotary positional embedding helper for head-wise Q/K rotation."""

    def __init__(
        self, dim: int, max_positions: int = 1_000_000, base: float = 10_000.0
    ):
        """Precompute rotary frequencies for a ``dim``-dimensional head space.

        :param dim: Per-head dimensionality (must be even).
        :type dim: int
        :param max_positions: Number of positions cached for reuse.
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
        # Precompute angles
        self.register_buffer(
            "angles", self._build_angles(max_positions, dim, base), persistent=False
        )

    @staticmethod
    def _build_angles(max_positions: int, dim: int, base: float) -> torch.Tensor:
        """Compute rotary angles for ``max_positions`` positions and ``dim`` channels.

        :param max_positions: Number of positions to precompute.
        :type max_positions: int
        :param dim: Per-head dimensionality of the rotary embedding.
        :type dim: int
        :param base: Exponential base controlling angular spacing.
        :type base: float
        :returns: Tensor of shape ``(max_positions, dim/2)`` holding phase angles.
        :rtype: torch.Tensor
        """
        half = dim // 2
        freqs = torch.exp(
            torch.arange(half, dtype=torch.float32) * -(math.log(base) / half)
        )
        t = torch.arange(max_positions, dtype=torch.float32).unsqueeze(
            1
        ) * freqs.unsqueeze(0)  # (T, half)
        return t

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
        angles = self.angles[start : start + length].to(device=device)
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

        stats_dtype = (
            torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
        )

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

        prev_count_clamped = torch.clamp(prev_count, min=1).to(stats_dtype)
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
    """Complex EMA layer that matches Megalodon's moving-average sub-block."""

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

        # Parameters per (D, N, 1) etc.
        self.alpha = nn.Parameter(
            torch.zeros(embed_dim, ndim, 1)
        )  # -> p = sigmoid(alpha)
        self.delta = nn.Parameter(
            torch.zeros(embed_dim, ndim, 1)
        )  # -> d = sigmoid(delta)
        self.theta = nn.Parameter(
            torch.zeros(embed_dim, 1, 1)
        )  # -> base angle multipliers
        self.gamma = nn.Parameter(
            torch.zeros(embed_dim, ndim, 2)
        )  # -> complex mixing (real, imag)
        self.omega = nn.Parameter(torch.zeros(embed_dim))  # residual scaling

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize EMA parameters following the reference distribution."""
        nn.init.normal_(self.alpha, mean=0.0, std=0.2)
        nn.init.normal_(self.delta, mean=0.0, std=0.2)
        nn.init.normal_(self.theta, mean=0.0, std=0.2)
        nn.init.normal_(self.gamma, mean=0.0, std=1.0)
        nn.init.normal_(self.omega, mean=0.0, std=1.0)

    @staticmethod
    def _r2c(z: torch.Tensor) -> torch.Tensor:
        """Convert real-valued two-channel tensors to complex values."""
        return torch.complex(z[..., 0], z[..., 1])

    def _coeffs(self):
        """Compute damping/decay coefficients for the sequential recurrence."""
        # All in fp32 for stability
        p = torch.sigmoid(self.alpha.float())
        d = torch.sigmoid(self.delta.float())
        wave = torch.arange(
            1, self.ndim + 1, dtype=torch.float32, device=self.alpha.device
        ).view(1, self.ndim, 1)
        base = torch.sigmoid(self.theta.float()) * (2.0 * math.pi / float(self.ndim))
        phi = wave * base  # (D, N, 1)
        q = (1.0 - p * d) * torch.exp(1j * phi)  # (D, N, 1) complex
        gamma = self._r2c(self.gamma.float()) * self.scale  # (D, N)
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
        p, q, gamma = self._coeffs()
        p = p.squeeze(-1).to(torch.complex64)
        q = q.squeeze(-1)  # (D, N)
        gamma = gamma.to(torch.complex64)
        x_c = x.to(torch.complex64)

        if hx is not None:
            hx_c = hx if hx.dtype.is_complex else torch.complex(hx[..., 0], hx[..., 1])
            h = hx_c.to(torch.complex64)
        else:
            h = torch.zeros(B, D, self.ndim, dtype=torch.complex64, device=x.device)

        out_c = torch.empty(B, D, L, dtype=torch.complex64, device=x.device)
        p_b = p.unsqueeze(0)  # (1, D, N)

        for t in range(L):
            xt = x_c[:, :, t].unsqueeze(-1)  # (B, D, 1)
            h = q * h + p_b * xt
            out_c[:, :, t] = (h * gamma.unsqueeze(0)).sum(dim=-1)

        y = torch.real(out_c).to(dtype=x.dtype)
        return y, h

    def _forward_fft(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """FFT-based convolution path used when no streaming state is required."""
        B, D, L = x.shape
        if L == 0:
            return x.new_zeros(B, D, L), None

        p, q, gamma = self._coeffs()
        p = p.squeeze(-1).to(torch.complex64)  # (D, N)
        q = q.squeeze(-1).to(torch.complex64)  # (D, N)
        gamma = gamma.to(torch.complex64)  # (D, N)

        t = torch.arange(L, device=x.device, dtype=torch.float32).view(1, 1, -1)
        q_pows = torch.pow(q.unsqueeze(-1), t)  # (D, N, L)
        kernel = (gamma.unsqueeze(-1) * p.unsqueeze(-1) * q_pows).sum(dim=1)  # (D, L)

        fft_len = 1 << (int(2 * L - 1).bit_length())
        x_fp32 = x.to(torch.float32)
        x_complex = torch.complex(x_fp32, torch.zeros_like(x_fp32))
        k_complex = kernel.to(torch.complex64)

        X = torch.fft.fft(x_complex, n=fft_len, dim=-1)
        K = torch.fft.fft(k_complex, n=fft_len, dim=-1)
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


class AttentionCache(NamedTuple):
    """Tuple storing cached keys, values, and token count for streaming attention.

    :param k: Cached key tensor shaped ``(batch, cached_length, heads, dim)``.
    :type k: torch.Tensor
    :param v: Cached value tensor shaped ``(batch, cached_length, heads, dim_v)``.
    :type v: torch.Tensor
    :param count: Total number of tokens processed so far.
    :type count: int
    """

    k: torch.Tensor  # (B, Lc, H, Dh)
    v: torch.Tensor  # (B, Lc, H, Dv)
    count: int  # total tokens seen (for RoPE index)


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
        rope_base: float,
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
        :type rope_base: float
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

    def _apply_dropkey(
        self, scores: Tensor, training: bool, keep_cols: Optional[Tensor] = None
    ) -> Tensor:
        """Apply DropKey-style dropout to the attention scores before softmax."""
        p = self.attention_dropout
        if not training or p <= 0.0:
            return scores
        drop_mask = (torch.rand_like(scores) < p) & ~torch.isinf(scores)
        if keep_cols is not None:
            keep_cols = keep_cols.to(device=scores.device, dtype=torch.long)
            rows = torch.arange(scores.size(-2), device=scores.device)
            valid = min(rows.numel(), keep_cols.numel())
            if valid > 0:
                rows = rows[:valid]
                cols = keep_cols[:valid]
                drop_mask[..., rows, cols] = False
        drop_mask[..., -1] = False  # safety for trailing position
        return scores.masked_fill(drop_mask, float("-inf"))

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        start_index: int,
        cache: Optional[AttentionCache],
        attn_mask: Optional[Tensor],
        training: bool,
    ) -> Tuple[Tensor, Optional[AttentionCache]]:
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
        :returns: Tuple of attention output and updated cache (if any).
        :rtype: Tuple[Tensor, Optional[AttentionCache]]
        """
        B, L, H, Dh = q.shape
        Dv = v.size(-1)
        device, dtype = q.device, q.dtype

        # RoPE rotates the *suffix* (current step/chunk) starting at `start_index`
        q, k = self.rope(q, k, start_index=start_index)

        # Cache handling: prefix context
        if cache is not None:
            prefix_len = cache.k.size(1)
            k = torch.cat([cache.k, k], dim=1)
            v = torch.cat([cache.v, v], dim=1)
            seen = cache.count
        else:
            prefix_len = 0
            seen = 0

        # Single-block path
        if L <= self.chunk_size:
            Lk = k.size(1)
            q_ = q.transpose(1, 2)  # (B,H,L,Dh)
            k_ = k.transpose(1, 2)  # (B,H,Lk,Dh)
            v_ = v.transpose(1, 2)  # (B,H,Lk,Dv)

            all_tokens = True
            if attn_mask is not None:
                all_tokens = bool(attn_mask.all().item())

            use_sdpa = (
                hasattr(F, "scaled_dot_product_attention")
                and prefix_len == 0
                and all_tokens
                and self.attention_dropout == 0.0
            )

            if use_sdpa:
                attn = F.scaled_dot_product_attention(
                    q_,
                    k_,
                    v_,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=True,
                )
                out = attn
            else:
                scores = torch.matmul(q_, k_.transpose(-2, -1)) / math.sqrt(Dh)
                scores = scores + self._causal_mask(
                    L, Lk, device, dtype, offset=prefix_len
                )

                if attn_mask is not None:
                    if prefix_len > 0:
                        prefix_mask = attn_mask.new_ones(B, prefix_len)
                        mask = torch.cat([prefix_mask, attn_mask], dim=1)
                    else:
                        mask = attn_mask
                    pad = (mask.to(dtype) - 1.0) * 1e9
                    scores = scores + pad.unsqueeze(1).unsqueeze(2)

                diag_idx = torch.arange(L, device=device) + prefix_len
                diag_idx = torch.clamp(diag_idx, max=Lk - 1)
                scores = self._apply_dropkey(scores, training, keep_cols=diag_idx)
                attn = torch.softmax(scores.float(), dim=-1).to(q_)
                out = torch.matmul(attn, v_)

            out = out.transpose(1, 2)  # (B,L,H,Dv)

            total = seen + L
            keep = total % self.chunk_size
            new_cache = (
                AttentionCache(k=k[:, -keep:], v=v[:, -keep:], count=total)
                if keep > 0
                else None
            )
            out = out.reshape(B, L, H * Dv)
            return out, new_cache

        # Multi-chunk: block-diagonal causal attention
        assert L % self.chunk_size == 0, (
            "For training, sequence length must be divisible by chunk_size"
        )
        nc = L // self.chunk_size
        q_chunks = q.view(B, nc, self.chunk_size, H, Dh)
        k_chunks = k[:, -L:].view(B, nc, self.chunk_size, H, Dh)
        v_chunks = v[:, -L:].view(B, nc, self.chunk_size, H, Dv)

        outs = []
        mask_block = self._causal_mask(self.chunk_size, self.chunk_size, device, dtype)
        if attn_mask is not None:
            attn_mask = attn_mask.view(B, nc, self.chunk_size)

        for i in range(nc):
            q_i = q_chunks[:, i].transpose(1, 2)  # (B,H,C,Dh)
            k_i = k_chunks[:, i].transpose(1, 2)
            v_i = v_chunks[:, i].transpose(1, 2)

            chunk_len = q_i.size(-2)
            scores = torch.matmul(q_i, k_i.transpose(-2, -1)) / math.sqrt(Dh)
            scores = scores + mask_block

            if attn_mask is not None:
                mask_i = attn_mask[:, i]
                pad = (mask_i.to(dtype) - 1.0) * 1e9
                scores = scores + pad.unsqueeze(1).unsqueeze(2)

            diag_idx = torch.arange(chunk_len, device=device)
            scores = self._apply_dropkey(scores, training, keep_cols=diag_idx)
            attn = torch.softmax(scores.float(), dim=-1).to(q_i)
            out_i = torch.matmul(attn, v_i).transpose(1, 2)  # (B,C,H,Dv)
            outs.append(out_i)

        out = torch.cat(outs, dim=1).reshape(B, L, H * Dv)
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
        cache: Optional[
            Tuple[
                Optional[AttentionCache],
                Tuple[Tensor, Tensor, Tensor],
                Optional[Tensor],
            ]
        ] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[
        Tensor,
        Optional[
            Tuple[
                Optional[AttentionCache],
                Tuple[Tensor, Tensor, Tensor],
                Optional[Tensor],
            ]
        ],
    ]:
        """Run the Megalodon attention block and return outputs plus cache.

        :param x: Input activations shaped ``(batch, length, dim)``.
        :type x: Tensor
        :param cache: Optional tuple of attention/norm/EMA caches for streaming.
        :type cache: Optional[Tuple[Optional[AttentionCache], Tuple[Tensor, Tensor, Tensor], Optional[Tensor]]]
        :param attn_mask: Optional attention mask with ones for valid tokens.
        :type attn_mask: Optional[Tensor]
        :returns: Tuple containing the updated activations and optional caches.
        :rtype: Tuple[Tensor, Optional[Tuple[Optional[AttentionCache], Tuple[Tensor, Tensor, Tensor], Optional[Tensor]]]]
        """
        B, L, D = x.shape
        residual = x

        # Unpack caches
        if cache is not None:
            if len(cache) == 3:
                attn_cache, norm_cache, ema_state = cache
            else:
                attn_cache, norm_cache = cache
                ema_state = None
            prev_count, prev_mean, prev_var = norm_cache
            hx = ema_state
        else:
            attn_cache = None
            prev_count = prev_mean = prev_var = None
            hx = None

        # 1) TimestepNorm (streaming)
        x_tn, new_count, new_mean, new_var = self.timenorm(
            x, prev_count, prev_mean, prev_var, attn_mask
        )

        # 2) Complex EMA over channels (B,D,L)
        y_cema, h_last = self.cema(x_tn.transpose(1, 2), hx=hx, compute_last_state=True)
        y_cema = y_cema.transpose(1, 2)

        # 3) RMSNorm + dropout
        mx = F.dropout(
            self.rmsnorm(y_cema), p=self.hidden_dropout, training=self.training
        )

        # 4) Shared Z, normalize per-head (RMS), then affine to Q/K
        z = self.wz(mx)  # (B, L, Z)
        z_heads = self._split_heads(z, self.z_head)  # (B, L, H, z_head)
        z_norm = z_heads / (
            z_heads.pow(2).mean(dim=-1, keepdim=True).add(self.norm_eps).sqrt()
        )
        z = self._merge_heads(z_norm)

        scale = (self.gamma + 1.0) / math.sqrt(self.z_head)  # (2, Z)
        z_aff = z.unsqueeze(2) * scale.unsqueeze(0).unsqueeze(0) + self.beta.unsqueeze(
            0
        ).unsqueeze(0)
        q, k = torch.unbind(z_aff, dim=2)  # (B, L, Z) each
        q = self._split_heads(q, self.z_head)  # (B, L, H, z_head)
        k = self._split_heads(k, self.z_head)

        # 5) Values and residual gate
        v = F.silu(self.wv(x_tn)).view(B, L, self.H, self.v_head)  # (B,L,H,v_head)
        r = F.silu(self.wr(mx))  # (B,L,E)

        # 6) Inner attention
        start_index = attn_cache.count if attn_cache is not None else 0
        out, new_attn = self.inner(
            q,
            k,
            v,
            start_index=start_index,
            cache=attn_cache,
            attn_mask=attn_mask,
            training=self.training,
        )

        # 7) Gate and project
        out = out * r
        h = self.wh1(mx) + self.wh2(out)
        h = F.dropout(h, p=self.dropout, training=self.training)
        y = h + residual

        ema_next = h_last.detach() if h_last is not None else None
        if attn_cache is not None or new_attn is not None or ema_next is not None:
            new_cache = (
                new_attn,
                (new_count.detach(), new_mean.detach(), new_var.detach()),
                ema_next,
            )
        else:
            new_cache = None
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
        self.norm = RMSNorm(D, eps=cfg.norm_eps, affine=cfg.norm_affine)
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

    def forward(self, x: Tensor) -> Tensor:
        """Run the normalized feed-forward block with optional SwiGLU."""
        residual = x
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

    def forward(self, x: Tensor, cache=None, attn_mask=None):
        """Apply attention + FFN returning updated states and cache."""
        x, cache = self.attn(x, cache=cache, attn_mask=attn_mask)
        x = self.ffn(x)
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
        self.layers = nn.ModuleList(
            [MegalodonBlock(config, i) for i in range(config.num_layers)]
        )
        self.norm = RMSNorm(D, eps=config.norm_eps, affine=config.norm_affine)
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
        past_key_values: Optional[List] = None,
        use_cache: bool = True,
        output_hidden_states: bool = False,
        output_attentions: bool = False,  # not used; kept for HF parity
    ):
        """Run embedding lookup and stacked decoder blocks over ``input_ids``.

        :param input_ids: Token ids shaped ``(batch, length)``.
        :type input_ids: torch.LongTensor
        :param attention_mask: Mask with ones for valid tokens.
        :type attention_mask: Optional[Tensor]
        :param past_key_values: Optional caches for streaming decoding.
        :type past_key_values: Optional[List]
        :param use_cache: Whether to return updated caches.
        :type use_cache: bool
        :param output_hidden_states: Whether to collect per-layer hidden states.
        :type output_hidden_states: bool
        :param output_attentions: Included for Hugging Face parity (unused).
        :type output_attentions: bool
        :returns: Tuple containing final hidden states, optional caches, and optional layer outputs.
        :rtype: Tuple
        """
        B, L = input_ids.shape
        requested_cache = use_cache
        self.config.gradient_checkpointing = self.gradient_checkpointing
        x = self.embed(input_ids) * self.scale

        use_cache = use_cache and not (self.gradient_checkpointing and self.training)

        caches = past_key_values or [None] * len(self.layers)
        all_hidden = [] if output_hidden_states else None

        for i, layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:

                def custom_forward(y, *, layer=layer):
                    return layer(y, cache=None, attn_mask=attention_mask)[0]

                x = self._gradient_checkpointing_func(custom_forward, x)
                caches[i] = None
            else:
                x, caches[i] = layer(x, cache=caches[i], attn_mask=attention_mask)
            if output_hidden_states:
                all_hidden.append(x)

        x = self.norm(x)

        out = (x,)
        if use_cache:
            out = out + (caches,)
        elif requested_cache:
            out = out + (None,)
        if output_hidden_states:
            out = out + (all_hidden,)
        return out


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
        past_key_values: Optional[List] = None,
        use_cache: bool = True,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ):
        """Run the decoder and LM head, optionally returning loss for labels.

        :param input_ids: Token ids shaped ``(batch, length)``.
        :type input_ids: torch.LongTensor
        :param attention_mask: Mask with ones for tokens to attend to.
        :type attention_mask: Optional[Tensor]
        :param labels: Optional labels for next-token prediction loss.
        :type labels: Optional[torch.LongTensor]
        :param past_key_values: Optional cache from a previous decoding step.
        :type past_key_values: Optional[List]
        :param use_cache: Whether to return updated past key values.
        :type use_cache: bool
        :param output_hidden_states: Whether to expose hidden states.
        :type output_hidden_states: bool
        :param output_attentions: Present for HF parity (unused).
        :type output_attentions: bool
        :returns: Tuple containing logits, optional loss, and optional caches.
        :rtype: Tuple
        """
        last_hidden, *rest = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        logits = self.lm_head(last_hidden)

        loss = None
        if labels is not None:
            # shift for CLM
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        cache = None
        hidden_states = None
        if use_cache and rest:
            cache = rest[0]
            rest = rest[1:]
        if output_hidden_states and rest:
            hidden_states = rest[0]

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
]
