# coding=utf-8
"""
modeling_megalodon.py

Pure-PyTorch Megalodon (decoder-only) implementation with:
  * Complex EMA long-memory (no custom kernels; FFT-based conv)
  * TimestepNorm (streaming group-wise norm, carries state across chunks)
  * Chunked, causal inner attention with Rotary Embeddings and caching
  * Normalized FFN (SwiGLU optional)
  * HF-compatible classes (Config + ForCausalLM) without relying on fused ops

Best practices:
  - Explicit shapes in docstrings
  - Minimal dtype casts for numerical stability (FFT in fp32, return to input dtype)
  - Deterministic cache semantics (remainder modulo chunk boundary)
"""

from __future__ import annotations

import math
from typing import List, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .configuration_megalodon import MegalodonConfig

try:
    from transformers import PreTrainedModel

    _HAS_HF = True
except Exception:
    PreTrainedModel = nn.Module  # fallback
    _HAS_HF = False


# -----------------------------------------------------------------------------
# Utilities / inits
# -----------------------------------------------------------------------------


def get_init_fn(mode: str, dim: Optional[int] = None):
    """Return a weight init function per `mode`."""
    if mode == "none":
        return lambda w: w
    if mode == "bert":
        std = 0.02
        return lambda w: nn.init.normal_(w, mean=0.0, std=std)
    if mode == "he":
        return lambda w: nn.init.kaiming_normal_(w, a=math.sqrt(5.0))
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
    """RMSNorm with optional affine scale.

    Shape:
        * Input:  (B, L, D)
        * Output: (B, L, D)
    """

    def __init__(self, dim: int, eps: float = 1e-6, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if affine else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        y = x * rms
        if self.weight is not None:
            y = y * self.weight
        return y


# -----------------------------------------------------------------------------
# Rotary positional embedding
# -----------------------------------------------------------------------------


class RotaryEmbedding(nn.Module):
    """RoPE for Q/K in each head.

    Args:
        dim: head dimension (must be even because we treat pairs as complex).
        max_positions: max positions cached.
        base: frequency base.

    Shapes:
        q, k: (B, T, H, Dh), with Dh = dim and even
    """

    def __init__(
        self, dim: int, max_positions: int = 1_000_000, base: float = 10_000.0
    ):
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
        half = dim // 2
        freqs = torch.exp(
            torch.arange(half, dtype=torch.float32) * -(math.log(base) / half)
        )
        t = torch.arange(max_positions, dtype=torch.float32).unsqueeze(
            1
        ) * freqs.unsqueeze(0)  # (T, half)
        return t

    def _get_cis(self, start: int, length: int, device, dtype):
        angles = self.angles[start : start + length].to(device=device)
        return torch.cos(angles).to(dtype), torch.sin(angles).to(dtype)

    @staticmethod
    def _pair_to_complex(x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=-1)
        return torch.complex(a, b)

    @staticmethod
    def _complex_to_pair(x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x.real, x.imag], dim=-1)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, start_index: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
    """Streaming group-wise normalization across time.

    The feature dimension D is split into G groups of size D//G. For each timestep,
    the group mean/var is updated (Welford) and used to normalize that timestep's features.
    State (count, mean, var) is optionally carried across chunks.

    Shapes
    ------
    x: (B, L, D)
    prev_count: (B,) int64 or None
    prev_mean: (B, G) or None
    prev_var: (B, G) or None
    padding_mask: (B, L) 1=token, 0=pad or None

    Returns
    -------
    y: (B, L, D)
    new_count: (B,)
    new_mean: (B, G)
    new_var: (B, G)
    """

    def __init__(
        self,
        num_features: int,
        num_groups: int,
        eps: float = 1e-5,
        affine: bool = True,
    ):
        super().__init__()
        if num_features % num_groups != 0:
            raise ValueError("num_features must be divisible by num_groups")
        self.num_features = num_features
        self.num_groups = num_groups
        self.group_size = num_features // num_groups
        self.eps = eps
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_buffer("weight", torch.ones(num_features), persistent=False)
            self.register_buffer("bias", torch.zeros(num_features), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        prev_count: Optional[torch.Tensor] = None,
        prev_mean: Optional[torch.Tensor] = None,
        prev_var: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

        x_groups = x.view(B, L, G, gs)
        y = torch.empty_like(x)

        mean = prev_mean
        var = prev_var
        count = prev_count.clone()

        # NOTE: stepwise loop keeps gradients (unrolled time dependency)
        for t in range(L):
            m_t = x_groups[:, t].mean(dim=-1)  # (B, G)
            mask_t = padding_mask[:, t]  # (B,)
            valid = mask_t.view(B, 1)

            c_new = count + mask_t.to(count.dtype)  # (B,)
            c_safe = torch.clamp(c_new, min=1)

            delta = m_t - mean
            mean = mean + (delta * valid) / c_safe.view(B, 1)

            m2 = var * torch.clamp(count, min=1).view(B, 1)
            delta2 = m_t - mean
            m2 = m2 + (delta * delta2 * valid)
            var = m2 / c_safe.view(B, 1)

            count = c_new

            mean_b = mean.unsqueeze(-1).expand(B, G, gs)
            var_b = var.unsqueeze(-1).expand(B, G, gs)

            x_t = x_groups[:, t]
            x_hat = (x_t - mean_b.to(x_t)) * torch.rsqrt(var_b.to(x_t) + self.eps)

            w = self.weight.view(1, G, gs).to(x_t)
            b = self.bias.view(1, G, gs).to(x_t)
            x_t_norm = x_hat * w + b

            y[:, t] = x_t_norm.view(B, D)

        return y, count, mean.to(dtype), var.to(dtype)


# -----------------------------------------------------------------------------
# Complex EMA (FFT-based conv; optional state recurrence for last hidden)
# -----------------------------------------------------------------------------


class ComplexEMA(nn.Module):
    """Multi-dimensional complex EMA used by Megalodon.

    Parameters
    ----------
    embed_dim: int
        Hidden size ``D``.
    ndim: int
        Number of EMA components ``N``.

    Notes
    -----
    * Computes a real-valued kernel via complex exponentials and applies
      1D convolution over time using FFT. Residual scaling is added.
    * When `compute_last_state=True`, we also return the last complex EMA
      state by explicitly rolling the recurrence (O(L*N)).
    """

    def __init__(self, embed_dim: int, ndim: int):
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

    def reset_parameters(self):
        nn.init.normal_(self.alpha, mean=0.0, std=0.2)
        nn.init.normal_(self.delta, mean=0.0, std=0.2)
        nn.init.normal_(self.theta, mean=0.0, std=0.2)
        nn.init.normal_(self.gamma, mean=0.0, std=1.0)
        nn.init.normal_(self.omega, mean=0.0, std=1.0)

    @staticmethod
    def _r2c(z: torch.Tensor) -> torch.Tensor:
        return torch.complex(z[..., 0], z[..., 1])

    def _coeffs(self):
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
        """Evaluate EMA recurrence sequentially (supports carry state)."""
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

    def forward(
        self,
        x: torch.Tensor,  # (B, D, L)
        hx: Optional[torch.Tensor] = None,  # (B, D, N) complex or last dim 2
        compute_last_state: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = x * self.omega.view(1, -1, 1).to(x)
        B, D, L = x.shape
        y_seq, h_last = self._forward_sequential(x, hx)
        y = y_seq + residual
        return y, (h_last if compute_last_state else None)


# -----------------------------------------------------------------------------
# Inner (chunked) attention
# -----------------------------------------------------------------------------


class AttentionCache(NamedTuple):
    k: torch.Tensor  # (B, Lc, H, Dh)
    v: torch.Tensor  # (B, Lc, H, Dv)
    count: int  # total tokens seen (for RoPE index)


class ChunkedSelfAttention(nn.Module):
    """Scaled dot-product attention with chunking + RoPE and cache.

    Args:
        num_heads: H
        head_dim:  Dh for Q/K
        value_head_dim: Dv for V
        chunk_size: block size used for causal chunking
        dropout: attention prob dropout

    Input shapes:
        q, k, v: (B, L, H, Dh/Dv)
        attn_mask: (B, L) 1=keep, 0=pad

    Returns:
        out: (B, L, H*Dv)
        new_cache: Optional[AttentionCache]
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
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.value_head_dim = value_head_dim
        self.chunk_size = chunk_size
        base = 10_000.0 if rope_base is None else rope_base
        self.rope = RotaryEmbedding(head_dim, base=base)
        self.attention_dropout = attention_dropout

    @staticmethod
    def _causal_mask(Lq: int, Lk: int, device, dtype):
        m = torch.full((Lq, Lk), float("-inf"), device=device, dtype=dtype)
        i = torch.arange(Lq, device=device).unsqueeze(1)
        j = torch.arange(Lk, device=device).unsqueeze(0)
        m[j <= i] = 0.0
        return m

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        start_index: int,
        cache: Optional[AttentionCache],
        attn_mask: Optional[torch.Tensor],
        training: bool,
    ) -> Tuple[torch.Tensor, Optional[AttentionCache]]:
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

            scores = torch.matmul(q_, k_.transpose(-2, -1)) / math.sqrt(Dh)
            scores = scores + self._causal_mask(L, Lk, device, dtype)

            if attn_mask is not None:
                if prefix_len > 0:
                    prefix_mask = attn_mask.new_ones(B, prefix_len)
                    mask = torch.cat([prefix_mask, attn_mask], dim=1)
                else:
                    mask = attn_mask
                pad = (mask.to(dtype) - 1.0) * 1e9
                scores = scores + pad.unsqueeze(1).unsqueeze(2)  # (B,1,1,Lk)

            attn = torch.softmax(scores.float(), dim=-1).to(q_)
            attn = F.dropout(attn, p=self.attention_dropout, training=training)
            out = torch.matmul(attn, v_)  # (B,H,L,Dv)
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
        assert (L % self.chunk_size) == 0, (
            "For training, L must be multiple of chunk_size"
        )
        nc = L // self.chunk_size
        q_chunks = q.view(B, nc, self.chunk_size, H, Dh)
        k_chunks = k[:, -L:].view(B, nc, self.chunk_size, H, Dh)
        v_chunks = v[:, -L:].view(B, nc, self.chunk_size, H, Dv)

        outs = []
        mask_block = self._causal_mask(self.chunk_size, self.chunk_size, device, dtype)
        for i in range(nc):
            q_i = q_chunks[:, i].transpose(1, 2)  # (B,H,C,Dh)
            k_i = k_chunks[:, i].transpose(1, 2)
            v_i = v_chunks[:, i].transpose(1, 2)
            scores = torch.matmul(q_i, k_i.transpose(-2, -1)) / math.sqrt(Dh)
            scores = scores + mask_block
            attn = torch.softmax(scores.float(), dim=-1).to(q_i)
            attn = F.dropout(attn, p=self.attention_dropout, training=training)
            out_i = torch.matmul(attn, v_i).transpose(1, 2)  # (B,C,H,Dv)
            outs.append(out_i)

        out = torch.cat(outs, dim=1).reshape(B, L, H * Dv)
        return out, None


# -----------------------------------------------------------------------------
# Megalodon Attention block (EMA → gates → chunked attention)
# -----------------------------------------------------------------------------


class MegalodonAttention(nn.Module):
    """EMA + gated chunked attention as in Megalodon.

    Forward
    -------
    x: (B, L, D)
    cache: Optional[ (AttentionCache, (count, mean, var)) ]
    attn_mask: (B, L) 1=token, 0=pad

    Returns:
        y: (B, L, D)
        new_cache: Optional[ (AttentionCache, (count, mean, var)) ]
    """

    def __init__(self, cfg: MegalodonConfig):
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

    def _split_heads(self, x: torch.Tensor, head_dim: int) -> torch.Tensor:
        B, L, T = x.shape
        return x.view(B, L, self.H, head_dim)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, L, H, Dh = x.shape
        return x.reshape(B, L, H * Dh)

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[
            Tuple[
                Optional[AttentionCache],
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                Optional[torch.Tensor],
            ]
        ] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        Optional[
            Tuple[
                Optional[AttentionCache],
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                Optional[torch.Tensor],
            ]
        ],
    ]:
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
    """(Optionally) SwiGLU FFN with RMSNorm pre/post and residual rescale."""

    def __init__(self, cfg: MegalodonConfig, layer_id: int):
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
        return x if self.alpha is None else (self.alpha * x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        super().__init__()
        self.attn = MegalodonAttention(cfg)
        self.ffn = NormalizedFFN(cfg, layer_id)

    def forward(self, x: torch.Tensor, cache=None, attn_mask=None):
        x, cache = self.attn(x, cache=cache, attn_mask=attn_mask)
        x = self.ffn(x)
        return x, cache


# -----------------------------------------------------------------------------
# Model + LM head
# -----------------------------------------------------------------------------


class MegalodonModel(PreTrainedModel):
    """Bare Megalodon decoder.

    Forward
    -------
    input_ids: (B, L) token ids
    attention_mask: (B, L) 1=token, 0=pad
    past_key_values: list[Optional[AttentionCache, (count, mean, var)]], len = num_layers
    use_cache: if True, return new caches for incremental decoding
    output_hidden_states: if True, also return per-layer hidden states

    Returns
    -------
    last_hidden_state: (B, L, D)
    past_key_values (optional)
    hidden_states (optional): List[(B, L, D)] per layer output
    """

    config_class = MegalodonConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ["MegalodonBlock"]

    def __init__(self, config: MegalodonConfig):
        super().__init__(config)
        D = config.model_dim
        self.embed = nn.Embedding(config.vocab_size, D, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList(
            [MegalodonBlock(config, i) for i in range(config.num_layers)]
        )
        self.norm = RMSNorm(D, eps=config.norm_eps, affine=config.norm_affine)
        self.scale = math.sqrt(D) if config.scale_emb else 1.0
        self.gradient_checkpointing = bool(config.gradient_checkpointing)
        if _HAS_HF:
            self.post_init()

    def get_input_embeddings(self):
        return self.embed

    def set_input_embeddings(self, value: nn.Embedding):
        self.embed = value

    def _gradient_checkpointing_func(self, func, *inputs):
        return torch.utils.checkpoint.checkpoint(func, *inputs, use_reentrant=False)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List] = None,
        use_cache: bool = True,
        output_hidden_states: bool = False,
        output_attentions: bool = False,  # not used; kept for HF parity
    ):
        B, L = input_ids.shape
        requested_cache = use_cache
        self.config.gradient_checkpointing = self.gradient_checkpointing
        x = self.embed(input_ids) * self.scale

        use_cache = use_cache and not (self.gradient_checkpointing and self.training)

        caches = past_key_values or [None] * len(self.layers)
        all_hidden = [] if output_hidden_states else None

        for i, layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:

                def custom_forward(y):
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
        if _HAS_HF:
            self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Embedding):
        self.model.set_input_embeddings(value)
        self.tie_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def _tie_weights(self):
        if self._tied_embeddings:
            self.lm_head.weight = self.model.embed.weight

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List] = None,
        use_cache: bool = True,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ):
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
