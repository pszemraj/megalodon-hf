# coding=utf-8
# Megalodon (pure PyTorch) — decoder-only CLM with chunking, EMA, and timestep norm
# Aligned to pszemraj/megalodon (official reference) but without custom CUDA kernels, model-parallel, or apex.
# This file is self-contained for learning/maintenance. Performance will be lower than the fused version,
# but the math mirrors the reference implementation as closely as possible.
#
# Key components mapped from the repo to pure PyTorch:
# - MovingAverageGatedAttention  -> MegalodonAttention (with inner chunked softmax attention)
# - MultiHeadComplexEMA          -> ComplexEMA (pure torch, FFT/Vandermonde based; optionally recurrent hidden-state)
# - TimestepNorm                 -> TimestepNorm (pure torch, streaming group stats across time with carryover)
# - FusedRMSNorm                 -> RMSNorm (pure torch)
# - Column/RowParallelLinear     -> nn.Linear
# - memory_efficient_dropout     -> F.dropout
#
# Notes:
# * Shapes follow the reference: batch-first (B, L, D) on the module boundaries; EMA works on (B, D, L).
# * Chunking enforces block-diagonal (causal) attention of chunk_size. Caches reset on chunk boundaries.
# * Rotary embeddings are included (apply to q/k before attention).

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Minimal config (Hugging Face compatible if available)
# -----------------------------------------------------------------------------

try:
    from transformers import PreTrainedModel
    from transformers.configuration_utils import PreTrainedConfig
    _HAS_HF = True
except Exception:
    PreTrainedModel = nn.Module  # fallback
    class PreTrainedConfig:  # minimal shim
        model_type: str = "megalodon"
        def __init__(self, **kwargs):
            for k, v in kwargs.items(): setattr(self, k, v)
    _HAS_HF = False


@dataclass
class MegalodonDefaults:
    vocab_size: int = 50257
    model_dim: int = 1024          # D
    num_layers: int = 24
    num_heads: int = 8
    z_dim: int = 512               # shared rep for Q/K (S)
    value_dim: int = 2048          # V projection/output (E)
    ffn_hidden_dim: int = 4096
    cema_ndim: int = 16            # number of complex EMA components (N)
    chunk_size: int = 2048
    norm_num_groups: int = 64      # groups for TimestepNorm (must divide D)
    dropout: float = 0.0
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    swiglu: bool = False
    rescale_nffn: bool = False
    scale_emb: bool = False
    share_emb: bool = False
    init_mode: str = "gaussian"    # {"gaussian","xavier","he","bert","none"}
    max_positions: int = 1_000_000 # rope cache upper bound
    pad_token_id: int = 0


class MegalodonConfig(PreTrainedConfig):
    model_type = "megalodon"
    def __init__(
        self,
        vocab_size=MegalodonDefaults.vocab_size,
        model_dim=MegalodonDefaults.model_dim,
        num_layers=MegalodonDefaults.num_layers,
        num_heads=MegalodonDefaults.num_heads,
        z_dim=MegalodonDefaults.z_dim,
        value_dim=MegalodonDefaults.value_dim,
        ffn_hidden_dim=MegalodonDefaults.ffn_hidden_dim,
        cema_ndim=MegalodonDefaults.cema_ndim,
        chunk_size=MegalodonDefaults.chunk_size,
        norm_num_groups=MegalodonDefaults.norm_num_groups,
        dropout=MegalodonDefaults.dropout,
        attention_dropout=MegalodonDefaults.attention_dropout,
        hidden_dropout=MegalodonDefaults.hidden_dropout,
        swiglu=MegalodonDefaults.swiglu,
        rescale_nffn=MegalodonDefaults.rescale_nffn,
        scale_emb=MegalodonDefaults.scale_emb,
        share_emb=MegalodonDefaults.share_emb,
        init_mode=MegalodonDefaults.init_mode,
        max_positions=MegalodonDefaults.max_positions,
        pad_token_id=MegalodonDefaults.pad_token_id,
        bos_token_id=1,
        eos_token_id=2,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs
        )
        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.z_dim = z_dim
        self.value_dim = value_dim
        self.ffn_hidden_dim = ffn_hidden_dim
        self.cema_ndim = cema_ndim
        self.chunk_size = chunk_size
        self.norm_num_groups = norm_num_groups
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout
        self.swiglu = swiglu
        self.rescale_nffn = rescale_nffn
        self.scale_emb = scale_emb
        self.share_emb = share_emb
        self.init_mode = init_mode
        self.max_positions = max_positions
        self.is_decoder = True
        self.use_cache = True


# -----------------------------------------------------------------------------
# Utilities / inits
# -----------------------------------------------------------------------------

def get_init_fn(mode: str, dim: Optional[int] = None):
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
# RMSNorm
# -----------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, affine: bool = True):
        super().__init__()
        self.eps = eps
        if affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x):
        # x: (..., D)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        y = x * rms
        if self.weight is not None:
            y = y * self.weight
        return y


# -----------------------------------------------------------------------------
# Rotary positional embedding (rope)
# -----------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_positions: int = 1_000_000, base: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.base = base
        self.max_positions = max_positions
        freqs = self._build_freqs(max_positions, dim, base)
        self.register_buffer("freqs", freqs, persistent=False)
        self.freqs_cis: Optional[torch.Tensor] = None  # cached complex

    @staticmethod
    def _build_freqs(max_positions, dim, base):
        half = dim // 2
        freqs = torch.exp(torch.arange(half, dtype=torch.float32) * -(math.log(base) / half))
        t = torch.arange(max_positions, dtype=torch.float32).unsqueeze(1) * freqs.unsqueeze(0)  # (T, half)
        return t  # store angles; sin/cos computed on the fly

    def _get_freqs_cis(self, start: int, end: int):
        # Build complex sin/cos (T, half) -> (T, half) complex cis
        angles = self.freqs[start:end]  # (T, half)
        # Convert to complex cis = cos + i sin
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        cis = torch.complex(cos, sin)  # (T, half)
        return cis

    def forward(self, q: torch.Tensor, k: torch.Tensor, start_index: int):
        # q,k: (B, T, H, D) with D even
        B, T, H, D = q.shape
        half = D // 2
        cis = self._get_freqs_cis(start_index, start_index + T).to(q.device)  # (T, half)
        # reshape to broadcast: (T, 1, 1, half)
        cis = cis.view(T, 1, 1, half)
        # cast q,k to complex pairs
        def to_complex(x):
            x1, x2 = x[..., :half], x[..., half:]
            return torch.complex(x1, x2)

        def from_complex(xc):
            x1 = torch.real(xc)
            x2 = torch.imag(xc)
            return torch.cat([x1, x2], dim=-1)

        q_c = to_complex(q)
        k_c = to_complex(k)
        q_rot = q_c * cis
        k_rot = k_c * cis
        return from_complex(q_rot.type_as(q)), from_complex(k_rot.type_as(k))


# -----------------------------------------------------------------------------
# TimestepNorm (streaming, per-group stats across time)
# -----------------------------------------------------------------------------

class TimestepNorm(nn.Module):
    """
    Normalize x over time using streaming group statistics.
    - Split the D features into G groups of size D/G.
    - Maintain running (per-sample, per-group) count/mean/var across timesteps.
    - For each timestep t, normalize features in each group by the *current* running mean/var.
    Returns normalized x and the updated count/mean/var to carry across chunks.
    """
    def __init__(self, num_features: int, num_groups: int, eps: float = 1e-5):
        super().__init__()
        assert num_features % num_groups == 0, "num_features must be divisible by num_groups"
        self.num_features = num_features
        self.num_groups = num_groups
        self.group_size = num_features // num_groups
        self.eps = eps
        # Per-feature affine (like LayerNorm)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(
        self,
        x: torch.Tensor,                         # (B, L, D)
        prev_count: Optional[torch.Tensor] = None,  # (B,)
        prev_mean: Optional[torch.Tensor] = None,   # (B, G)
        prev_var: Optional[torch.Tensor] = None,    # (B, G)
        padding_mask: Optional[torch.Tensor] = None # (B, L) 1=token,0=pad
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        G = self.num_groups
        gs = self.group_size
        device = x.device
        dtype = x.dtype

        if prev_count is None:
            prev_count = torch.zeros(B, dtype=torch.long, device=device)
        if prev_mean is None:
            prev_mean = torch.zeros(B, G, dtype=torch.float32, device=device)
        if prev_var is None:
            prev_var = torch.ones(B, G, dtype=torch.float32, device=device)

        # Work in float32 for stability
        mean = prev_mean
        var = prev_var
        count = prev_count.clone()

        # Output buffer
        y = torch.empty_like(x)

        # Prepare views to compute group means quickly
        x_ = x.view(B, L, G, gs)  # (B, L, G, group_size)

        if padding_mask is None:
            padding_mask = torch.ones(B, L, dtype=torch.bool, device=device)

        # Streaming Welford updates per timestep
        for t in range(L):
            m_t = x_[:, t].mean(dim=-1)  # (B, G)
            mask_t = padding_mask[:, t]  # (B,)
            # Only update for valid tokens
            valid = mask_t.view(B, 1)
            # Update counts
            c_new = count + mask_t.to(count.dtype)
            # Avoid division by zero
            c_safe = torch.clamp(c_new, min=1)
            # Delta and new mean
            delta = (m_t - mean)
            mean = mean + (delta * valid) / c_safe.view(B, 1)
            # For variance, maintain running second moment around mean
            # Approximate running var with per-timestep group mean variance
            # M2 update: var * count -> new M2
            m2 = var * torch.clamp(count, min=1).view(B, 1)
            delta2 = (m_t - mean)
            m2 = m2 + (delta * delta2 * valid)
            var = m2 / c_safe.view(B, 1)
            count = c_new

            # Normalize current timestep features using current stats
            # Broadcast group mean/var to feature dimension
            mean_b = mean.unsqueeze(-1).expand(B, G, gs)
            var_b = var.unsqueeze(-1).expand(B, G, gs)
            x_t = x_[:, t]
            x_hat = (x_t - mean_b.to(x_t)) * torch.rsqrt(var_b.to(x_t) + self.eps)
            # Per-feature affine
            w = self.weight.view(1, G, gs).to(x_t)
            b = self.bias.view(1, G, gs).to(x_t)
            x_hat = x_hat * w + b
            y[:, t] = x_hat.view(B, D)

        return y, count, mean.to(dtype), var.to(dtype)


# -----------------------------------------------------------------------------
# Complex EMA (pure torch): kernel via Vandermonde, plus optional bias from initial state
# -----------------------------------------------------------------------------

class ComplexEMA(nn.Module):
    """
    Multi-dimensional complex EMA used in Megalodon.
    Parameters:
        alpha, delta: D x N x 1 (before sigmoid) -> p = sigm(alpha) \in (0,1), d = sigm(delta)\in(0,1)
        theta:        D x 1 x 1 (before sigmoid) -> base angle; per-component angle = sigm(theta) * 2π/N * n
        gamma:        D x N x 2 (real, imag)     -> complex mixing; scaled by sqrt(1/N)
        omega:        D (residual scaling)
    Forward (x: B x D x L) returns y (same shape) and optional last hidden state h_L: (B x D x N) (complex).
    Math:
        q_dn = (1 - p_dn * d_dn) * exp(i * φ_n);  φ_n = sigm(theta_d) * 2π/N * n
        kernel_d[t] = Re( sum_n p_dn * gamma_dn * q_dn^t )
        y = x (*) kernel   (+ residual omega * x)
        If hx provided, an additive bias term b(t) = Re( sum_n gamma_dn * q_dn^t * hx_dn ).
    """
    def __init__(self, embed_dim: int, ndim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.ndim = ndim
        self.scale = math.sqrt(1.0 / float(ndim))

        # Parameters per (D, N, 1) etc.
        self.alpha = nn.Parameter(torch.zeros(embed_dim, ndim, 1))
        self.delta = nn.Parameter(torch.zeros(embed_dim, ndim, 1))
        self.theta = nn.Parameter(torch.zeros(embed_dim, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(embed_dim, ndim, 2))
        self.omega = nn.Parameter(torch.zeros(embed_dim))

        # Cache for eval
        self._cached = {}

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize following reference heuristics
        nn.init.normal_(self.alpha, mean=0.0, std=0.2)
        nn.init.normal_(self.delta, mean=0.0, std=0.2)
        nn.init.normal_(self.theta, mean=0.0, std=0.2)
        nn.init.normal_(self.gamma, mean=0.0, std=1.0)  # will be scaled by 1/sqrt(N)
        nn.init.normal_(self.omega, mean=0.0, std=1.0)

    @staticmethod
    def _r2c(z: torch.Tensor) -> torch.Tensor:
        # last dim size 2 -> complex
        return torch.complex(z[..., 0], z[..., 1])

    def _coeffs(self):
        # D x N x 1
        p = torch.sigmoid(self.alpha.float())
        d = torch.sigmoid(self.delta.float())
        # Angle per component n \in [1..N]
        wave = torch.arange(1, self.ndim + 1, dtype=torch.float32, device=self.alpha.device).view(1, self.ndim, 1)
        base = torch.sigmoid(self.theta.float()) * (2.0 * math.pi / float(self.ndim))  # D x 1 x 1
        phi = wave * base  # D x N x 1
        q = (1.0 - p * d) * torch.exp(1j * phi)  # D x N x 1 (complex)
        gamma = self._r2c(self.gamma.float()) * self.scale  # D x N (complex)
        return p, q, gamma

    def _kernel_and_bias(self, L: int, hx: Optional[torch.Tensor]):
        # Returns kernel: (D, L) real, and bias: (B, D, L) real or None
        key = (L, hx is not None)
        if (not self.training) and key in self._cached and hx is None:
            return self._cached[key]

        p, q, gamma = self._coeffs()              # p: D x N x 1 ; q: D x N x 1 (complex) ; gamma: D x N (complex)
        D, N, _ = p.shape
        device = p.device
        # Vandermonde powers: (N, L) complex
        t = torch.arange(L, dtype=torch.float32, device=device).view(1, 1, L)  # 1x1xL
        V = torch.pow(q, t)                  # D x N x L (complex)
        kernel_c = (p * V)                   # D x N x L
        kernel_c = (kernel_c * gamma.unsqueeze(-1)).sum(dim=1)  # D x L (complex)
        kernel = torch.real(kernel_c)        # real-valued kernel per channel

        bias = None
        if hx is not None:
            # hx expected shape: (B, D, N) complex or real pair (...,2). Accept both.
            if hx.dtype.is_complex:
                hx_c = hx
            else:
                hx_c = torch.complex(hx[..., 0], hx[..., 1])
            # bias_d[t] = Re( sum_n gamma_dn * q_dn^t * hx_dn )
            bias_c = (gamma.unsqueeze(0).unsqueeze(-1) * torch.pow(q.unsqueeze(0), t)).sum(dim=2)  # (1,B?) wait shapes
            # q: D x N x 1 ; need (D,N,L)
            qL = torch.pow(q, t)                  # D x N x L
            # (B, D, N) * (D, N, L) -> (B, D, L)
            bias_c = (hx_c.unsqueeze(-1) * qL.unsqueeze(0)).sum(dim=2) * gamma.unsqueeze(0).unsqueeze(-1)
            bias = torch.real(bias_c).contiguous()  # (B, D, L)

        if (not self.training) and hx is None:
            self._cached[key] = (kernel, None)

        return kernel.to(torch.float32), (None if bias is None else bias.to(torch.float32))

    @staticmethod
    def _fftconv_real(x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        # x: (B, D, L), k: (D, Lk) => y: (B, D, L)
        B, D, L = x.shape
        Lk = k.shape[-1]
        n = L + Lk - 1
        n_fft = 1 << (n - 1).bit_length()  # next pow2
        X = torch.fft.rfft(x.float(), n=n_fft)
        K = torch.fft.rfft(k.float(), n=n_fft).unsqueeze(0)  # (1, D, n_fft//2+1)
        Y = X * K
        y_full = torch.fft.irfft(Y, n=n_fft)
        y = y_full[..., :L]  # causal valid part
        return y.type_as(x)

    def forward(
        self,
        x: torch.Tensor,                     # (B, D, L)
        hx: Optional[torch.Tensor] = None,   # (B, D, N) optional complex/real2
        compute_last_state: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = x * self.omega.view(1, -1, 1).to(x)
        B, D, L = x.shape
        k, b = self._kernel_and_bias(L, hx)
        y = self._fftconv_real(x, k)
        if b is not None:
            y = y + b.to(y)
        y = y + residual

        # Optional final hidden state via recurrence to avoid huge powers
        h_last = None
        if compute_last_state:
            p, q, _ = self._coeffs()         # D x N x 1 complex
            # Iterate along time to get last state: h_t = q*h_{t-1} + p*x_t
            # Keep complex h; initialize from hx if provided else zeros.
            if hx is not None:
                h = hx if hx.dtype.is_complex else torch.complex(hx[..., 0], hx[..., 1])
            else:
                h = torch.zeros(B, D, self.ndim, dtype=torch.complex64, device=x.device)
            # x: (B,D,L) -> (B,D,L,1)
            px = p.unsqueeze(0) * x.unsqueeze(-1).to(torch.complex64)
            for t in range(L):
                h = q.squeeze(-1) * h + px[..., t, :]
            h_last = h  # (B, D, N) complex
        return y, h_last


# -----------------------------------------------------------------------------
# Inner (chunked) attention: causal, rope, optional cache
# -----------------------------------------------------------------------------

class ChunkedSelfAttention(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, value_head_dim: int, chunk_size: int,
                 dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.value_head_dim = value_head_dim
        self.chunk_size = chunk_size
        self.dropout = dropout
        self.rope = RotaryEmbedding(head_dim * 2)  # q/k are split into real/imag halves

    def _causal_mask(self, Lq: int, Lk: int, device, dtype):
        # return additive mask (B,H,Lq,Lk) -> here independent of B,H; we'll broadcast later
        mask = torch.full((Lq, Lk), float("-inf"), device=device, dtype=dtype)
        idx = torch.arange(Lq, device=device)
        idy = torch.arange(Lk, device=device)
        # allow j <= i (causal)
        causal = (idy.unsqueeze(0) <= idx.unsqueeze(1))
        mask[causal] = 0.0
        return mask  # (Lq, Lk)

    def forward(
        self,
        q: torch.Tensor,  # (B, L, H, Dh)  with Dh even (rope expects even split)
        k: torch.Tensor,  # (B, L, H, Dh)
        v: torch.Tensor,  # (B, L, H, Dv)
        start_index: int = 0,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor, int]] = None,  # (k_cache, v_cache, count)
        attn_mask: Optional[torch.Tensor] = None,  # (B, L) 1=keep
        training: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor, int]]]:
        B, L, H, Dh = q.shape
        Dv = v.size(-1)
        device = q.device
        dtype = q.dtype

        # RoPE (use start_index if caching)
        q, k = self.rope(q, k, start_index=start_index)

        # Concatenate cache (prefix) if present
        new_cache = None
        if cache is not None:
            k_cache, v_cache, count = cache
            k = torch.cat([k_cache, k], dim=1)
            v = torch.cat([v_cache, v], dim=1)
        else:
            count = 0

        # Chunking: if L <= chunk_size, single block
        outputs = []
        if L <= self.chunk_size:
            # Single-block attention to current context (may include cache prefix)
            Lk = k.size(1)
            # Build causal mask for the suffix positions (only for current L positions versus entire context)
            base_mask = self._causal_mask(L, Lk, device, dtype)  # (L, Lk)
            if attn_mask is not None:
                # Broadcast padding: (B,1,1,Lk) additive
                pad = (attn_mask.to(dtype) - 1.0) * 1e9  # 0 -> -1e9, 1 -> 0
                base_mask = base_mask + pad.unsqueeze(1)  # (B, L, Lk) after broadcast
            # Compute scaled dot-product attention
            q_ = q.transpose(1, 2)  # (B,H,L,Dh)
            k_ = k.transpose(1, 2)  # (B,H,Lk,Dh)
            v_ = v.transpose(1, 2)  # (B,H,Lk,Dv)
            scores = torch.matmul(q_, k_.transpose(-2, -1)) / math.sqrt(Dh)
            # add causal + pad masks
            scores = scores + base_mask
            attn = F.softmax(scores.float(), dim=-1).to(q_)
            attn = F.dropout(attn, p=self.dropout, training=training)
            out = torch.matmul(attn, v_)  # (B,H,L,Dv)
            out = out.transpose(1, 2)     # (B,L,H,Dv)
            outputs.append(out)
            # update cache: keep last (k,v) upto chunk_size boundary
            # We keep (count + L) % chunk_size tokens (remainder) as the next prefix.
            total = count + L
            keep = total % self.chunk_size
            if keep > 0:
                new_cache = (k[:, -keep:].contiguous(), v[:, -keep:].contiguous(), total)
            else:
                new_cache = None
        else:
            # Multiple chunks; no cross-chunk attention (block-diagonal)
            assert (L % self.chunk_size) == 0, "For training, L must be multiple of chunk_size"
            nc = L // self.chunk_size
            q_chunks = q.view(B, nc, self.chunk_size, H, Dh)
            k_chunks = k[:, -L:].view(B, nc, self.chunk_size, H, Dh)  # restrict to current L only
            v_chunks = v[:, -L:].view(B, nc, self.chunk_size, H, Dv)
            for i in range(nc):
                q_i = q_chunks[:, i]  # (B,C,H,Dh)
                k_i = k_chunks[:, i]
                v_i = v_chunks[:, i]
                base_mask = self._causal_mask(self.chunk_size, self.chunk_size, device, dtype)  # (C,C)
                q_i_ = q_i.transpose(1, 2)  # (B,H,C,Dh)
                k_i_ = k_i.transpose(1, 2)
                v_i_ = v_i.transpose(1, 2)
                scores = torch.matmul(q_i_, k_i_.transpose(-2, -1)) / math.sqrt(Dh)
                scores = scores + base_mask
                attn = F.softmax(scores.float(), dim=-1).to(q_i_)
                attn = F.dropout(attn, p=self.dropout, training=training)
                out_i = torch.matmul(attn, v_i_)  # (B,H,C,Dv)
                out_i = out_i.transpose(1, 2)     # (B,C,H,Dv)
                outputs.append(out_i)
            out = torch.cat(outputs, dim=1)  # (B,L,H,Dv)

        # Merge heads
        out = out.reshape(B, -1, H * Dv)
        return out, new_cache


# -----------------------------------------------------------------------------
# Megalodon Attention Block (EMA + gated attention)
# -----------------------------------------------------------------------------

class MegalodonAttention(nn.Module):
    def __init__(
        self,
        cfg: MegalodonConfig,
    ):
        super().__init__()
        D = cfg.model_dim
        H = cfg.num_heads
        Z = cfg.z_dim
        E = cfg.value_dim
        self.cfg = cfg
        self.n_heads = H
        self.z_head_dim = Z // H
        self.v_head_dim = E // H

        self.timenorm = TimestepNorm(D, cfg.norm_num_groups, eps=1e-5)
        self.cema = ComplexEMA(D, cfg.cema_ndim)
        self.rmsnorm = RMSNorm(D, eps=1e-5)

        init_fn = get_init_fn(cfg.init_mode)
        # Linear layers (no model-parallel in pure version)
        self.wz = nn.Linear(D, Z, bias=True)
        self.wv = nn.Linear(D, E, bias=True)
        self.wr = nn.Linear(D, E, bias=True)
        self.wh1 = nn.Linear(D, D, bias=True)
        self.wh2 = nn.Linear(E, D, bias=True)

        for lin in [self.wz, self.wv, self.wr, self.wh1, self.wh2]:
            init_fn(lin.weight)
            if lin.bias is not None:
                nn.init.zeros_(lin.bias)

        # Per-dimension affine to generate q/k from shared z
        self.gamma = nn.Parameter(torch.zeros(2, Z))  # scales for q and k
        self.beta = nn.Parameter(torch.zeros(2, Z))   # biases for q and k

        self.inner_attn = ChunkedSelfAttention(H, self.z_head_dim, self.v_head_dim, cfg.chunk_size,
                                               dropout=cfg.attention_dropout)

        self.dropout = cfg.dropout
        self.hidden_dropout = cfg.hidden_dropout

    def _split_heads(self, z: torch.Tensor) -> torch.Tensor:
        B, L, Z = z.shape
        return z.view(B, L, self.n_heads, self.z_head_dim)

    def _merge_heads(self, h: torch.Tensor) -> torch.Tensor:
        B, L, H, Dh = h.shape
        return h.reshape(B, L, H * Dh)

    def forward(
        self,
        x: torch.Tensor,  # (B, L, D)
        cache: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor, int],  # attn cache
                              Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = None,
        attn_mask: Optional[torch.Tensor] = None,  # (B, L) 1=keep
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        B, L, D = x.shape
        residual = x

        # Unpack caches
        if cache is not None:
            cache_attn, cache_norm = cache
            prev_count, prev_mean, prev_var = cache_norm
            k_cache, v_cache, count = cache_attn
            hx = None  # we do not carry EMA hidden state between chunks in this pure variant by default
        else:
            k_cache = v_cache = None
            count = 0
            prev_count = prev_mean = prev_var = None
            hx = None

        # TimestepNorm
        out_tsn, new_count, new_mean, new_var = self.timenorm(x, prev_count, prev_mean, prev_var, attn_mask)

        # Complex EMA (on B, D, L)
        out_cema, hx_last = self.cema(out_tsn.transpose(1, 2), hx=hx, compute_last_state=(k_cache is not None))
        out_cema = out_cema.transpose(1, 2)  # back to (B, L, D)

        # RMSNorm + dropout
        mx = self.rmsnorm(out_cema)
        mx = F.dropout(mx, p=self.hidden_dropout, training=self.training)

        # Shared z, then per-head RMSNorm-like normalization and construct q/k via affine
        z = self.wz(mx)                        # (B, L, Z)
        z_heads = self._split_heads(z)         # (B, L, H, Z/H)
        # Normalize per head (RMS)
        z_norm = z_heads / (z_heads.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt())
        z = self._merge_heads(z_norm)          # (B, L, Z)

        gamma = (self.gamma + 1.0) / math.sqrt(self.z_head_dim)  # (2, Z)
        z_aff = z.unsqueeze(2) * gamma.unsqueeze(0).unsqueeze(0) + self.beta.unsqueeze(0).unsqueeze(0)
        q, k = torch.unbind(z_aff, dim=2)      # (B, L, Z) each
        # Split to heads
        q = self._split_heads(q)               # (B, L, H, Dh)
        k = self._split_heads(k)               # (B, L, H, Dh)

        # Values and gating
        v = torch.silu(self.wv(out_tsn))       # (B, L, E)
        r = torch.silu(self.wr(mx))            # (B, L, E)
        v = v.view(B, L, self.n_heads, self.v_head_dim)

        # Inner attention with chunking + RoPE
        cache_tuple = (k_cache, v_cache, count) if (k_cache is not None and v_cache is not None) else None
        attn_out, new_attn_cache = self.inner_attn(q, k, v, start_index=count, cache=cache_tuple,
                                                   attn_mask=attn_mask, training=self.training)

        attn_out = attn_out * r  # gating
        # Project and residual
        h = self.wh1(mx) + self.wh2(attn_out)
        h = F.dropout(h, p=self.dropout, training=self.training)
        out = h + residual

        new_cache = None
        if cache is not None:
            new_cache = (new_attn_cache, (new_count.detach(), new_mean.detach(), new_var.detach()))

        return out, new_cache


# -----------------------------------------------------------------------------
# FFN
# -----------------------------------------------------------------------------

class NormalizedFFN(nn.Module):
    def __init__(self, cfg: MegalodonConfig, layer_id: int):
        super().__init__()
        D = cfg.model_dim
        H = cfg.ffn_hidden_dim
        self.norm = RMSNorm(D, eps=1e-5)
        self.swiglu = cfg.swiglu
        self.alpha = (0.1 * (0.5 ** layer_id)) if cfg.rescale_nffn else None
        if self.swiglu:
            self.fc1 = nn.Linear(D, H, bias=True)
            self.fc3 = nn.Linear(D, H, bias=True)
            self.fc2 = nn.Linear(H, D, bias=True)
        else:
            self.fc1 = nn.Linear(D, H, bias=True)
            self.fc2 = nn.Linear(H, D, bias=True)
        for lin in [self.fc1, self.fc2] + ([self.fc3] if self.swiglu else []):
            get_init_fn(cfg.init_mode)(lin.weight)
            if lin.bias is not None:
                nn.init.zeros_(lin.bias)
        self.hidden_dropout = cfg.hidden_dropout

    def rescale(self, x): return x if self.alpha is None else (self.alpha * x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        if self.swiglu:
            hidden = torch.silu(self.fc1(x)) * self.fc3(x)
            hidden = F.dropout(hidden, p=self.hidden_dropout, training=self.training)
            out = self.fc2(hidden)
        else:
            hidden = torch.silu(self.fc1(x))
            hidden = F.dropout(hidden, p=self.hidden_dropout, training=self.training)
            out = self.fc2(hidden)
        out = F.dropout(out, p=self.hidden_dropout, training=self.training)
        out = self.rescale(out) + residual
        return out


# -----------------------------------------------------------------------------
# Transformer block
# -----------------------------------------------------------------------------

class MegalodonBlock(nn.Module):
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
    config_class = MegalodonConfig

    def __init__(self, config: MegalodonConfig):
        super().__init__()
        self.config = config
        D = config.model_dim
        self.embed = nn.Embedding(config.vocab_size, D, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList([MegalodonBlock(config, i) for i in range(config.num_layers)])
        self.norm = RMSNorm(D, eps=1e-5)

        if config.scale_emb:
            self.scale = math.sqrt(D)
        else:
            self.scale = 1.0

        if _HAS_HF:
            self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List] = None,  # per layer caches
        use_cache: bool = True,
        output_hidden_states: bool = False,
        output_attentions: bool = False,  # not used
    ):
        B, L = input_ids.shape
        x = self.embed(input_ids) * self.scale  # (B, L, D)
        caches = past_key_values or [None] * len(self.layers)

        all_hidden = [] if output_hidden_states else None

        for i, layer in enumerate(self.layers):
            x, caches[i] = layer(x, cache=caches[i], attn_mask=attention_mask)
            if output_hidden_states:
                all_hidden.append(x)

        x = self.norm(x)  # (B, L, D)
        out = (x,)
        if use_cache:
            out = out + (caches,)
        if output_hidden_states:
            out = out + (all_hidden,)
        return out


class MegalodonForCausalLM(PreTrainedModel):
    config_class = MegalodonConfig

    def __init__(self, config: MegalodonConfig):
        super().__init__()
        self.config = config
        self.model = MegalodonModel(config)
        self.lm_head = nn.Linear(config.model_dim, config.vocab_size, bias=False)
        # tie weights
        self.lm_head.weight = self.model.embed.weight
        if _HAS_HF:
            self.post_init()

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
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        hidden = outputs[0]  # (B, L, D)
        logits = self.lm_head(hidden)  # (B, L, V)

        loss = None
        if labels is not None:
            # shift for causal LM
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        out = (logits,)
        if use_cache:
            out = out + (outputs[1],)
        if output_hidden_states:
            out = out + (outputs[-1],)
        if loss is not None:
            out = (loss,) + out
        return out
