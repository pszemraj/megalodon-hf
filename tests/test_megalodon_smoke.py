# tests/test_megalodon_smoke.py
import math
from typing import Tuple

import pytest
import torch

from megalodon import MegalodonConfig, MegalodonForCausalLM, MegalodonModel
from megalodon.modeling_megalodon import (
    AttentionCache,
    ChunkedSelfAttention,
    ComplexEMA,
    RMSNorm,
    TimestepNorm,
)

TOL = 5e-4  # allow tiny differences due to FFT and accumulation order


@torch.no_grad()
def test_forward_single_chunk_cpu():
    torch.manual_seed(0)
    cfg = MegalodonConfig()
    base = MegalodonModel(cfg).eval()
    lm = MegalodonForCausalLM(cfg).eval()

    B, L = 1, 32  # well within chunk_size (2048)
    x = torch.randint(0, cfg.vocab_size, (B, L))
    attn = torch.ones(B, L, dtype=torch.long)

    # Base
    last, pkv = base(x, attention_mask=attn, use_cache=True)[:2]
    assert last.shape == (B, L, cfg.model_dim)
    assert isinstance(pkv, list) and len(pkv) == cfg.num_layers

    # LM
    labels = torch.randint(0, cfg.vocab_size, (B, L))
    loss, logits, pkv2 = lm(
        input_ids=x, attention_mask=attn, labels=labels, use_cache=True
    )
    assert logits.shape == (B, L, cfg.vocab_size)
    assert math.isfinite(float(loss))
    assert isinstance(pkv2, list) and len(pkv2) == cfg.num_layers


def _reference_timestep_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    prev_count: torch.Tensor,
    prev_mean: torch.Tensor,
    prev_var: torch.Tensor,
    padding_mask: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, L, D = x.shape
    G = prev_mean.size(1)
    gs = D // G
    x_groups = x.view(B, L, G, gs)
    y = torch.empty_like(x)
    mean = prev_mean.clone()
    var = prev_var.clone()
    count = prev_count.clone()
    for t in range(L):
        m_t = x_groups[:, t].mean(dim=-1)
        mask_t = padding_mask[:, t]
        valid = mask_t.view(B, 1)
        c_new = count + mask_t.to(count.dtype)
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
        x_hat = (x_t - mean_b.to(x_t)) * torch.rsqrt(var_b.to(x_t) + eps)
        w = (weight + 1.0).view(1, G, gs).to(x_t)
        b = bias.view(1, G, gs).to(x_t)
        y[:, t] = (x_hat * w + b).reshape(B, D)
    return y, count, mean, var


def test_timestep_norm_matches_reference():
    torch.manual_seed(0)
    B, L, D, G = 2, 7, 12, 3
    module = TimestepNorm(D, G, eps=1e-5, affine=True)
    module.weight.data.normal_()
    module.bias.data.uniform_(-0.5, 0.5)

    x = torch.randn(B, L, D)
    padding_mask = torch.rand(B, L) > 0.3
    prev_count = torch.tensor([0, 3], dtype=torch.long)
    prev_mean = torch.randn(B, G)
    prev_var = torch.rand(B, G) + 0.5

    y, count, mean, var = module(
        x,
        prev_count=prev_count,
        prev_mean=prev_mean,
        prev_var=prev_var,
        padding_mask=padding_mask,
    )

    ref_y, ref_count, ref_mean, ref_var = _reference_timestep_norm(
        x,
        module.weight.detach(),
        module.bias.detach(),
        prev_count.clone(),
        prev_mean.clone(),
        prev_var.clone(),
        padding_mask,
        module.eps,
    )

    assert torch.allclose(y, ref_y, atol=1e-5, rtol=1e-5)
    assert torch.equal(count, ref_count)
    assert torch.allclose(mean, ref_mean.to(mean), atol=1e-5, rtol=1e-5)
    assert torch.allclose(var, ref_var.to(var), atol=1e-5, rtol=1e-5)


@torch.no_grad()
def test_forward_multi_chunk_cpu():
    torch.manual_seed(0)
    cfg = MegalodonConfig()
    base = MegalodonModel(cfg).eval()

    B, L = 1, cfg.chunk_size * 2  # multiple of chunk_size -> block-diagonal path
    x = torch.randint(0, cfg.vocab_size, (B, L))
    attn = torch.ones(B, L, dtype=torch.long)

    out = base(x, attention_mask=attn, use_cache=False)[0]
    assert out.shape == (B, L, cfg.model_dim)


@torch.no_grad()
def test_chunked_attention_is_block_diagonal():
    torch.manual_seed(0)
    chunk_size = 8
    num_chunks = 2
    B, H, Dh, Dv = 1, 2, 4, 4
    attn = ChunkedSelfAttention(
        num_heads=H,
        head_dim=Dh,
        value_head_dim=Dv,
        chunk_size=chunk_size,
        rope_base=10_000.0,
        attention_dropout=0.0,
    )
    L = chunk_size * num_chunks
    q = torch.randn(B, L, H, Dh)
    k = torch.randn(B, L, H, Dh)
    v = torch.randn(B, L, H, Dv)
    mask = torch.ones(B, L, dtype=torch.long)

    out_full, _ = attn(
        q, k, v, start_index=0, cache=None, attn_mask=mask, training=False
    )

    v_zero = v.clone()
    v_zero[:, :chunk_size] = 0.0
    out_zero, _ = attn(
        q, k, v_zero, start_index=0, cache=None, attn_mask=mask, training=False
    )

    # First chunk changes, second chunk identical
    assert not torch.allclose(out_full[:, :chunk_size], out_zero[:, :chunk_size])
    assert torch.allclose(out_full[:, chunk_size:], out_zero[:, chunk_size:], atol=1e-5)


def test_dropkey_preserves_current_position():
    torch.manual_seed(0)
    chunk_size = 4
    B, H, Dh, Dv = 1, 1, 2, 2
    attn = ChunkedSelfAttention(
        num_heads=H,
        head_dim=Dh,
        value_head_dim=Dv,
        chunk_size=chunk_size,
        rope_base=10_000.0,
        attention_dropout=1.0,
    )
    attn_mask = torch.ones(B, chunk_size, dtype=torch.long)
    q = torch.randn(B, chunk_size, H, Dh)
    k = torch.randn(B, chunk_size, H, Dh)
    v = torch.randn(B, chunk_size, H, Dv)

    out, _ = attn(
        q,
        k,
        v,
        start_index=0,
        cache=None,
        attn_mask=attn_mask,
        training=True,
    )
    assert torch.isfinite(out).all()


def test_complex_ema_impulse_response_decays():
    torch.manual_seed(0)
    cema = ComplexEMA(embed_dim=1, ndim=1)
    with torch.no_grad():
        cema.alpha.fill_(0.0)  # p = 0.5
        cema.delta.fill_(0.0)  # d = 0.5
        cema.theta.fill_(-10.0)  # phi ≈ 0
        cema.gamma.zero_()
        cema.gamma[..., 0] = 1.0  # real mixing = 1
        cema.omega.zero_()

    x = torch.zeros(1, 1, 6)
    x[..., 0] = 1.0
    y, _ = cema(x, compute_last_state=False)

    expected = torch.tensor([0.5 * (0.75**t) for t in range(6)], dtype=y.dtype)
    assert torch.allclose(y.squeeze(0).squeeze(0), expected, atol=1e-5, rtol=1e-5)


def test_sdpa_with_prefix_and_padding_matches_reference():
    torch.manual_seed(0)
    chunk_size = 4
    prefix_len = 3
    num_heads, head_dim, value_head_dim = 2, 4, 4
    attn = ChunkedSelfAttention(
        num_heads=num_heads,
        head_dim=head_dim,
        value_head_dim=value_head_dim,
        chunk_size=chunk_size,
        rope_base=10_000.0,
        attention_dropout=0.0,
    )

    B = 1
    q = torch.randn(B, chunk_size, num_heads, head_dim)
    k = torch.randn(B, chunk_size, num_heads, head_dim)
    v = torch.randn(B, chunk_size, num_heads, value_head_dim)
    cache_k = torch.randn(B, prefix_len, num_heads, head_dim)
    cache_v = torch.randn(B, prefix_len, num_heads, value_head_dim)
    cache = AttentionCache(cache_k, cache_v, prefix_len)
    attn_mask = torch.tensor([[1, 0, 1, 1]], dtype=torch.long)

    out_sdpa, _ = attn(
        q,
        k,
        v,
        start_index=prefix_len,
        cache=cache,
        attn_mask=attn_mask,
        training=False,
    )

    q_rot, k_rot = attn.rope(q, k, start_index=prefix_len)
    k_full = torch.cat([cache_k, k_rot], dim=1)
    v_full = torch.cat([cache_v, v], dim=1)

    q_ = q_rot.transpose(1, 2)
    k_ = k_full.transpose(1, 2)
    v_ = v_full.transpose(1, 2)

    scores = torch.matmul(q_, k_.transpose(-2, -1)) / math.sqrt(head_dim)
    causal = attn._causal_mask(
        chunk_size,
        prefix_len + chunk_size,
        q.device,
        q.dtype,
        offset=prefix_len,
    )
    scores = scores + causal

    prefix_mask = attn_mask.new_ones(B, prefix_len)
    mask_tokens = torch.cat([prefix_mask, attn_mask], dim=1)
    invalid = mask_tokens == 0
    scores = scores.masked_fill(invalid.view(B, 1, 1, -1), float("-inf"))

    weights = torch.softmax(scores.float(), dim=-1).to(q_)
    ref = torch.matmul(weights, v_).transpose(1, 2).reshape(B, chunk_size, -1)
    assert torch.allclose(out_sdpa, ref, atol=1e-5, rtol=1e-5)


def test_timestep_norm_streaming_matches_full():
    torch.manual_seed(0)
    norm = TimestepNorm(num_features=8, num_groups=4)
    x = torch.randn(2, 9, 8)
    mask = torch.ones(2, 9, dtype=torch.bool)

    full, c_full, m_full, v_full = norm(x, padding_mask=mask)

    count = mean = var = None
    chunks = []
    for start in range(0, 9, 3):
        end = start + 3
        y_chunk, count, mean, var = norm(
            x[:, start:end],
            prev_count=count,
            prev_mean=mean,
            prev_var=var,
            padding_mask=mask[:, start:end],
        )
        chunks.append(y_chunk)

    streamed = torch.cat(chunks, dim=1)
    assert torch.allclose(streamed, full, atol=1e-5, rtol=1e-5)
    assert torch.equal(count, c_full)
    assert torch.allclose(mean, m_full, atol=1e-5, rtol=1e-5)
    assert torch.allclose(var, v_full, atol=1e-5, rtol=1e-5)


def test_rmsnorm_plus_one_reparameterization():
    torch.manual_seed(0)
    rms = RMSNorm(dim=6)
    x = torch.randn(2, 5, 6)
    y = rms(x)
    base = x / x.pow(2).mean(dim=-1, keepdim=True).add(rms.eps).sqrt()
    assert torch.allclose(y, base, atol=1e-6, rtol=1e-6)


def test_complex_ema_fft_matches_sequential():
    torch.manual_seed(0)
    D, N, L = 4, 3, 64
    cema = ComplexEMA(D, N)
    x = torch.randn(2, D, L)

    y_fft, state_fft = cema(x, compute_last_state=False)
    y_seq, state_seq = cema(x, compute_last_state=True)

    assert state_fft is None
    assert state_seq is not None
    assert torch.allclose(y_fft, y_seq, atol=1e-5, rtol=1e-5)


def test_complex_ema_streaming_state():
    torch.manual_seed(0)
    D, N, L = 2, 2, 16
    cema = ComplexEMA(D, N)
    x = torch.randn(1, D, L)
    hx = torch.zeros(1, D, N, dtype=torch.complex64)

    y, h_next = cema(x, hx=hx, compute_last_state=True)

    assert y.shape == (1, D, L)
    assert h_next is not None
    assert torch.is_complex(h_next)


@torch.no_grad()
def test_cache_equivalence_tail_logits():
    torch.manual_seed(0)
    cfg = MegalodonConfig()
    lm = MegalodonForCausalLM(cfg).eval()

    B, prefix_len, suffix_len = 1, 64, 32
    L = prefix_len + suffix_len
    x_all = torch.randint(0, cfg.vocab_size, (B, L))
    attn_all = torch.ones(B, L, dtype=torch.long)

    # baseline, no cache
    logits_all = lm(input_ids=x_all, attention_mask=attn_all, use_cache=False)[0]

    # cached incremental
    logits_pref, pkv = lm(
        input_ids=x_all[:, :prefix_len],
        attention_mask=attn_all[:, :prefix_len],
        use_cache=True,
    )[:2]
    logits_suf = lm(
        input_ids=x_all[:, prefix_len:],
        attention_mask=attn_all[:, prefix_len:],
        past_key_values=pkv,
        use_cache=True,
    )[0]

    ref_tail = logits_all[:, -suffix_len:, :]
    max_diff = (ref_tail - logits_suf).abs().max().item()
    assert max_diff <= TOL, (
        f"cached vs one-shot tail logits differ by {max_diff:.3e} > {TOL}"
    )


@pytest.mark.cuda
@torch.no_grad()
def test_cuda_smoke():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA available")
    torch.manual_seed(0)
    cfg = MegalodonConfig()
    lm = MegalodonForCausalLM(cfg).eval().cuda()

    B, L = 1, 32
    x = torch.randint(0, cfg.vocab_size, (B, L), device="cuda")
    attn = torch.ones(B, L, dtype=torch.long, device="cuda")
    logits = lm(input_ids=x, attention_mask=attn, use_cache=False)[0]
    assert logits.is_cuda and logits.shape == (B, L, cfg.vocab_size)


def test_sdpa_matches_reference():
    torch.manual_seed(0)
    chunk_size = 16
    num_heads, head_dim, value_dim = 2, 8, 8
    attn = ChunkedSelfAttention(
        num_heads=num_heads,
        head_dim=head_dim,
        value_head_dim=value_dim,
        chunk_size=chunk_size,
        rope_base=10_000.0,
        attention_dropout=0.0,
    )
    B, L = 1, chunk_size
    q = torch.randn(B, L, num_heads, head_dim)
    k = torch.randn(B, L, num_heads, head_dim)
    v = torch.randn(B, L, num_heads, value_dim)

    out_manual, _ = attn(
        q,
        k,
        v,
        start_index=0,
        cache=None,
        attn_mask=None,
        training=False,
    )

    q_rot, k_rot = attn.rope(q, k, start_index=0)
    q_rot = q_rot.transpose(1, 2)
    k_rot = k_rot.transpose(1, 2)
    v_ = v.transpose(1, 2)

    scores = torch.matmul(q_rot, k_rot.transpose(-2, -1)) / math.sqrt(head_dim)
    mask = attn._causal_mask(L, L, q.device, q.dtype)
    scores = scores + mask
    weights = torch.softmax(scores.float(), dim=-1).to(q)
    ref = torch.matmul(weights, v_).transpose(1, 2).reshape(B, L, -1)

    assert torch.allclose(out_manual, ref, atol=1e-5, rtol=1e-5)
