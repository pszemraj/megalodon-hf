# tests/test_megalodon_smoke.py
import math
from typing import Tuple

import pytest
import torch

from megalodon import MegalodonConfig, MegalodonForCausalLM, MegalodonModel
from megalodon.modeling_megalodon import TimestepNorm

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
        w = weight.view(1, G, gs).to(x_t)
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
    padding_mask = (torch.rand(B, L) > 0.3)
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
def test_chunked_attention_matches_full_block():
    torch.manual_seed(0)
    seq_len = 192
    cfg_chunked = MegalodonConfig(chunk_size=64)
    cfg_full = MegalodonConfig(chunk_size=seq_len)

    model_chunked = MegalodonModel(cfg_chunked).eval()
    model_full = MegalodonModel(cfg_full).eval()
    model_full.load_state_dict(model_chunked.state_dict())

    x = torch.randint(0, cfg_chunked.vocab_size, (1, seq_len))
    attn = torch.ones(1, seq_len, dtype=torch.long)

    out_chunked = model_chunked(x, attention_mask=attn, use_cache=False)[0]
    out_full = model_full(x, attention_mask=attn, use_cache=False)[0]
    max_diff = (out_chunked - out_full).abs().max().item()
    assert max_diff <= TOL, f"chunked vs full attention diff {max_diff:.3e} > {TOL}"


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
