# tests/test_megalodon_smoke.py
import math

import pytest
import torch

from megalodon import MegalodonConfig, MegalodonForCausalLM, MegalodonModel

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
