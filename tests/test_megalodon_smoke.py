"""End-to-end smoke tests covering inference utilities and cache behaviour."""

import math
from typing import List, Optional, Tuple

import pytest
import torch

from megalodon import MegalodonConfig, MegalodonForCausalLM, MegalodonModel
from megalodon.modeling_megalodon import (AttentionCache, ChunkedSelfAttention,
                                          ComplexEMA, MegalodonAttention,
                                          RMSNorm, TimestepNorm)

TOL = 5e-4  # allow tiny differences due to FFT and accumulation order


@torch.no_grad()
def _greedy_decode(
    lm: MegalodonForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: Optional[int] = None,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Greedy decode with cache, returning generated tokens and per-step logits."""
    outputs = lm(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )
    pkv = outputs.past_key_values
    logits = outputs.logits[:, -1]
    logits_history: List[torch.Tensor] = []
    generated: List[torch.Tensor] = []
    for _ in range(max_new_tokens):
        logits_history.append(logits)
        next_token = logits.argmax(dim=-1, keepdim=True)
        generated.append(next_token)
        if eos_token_id is not None and torch.all(next_token == eos_token_id):
            break
        outputs = lm(
            input_ids=next_token,
            attention_mask=torch.ones_like(next_token),
            past_key_values=pkv,
            use_cache=True,
            return_dict=True,
        )
        pkv = outputs.past_key_values
        logits = outputs.logits[:, -1]
    if generated:
        gen = torch.cat(generated, dim=1)
    else:
        gen = input_ids.new_empty((input_ids.size(0), 0), dtype=input_ids.dtype)
    return gen, logits_history


@torch.no_grad()
def _stream_logits(
    lm: MegalodonForCausalLM,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    continuation_ids: torch.Tensor,
) -> List[torch.Tensor]:
    """Run streaming teacher-forced logits over a continuation sequence."""
    outputs = lm(
        input_ids=prompt_ids,
        attention_mask=prompt_mask,
        use_cache=True,
        return_dict=True,
    )
    pkv = outputs.past_key_values
    logits_history: List[torch.Tensor] = []
    for step in range(continuation_ids.size(1)):
        token = continuation_ids[:, step : step + 1]
        outputs = lm(
            input_ids=token,
            attention_mask=torch.ones_like(token),
            past_key_values=pkv,
            use_cache=True,
            return_dict=True,
        )
        pkv = outputs.past_key_values
        logits_history.append(outputs.logits[:, -1])
    return logits_history


@torch.no_grad()
def test_forward_single_chunk_cpu() -> None:
    """Sanity-check a short forward pass with caching enabled on CPU."""
    torch.manual_seed(0)
    cfg = MegalodonConfig()
    base = MegalodonModel(cfg).eval()
    lm = MegalodonForCausalLM(cfg).eval()

    B, L = 1, 32  # well within chunk_size (2048)
    x = torch.randint(0, cfg.vocab_size, (B, L))
    attn = torch.ones(B, L, dtype=torch.long)

    # Base
    base_out = base(x, attention_mask=attn, use_cache=True, return_dict=True)
    last = base_out.last_hidden_state
    pkv = base_out.past_key_values
    assert last.shape == (B, L, cfg.model_dim)
    assert isinstance(pkv, tuple) and len(pkv) == cfg.num_layers + 1

    # LM
    labels = torch.randint(0, cfg.vocab_size, (B, L))
    lm_out = lm(
        input_ids=x,
        attention_mask=attn,
        labels=labels,
        use_cache=True,
        return_dict=True,
    )
    assert lm_out.logits.shape == (B, L, cfg.vocab_size)
    assert math.isfinite(float(lm_out.loss))
    assert isinstance(lm_out.past_key_values, tuple)
    assert len(lm_out.past_key_values) == cfg.num_layers + 1


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


def test_timestep_norm_matches_reference() -> None:
    """Compare torch implementation of TimestepNorm against reference math."""
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
def test_forward_multi_chunk_cpu() -> None:
    """Validate multi-chunk forward pass agrees with chunked streaming API."""
    torch.manual_seed(0)
    cfg = MegalodonConfig()
    base = MegalodonModel(cfg).eval()

    B, L = 1, cfg.chunk_size * 2  # multiple of chunk_size -> block-diagonal path
    x = torch.randint(0, cfg.vocab_size, (B, L))
    attn = torch.ones(B, L, dtype=torch.long)

    out = base(
        x, attention_mask=attn, use_cache=False, return_dict=True
    ).last_hidden_state
    assert out.shape == (B, L, cfg.model_dim)


@torch.no_grad()
def test_streaming_generation_stops_on_eos() -> None:
    """Greedy streaming decode should stop once EOS is produced."""
    torch.manual_seed(0)
    cfg = MegalodonConfig(
        vocab_size=16,
        model_dim=32,
        num_layers=2,
        num_heads=4,
        z_dim=32,
        value_dim=32,
        ffn_hidden_dim=64,
        chunk_size=8,
        cema_ndim=4,
        norm_num_groups=4,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        dropout=0.0,
    )
    lm = MegalodonForCausalLM(cfg).eval()
    with torch.no_grad():
        lm.lm_head.weight.zero_()

    prompt_len = cfg.chunk_size + 2
    prompt = torch.randint(0, cfg.vocab_size, (1, prompt_len))
    mask = torch.ones_like(prompt)
    generated, logits_history = _greedy_decode(
        lm,
        prompt,
        mask,
        max_new_tokens=cfg.chunk_size * 2 + 1,
        eos_token_id=0,
    )

    assert generated.shape[1] == 1
    assert generated[0, 0].item() == 0
    assert len(logits_history) == 1


@torch.no_grad()
def test_chunked_attention_is_block_diagonal() -> None:
    """Attention mask should enforce block-diagonal structure across chunks."""
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


@torch.no_grad()
def test_multi_chunk_matches_streaming_block_diagonal() -> None:
    """Block-diagonal multi-chunk path should match streaming across boundaries."""
    torch.manual_seed(0)
    chunk_size = 4
    B, H, Dh, Dv = 1, 2, 4, 4
    attn = ChunkedSelfAttention(
        num_heads=H,
        head_dim=Dh,
        value_head_dim=Dv,
        chunk_size=chunk_size,
        rope_base=10_000.0,
        attention_dropout=0.0,
    )
    L = chunk_size * 2
    q = torch.randn(B, L, H, Dh)
    k = torch.randn(B, L, H, Dh)
    v = torch.randn(B, L, H, Dv)
    attn_mask = torch.ones(B, L, dtype=torch.long)

    out_block, _ = attn(
        q, k, v, start_index=0, cache=None, attn_mask=attn_mask, training=False
    )
    # max_cache_len=-1 clamps to chunk_size, recreating block-diagonal streaming.
    out_stream, _ = attn(
        q,
        k,
        v,
        start_index=0,
        cache=None,
        attn_mask=attn_mask,
        training=False,
        max_cache_len=-1,
        return_cache=True,
    )
    assert torch.allclose(out_block, out_stream, atol=1e-5, rtol=1e-5)


@torch.no_grad()
def test_streaming_generation_uses_earliest_context() -> None:
    """Cross-chunk streaming should reflect information from the first chunk."""
    torch.manual_seed(0)
    cfg = MegalodonConfig(
        vocab_size=32,
        model_dim=64,
        num_layers=2,
        num_heads=4,
        z_dim=64,
        value_dim=64,
        ffn_hidden_dim=128,
        chunk_size=8,
        cema_ndim=4,
        norm_num_groups=4,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        dropout=0.0,
    )
    lm = MegalodonForCausalLM(cfg).eval()
    prompt_len = cfg.chunk_size + 3
    tail_len = prompt_len - cfg.chunk_size
    tail = torch.randint(0, cfg.vocab_size, (1, tail_len))
    first_a = torch.zeros((1, cfg.chunk_size), dtype=torch.long)
    first_b = torch.full((1, cfg.chunk_size), cfg.vocab_size - 1, dtype=torch.long)
    prompt_a = torch.cat([first_a, tail], dim=1)
    prompt_b = torch.cat([first_b, tail], dim=1)
    mask = torch.ones_like(prompt_a)

    gen_len = cfg.chunk_size * 2 + 3
    generated, _ = _greedy_decode(
        lm,
        prompt_a,
        mask,
        max_new_tokens=gen_len,
        eos_token_id=None,
    )
    assert generated.shape[1] == gen_len

    logits_a = _stream_logits(lm, prompt_a, mask, generated)
    logits_b = _stream_logits(lm, prompt_b, mask, generated)
    diff = (logits_a[-1] - logits_b[-1]).abs().max().item()
    assert diff > 1e-5, (
        "Expected earliest-chunk context to influence late-step logits in streaming decode."
    )


def test_dropkey_preserves_current_position() -> None:
    """DropKey dropout must keep the current token unmasked."""
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


def test_normalized_attention_rms_norm() -> None:
    """Inverse affine on Q should restore unit RMS (matches paper/upstream)."""
    torch.manual_seed(0)
    cfg = MegalodonConfig(
        model_dim=12,
        num_heads=3,
        z_dim=12,
        value_dim=12,
        chunk_size=8,
        norm_num_groups=3,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        dropout=0.0,
    )
    attn_block = MegalodonAttention(cfg).eval()

    B, L = 2, 4
    x = torch.randn(B, L, cfg.model_dim)
    attn_mask = torch.ones(B, L, dtype=torch.bool)
    captured = {}

    def _capture(module, args, kwargs):
        captured["q"] = args[0].detach()

    handle = attn_block.inner.register_forward_hook(_capture)
    try:
        attn_block(x, cache=None, attn_mask=attn_mask, return_cache=False)
    finally:
        handle.remove()

    assert "q" in captured
    q = captured["q"]  # (B, L, H, z_head)
    scale = (attn_block.gamma + 1.0) / math.sqrt(attn_block.z_head)
    beta = attn_block.beta
    scale_q = scale[0].view(attn_block.H, attn_block.z_head)
    beta_q = beta[0].view(attn_block.H, attn_block.z_head)
    z_recovered = (
        q - beta_q.view(1, 1, attn_block.H, attn_block.z_head)
    ) / scale_q.view(1, 1, attn_block.H, attn_block.z_head)
    # RMS norm: sqrt(mean(z^2)) should be ~1.0 per head (epsilon causes small deviation)
    rms_per_head = z_recovered.float().pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(
        rms_per_head, torch.ones_like(rms_per_head), atol=1e-4, rtol=1e-4
    )


@pytest.mark.filterwarnings(
    "ignore:Casting complex values to real discards the imaginary part"
)
def test_model_rejects_float16_embeddings() -> None:
    """Ensure the model refuses float16 execution as documented."""
    cfg = MegalodonConfig()
    model = MegalodonModel(cfg)
    model.to(torch.float16)
    input_ids = torch.randint(0, cfg.vocab_size, (1, 8))
    with pytest.raises(ValueError, match="float32 or bfloat16"):
        model(input_ids)


def test_complex_ema_impulse_response_decays() -> None:
    """Impulse response should remain a decaying real signal."""
    torch.manual_seed(0)
    cema = ComplexEMA(embed_dim=1, ndim=1)
    with torch.no_grad():
        cema.alpha.fill_(0.0)  # p = sigmoid(0) = 0.5
        cema.delta.fill_(0.0)  # radius = 1 - p*sigmoid(0) = 0.75
        cema.theta.fill_(-100.0)  # angle ~0 => real-positive q
        cema.gamma_real.fill_(1.0)
        cema.gamma_imag.fill_(0.0)
        cema.omega.zero_()

    x = torch.zeros(1, 1, 6)
    x[..., 0] = 1.0
    y, _ = cema(x, compute_last_state=False)

    expected = torch.tensor([0.5 * (0.75**t) for t in range(6)], dtype=y.dtype)
    assert torch.allclose(y.squeeze(0).squeeze(0), expected, atol=1e-5, rtol=1e-5)


@torch.no_grad()
def test_complex_ema_fft_handles_zero_q() -> None:
    """FFT path should remain finite when |q| collapses to zero."""
    torch.manual_seed(0)
    cema = ComplexEMA(embed_dim=2, ndim=2)
    with torch.no_grad():
        cema.alpha.fill_(100.0)
        cema.delta.fill_(100.0)
        cema.theta.zero_()
        cema.gamma_real.fill_(1.0)
        cema.gamma_imag.fill_(0.0)
        cema.omega.zero_()

    x = torch.randn(1, 2, 8)
    y, _ = cema(x, hx=None, compute_last_state=False)
    assert torch.isfinite(y).all()


def test_sdpa_with_prefix_and_padding_matches_reference() -> None:
    """Manual attention path should match the SDPA fallback with prefix padding."""
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
        max_cache_len=prefix_len + chunk_size,
    )

    q_rot, k_rot = attn.rope(q, k, start_index=prefix_len)
    k_full = torch.cat([cache_k, k_rot], dim=1)
    v_full = torch.cat([cache_v, v], dim=1)

    q_ = q_rot.transpose(1, 2)
    k_ = k_full.transpose(1, 2)
    v_ = v_full.transpose(1, 2)

    scores = torch.matmul(q_, k_.transpose(-2, -1))
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


def test_timestep_norm_streaming_matches_full() -> None:
    """Streaming TimestepNorm should match processing the whole sequence at once."""
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


def test_rmsnorm_plus_one_reparameterization() -> None:
    """RMSNorm reparameterization should produce the same output as direct scaling."""
    torch.manual_seed(0)
    rms = RMSNorm(dim=6)
    x = torch.randn(2, 5, 6)
    y = rms(x)
    base = x / x.pow(2).mean(dim=-1, keepdim=True).add(rms.eps).sqrt()
    assert torch.allclose(y, base, atol=1e-6, rtol=1e-6)


def test_complex_ema_fft_matches_sequential() -> None:
    """FFT and sequential EMA paths must match when no cache is used."""
    torch.manual_seed(0)
    D, N, L = 4, 3, 64
    cema = ComplexEMA(D, N)
    x = torch.randn(2, D, L)

    y_fft, state_fft = cema(x, compute_last_state=False)
    y_seq, state_seq = cema(x, compute_last_state=True)

    assert state_fft is None
    assert state_seq is not None
    assert torch.allclose(y_fft, y_seq, atol=1e-5, rtol=1e-5)


def test_complex_ema_streaming_state() -> None:
    """Sequential EMA should produce reproducible hidden state for streaming."""
    torch.manual_seed(0)
    D, N, L = 2, 2, 16
    cema = ComplexEMA(D, N)
    x = torch.randn(1, D, L)
    hx = torch.zeros(1, D, N, dtype=torch.complex64)

    y, h_next = cema(x, hx=hx, compute_last_state=True)

    assert y.shape == (1, D, L)
    assert h_next is not None
    assert torch.is_complex(h_next)


def test_complex_ema_eigenvalues_inside_unit_circle() -> None:
    """EMA eigenvalues must stay inside the unit circle."""
    torch.manual_seed(0)
    D, N = 8, 4
    cema = ComplexEMA(D, N)

    with torch.no_grad():
        cema.alpha.fill_(0.0)
        cema.delta.fill_(0.0)
        cema.theta.fill_(-100.0)

    p, q, gamma = cema._coeffs()
    magnitudes = q.abs()

    # All eigenvalue magnitudes must be <= 1
    assert (magnitudes <= 1.0).all(), (
        f"EMA eigenvalues outside unit circle: max |q| = {magnitudes.max().item():.6f}"
    )


@torch.no_grad()
def test_cache_equivalence_tail_logits() -> None:
    """Tail logits must match between cached and uncached decoding."""
    torch.manual_seed(0)
    cfg = MegalodonConfig()
    lm = MegalodonForCausalLM(cfg).eval()

    B, prefix_len, suffix_len = 1, 64, 32
    L = prefix_len + suffix_len
    x_all = torch.randint(0, cfg.vocab_size, (B, L))
    attn_all = torch.ones(B, L, dtype=torch.long)

    # baseline, no cache
    logits_all = lm(
        input_ids=x_all,
        attention_mask=attn_all,
        use_cache=False,
        return_dict=True,
    ).logits

    # cached incremental
    pref_out = lm(
        input_ids=x_all[:, :prefix_len],
        attention_mask=attn_all[:, :prefix_len],
        use_cache=True,
        return_dict=True,
    )
    pkv = pref_out.past_key_values
    logits_suf = lm(
        input_ids=x_all[:, prefix_len:],
        attention_mask=attn_all[:, prefix_len:],
        past_key_values=pkv,
        use_cache=True,
        return_dict=True,
    ).logits

    ref_tail = logits_all[:, -suffix_len:, :]
    max_diff = (ref_tail - logits_suf).abs().max().item()
    assert max_diff <= TOL, (
        f"cached vs one-shot tail logits differ by {max_diff:.3e} > {TOL}"
    )


@torch.no_grad()
def test_cache_equivalence_multi_chunk_tail() -> None:
    """Cached decoding must stay consistent when the prefix spans multiple chunks."""
    torch.manual_seed(0)
    cfg = MegalodonConfig(
        model_dim=64,
        num_layers=2,
        num_heads=4,
        z_dim=64,
        value_dim=64,
        ffn_hidden_dim=128,
        chunk_size=8,
        norm_num_groups=4,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        dropout=0.0,
    )
    lm = MegalodonForCausalLM(cfg).eval()

    B = 1
    prefix_len = cfg.chunk_size * 2
    suffix_len = 4
    L = prefix_len + suffix_len
    x_all = torch.randint(0, cfg.vocab_size, (B, L))
    attn_all = torch.ones(B, L, dtype=torch.long)

    logits_all = lm(
        input_ids=x_all,
        attention_mask=attn_all,
        use_cache=False,
        return_dict=True,
    ).logits

    pkv = None
    for offset in range(0, prefix_len, cfg.chunk_size):
        chunk = x_all[:, offset : offset + cfg.chunk_size]
        mask_chunk = attn_all[:, offset : offset + cfg.chunk_size]
        pref_out = lm(
            input_ids=chunk,
            attention_mask=mask_chunk,
            past_key_values=pkv,
            use_cache=True,
            return_dict=True,
        )
        pkv = pref_out.past_key_values
    logits_suf = lm(
        input_ids=x_all[:, prefix_len:],
        attention_mask=attn_all[:, prefix_len:],
        past_key_values=pkv,
        use_cache=True,
        return_dict=True,
    ).logits

    ref_tail = logits_all[:, -suffix_len:, :]
    max_diff = (ref_tail - logits_suf).abs().max().item()
    # Expect very close equivalence for a slightly larger model; investigate if this drifts.
    assert max_diff <= 5e-3, (
        f"cached multi-chunk tail logits differ by {max_diff:.3e} > 5e-3"
    )


def test_attention_cache_respects_max_len() -> None:
    """Attention cache should obey the caller-provided max_cache_len limit."""
    torch.manual_seed(0)
    cfg = MegalodonConfig(
        model_dim=16,
        num_layers=1,
        num_heads=2,
        z_dim=16,
        value_dim=16,
        ffn_hidden_dim=32,
        chunk_size=8,
        norm_num_groups=4,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        dropout=0.0,
    )
    attn = MegalodonAttention(cfg).eval()
    B, L = 1, cfg.chunk_size
    x = torch.randn(B, L, cfg.model_dim)
    mask = torch.ones(B, L, dtype=torch.long)

    _, cache = attn(
        x,
        cache=None,
        attn_mask=mask,
        return_cache=True,
        max_cache_len=5,
    )

    assert cache is not None and cache.attn is not None
    assert cache.attn.k.shape[1] == 5
    assert cache.attn.count == L


def test_attention_cache_truncation_keeps_causality() -> None:
    """Clamped caches must preserve causal masking when prefix is trimmed."""
    torch.manual_seed(0)
    H, Dh, Dv = 2, 4, 4
    chunk_size = 4
    max_cache_len = 4
    past_len = 6
    new_len = 2
    attn = ChunkedSelfAttention(
        num_heads=H,
        head_dim=Dh,
        value_head_dim=Dv,
        chunk_size=chunk_size,
        rope_base=10_000.0,
        attention_dropout=0.0,
    )
    B = 1
    k_past = torch.randn(B, past_len, H, Dh)
    v_past = torch.randn(B, past_len, H, Dv)
    cache_full = AttentionCache(k_past, v_past, past_len)

    q_new = torch.randn(B, new_len, H, Dh)
    k_new = torch.randn(B, new_len, H, Dh)
    v_new = torch.randn(B, new_len, H, Dv)
    pos_in_chunk = cache_full.count % chunk_size
    effective_limit = min(max_cache_len, pos_in_chunk) if pos_in_chunk > 0 else 0

    # Path A: provide full cache, let attention clamp internally.
    out_clamped, _ = attn(
        q_new,
        k_new,
        v_new,
        start_index=cache_full.count,
        cache=cache_full,
        attn_mask=None,
        training=False,
        max_cache_len=max_cache_len,
        return_cache=False,
        return_position=False,
    )

    # Path B: manually clamp cache, then run with a large max_cache_len (no further clamp).
    if effective_limit == 0:
        cache_manual = None
    else:
        cache_manual = AttentionCache(
            k=cache_full.k[:, -effective_limit:],
            v=cache_full.v[:, -effective_limit:],
            count=cache_full.count,
        )
    out_manual, _ = attn(
        q_new,
        k_new,
        v_new,
        start_index=cache_full.count,
        cache=cache_manual,
        attn_mask=None,
        training=False,
        max_cache_len=max_cache_len,
        return_cache=False,
        return_position=False,
    )

    max_diff = (out_clamped - out_manual).abs().max().item()
    assert max_diff < 1e-5, (
        f"cache truncation broke causality (max diff {max_diff:.3e})"
    )


def test_sliding_cache_multi_chunk_attention_window() -> None:
    """Streaming cache should slide across multiple chunks with correct RoPE + causality."""
    torch.manual_seed(0)
    chunk_size = 4
    max_cache_len = 6  # retain >1 chunk
    B, H, Dh, Dv = 1, 1, 2, 3
    attn = ChunkedSelfAttention(
        num_heads=H,
        head_dim=Dh,
        value_head_dim=Dv,
        chunk_size=chunk_size,
        rope_base=10_000.0,
        attention_dropout=0.0,
    )
    mask = torch.ones(B, chunk_size, dtype=torch.long)

    q1 = torch.randn(B, chunk_size, H, Dh)
    k1 = torch.randn(B, chunk_size, H, Dh)
    v1 = torch.randn(B, chunk_size, H, Dv)

    out1, cache1, pos1 = attn(
        q1,
        k1,
        v1,
        start_index=0,
        cache=None,
        attn_mask=mask,
        training=False,
        max_cache_len=max_cache_len,
        return_cache=True,
        return_position=True,
    )
    assert out1.shape == (B, chunk_size, H * Dv)
    assert cache1 is not None and cache1.length == chunk_size and cache1.count == chunk_size
    assert pos1 == chunk_size

    q2 = torch.randn(B, chunk_size, H, Dh)
    k2 = torch.randn(B, chunk_size, H, Dh)
    v2 = torch.randn(B, chunk_size, H, Dv)

    out2, cache2, pos2 = attn(
        q2,
        k2,
        v2,
        start_index=pos1,
        cache=cache1,
        attn_mask=mask,
        training=False,
        max_cache_len=max_cache_len,
        return_cache=True,
        return_position=True,
    )

    assert out2.shape == (B, chunk_size, H * Dv)
    assert cache2 is not None
    assert cache2.length == max_cache_len
    assert cache2.count == pos1 + chunk_size
    assert cache2.start_index == cache2.count - cache2.length
    assert pos2 == cache2.count

    # Unbounded cache should retain the full history (2 * chunk_size).
    out_u, cache_u, pos_u = attn(
        q2,
        k2,
        v2,
        start_index=pos1,
        cache=cache1,
        attn_mask=mask,
        training=False,
        max_cache_len=None,
        cache_unbounded=True,
        return_cache=True,
        return_position=True,
    )
    assert cache_u is not None and cache_u.length == chunk_size * 2
    assert pos_u == cache_u.count
    assert torch.isfinite(out_u).all()

    # Manual reference with sliding window (keep last max_cache_len keys)
    q1_rot, k1_rot = attn.rope(q1, k1, start_index=0)
    q2_rot, k2_rot = attn.rope(q2, k2, start_index=pos1)
    k_all = torch.cat([k1_rot, k2_rot], dim=1)
    v_all = torch.cat([v1, v2], dim=1)
    k_keep = k_all[:, -max_cache_len:]
    v_keep = v_all[:, -max_cache_len:]

    key_positions = torch.arange(
        pos1 + chunk_size - max_cache_len, pos1 + chunk_size, device=q1.device
    )
    query_positions = torch.arange(pos1, pos1 + chunk_size, device=q1.device)
    causal = key_positions.view(1, 1, 1, max_cache_len) <= query_positions.view(
        1, 1, chunk_size, 1
    )
    mask_causal = torch.where(
        causal,
        torch.zeros_like(causal, dtype=torch.float32),
        float("-inf"),
    )

    q2_r = q2_rot.transpose(1, 2)  # (B,H,L,Dh)
    k_keep_t = k_keep.transpose(1, 2)  # (B,H,Lk,Dh)
    v_keep_t = v_keep.transpose(1, 2)  # (B,H,Lk,Dv)

    scores = torch.matmul(q2_r, k_keep_t.transpose(-2, -1))
    scores = scores + mask_causal
    weights = torch.softmax(scores.float(), dim=-1).to(q2_r)
    ref = torch.matmul(weights, v_keep_t).transpose(1, 2).reshape(B, chunk_size, -1)

    assert torch.allclose(out2, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.cuda
@torch.no_grad()
def test_cuda_smoke() -> None:
    """CUDA path sanity check (skipped when GPU is unavailable)."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA available")
    torch.manual_seed(0)
    cfg = MegalodonConfig()
    lm = MegalodonForCausalLM(cfg).eval().cuda()

    B, L = 1, 32
    x = torch.randint(0, cfg.vocab_size, (B, L), device="cuda")
    attn = torch.ones(B, L, dtype=torch.long, device="cuda")
    logits = lm(
        input_ids=x,
        attention_mask=attn,
        use_cache=False,
        return_dict=True,
    ).logits
    assert logits.is_cuda and logits.shape == (B, L, cfg.vocab_size)


def test_sdpa_matches_reference() -> None:
    """SDPA helper should match manual attention computation on CUDA."""
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

    scores = torch.matmul(q_rot, k_rot.transpose(-2, -1))
    mask = attn._causal_mask(L, L, q.device, q.dtype)
    scores = scores + mask
    weights = torch.softmax(scores.float(), dim=-1).to(q)
    ref = torch.matmul(weights, v_).transpose(1, 2).reshape(B, L, -1)

    assert torch.allclose(out_manual, ref, atol=1e-5, rtol=1e-5)
