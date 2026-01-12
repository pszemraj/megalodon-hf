# coding=utf-8
"""Path-matrix tests for ``ChunkedSelfAttention.forward``.

The nested ``attend_single_chunk`` helper is not the only source of masking
regressions. ``ChunkedSelfAttention.forward`` itself has several major branches:

* streaming mode (return_cache or cache provided) vs non-streaming
* single-chunk vs multi-chunk
* multi-chunk vectorized SDPA path (attn_mask is None) vs per-chunk loop (attn_mask not None)
* non-divisible sequence length padding

These tests pin equivalence across those branches for cases where semantics
should match.
"""

from __future__ import annotations

import pytest
import torch

from megalodon.modeling_megalodon import AttentionCache, ChunkedSelfAttention


@torch.no_grad()
def test_multi_chunk_vectorized_matches_loop_all_ones_mask() -> None:
    """Vectorized SDPA path (mask=None) must match loop path (mask=all ones)."""
    torch.manual_seed(0)

    B, H, Dh, Dv = 2, 3, 4, 5
    chunk_size = 4
    L = chunk_size * 3

    attn = ChunkedSelfAttention(
        num_heads=H,
        head_dim=Dh,
        value_head_dim=Dv,
        chunk_size=chunk_size,
        rope_base=10_000.0,
        attention_dropout=0.0,
    ).eval()

    q = torch.randn(B, L, H, Dh)
    k = torch.randn(B, L, H, Dh)
    v = torch.randn(B, L, H, Dv)

    out_vec, _ = attn(
        q,
        k,
        v,
        start_index=0,
        cache=None,
        attn_mask=None,
        training=False,
    )

    mask = torch.ones(B, L, dtype=torch.bool)
    out_loop, _ = attn(
        q,
        k,
        v,
        start_index=0,
        cache=None,
        attn_mask=mask,
        training=False,
    )

    assert torch.allclose(out_vec, out_loop, atol=1e-5, rtol=1e-5)


@torch.no_grad()
def test_multi_chunk_streaming_matches_non_streaming_when_mask_none() -> None:
    """Streaming (return_cache=True) must match non-streaming for chunk-local attention."""
    torch.manual_seed(0)

    B, H, Dh, Dv = 1, 2, 4, 3
    chunk_size = 4
    L = chunk_size * 2

    attn = ChunkedSelfAttention(
        num_heads=H,
        head_dim=Dh,
        value_head_dim=Dv,
        chunk_size=chunk_size,
        rope_base=10_000.0,
        attention_dropout=0.0,
    ).eval()

    q = torch.randn(B, L, H, Dh)
    k = torch.randn(B, L, H, Dh)
    v = torch.randn(B, L, H, Dv)

    out_block, _ = attn(
        q,
        k,
        v,
        start_index=0,
        cache=None,
        attn_mask=None,
        training=False,
        return_cache=False,
    )

    out_stream, _cache = attn(
        q,
        k,
        v,
        start_index=0,
        cache=None,
        attn_mask=None,
        training=False,
        return_cache=True,
        max_cache_len=-1,
    )

    assert torch.allclose(out_block, out_stream, atol=1e-5, rtol=1e-5)


@torch.no_grad()
def test_multi_chunk_streaming_matches_manual_split_with_padding_mask() -> None:
    """With a padding mask, streaming chunk-local attention should match a manual split."""
    torch.manual_seed(0)

    B, H, Dh, Dv = 2, 2, 2, 2
    chunk_size = 4
    L = chunk_size * 2

    attn = ChunkedSelfAttention(
        num_heads=H,
        head_dim=Dh,
        value_head_dim=Dv,
        chunk_size=chunk_size,
        rope_base=10_000.0,
        attention_dropout=0.0,
    ).eval()
    attn._sdpa_available = False

    q = torch.randn(B, L, H, Dh)
    k = torch.randn(B, L, H, Dh)
    v = torch.randn(B, L, H, Dv)

    mask = torch.tensor(
        [
            [1, 1, 0, 1, 1, 0, 1, 1],
            [1, 0, 1, 1, 1, 1, 0, 1],
        ],
        dtype=torch.bool,
    )

    out_stream, _cache = attn(
        q,
        k,
        v,
        start_index=0,
        cache=None,
        attn_mask=mask,
        training=False,
        return_cache=True,
        max_cache_len=-1,
    )

    out_a, cache_a = attn(
        q[:, :chunk_size],
        k[:, :chunk_size],
        v[:, :chunk_size],
        start_index=0,
        cache=None,
        attn_mask=mask[:, :chunk_size],
        training=False,
        return_cache=True,
        max_cache_len=-1,
    )
    out_b, _cache_b = attn(
        q[:, chunk_size:],
        k[:, chunk_size:],
        v[:, chunk_size:],
        start_index=0,
        cache=cache_a,
        attn_mask=mask[:, chunk_size:],
        training=False,
        return_cache=True,
        max_cache_len=-1,
    )
    out_manual = torch.cat([out_a, out_b], dim=1)

    assert torch.allclose(out_stream, out_manual, atol=1e-5, rtol=1e-5)


@torch.no_grad()
def test_future_tokens_do_not_affect_previous_outputs_within_chunk() -> None:
    """Changing *future* K/V must not change outputs for earlier positions.

    This guards against causal-mask regressions when sequences are padded to a
    multiple of chunk_size.
    """
    torch.manual_seed(0)

    B, H, Dh, Dv = 1, 1, 2, 2
    chunk_size = 8

    real_L = chunk_size + 2
    padded_L = chunk_size * 2

    attn = ChunkedSelfAttention(
        num_heads=H,
        head_dim=Dh,
        value_head_dim=Dv,
        chunk_size=chunk_size,
        rope_base=10_000.0,
        attention_dropout=0.0,
    ).eval()

    q = torch.randn(B, padded_L, H, Dh)
    k = torch.randn(B, padded_L, H, Dh)
    v = torch.randn(B, padded_L, H, Dv)

    out_full, _ = attn(
        q,
        k,
        v,
        start_index=0,
        cache=None,
        attn_mask=None,
        training=False,
    )

    k2 = k.clone()
    v2 = v.clone()
    k2[:, real_L:] = 0.0
    v2[:, real_L:] = 0.0
    out_zero, _ = attn(
        q,
        k2,
        v2,
        start_index=0,
        cache=None,
        attn_mask=None,
        training=False,
    )

    assert torch.allclose(
        out_full[:, :real_L], out_zero[:, :real_L], atol=1e-5, rtol=1e-5
    )


def test_forward_rejects_bad_cache_type() -> None:
    """ChunkedSelfAttention should fail fast on non-AttentionCache cache objects."""
    torch.manual_seed(0)

    attn = ChunkedSelfAttention(
        num_heads=1,
        head_dim=2,
        value_head_dim=2,
        chunk_size=4,
        rope_base=10_000.0,
        attention_dropout=0.0,
    )

    q = torch.randn(1, 2, 1, 2)
    k = torch.randn(1, 2, 1, 2)
    v = torch.randn(1, 2, 1, 2)

    with pytest.raises(TypeError, match="AttentionCache"):
        _ = attn(
            q, k, v, start_index=0, cache=(None, None), attn_mask=None, training=False
        )


@torch.no_grad()
def test_cache_clamp_trims_mask_and_kv_together() -> None:
    """Clamping the cache must trim K, V, and cache.mask identically."""
    torch.manual_seed(0)

    B, H, Dh, Dv = 1, 1, 2, 2
    chunk_size = 4
    attn = ChunkedSelfAttention(
        num_heads=H,
        head_dim=Dh,
        value_head_dim=Dv,
        chunk_size=chunk_size,
        rope_base=10_000.0,
        attention_dropout=0.0,
    ).eval()
    attn._sdpa_available = False

    past_len = 6
    k_past = torch.randn(B, past_len, H, Dh)
    v_past = torch.randn(B, past_len, H, Dv)
    mask_past = torch.tensor([[1, 0, 1, 1, 0, 1]], dtype=torch.bool)

    cache = AttentionCache(
        k=k_past,
        v=v_past,
        count=torch.tensor([7], dtype=torch.long),
        mask=mask_past,
    )

    q = torch.randn(B, 1, H, Dh)
    k = torch.randn(B, 1, H, Dh)
    v = torch.randn(B, 1, H, Dv)

    _out, new_cache, _pos = attn(
        q,
        k,
        v,
        start_index=past_len,
        cache=cache,
        attn_mask=torch.ones(B, 1, dtype=torch.bool),
        training=False,
        max_cache_len=4,
        return_cache=True,
        return_position=True,
    )

    assert new_cache is not None
    assert new_cache.length == 4
    assert new_cache.mask is not None
    expected = torch.cat([mask_past[:, -3:], torch.ones(B, 1, dtype=torch.bool)], dim=1)
    assert torch.equal(new_cache.mask, expected)
