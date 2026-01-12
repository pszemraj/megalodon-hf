# coding=utf-8
"""Additional path coverage for :class:`~megalodon.modeling_megalodon.ChunkedSelfAttention`.

The repo already has very thorough tests for:

* SDPA vs manual masking
* sliding-window cache semantics
* multi-chunk block-diagonal equivalence

However, a few branches are still easy to miss in reviews:

* **Chunk-boundary handling** when start_index is *not* aligned to chunk_size
* **Multi-chunk, all-ones mask** (attn_mask provided) should match the
  optimized attn_mask=None vectorized path
* **Non-divisible lengths**: non-streaming pad/unpad path should match the
  streaming chunk-splitting path

These tests aim to pin those behaviors down.
"""

from __future__ import annotations

import torch

from megalodon.modeling_megalodon import ChunkedSelfAttention


TOL = 1e-5


@torch.no_grad()
def test_chunk_local_reset_when_start_pos_is_misaligned() -> None:
    """Faithful chunk-local streaming must reset cache at *absolute* chunk boundaries.

    Scenario:
      chunk_size=4
      start_index=3 (we are 1 token before a chunk boundary at position 4)
      L=3 tokens => absolute positions [3,4,5]

    Correct behavior (chunk-local):
      - token at position 4 must NOT attend to token at position 3.

    This test compares the built-in streaming chunk-splitting behavior to a
    manual split exactly at the absolute boundary.
    """
    torch.manual_seed(0)

    chunk_size = 4
    attn = ChunkedSelfAttention(
        num_heads=1,
        head_dim=2,
        value_head_dim=2,
        chunk_size=chunk_size,
        rope_base=10_000.0,
        attention_dropout=0.0,
    ).eval()
    attn._sdpa_available = False

    B, L = 1, 3
    start_index = 3

    q = torch.zeros(B, L, 1, 2)
    k = torch.zeros(B, L, 1, 2)

    v = torch.tensor(
        [[[[100.0, 0.0]], [[1.0, 0.0]], [[2.0, 0.0]]]],
        dtype=torch.float32,
    )

    out_stream, _cache_stream = attn(
        q,
        k,
        v,
        start_index=start_index,
        cache=None,
        attn_mask=None,
        training=False,
        max_cache_len=chunk_size,
        cache_unbounded=False,
        return_cache=True,
    )

    out_a, _cache_a = attn(
        q[:, :1],
        k[:, :1],
        v[:, :1],
        start_index=start_index,
        cache=None,
        attn_mask=None,
        training=False,
        max_cache_len=chunk_size,
        cache_unbounded=False,
        return_cache=True,
    )
    out_b, _cache_b = attn(
        q[:, 1:],
        k[:, 1:],
        v[:, 1:],
        start_index=start_index + 1,
        cache=None,
        attn_mask=None,
        training=False,
        max_cache_len=chunk_size,
        cache_unbounded=False,
        return_cache=True,
    )
    out_manual = torch.cat([out_a, out_b], dim=1)

    assert torch.allclose(out_stream, out_manual, atol=TOL, rtol=TOL), (
        "Chunk boundary handling mismatch when start_index is misaligned. "
        f"max diff={(out_stream - out_manual).abs().max().item():.6g}"
    )

    assert torch.allclose(
        out_stream[:, 1], v[:, 1].reshape(1, -1), atol=1e-6, rtol=1e-6
    )


@torch.no_grad()
def test_multi_chunk_all_ones_mask_matches_no_mask() -> None:
    """Providing an all-ones attn_mask must not change results.

    This pins equivalence between:
      - optimized vectorized multi-chunk path (attn_mask=None)
      - explicit masked multi-chunk fallback loop (attn_mask provided)
    """
    torch.manual_seed(0)

    chunk_size = 4
    attn = ChunkedSelfAttention(
        num_heads=2,
        head_dim=4,
        value_head_dim=4,
        chunk_size=chunk_size,
        rope_base=10_000.0,
        attention_dropout=0.0,
    ).eval()
    attn._sdpa_available = True

    B, L = 2, 8
    q = torch.randn(B, L, 2, 4)
    k = torch.randn(B, L, 2, 4)
    v = torch.randn(B, L, 2, 4)

    out_nomask, _ = attn(
        q,
        k,
        v,
        start_index=0,
        cache=None,
        attn_mask=None,
        training=False,
        max_cache_len=chunk_size,
        cache_unbounded=False,
        return_cache=False,
    )

    mask_all = torch.ones(B, L, dtype=torch.bool)
    out_masked, _ = attn(
        q,
        k,
        v,
        start_index=0,
        cache=None,
        attn_mask=mask_all,
        training=False,
        max_cache_len=chunk_size,
        cache_unbounded=False,
        return_cache=False,
    )

    assert torch.allclose(out_nomask, out_masked, atol=TOL, rtol=TOL), (
        "All-ones mask changed multi-chunk output. "
        f"max diff={(out_nomask - out_masked).abs().max().item():.6g}"
    )


@torch.no_grad()
def test_non_divisible_length_streaming_matches_non_streaming() -> None:
    """Non-streaming pad/unpad should match streaming chunk splitting.

    This test hits the `pad_len` branch (L > chunk_size and L % chunk_size != 0).
    """
    torch.manual_seed(0)

    chunk_size = 4
    attn = ChunkedSelfAttention(
        num_heads=2,
        head_dim=4,
        value_head_dim=4,
        chunk_size=chunk_size,
        rope_base=10_000.0,
        attention_dropout=0.0,
    ).eval()

    B, L = 1, 6
    q = torch.randn(B, L, 2, 4)
    k = torch.randn(B, L, 2, 4)
    v = torch.randn(B, L, 2, 4)

    out_full, _ = attn(
        q,
        k,
        v,
        start_index=0,
        cache=None,
        attn_mask=None,
        training=False,
        max_cache_len=chunk_size,
        cache_unbounded=False,
        return_cache=False,
    )

    out_stream, _ = attn(
        q,
        k,
        v,
        start_index=0,
        cache=None,
        attn_mask=None,
        training=False,
        max_cache_len=chunk_size,
        cache_unbounded=False,
        return_cache=True,
    )

    assert torch.allclose(out_full, out_stream, atol=TOL, rtol=TOL), (
        "Non-divisible length mismatch between streaming and non-streaming paths. "
        f"max diff={(out_full - out_stream).abs().max().item():.6g}"
    )
