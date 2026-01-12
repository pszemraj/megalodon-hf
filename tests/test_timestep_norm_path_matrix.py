# coding=utf-8
"""Exhaustive path-matrix tests for :class:`~megalodon.modeling_megalodon.TimestepNorm`.

TimestepNorm is the other stateful "long memory" component in Megalodon.
It has multiple branches (prev state present vs not, padding mask vs None,
L==0 edge-case, bf16 stats upcasting) that are easy to regress.

These tests intentionally enumerate combinations rather than sampling.
"""

from __future__ import annotations

import pytest
import torch

from megalodon.modeling_megalodon import TimestepNorm


TOL_FP32 = 1e-5
TOL_BF16 = 5e-2


def _make_mask(B: int, L: int) -> torch.Tensor:
    """Mask with a mixture of right-pad, left-pad, and internal gaps."""
    mask = torch.ones(B, L, dtype=torch.bool)
    if L >= 2:
        mask[0, -2:] = False
    if B >= 2 and L >= 3:
        mask[1, :3] = False
    if B >= 3 and L >= 5:
        mask[2, 2] = False
    return mask


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mask_present", [False, True])
@pytest.mark.parametrize("prev_state_present", [False, True])
@pytest.mark.parametrize("split_at", [0, 3])
def test_timestep_norm_streaming_matches_full_across_paths(
    dtype: torch.dtype,
    mask_present: bool,
    prev_state_present: bool,
    split_at: int,
) -> None:
    """All key branches should preserve the full-vs-streamed equivalence."""
    torch.manual_seed(0)

    B, L, D, G = 3, 7, 12, 3
    norm = TimestepNorm(num_features=D, num_groups=G).eval()

    x = torch.randn(B, L, D, dtype=dtype)
    mask = _make_mask(B, L) if mask_present else None

    prev_count = prev_mean = prev_var = None
    if prev_state_present:
        prev_count = torch.tensor([5, 7, 1], dtype=torch.long)
        prev_mean = torch.randn(B, G, dtype=torch.float32)
        prev_var = torch.rand(B, G, dtype=torch.float32) + 0.5

    y_full, c_full, m_full, v_full = norm(
        x,
        prev_count=prev_count,
        prev_mean=prev_mean,
        prev_var=prev_var,
        padding_mask=mask,
    )

    count = prev_count
    mean = prev_mean
    var = prev_var
    chunks = []
    for start, end in [(0, split_at), (split_at, L)]:
        chunk_mask = mask[:, start:end] if mask is not None else None
        y_chunk, count, mean, var = norm(
            x[:, start:end],
            prev_count=count,
            prev_mean=mean,
            prev_var=var,
            padding_mask=chunk_mask,
        )
        chunks.append(y_chunk)

    y_stream = torch.cat(chunks, dim=1)

    tol = TOL_BF16 if dtype == torch.bfloat16 else TOL_FP32
    assert torch.allclose(y_stream, y_full, atol=tol, rtol=tol), (
        "TimestepNorm streamed output != full output for combo "
        f"dtype={dtype}, mask_present={mask_present}, prev_state_present={prev_state_present}, "
        f"split_at={split_at}. max diff={(y_stream - y_full).abs().max().item():.6g}"
    )

    assert torch.equal(count, c_full)
    assert torch.allclose(mean, m_full, atol=tol, rtol=tol)
    assert torch.allclose(var, v_full, atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_timestep_norm_all_masked_chunk_does_not_update_state(
    dtype: torch.dtype,
) -> None:
    """If an entire segment is masked out, stats must remain unchanged."""
    torch.manual_seed(0)

    B, L, D, G = 2, 5, 12, 3
    norm = TimestepNorm(num_features=D, num_groups=G).eval()

    prev_count = torch.tensor([4, 9], dtype=torch.long)
    prev_mean = torch.randn(B, G, dtype=torch.float32)
    prev_var = torch.rand(B, G, dtype=torch.float32) + 0.5

    x = torch.randn(B, L, D, dtype=dtype)
    mask = torch.zeros(B, L, dtype=torch.bool)

    y, c, m, v = norm(
        x,
        prev_count=prev_count,
        prev_mean=prev_mean,
        prev_var=prev_var,
        padding_mask=mask,
    )

    assert torch.isfinite(y).all()

    assert torch.equal(c, prev_count)
    tol = TOL_BF16 if dtype == torch.bfloat16 else TOL_FP32
    assert torch.allclose(m, prev_mean.to(m.dtype), atol=tol, rtol=tol)
    assert torch.allclose(v, prev_var.to(v.dtype), atol=tol, rtol=tol)
