# coding=utf-8
"""RoPE (rotary embedding) position correctness and input validation.

Why this file exists
--------------------
RoPE bugs are usually "silent" (everything runs, but positions are off), and the
symptoms look like generic quality drops. The two highest value invariants are:

1) **Slice vs offset equivalence**:
   Rotating a window with start_index=S must equal rotating the full sequence
   from start_index=0 and slicing positions [S:S+L].

2) **Negative start_index is invalid** (should fail fast).

This test uses the public `ChunkedSelfAttention.rope` object (whatever its
concrete class is), because the model consistently calls `attn.rope(q, k,
start_index=...)`.
"""

from __future__ import annotations

import pytest
import torch

from megalodon.modeling_megalodon import ChunkedSelfAttention


TOL_FP32 = 1e-5
TOL_BF16 = 5e-4


@pytest.mark.parametrize("start_index", [0, 1, 3, 4, 57], ids=lambda s: f"start={s}")
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16], ids=lambda d: str(d).split(".")[-1]
)
@torch.no_grad()
def test_rope_slice_vs_offset_equivalence(start_index: int, dtype: torch.dtype) -> None:
    torch.manual_seed(0)

    chunk_size = 8
    B, H, Dh = 2, 2, 8
    L = 11
    total = start_index + L

    attn = ChunkedSelfAttention(
        num_heads=H,
        head_dim=Dh,
        value_head_dim=Dh,
        chunk_size=chunk_size,
        rope_base=10_000.0,
        attention_dropout=0.0,
    ).eval()

    rope = attn.rope

    q_full = torch.randn(B, total, H, Dh, dtype=dtype)
    k_full = torch.randn(B, total, H, Dh, dtype=dtype)

    q_slice = q_full[:, start_index:]
    k_slice = k_full[:, start_index:]

    q_rot_slice, k_rot_slice = rope(q_slice, k_slice, start_index=start_index)
    q_rot_full, k_rot_full = rope(q_full, k_full, start_index=0)

    q_rot_full_slice = q_rot_full[:, start_index:]
    k_rot_full_slice = k_rot_full[:, start_index:]

    tol = TOL_BF16 if dtype == torch.bfloat16 else TOL_FP32
    assert torch.allclose(
        q_rot_slice.float(), q_rot_full_slice.float(), atol=tol, rtol=tol
    )
    assert torch.allclose(
        k_rot_slice.float(), k_rot_full_slice.float(), atol=tol, rtol=tol
    )


def test_rope_negative_start_index_rejected() -> None:
    torch.manual_seed(0)

    attn = ChunkedSelfAttention(
        num_heads=1,
        head_dim=8,
        value_head_dim=8,
        chunk_size=8,
        rope_base=10_000.0,
        attention_dropout=0.0,
    ).eval()

    q = torch.randn(1, 3, 1, 8)
    k = torch.randn(1, 3, 1, 8)

    with pytest.raises(
        (ValueError, RuntimeError), match=r"start_index|start|position|pos"
    ):
        _ = attn.rope(q, k, start_index=-1)
