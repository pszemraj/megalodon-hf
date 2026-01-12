# coding=utf-8
"""Exhaustive test suite for ChunkedSelfAttention masking and caching.

This file exists because multiple non-trivial bugs have appeared in
`ChunkedSelfAttention.attend_single_chunk`. The root cause is combinatorial:
correctness depends on the interaction of:

    - cache_blk present vs absent
    - mask_blk present vs absent
    - cache_blk.mask present vs absent
    - trimming (sliding window) active vs inactive
    - SDPA vs manual attention path

This suite:
    1. Enumerates all valid combinations in a parametrized matrix test
    2. Provides named regression tests for each known bug class
    3. Uses loud test values so masking failures are obvious (not epsilon diffs)
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Optional

import pytest
import torch

pytest.importorskip("transformers")

from megalodon import MegalodonConfig
from megalodon.modeling_megalodon import AttentionCache, ChunkedSelfAttention

# =============================================================================
# Test Fixtures and Helpers
# =============================================================================


@dataclass(frozen=True)
class _Case:
    """A single test case configuration.

    :ivar bool cache_present: Whether an attention cache is provided.
    :ivar bool mask_present: Whether a current attention mask is provided.
    :ivar bool cache_mask_present: Whether the cache carries a mask.
    :ivar bool trim_active: Whether cache trimming is enabled.
    :ivar bool sdpa: Whether SDPA is enabled for attention.
    """

    cache_present: bool
    mask_present: bool
    cache_mask_present: bool
    trim_active: bool
    sdpa: bool

    @property
    def id(self) -> str:
        return (
            f"cache={int(self.cache_present)}_"
            f"mask={int(self.mask_present)}_"
            f"cachemask={int(self.cache_mask_present)}_"
            f"trim={int(self.trim_active)}_"
            f"sdpa={int(self.sdpa)}"
        )

    @property
    def is_valid(self) -> bool:
        """Some combinations are invalid and should be skipped or should raise.

        :return bool: ``True`` when the case is valid and should be tested.
        """
        # cache_mask without cache is meaningless (skip)
        if self.cache_mask_present and not self.cache_present:
            return False
        return True

    @property
    def should_raise(self) -> bool:
        """Trim without cache requires max_cache_len < chunk_size, which must raise.

        :return bool: ``True`` when the case should raise a runtime error.
        """
        return self.trim_active and not self.cache_present


def _make_attn(*, chunk_size: int = 4, sdpa: bool = True) -> ChunkedSelfAttention:
    """Create a small attention module for testing.

    :param int chunk_size: Chunk size for attention, defaults to ``4``.
    :param bool sdpa: Whether to enable the SDPA path, defaults to ``True``.
    :return ChunkedSelfAttention: Configured attention module in eval mode.
    """
    attn = ChunkedSelfAttention(
        num_heads=2,
        head_dim=4,  # must be even for RoPE
        value_head_dim=1,  # 1D values make expected outputs easy to compute
        chunk_size=chunk_size,
        rope_base=10_000.0,
        attention_dropout=0.0,
    )
    attn._sdpa_available = sdpa
    return attn.eval()


def _zeros_qk(
    batch: int, length: int, heads: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create zero Q/K tensors. With q=k=0, attention is uniform over allowed keys.

    :param int batch: Batch size.
    :param int length: Sequence length.
    :param int heads: Number of attention heads.
    :param int head_dim: Per-head dimensionality.
    :return tuple[torch.Tensor, torch.Tensor]: Zero-valued ``(q, k)`` tensors.
    """
    shape = (batch, length, heads, head_dim)
    return torch.zeros(shape), torch.zeros(shape)


def _make_v_loud(
    batch: int,
    length: int,
    heads: int,
    base_values: list[float],
    mask: Optional[torch.Tensor] = None,
    loud_value: float = 1000.0,
) -> torch.Tensor:
    """Create V tensor where masked positions have loud (large) values.

    This makes masking bugs obvious: if a masked key leaks through,
    the output will be off by hundreds, not by floating-point epsilon.

    :param int batch: Batch size.
    :param int length: Sequence length.
    :param int heads: Number of attention heads.
    :param list[float] base_values: Per-position base values for ``v``.
    :param Optional[torch.Tensor] mask: Optional validity mask, defaults to ``None``.
    :param float loud_value: Value added to masked positions, defaults to ``1000.0``.
    :return torch.Tensor: Value tensor shaped ``(batch, length, heads, 1)``.
    """
    assert len(base_values) == length
    # Shape: (1, L, 1, 1) -> broadcast to (B, L, H, 1)
    v = torch.tensor(base_values, dtype=torch.float32).view(1, length, 1, 1)
    v = v.expand(batch, length, heads, 1).clone()

    if mask is not None:
        # mask: (B, L) with True=valid, False=masked
        # Add loud_value to masked positions
        invalid = (~mask).float().view(batch, length, 1, 1)
        v = v + invalid * loud_value

    return v


def _make_mask(batch: int, pattern: list[int]) -> torch.Tensor:
    """Create boolean mask from 0/1 pattern. 1=valid, 0=masked.

    :param int batch: Batch size.
    :param list[int] pattern: Mask pattern with 1=valid and 0=masked.
    :return torch.Tensor: Boolean mask shaped ``(batch, length)``.
    """
    m = torch.tensor(pattern, dtype=torch.bool).view(1, -1)
    return m.expand(batch, -1).clone()


# =============================================================================
# Reference Implementation (Intended Semantics)
# =============================================================================


def _reference_attend(
    attn: ChunkedSelfAttention,
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    start_index: int,
    cache: Optional[AttentionCache],
    mask_blk: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor] = None,
    max_cache_len: int,
) -> tuple[torch.Tensor, AttentionCache]:
    """Reference implementation encoding the intended semantics.

    Key semantic invariants:
    1. cache.mask is ALWAYS respected when present, even if mask_blk is None
    2. Missing masks default to all-ones (all valid)
    3. Trimming uses suffix alignment: kept = full[-keep:], not full[:keep]
    4. Returned cache.mask matches returned K/V length

    :param ChunkedSelfAttention attn: Attention module under test.
    :param torch.Tensor q: Query tensor shaped ``(batch, length, heads, dim)``.
    :param torch.Tensor k: Key tensor shaped ``(batch, length, heads, dim)``.
    :param torch.Tensor v: Value tensor shaped ``(batch, length, heads, dim_v)``.
    :param int start_index: Absolute start index for RoPE positions.
    :param Optional[AttentionCache] cache: Optional attention cache.
    :param Optional[torch.Tensor] mask_blk: Optional current chunk mask.
    :param Optional[torch.Tensor] position_ids: Optional explicit position ids, defaults to ``None``.
    :param int max_cache_len: Maximum cache length for trimming.
    :return tuple[torch.Tensor, AttentionCache]: Output tensor and updated cache.
    """
    B, Lq, H, Dh = q.shape
    Dv = v.size(-1)
    chunk_size = attn.chunk_size

    # Validation
    if max_cache_len < chunk_size:
        raise ValueError(
            f"max_cache_len ({max_cache_len}) must be >= chunk_size ({chunk_size})"
        )

    # Determine effective start position
    if cache is not None:
        start_pos = cache.count
    else:
        start_pos = torch.full((B,), start_index, dtype=torch.long, device=q.device)

    faithful_chunk_local = max_cache_len == chunk_size
    if cache is not None and faithful_chunk_local:
        pos_in_chunk = start_pos % chunk_size
        max_pos_in_chunk = int(pos_in_chunk.max().item())
        all_at_boundary = (pos_in_chunk == 0).all().item()
        if all_at_boundary:
            cache = None
        else:
            if cache.length > max_pos_in_chunk:
                cache = AttentionCache(
                    k=cache.k[:, -max_pos_in_chunk:],
                    v=cache.v[:, -max_pos_in_chunk:],
                    count=cache.count,
                    mask=cache.mask[:, -max_pos_in_chunk:]
                    if cache.mask is not None
                    else None,
                )
            if cache.length > 0:
                keep_counts = pos_in_chunk.clamp(max=cache.length)
                idx = torch.arange(cache.length, device=q.device).unsqueeze(0)
                chunk_mask = idx >= (cache.length - keep_counts).unsqueeze(1)
                if cache.mask is None:
                    if not chunk_mask.all():
                        cache = AttentionCache(
                            k=cache.k,
                            v=cache.v,
                            count=cache.count,
                            mask=chunk_mask,
                        )
                else:
                    cache = AttentionCache(
                        k=cache.k,
                        v=cache.v,
                        count=cache.count,
                        mask=cache.mask & chunk_mask,
                    )

    if cache is not None:
        prefix_len = cache.length
    else:
        prefix_len = 0

    if position_ids is None:
        if mask_blk is not None:
            mask_long = mask_blk.to(torch.long)
            pos_offsets = mask_long.cumsum(dim=1) - 1
            pos_offsets = pos_offsets.clamp(min=0)
            position_ids = start_pos.unsqueeze(1) + pos_offsets
        else:
            offsets = torch.arange(Lq, device=q.device, dtype=torch.long)
            position_ids = start_pos.unsqueeze(1) + offsets.unsqueeze(0)
    else:
        position_ids = position_ids.to(device=q.device, dtype=torch.long)

    # Apply RoPE (no-op for zero q/k, but maintain parity)
    q_rot, k_rot = attn.rope(q, k, start_index=0, position_ids=position_ids)

    # Concatenate with cache
    if cache is not None:
        k_cat = torch.cat([cache.k, k_rot], dim=1)
        v_cat = torch.cat([cache.v, v], dim=1)
    else:
        k_cat = k_rot
        v_cat = v

    # Build full mask BEFORE trimming
    # Invariant: missing segments default to all-ones
    if cache is not None:
        if cache.mask is not None:
            prefix_mask = cache.mask
        else:
            prefix_mask = torch.ones(B, prefix_len, dtype=torch.bool, device=q.device)
    else:
        prefix_mask = torch.ones(B, 0, dtype=torch.bool, device=q.device)

    if mask_blk is not None:
        cur_mask = mask_blk
    else:
        cur_mask = torch.ones(B, Lq, dtype=torch.bool, device=q.device)

    full_mask = torch.cat([prefix_mask, cur_mask], dim=1)

    # Apply trimming (suffix alignment)
    total_len = k_cat.size(1)
    keep = min(max_cache_len, total_len)
    if keep < total_len:
        k_cat = k_cat[:, -keep:]
        v_cat = v_cat[:, -keep:]
        full_mask = full_mask[:, -keep:]  # suffix, not prefix

    Lk = k_cat.size(1)
    if cache is not None:
        if cache.mask is None:
            cache_offset = cache.count - cache.length
            cache_positions = cache_offset.unsqueeze(1) + torch.arange(
                prefix_len, device=q.device
            ).unsqueeze(0)
        else:
            cache_mask_long = cache.mask.to(torch.long)
            cache_valid = cache_mask_long.sum(dim=1)
            cache_offset = cache.count - cache_valid
            cache_positions = cache_mask_long.cumsum(dim=1) - 1
            cache_positions = cache_positions.clamp(min=0)
            cache_positions = cache_offset.unsqueeze(1) + cache_positions
        key_positions = torch.cat([cache_positions, position_ids], dim=1)
    else:
        key_positions = position_ids
    if key_positions.size(1) > Lk:
        key_positions = key_positions[:, -Lk:]

    # Manual attention computation
    q_ = q_rot.transpose(1, 2)  # (B, H, Lq, Dh)
    k_ = k_cat.transpose(1, 2)  # (B, H, Lk, Dh)
    v_ = v_cat.transpose(1, 2)  # (B, H, Lk, Dv)

    # Attention scores
    scores = torch.matmul(q_, k_.transpose(-2, -1))  # (B, H, Lq, Lk)

    # Causal mask (position-based to match implementation)
    query_positions = position_ids
    causal = key_positions.unsqueeze(1) <= query_positions.unsqueeze(2)
    causal_mask = torch.where(
        causal,
        torch.zeros_like(causal, dtype=scores.dtype),
        scores.new_tensor(float("-inf")),
    )
    scores = scores + causal_mask.unsqueeze(1)

    # Key validity mask
    # full_mask: (B, Lk) -> (B, 1, 1, Lk)
    key_mask = full_mask.float().view(B, 1, 1, Lk)
    scores = scores + (key_mask - 1.0) * 1e9  # -inf for invalid keys

    # Softmax and output
    weights = torch.softmax(scores, dim=-1)
    out = torch.matmul(weights, v_)  # (B, H, Lq, Dv)
    out = out.transpose(1, 2).reshape(B, Lq, H * Dv)
    if mask_blk is not None:
        out = torch.where(mask_blk.unsqueeze(-1), out, out.new_zeros(()))

    # Build return cache
    if mask_blk is not None:
        valid_counts = mask_blk.to(torch.long).sum(dim=1)
    else:
        valid_counts = torch.full((B,), Lq, dtype=torch.long, device=q.device)
    new_count = start_pos + valid_counts

    # Cache mask: present if ANY mask info existed
    if cache is not None and cache.mask is not None:
        new_mask = full_mask
    elif mask_blk is not None:
        new_mask = full_mask
    else:
        new_mask = None

    new_cache = AttentionCache(k=k_cat, v=v_cat, count=new_count, mask=new_mask)

    return out, new_cache


# =============================================================================
# Matrix Test: All Valid Combinations
# =============================================================================


def _generate_cases() -> list[_Case]:
    """Generate all valid test cases.

    :return list[_Case]: List of valid test case configurations.
    """
    cases = []
    for combo in itertools.product([False, True], repeat=5):
        case = _Case(*combo)
        if case.is_valid:
            cases.append(case)
    return cases


@pytest.mark.parametrize("case", _generate_cases(), ids=lambda c: c.id)
@torch.no_grad()
def test_attend_single_chunk_matrix(case: _Case) -> None:
    """Exhaustive matrix test for all conditional combinations.

    :param _Case case: Test case configuration.
    :return None: This test returns ``None``.
    """

    torch.manual_seed(42)

    B = 2
    H = 2
    Dh = 4
    chunk_size = 4
    Lq = chunk_size  # current chunk is always full size

    attn = _make_attn(chunk_size=chunk_size, sdpa=case.sdpa)

    # Configure cache
    if case.cache_present:
        cache_len = chunk_size if case.trim_active else 2
        cache_values = [10.0, 20.0, 30.0, 40.0][:cache_len]

        if case.cache_mask_present:
            # Place zeros in positions that will be dropped if trimming
            # This catches slice-direction bugs
            if case.trim_active:
                cache_mask = _make_mask(B, [0, 0, 1, 1])  # first two invalid
            else:
                cache_mask = _make_mask(B, [1, 0])  # second invalid
        else:
            cache_mask = None

        k_cache, _ = _zeros_qk(B, cache_len, H, Dh)
        v_cache = _make_v_loud(B, cache_len, H, cache_values, cache_mask)
        cache = AttentionCache(
            k=k_cache,
            v=v_cache,
            count=torch.full((B,), cache_len, dtype=torch.long),
            mask=cache_mask,
        )
    else:
        cache = None
        cache_len = 0
        cache_mask = None

    # Configure current chunk mask
    if case.mask_present:
        # Ensure at least one valid position to avoid all-masked rows
        mask_blk = _make_mask(B, [1, 0, 1, 0])
    else:
        mask_blk = None

    # Configure max_cache_len
    if case.trim_active and not case.cache_present:
        # This configuration should raise
        max_cache_len = chunk_size - 1
    else:
        # Allow some trimming when cache is present
        max_cache_len = chunk_size + 2  # keep 6

    # Create inputs
    q, k = _zeros_qk(B, Lq, H, Dh)
    new_values = [1.0, 2.0, 3.0, 4.0]
    v = _make_v_loud(B, Lq, H, new_values, mask_blk)

    # Handle should-raise cases
    if case.should_raise:
        with pytest.raises(ValueError, match=r"max_cache_len.*chunk_size"):
            attn(
                q,
                k,
                v,
                start_index=0,
                cache=cache,
                attn_mask=mask_blk,
                training=False,
                max_cache_len=max_cache_len,
                return_cache=True,
            )
        return

    # Run actual implementation
    out, new_cache = attn(
        q,
        k,
        v,
        start_index=0,
        cache=cache,
        attn_mask=mask_blk,
        training=False,
        max_cache_len=max_cache_len,
        return_cache=True,
    )

    # Run reference
    ref_out, ref_cache = _reference_attend(
        attn,
        q=q,
        k=k,
        v=v,
        start_index=0,
        cache=cache,
        mask_blk=mask_blk,
        max_cache_len=max_cache_len,
    )

    # Compare outputs
    torch.testing.assert_close(out, ref_out, atol=1e-5, rtol=1e-5)

    # Compare cache metadata
    assert torch.equal(new_cache.count, ref_cache.count)
    assert new_cache.length == ref_cache.length

    # Compare cache mask
    if ref_cache.mask is None:
        assert new_cache.mask is None, "Expected no cache mask, got one"
    else:
        assert new_cache.mask is not None, "Expected cache mask, got None"
        torch.testing.assert_close(
            new_cache.mask.to(torch.long),
            ref_cache.mask.to(torch.long),
        )


# =============================================================================
# Named Regression Tests: One Per Bug Class
# =============================================================================


@torch.no_grad()
def test_mask_slice_direction_under_trimming() -> None:
    """Issue 1: mask must use suffix slice [-keep:], not prefix [:keep].

    When trimming drops old keys, the kept mask must align with the kept K/V.
    The classic bug: K/V are sliced as [-keep:] but mask is sliced as [:keep].

    This test places invalid keys in positions that will be dropped, then
    verifies those invalid keys do not affect the output. If the slice direction
    is wrong, the dropped invalid keys will actually be kept, and their
    loud values will corrupt the output.

    :return None: This test returns ``None``.
    """
    attn = _make_attn(chunk_size=4, sdpa=True)
    B, H = 1, 2

    # Cache: 4 tokens, first 2 are invalid (will be dropped by trim)
    cache_mask = _make_mask(B, [0, 0, 1, 1])  # positions 0,1 invalid
    cache_values = [1000.0, 1000.0, 30.0, 40.0]  # loud values for invalid
    k_cache, _ = _zeros_qk(B, 4, H, 4)
    v_cache = _make_v_loud(B, 4, H, cache_values, mask=None)  # values already set
    cache = AttentionCache(
        k=k_cache, v=v_cache, count=torch.tensor([4], dtype=torch.long), mask=cache_mask
    )

    # Current chunk: all valid
    mask_blk = _make_mask(B, [1, 1, 1, 1])
    new_values = [1.0, 2.0, 3.0, 4.0]
    q, k = _zeros_qk(B, 4, H, 4)
    v = _make_v_loud(B, 4, H, new_values, mask=None)

    # max_cache_len=6 means we drop 2 tokens from 8 total
    # Correct: drop positions 0,1 (the invalid ones)
    # Bug: drop positions 6,7 (valid new tokens)
    out, _ = attn(
        q,
        k,
        v,
        start_index=0,
        cache=cache,
        attn_mask=mask_blk,
        training=False,
        max_cache_len=6,
        return_cache=True,
    )

    # First query can attend to: kept cache [30, 40] + new [1] = positions 0,1,2
    # With q=k=0, attention is uniform: mean([30, 40, 1]) = 71/3 ~= 23.67
    # If slice is wrong, we would see 1000s in the output
    expected = 71.0 / 3.0
    actual = out[0, 0, 0].item()

    assert abs(actual - expected) < 1.0, (
        f"Slice direction bug: expected ~{expected:.2f}, got {actual:.2f}. "
        "If actual >> expected, the mask slice used prefix instead of suffix."
    )


@torch.no_grad()
def test_cache_mask_applied_when_current_mask_none() -> None:
    """Issue 3B: cache mask must be respected even when current mask is None.

    The bug: prefix mask is only applied inside `if mask_blk is not None`,
    so passing mask_blk=None makes cached padding keys valid again.

    :return None: This test returns ``None``.
    """
    attn = _make_attn(chunk_size=4, sdpa=True)
    B, H = 1, 2

    # Cache: position 1 is invalid with a loud value
    cache_mask = _make_mask(B, [1, 0])  # position 1 invalid
    cache_values = [1.0, 1000.0]  # loud value for invalid
    k_cache, _ = _zeros_qk(B, 2, H, 4)
    v_cache = _make_v_loud(B, 2, H, cache_values, mask=None)
    cache = AttentionCache(
        k=k_cache, v=v_cache, count=torch.tensor([2], dtype=torch.long), mask=cache_mask
    )

    # Current chunk: no mask (this is the trigger)
    new_values = [10.0, 20.0, 30.0, 40.0]
    q, k = _zeros_qk(B, 4, H, 4)
    v = _make_v_loud(B, 4, H, new_values, mask=None)

    out, _ = attn(
        q,
        k,
        v,
        start_index=0,
        cache=cache,
        attn_mask=None,  # no current mask
        training=False,
        max_cache_len=8,
        return_cache=True,
    )

    # First query can attend to: cache[0]=1 (valid), cache[1]=1000 (invalid), new[0]=10
    # Correct: mean([1, 10]) = 5.5
    # Bug: mean([1, 1000, 10]) = 337
    expected = 5.5
    actual = out[0, 0, 0].item()

    assert abs(actual - expected) < 1.0, (
        f"Cache mask ignored: expected ~{expected:.2f}, got {actual:.2f}. "
        "The cached padding mask was not applied because current mask was None."
    )


@torch.no_grad()
def test_cache_mask_extended_when_current_mask_none() -> None:
    """Issue 3B: returned cache.mask must include new tokens even if current mask is None.

    When cache has a mask but current mask is None, the returned cache.mask
    must be extended with ones for the new tokens (they are all valid).

    :return None: This test returns ``None``.
    """
    attn = _make_attn(chunk_size=4, sdpa=True)
    B, H = 1, 2

    cache_mask = _make_mask(B, [1, 0])
    k_cache, _ = _zeros_qk(B, 2, H, 4)
    v_cache = _make_v_loud(B, 2, H, [1.0, 2.0], mask=None)
    cache = AttentionCache(
        k=k_cache, v=v_cache, count=torch.tensor([2], dtype=torch.long), mask=cache_mask
    )

    q, k = _zeros_qk(B, 4, H, 4)
    v = _make_v_loud(B, 4, H, [10.0, 20.0, 30.0, 40.0], mask=None)

    _, new_cache = attn(
        q,
        k,
        v,
        start_index=0,
        cache=cache,
        attn_mask=None,
        training=False,
        max_cache_len=8,
        return_cache=True,
    )

    assert new_cache.mask is not None, "Cache mask should exist (inherited from prefix)"
    assert new_cache.mask.shape == (B, 6), (
        f"Expected shape (1, 6), got {new_cache.mask.shape}"
    )

    # Expected: [1, 0] from cache + [1, 1, 1, 1] for new tokens
    expected = torch.tensor([[True, False, True, True, True, True]])
    torch.testing.assert_close(new_cache.mask, expected)


@torch.no_grad()
def test_prefix_ones_when_cache_mask_none_current_present() -> None:
    """Issue 3A: when cache.mask is None but current mask exists, prefix defaults to all-ones.

    The returned cache.mask must include the prefix as all-valid (ones),
    not be truncated to just the current mask.

    :return None: This test returns ``None``.
    """
    attn = _make_attn(chunk_size=4, sdpa=True)
    B, H = 1, 2

    # Cache without mask
    k_cache, _ = _zeros_qk(B, 2, H, 4)
    v_cache = _make_v_loud(B, 2, H, [1.0, 2.0], mask=None)
    cache = AttentionCache(
        k=k_cache, v=v_cache, count=torch.tensor([2], dtype=torch.long), mask=None
    )

    # Current chunk with mask
    mask_blk = _make_mask(B, [1, 0, 1, 0])
    q, k = _zeros_qk(B, 4, H, 4)
    v = _make_v_loud(B, 4, H, [10.0, 20.0, 30.0, 40.0], mask=None)

    _, new_cache = attn(
        q,
        k,
        v,
        start_index=0,
        cache=cache,
        attn_mask=mask_blk,
        training=False,
        max_cache_len=8,
        return_cache=True,
    )

    assert new_cache.mask is not None, (
        "Cache mask should exist (current mask was provided)"
    )
    assert new_cache.mask.shape == (B, 6), (
        f"Expected shape (1, 6), got {new_cache.mask.shape}"
    )

    # Expected: [1, 1] for prefix (defaulted) + [1, 0, 1, 0] from current
    expected = torch.tensor([[True, True, True, False, True, False]])
    torch.testing.assert_close(new_cache.mask, expected)


@torch.no_grad()
def test_max_cache_len_lt_chunk_size_rejected_runtime() -> None:
    """Issue 2: max_cache_len < chunk_size must raise at runtime.

    Allowing this silently breaks causal invariants: queries in the current
    chunk may not be able to attend to earlier tokens in the same chunk.

    :return None: This test returns ``None``.
    """
    attn = _make_attn(chunk_size=8, sdpa=True)
    q, k = _zeros_qk(1, 4, 2, 4)
    v = _make_v_loud(1, 4, 2, [1.0, 2.0, 3.0, 4.0], mask=None)

    with pytest.raises(ValueError, match=r"max_cache_len.*chunk_size"):
        attn(
            q,
            k,
            v,
            start_index=0,
            cache=None,
            attn_mask=None,
            training=False,
            max_cache_len=4,  # < chunk_size=8
            return_cache=True,
        )

    cache_len = 8
    cache_mask = torch.ones(1, cache_len, dtype=torch.bool)
    cache = AttentionCache(
        k=torch.zeros(1, cache_len, 2, 4),
        v=torch.zeros(1, cache_len, 2, 1),
        count=cache_mask.to(torch.long).sum(dim=1),
        mask=cache_mask,
    )
    with pytest.raises(ValueError, match=r"max_cache_len.*chunk_size"):
        attn(
            q,
            k,
            v,
            start_index=0,
            cache=cache,
            attn_mask=None,
            training=False,
            max_cache_len=4,  # < chunk_size=8
            return_cache=True,
        )


def test_max_cache_len_lt_chunk_size_rejected_config() -> None:
    """Issue 2: MegalodonConfig should reject max_cache_len < chunk_size.

    Catching this at config time is better than at runtime.

    :return None: This test returns ``None``.
    """
    with pytest.raises(ValueError):
        MegalodonConfig(chunk_size=8, max_cache_len=4)


# =============================================================================
# Edge Case Tests
# =============================================================================


@torch.no_grad()
def test_no_mask_no_cache_baseline() -> None:
    """Baseline: no mask, no cache should work and produce causal attention.

    :return None: This test returns ``None``.
    """
    attn = _make_attn(chunk_size=4, sdpa=True)
    B, H = 1, 2

    q, k = _zeros_qk(B, 4, H, 4)
    v = _make_v_loud(B, 4, H, [1.0, 2.0, 3.0, 4.0], mask=None)

    out, new_cache = attn(
        q,
        k,
        v,
        start_index=0,
        cache=None,
        attn_mask=None,
        training=False,
        max_cache_len=8,
        return_cache=True,
    )

    # With q=k=0, each query attends uniformly to all causal keys
    # Query 0: mean([1]) = 1
    # Query 1: mean([1, 2]) = 1.5
    # Query 2: mean([1, 2, 3]) = 2
    # Query 3: mean([1, 2, 3, 4]) = 2.5
    expected = torch.tensor([[[1.0], [1.5], [2.0], [2.5]]])
    expected = expected.expand(B, 4, H)

    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)
    assert new_cache.mask is None, "No mask info -> cache.mask should be None"


@torch.no_grad()
def test_sdpa_and_manual_paths_match() -> None:
    """SDPA and manual attention paths should produce identical results.

    :return None: This test returns ``None``.
    """
    B, H = 2, 2

    # Create a non-trivial scenario: cache with mask + current with mask
    cache_mask = _make_mask(B, [1, 0])
    mask_blk = _make_mask(B, [1, 1, 0, 1])

    k_cache, _ = _zeros_qk(B, 2, H, 4)
    v_cache = _make_v_loud(B, 2, H, [5.0, 500.0], mask=None)  # loud invalid
    cache = AttentionCache(
        k=k_cache,
        v=v_cache,
        count=torch.full((B,), 2, dtype=torch.long),
        mask=cache_mask,
    )

    q, k = _zeros_qk(B, 4, H, 4)
    v = _make_v_loud(B, 4, H, [1.0, 2.0, 300.0, 4.0], mask=None)  # loud invalid

    attn_sdpa = _make_attn(chunk_size=4, sdpa=True)
    attn_manual = _make_attn(chunk_size=4, sdpa=False)

    out_sdpa, cache_sdpa = attn_sdpa(
        q,
        k,
        v,
        start_index=0,
        cache=cache,
        attn_mask=mask_blk,
        training=False,
        max_cache_len=8,
        return_cache=True,
    )

    out_manual, cache_manual = attn_manual(
        q,
        k,
        v,
        start_index=0,
        cache=cache,
        attn_mask=mask_blk,
        training=False,
        max_cache_len=8,
        return_cache=True,
    )

    torch.testing.assert_close(out_sdpa, out_manual, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(cache_sdpa.mask, cache_manual.mask)


@torch.no_grad()
def test_cache_kv_mask_length_invariant() -> None:
    """Invariant: cache.mask.shape[1] must always equal cache.k.shape[1].

    :return None: This test returns ``None``.
    """
    attn = _make_attn(chunk_size=4, sdpa=True)
    B, H = 1, 2

    # Run through several forward passes, checking invariant after each
    cache = None
    for step in range(5):
        q, k = _zeros_qk(B, 4, H, 4)
        v = _make_v_loud(B, 4, H, [1.0, 2.0, 3.0, 4.0], mask=None)

        # Alternate: sometimes provide mask, sometimes do not
        mask_blk = _make_mask(B, [1, 1, 0, 1]) if step % 2 == 0 else None

        _, cache = attn(
            q,
            k,
            v,
            start_index=0 if cache is None else 0,
            cache=cache,
            attn_mask=mask_blk,
            training=False,
            max_cache_len=8,
            return_cache=True,
        )

        if cache.mask is not None:
            assert cache.mask.shape[1] == cache.k.shape[1], (
                f"Step {step}: mask length {cache.mask.shape[1]} != "
                f"KV length {cache.k.shape[1]}"
            )


@pytest.mark.parametrize("explicit_position_ids", [False, True])
@torch.no_grad()
def test_multichunk_padding_extends_position_ids(
    explicit_position_ids: bool,
) -> None:
    """Padding to chunk_size must keep position_ids aligned with padded length.

    :param bool explicit_position_ids: Whether to pass explicit position ids.
    :return None: This test returns ``None``.
    """
    attn = _make_attn(chunk_size=8, sdpa=True)
    B, H = 2, 2
    L = 10  # not divisible by chunk_size (8) -> triggers padding

    q, k = _zeros_qk(B, L, H, 4)
    v = _make_v_loud(B, L, H, list(range(1, L + 1)), mask=None)
    attn_mask = torch.ones(B, L, dtype=torch.bool)
    attn_mask[0, -1] = False

    position_ids = None
    if explicit_position_ids:
        position_ids = torch.arange(L, device=q.device).unsqueeze(0).expand(B, -1)

    out, _ = attn(
        q,
        k,
        v,
        start_index=0,
        cache=None,
        attn_mask=attn_mask,
        training=False,
        position_ids=position_ids,
    )

    assert out.shape == (B, L, H)


@torch.no_grad()
def test_reference_attend_uses_position_ids_for_causality() -> None:
    """Reference should match position-based causal masking when provided.

    :return None: This test returns ``None``.
    """
    attn = _make_attn(chunk_size=4, sdpa=True)
    B, H = 1, 2
    L = 4

    q, k = _zeros_qk(B, L, H, 4)
    v = _make_v_loud(B, L, H, [1.0, 2.0, 3.0, 4.0], mask=None)
    mask_blk = torch.ones(B, L, dtype=torch.bool)
    position_ids = torch.tensor([[0, 2, 2, 5]], dtype=torch.long)

    out, new_cache = attn(
        q,
        k,
        v,
        start_index=0,
        cache=None,
        attn_mask=mask_blk,
        training=False,
        max_cache_len=8,
        return_cache=True,
        position_ids=position_ids,
    )

    ref_out, ref_cache = _reference_attend(
        attn,
        q=q,
        k=k,
        v=v,
        start_index=0,
        cache=None,
        mask_blk=mask_blk,
        position_ids=position_ids,
        max_cache_len=8,
    )

    torch.testing.assert_close(out, ref_out, atol=1e-5, rtol=1e-5)
    assert new_cache.count == ref_cache.count


def _empty_sequence_cases() -> list[tuple[bool, bool, bool, bool, bool]]:
    cases = []
    for cache_present in [False, True]:
        for cache_mask_present in [False, True]:
            for return_cache in [False, True]:
                for mask_present in [False, True]:
                    for position_ids_present in [False, True]:
                        if cache_mask_present and not cache_present:
                            continue
                        cases.append(
                            (
                                cache_present,
                                cache_mask_present,
                                return_cache,
                                mask_present,
                                position_ids_present,
                            )
                        )
    return cases


@pytest.mark.parametrize(
    "case",
    _empty_sequence_cases(),
    ids=(
        lambda c: (
            f"cache={int(c[0])}_"
            f"cachemask={int(c[1])}_"
            f"return={int(c[2])}_"
            f"mask={int(c[3])}_"
            f"pos={int(c[4])}"
        )
    ),
)
@torch.no_grad()
def test_empty_sequence_returns_empty(
    case: tuple[bool, bool, bool, bool, bool],
) -> None:
    """Empty sequences should return empty outputs without errors.

    :param tuple[bool, bool, bool, bool, bool] case: Case tuple controlling cache/mask behavior.
    :return None: This test returns ``None``.
    """
    (
        cache_present,
        cache_mask_present,
        return_cache,
        mask_present,
        position_ids_present,
    ) = case

    attn = _make_attn(chunk_size=8, sdpa=True)
    B, H = 2, 2
    L = 0

    q, k = _zeros_qk(B, L, H, 4)
    v = torch.zeros(B, L, H, 1)
    attn_mask = torch.ones(B, L, dtype=torch.bool) if mask_present else None
    position_ids = torch.zeros(B, L, dtype=torch.long) if position_ids_present else None

    cache = None
    if cache_present:
        cache_len = 3
        k_cache, _ = _zeros_qk(B, cache_len, H, 4)
        v_cache = torch.zeros(B, cache_len, H, 1)
        cache_mask = None
        if cache_mask_present:
            cache_mask = torch.tensor(
                [[True, True, False], [True, False, False]], dtype=torch.bool
            )
        cache = AttentionCache(
            k=k_cache,
            v=v_cache,
            count=torch.tensor([3, 1], dtype=torch.long),
            mask=cache_mask,
        )

    out, new_cache = attn(
        q,
        k,
        v,
        start_index=5,
        cache=cache,
        attn_mask=attn_mask,
        training=False,
        max_cache_len=8,
        return_cache=return_cache,
        position_ids=position_ids,
    )

    assert out.shape == (B, 0, H)
    if return_cache and cache_present:
        assert new_cache is not None
        assert new_cache.length == cache.length
        torch.testing.assert_close(new_cache.count, cache.count)
        if cache.mask is None:
            assert new_cache.mask is None
        else:
            torch.testing.assert_close(
                new_cache.mask.to(torch.long), cache.mask.to(torch.long)
            )
    else:
        assert new_cache is None


@pytest.mark.parametrize("cache_present", [False, True])
@torch.no_grad()
def test_empty_sequence_returns_position(cache_present: bool) -> None:
    """Empty sequences should not advance cached positions.

    :param bool cache_present: Whether to provide a cache.
    :return None: This test returns ``None``.
    """
    attn = _make_attn(chunk_size=8, sdpa=True)
    B, H = 2, 2
    L = 0

    q, k = _zeros_qk(B, L, H, 4)
    v = torch.zeros(B, L, H, 1)

    cache = None
    expected_pos = torch.full((B,), 7, dtype=torch.long)
    if cache_present:
        cache_len = 3
        k_cache, _ = _zeros_qk(B, cache_len, H, 4)
        v_cache = torch.zeros(B, cache_len, H, 1)
        cache = AttentionCache(
            k=k_cache,
            v=v_cache,
            count=torch.tensor([3, 1], dtype=torch.long),
            mask=None,
        )
        expected_pos = cache.count

    out, new_cache, pos = attn(
        q,
        k,
        v,
        start_index=7,
        cache=cache,
        attn_mask=None,
        training=False,
        max_cache_len=8,
        return_cache=True,
        return_position=True,
    )

    assert out.shape == (B, 0, H)
    assert (new_cache is None) == (not cache_present)
    torch.testing.assert_close(pos, expected_pos)


def _fully_masked_cases() -> list[tuple[bool, bool, bool, bool, int]]:
    cases = []
    for cache_present in [False, True]:
        for cache_mask_present in [False, True]:
            if cache_mask_present and not cache_present:
                continue
            for return_cache in [False, True]:
                for sdpa in [False, True]:
                    for length in [4, 10]:
                        cases.append(
                            (
                                cache_present,
                                cache_mask_present,
                                return_cache,
                                sdpa,
                                length,
                            )
                        )
    return cases


@pytest.mark.parametrize(
    "case",
    _fully_masked_cases(),
    ids=lambda c: (
        f"cache={int(c[0])}_"
        f"cachemask={int(c[1])}_"
        f"return={int(c[2])}_"
        f"sdpa={int(c[3])}_"
        f"L={c[4]}"
    ),
)
@torch.no_grad()
def test_fully_masked_rows_are_finite(case: tuple[bool, bool, bool, bool, int]) -> None:
    """Fully-masked rows should be finite and zeroed (no NaNs).

    :param tuple[bool, bool, bool, bool, int] case: Case tuple controlling cache/mask/length behavior.
    :return None: This test returns ``None``.
    """
    cache_present, cache_mask_present, return_cache, sdpa, length = case
    attn = _make_attn(chunk_size=8, sdpa=sdpa)
    B, H = 2, 2

    q, k = _zeros_qk(B, length, H, 4)
    v = _make_v_loud(B, length, H, list(range(1, length + 1)), mask=None)
    attn_mask = torch.ones(B, length, dtype=torch.bool)
    attn_mask[0].fill_(False)

    cache = None
    if cache_present:
        cache_len = 3
        k_cache, _ = _zeros_qk(B, cache_len, H, 4)
        v_cache = torch.zeros(B, cache_len, H, 1)
        cache_mask = None
        if cache_mask_present:
            cache_mask = torch.tensor(
                [[True, True, False], [True, False, False]], dtype=torch.bool
            )
        cache = AttentionCache(
            k=k_cache,
            v=v_cache,
            count=torch.tensor([3, 1], dtype=torch.long),
            mask=cache_mask,
        )

    out, _cache = attn(
        q,
        k,
        v,
        start_index=0,
        cache=cache,
        attn_mask=attn_mask,
        training=False,
        max_cache_len=8,
        return_cache=return_cache,
    )

    assert torch.isfinite(out).all()
    assert torch.allclose(out[0], torch.zeros_like(out[0]), atol=0.0, rtol=0.0)


@torch.no_grad()
def test_chunk_local_mixed_positions_match_per_sample() -> None:
    """Mixed cached positions should not cross chunk boundaries.

    :return None: This test returns ``None``.
    """
    attn = _make_attn(chunk_size=16, sdpa=True)
    B, H = 2, 2
    L = 10

    q, k = _zeros_qk(B, L, H, 4)
    v = _make_v_loud(B, L, H, list(range(1, L + 1)), mask=None)

    cache_len = 15
    cache_mask = torch.zeros(B, cache_len, dtype=torch.bool)
    cache_mask[0, :] = True
    cache_mask[1, -1] = True
    k_cache, _ = _zeros_qk(B, cache_len, H, 4)
    v_cache = torch.zeros(B, cache_len, H, 1)
    v_cache[0].fill_(1000.0)
    v_cache[1].fill_(1.0)
    cache = AttentionCache(
        k=k_cache,
        v=v_cache,
        count=torch.tensor([15, 1], dtype=torch.long),
        mask=cache_mask,
    )

    out_batched, _ = attn(
        q,
        k,
        v,
        start_index=0,
        cache=cache,
        attn_mask=None,
        training=False,
        return_cache=False,
    )

    cache0 = AttentionCache(
        k=k_cache[:1],
        v=v_cache[:1],
        count=torch.tensor([15], dtype=torch.long),
        mask=cache_mask[:1],
    )
    cache1 = AttentionCache(
        k=k_cache[1:2],
        v=v_cache[1:2],
        count=torch.tensor([1], dtype=torch.long),
        mask=cache_mask[1:2],
    )
    out0, _ = attn(
        q[:1],
        k[:1],
        v[:1],
        start_index=0,
        cache=cache0,
        attn_mask=None,
        training=False,
        return_cache=False,
    )
    out1, _ = attn(
        q[1:2],
        k[1:2],
        v[1:2],
        start_index=0,
        cache=cache1,
        attn_mask=None,
        training=False,
        return_cache=False,
    )
    out_expected = torch.cat([out0, out1], dim=0)

    torch.testing.assert_close(out_batched, out_expected, atol=1e-5, rtol=1e-5)
