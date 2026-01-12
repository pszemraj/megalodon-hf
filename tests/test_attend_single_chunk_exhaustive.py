# coding=utf-8
"""Exhaustive path tests for ``ChunkedSelfAttention.attend_single_chunk``.

The model's masking/caching bugs have historically shown up in the nested
``attend_single_chunk`` helper inside ``ChunkedSelfAttention.forward``.

This file intentionally enumerates the key conditionals that affect masking:
  - cache present vs absent
  - current chunk mask present vs absent
  - cache mask present vs absent
  - trimming (sliding-window drop) active vs inactive

Instead of "representative" tests, we run one test per combination.
"""

from __future__ import annotations

import itertools

import pytest
import torch
import torch.nn.functional as F

from megalodon.modeling_megalodon import AttentionCache, ChunkedSelfAttention


def _make_values(
    *,
    batch: int,
    length: int,
    heads: int,
    value_dim: int,
    abs_start_pos: int | torch.Tensor,
    key_valid_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Build V so that masked keys have very large values.

    We intentionally make masked keys "loud" (+1000) so that any masking
    regression produces a large numerical difference.
    """
    assert value_dim == 1, "Test helper currently assumes value_dim == 1"
    if torch.is_tensor(abs_start_pos):
        device = abs_start_pos.device
    else:
        device = key_valid_mask.device if key_valid_mask is not None else None
    pos = torch.arange(length, device=device).view(1, length, 1, 1)
    if torch.is_tensor(abs_start_pos):
        base = abs_start_pos.to(device=device).view(batch, 1, 1, 1)
        pos = pos + base
    else:
        pos = pos + abs_start_pos
    b_off = torch.arange(batch, device=device).view(batch, 1, 1, 1) * 100
    h_off = torch.arange(heads, device=device).view(1, 1, heads, 1) * 10
    v = (pos + b_off + h_off).to(torch.float32)
    if key_valid_mask is not None:
        inv = (~key_valid_mask).to(torch.float32).view(batch, length, 1, 1)
        v = v + inv * 1000.0
    return v


def _clamp_cache(
    cache: AttentionCache | None, limit: int | None
) -> AttentionCache | None:
    if cache is None or limit is None:
        return cache
    if cache.length <= limit:
        return cache
    return AttentionCache(
        k=cache.k[:, -limit:],
        v=cache.v[:, -limit:],
        count=cache.count,
        mask=cache.mask[:, -limit:] if cache.mask is not None else None,
    )


def _reference_single(
    attn: ChunkedSelfAttention,
    *,
    q_blk: torch.Tensor,
    k_blk: torch.Tensor,
    v_blk: torch.Tensor,
    start_index: int,
    cache: AttentionCache | None,
    mask_blk: torch.Tensor | None,
    max_cache_len: int | None,
) -> tuple[torch.Tensor, AttentionCache]:
    """Reference implementation for a single forward call (L <= chunk_size).

    This matches the *intended* semantics:
      - key masking always respects cache.mask when present
      - if either cache.mask or mask_blk is present, we construct a full key mask
        for (prefix + current) where missing segments default to all-True.
      - sliding-window trimming drops from the left and masks are trimmed the same way.
    """
    if max_cache_len is None or max_cache_len == -1:
        cache_limit = attn.chunk_size
    else:
        cache_limit = max_cache_len

    if cache_limit is not None and cache_limit < attn.chunk_size:
        raise ValueError(
            f"max_cache_len ({cache_limit}) must be >= chunk_size ({attn.chunk_size})"
        )

    cache = _clamp_cache(cache, cache_limit)

    B, Lq, H, _Dh = q_blk.shape
    device = q_blk.device

    if cache is not None:
        start_pos = cache.count
    else:
        start_pos = torch.full((B,), start_index, dtype=torch.long, device=device)

    faithful_chunk_local = cache_limit == attn.chunk_size
    if cache is not None and faithful_chunk_local:
        pos_in_chunk = start_pos % attn.chunk_size
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
                idx = torch.arange(cache.length, device=device).unsqueeze(0)
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

    if mask_blk is not None:
        mask_long = mask_blk.to(torch.long)
        pos_offsets = mask_long.cumsum(dim=1) - 1
        pos_offsets = pos_offsets.clamp(min=0)
        position_ids = start_pos.unsqueeze(1) + pos_offsets
    else:
        offsets = torch.arange(Lq, device=device, dtype=torch.long)
        position_ids = start_pos.unsqueeze(1) + offsets.unsqueeze(0)

    q_rot, k_rot = attn.rope(q_blk, k_blk, start_index=0, position_ids=position_ids)

    if cache is not None:
        k_cat = torch.cat([cache.k, k_rot], dim=1)
        v_cat = torch.cat([cache.v, v_blk], dim=1)
    else:
        k_cat = k_rot
        v_cat = v_blk

    total_len = k_cat.size(1)
    keep = total_len if cache_limit is None else min(cache_limit, total_len)
    if total_len > keep:
        k_cat = k_cat[:, -keep:]
        v_cat = v_cat[:, -keep:]
    Lk = k_cat.size(1)

    mask_tokens = None
    has_prefix_mask = cache is not None and cache.mask is not None
    has_current_mask = mask_blk is not None
    if has_prefix_mask or has_current_mask:
        if cache is not None and cache.length > 0:
            if has_prefix_mask:
                prefix_mask = cache.mask
            else:
                prefix_mask = torch.ones(
                    B, cache.length, dtype=torch.bool, device=device
                )
        else:
            prefix_mask = None
        if has_current_mask:
            current_mask = mask_blk.to(torch.bool)
        else:
            current_mask = torch.ones(B, Lq, dtype=torch.bool, device=device)
        if prefix_mask is not None:
            mask_tokens = torch.cat([prefix_mask, current_mask], dim=1)
        else:
            mask_tokens = current_mask
        if mask_tokens.size(1) > Lk:
            mask_tokens = mask_tokens[:, -Lk:]
        elif mask_tokens.size(1) < Lk:
            pad_len = Lk - mask_tokens.size(1)
            mask_tokens = F.pad(mask_tokens, (pad_len, 0), value=1)

    if cache is not None:
        if cache.mask is None:
            cache_offset = cache.count - cache.length
            cache_positions = cache_offset.unsqueeze(1) + torch.arange(
                cache.length, device=device
            ).unsqueeze(0)
        else:
            cache_mask_long = cache.mask.to(torch.long)
            cache_valid = cache_mask_long.sum(dim=1)
            cache_offset = cache.count - cache_valid
            cache_positions = cache_mask_long.cumsum(dim=1) - 1
            cache_positions = cache_positions.clamp(min=0)
            cache_positions = cache_offset.unsqueeze(1) + cache_positions
        key_positions = torch.cat([cache_positions, position_ids.to(torch.long)], dim=1)
    else:
        key_positions = position_ids.to(torch.long)
    if key_positions.size(1) > Lk:
        key_positions = key_positions[:, -Lk:]

    base_mask = None
    prefix_len_kept = max(0, Lk - Lq)
    if prefix_len_kept > 0 or mask_tokens is not None:
        query_positions = position_ids.to(torch.long)
        causal = key_positions.unsqueeze(1) <= query_positions.unsqueeze(2)
        base_mask = torch.where(
            causal,
            torch.zeros_like(causal, dtype=q_blk.dtype),
            torch.tensor(float("-inf"), dtype=q_blk.dtype, device=device),
        )
        base_mask = base_mask.unsqueeze(1)
        if mask_tokens is not None:
            base_mask = base_mask.masked_fill(
                (mask_tokens == 0).view(B, 1, 1, Lk), float("-inf")
            )

    q_ = q_rot.transpose(1, 2)
    k_ = k_cat.transpose(1, 2)
    v_ = v_cat.transpose(1, 2)

    scores = torch.matmul(q_, k_.transpose(-2, -1)).float()
    if base_mask is not None:
        scores = scores + base_mask.to(scores.dtype)
    else:
        scores = scores + attn._causal_mask(
            Lq, Lk, device, torch.float32, offset=prefix_len_kept
        )

    weights = torch.softmax(scores, dim=-1).to(q_)
    out = torch.matmul(weights, v_).transpose(1, 2).reshape(B, Lq, H * v_blk.size(-1))
    if mask_blk is not None:
        out = torch.where(mask_blk.unsqueeze(-1), out, out.new_zeros(()))

    if mask_blk is not None:
        valid_counts = mask_blk.to(torch.long).sum(dim=1)
    else:
        valid_counts = torch.full((B,), Lq, dtype=torch.long, device=device)
    base_count = cache.count if cache is not None else start_pos
    new_count = base_count + valid_counts
    new_cache = AttentionCache(
        k=k_cat[:, -keep:], v=v_cat[:, -keep:], count=new_count, mask=mask_tokens
    )
    return out, new_cache


def _case_ids() -> list[str]:
    ids: list[str] = []
    for has_cache, has_mask, has_cache_mask, do_trim in itertools.product(
        [False, True], repeat=4
    ):
        ids.append(
            f"cache={int(has_cache)}_mask={int(has_mask)}_"
            f"cachemask={int(has_cache_mask)}_trim={int(do_trim)}"
        )
    return ids


@pytest.mark.parametrize(
    "has_cache,has_mask,has_cache_mask,do_trim",
    list(itertools.product([False, True], repeat=4)),
    ids=_case_ids(),
)
@torch.no_grad()
def test_attend_single_chunk_exhaustive_paths(
    has_cache: bool, has_mask: bool, has_cache_mask: bool, do_trim: bool
) -> None:
    torch.manual_seed(0)

    chunk_size = 8
    B, H, Dh, Dv = 2, 2, 2, 1
    L_new = 3

    attn = ChunkedSelfAttention(
        num_heads=H,
        head_dim=Dh,
        value_head_dim=Dv,
        chunk_size=chunk_size,
        rope_base=10_000.0,
        attention_dropout=0.0,
    ).eval()
    attn._sdpa_available = False

    q_new = torch.zeros(B, L_new, H, Dh)
    k_new = torch.zeros(B, L_new, H, Dh)

    if has_cache:
        cache_len = chunk_size if do_trim else chunk_size // 2
        k_cache = torch.zeros(B, cache_len, H, Dh)
        cache_mask = None
        if has_cache_mask:
            pat0 = [1 if i % 2 == 0 else 0 for i in range(cache_len)]
            pat1 = [0 if i % 2 == 0 else 1 for i in range(cache_len)]
            cache_mask = torch.stack(
                [
                    torch.tensor(pat0, dtype=torch.bool),
                    torch.tensor(pat1, dtype=torch.bool),
                ],
                dim=0,
            )
            assert cache_mask.shape == (B, cache_len)
        if cache_mask is None:
            cache_count = torch.full((B,), cache_len, dtype=torch.long)
        else:
            cache_count = cache_mask.to(torch.long).sum(dim=1)
        cache = AttentionCache(
            k=k_cache,
            v=torch.empty(B, cache_len, H, Dv),
            count=cache_count,
            mask=cache_mask,
        )
        start_index = 0
        start_pos = cache.count
    else:
        cache_len = 0
        cache_mask = None
        cache = None
        start_index = 0
        start_pos = torch.full((B,), start_index, dtype=torch.long)

    mask_blk = None
    if has_mask:
        base0 = [1, 0, 1] if not has_cache else [1, 1, 0]
        base1 = [1, 1, 0] if not has_cache else [1, 0, 1]
        m0 = torch.tensor(base0, dtype=torch.bool).view(1, -1)
        m1 = torch.tensor(base1, dtype=torch.bool).view(1, -1)
        mask_blk = torch.cat([m0, m1], dim=0)
        assert mask_blk.shape == (B, L_new)

    if has_cache:
        prefix_eff = cache_mask
        if prefix_eff is None and has_mask:
            prefix_eff = torch.ones(B, cache_len, dtype=torch.bool)
        cur_eff = mask_blk
        if cur_eff is None and prefix_eff is not None:
            cur_eff = torch.ones(B, L_new, dtype=torch.bool)
        if prefix_eff is None and cur_eff is None:
            full_key_mask = None
        else:
            prefix_part = (
                prefix_eff
                if prefix_eff is not None
                else torch.empty(B, 0, dtype=torch.bool)
            )
            cur_part = (
                cur_eff if cur_eff is not None else torch.empty(B, 0, dtype=torch.bool)
            )
            full_key_mask = torch.cat([prefix_part, cur_part], dim=1)
    else:
        full_key_mask = mask_blk

    if has_cache:
        v_cache = _make_values(
            batch=B,
            length=cache_len,
            heads=H,
            value_dim=Dv,
            abs_start_pos=0,
            key_valid_mask=(
                full_key_mask[:, :cache_len] if full_key_mask is not None else None
            ),
        )
        cache = AttentionCache(k=cache.k, v=v_cache, count=cache.count, mask=cache.mask)
    v_new = _make_values(
        batch=B,
        length=L_new,
        heads=H,
        value_dim=Dv,
        abs_start_pos=start_pos,
        key_valid_mask=(
            full_key_mask[:, cache_len:] if full_key_mask is not None else None
        ),
    )

    if do_trim and not has_cache:
        max_cache_len = chunk_size - 1
        with pytest.raises(ValueError, match=r"max_cache_len.*chunk_size"):
            attn(
                q_new,
                k_new,
                v_new,
                start_index=start_index,
                cache=None,
                attn_mask=mask_blk,
                training=False,
                max_cache_len=max_cache_len,
                return_cache=True,
            )
        return

    max_cache_len = chunk_size

    out, out_cache = attn(
        q_new,
        k_new,
        v_new,
        start_index=start_index,
        cache=cache,
        attn_mask=mask_blk,
        training=False,
        max_cache_len=max_cache_len,
        return_cache=True,
    )
    assert out_cache is not None

    ref_out, ref_cache = _reference_single(
        attn,
        q_blk=q_new,
        k_blk=k_new,
        v_blk=v_new,
        start_index=start_index,
        cache=cache,
        mask_blk=mask_blk,
        max_cache_len=max_cache_len,
    )

    assert torch.allclose(out, ref_out, atol=1e-6, rtol=0.0)
    assert torch.equal(out_cache.count, ref_cache.count)
    assert out_cache.length == ref_cache.length
    if ref_cache.mask is None:
        assert out_cache.mask is None
    else:
        assert out_cache.mask is not None
        assert torch.equal(out_cache.mask, ref_cache.mask)


@torch.no_grad()
def test_max_cache_len_smaller_than_chunk_size_is_rejected() -> None:
    """Regression test: max_cache_len < chunk_size must raise.

    When the cache horizon is smaller than a single chunk, the model cannot
    guarantee intra-chunk causality/masking semantics (the current token may
    require attending to earlier tokens within the same chunk that would be
    forcibly dropped).
    """
    torch.manual_seed(0)
    chunk_size = 8
    B, H, Dh, Dv = 1, 1, 2, 1
    L = 3
    attn = ChunkedSelfAttention(
        num_heads=H,
        head_dim=Dh,
        value_head_dim=Dv,
        chunk_size=chunk_size,
        rope_base=10_000.0,
        attention_dropout=0.0,
    ).eval()
    attn._sdpa_available = False

    q = torch.zeros(B, L, H, Dh)
    k = torch.zeros(B, L, H, Dh)
    v = torch.zeros(B, L, H, Dv)

    with pytest.raises(ValueError, match=r"max_cache_len.*chunk_size"):
        attn(
            q,
            k,
            v,
            start_index=0,
            cache=None,
            attn_mask=None,
            training=False,
            max_cache_len=chunk_size - 1,
            return_cache=True,
        )

    cache_len = chunk_size
    mask = torch.ones(B, cache_len, dtype=torch.bool)
    cache = AttentionCache(
        k=torch.zeros(B, cache_len, H, Dh),
        v=torch.zeros(B, cache_len, H, Dv),
        count=mask.to(torch.long).sum(dim=1),
        mask=mask,
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
            max_cache_len=chunk_size - 1,
            return_cache=True,
        )
