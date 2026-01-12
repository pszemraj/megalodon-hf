# coding=utf-8
"""Exhaustive edge-case tests for ComplexEMA masking + streaming.

Why this file exists
--------------------
ComplexEMA contains a few subtle masking/caching interactions that are easy to
regress without *explicit* tests:

- When a padding mask has trailing zeros, the cached state returned by
  ``compute_last_state=True`` should correspond to the **last valid token**, not
  the state after decaying through padding.

- When a whole chunk is masked (no valid tokens for some batch elements), the
  returned state should **not decay at all** for those elements. This matters
  in batched streaming / chunked prefill.

- Streaming equivalence: splitting a sequence into chunks with ``hx`` should
  reproduce a one-shot sequential run.

These tests intentionally keep dimensions tiny so they can run on CPU quickly
and deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from megalodon.modeling_megalodon import ComplexEMA


def _make_mask(pattern: list[int] | None, *, batch: int) -> torch.Tensor | None:
    """Build a batch mask from a 0/1 pattern.

    :param Optional[list[int]] pattern: 0/1 pattern or ``None`` to disable masking.
    :param int batch: Batch size.
    :return Optional[torch.Tensor]: Boolean mask shaped ``(batch, length)``.
    """
    if pattern is None:
        return None
    m = torch.tensor(pattern, dtype=torch.bool).view(1, -1)
    return m.expand(batch, -1).clone()


def _masked_x(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Apply a padding mask to a tensor.

    :param torch.Tensor x: Input tensor shaped ``(batch, dim, length)``.
    :param Optional[torch.Tensor] mask: Boolean mask shaped ``(batch, length)``.
    :return torch.Tensor: Masked tensor with padded positions zeroed.
    """
    if mask is None:
        return x
    return torch.where(mask.unsqueeze(1), x, x.new_zeros(()))


def _last_true_idx(mask: torch.Tensor) -> torch.Tensor:
    """Return last True index per batch element, without losing the all-false case."""
    B, L = mask.shape
    idx = torch.arange(L, device=mask.device).view(1, L).expand(B, L)
    masked = torch.where(mask, idx, torch.full_like(idx, -1))
    return masked.max(dim=-1).values


def _manual_cema_state_at_indices(
    cema: ComplexEMA,
    x: torch.Tensor,
    hx: torch.Tensor | None,
    capture_idx: torch.Tensor,
) -> torch.Tensor:
    """Run the CEMA recurrence and capture h at per-batch indices.

    ``capture_idx`` is allowed to be -1, meaning "capture the initial hx".
    """
    B, D, L = x.shape
    p, q, _gamma = cema._coeffs()
    p_b = p.unsqueeze(0)
    q_b = q.unsqueeze(0)

    h = (
        torch.zeros(B, D, cema.ndim, device=x.device, dtype=torch.complex64)
        if hx is None
        else hx.to(torch.complex64)
    )

    out = torch.empty_like(h)

    take_initial = (capture_idx == -1).view(B, 1, 1)
    out = torch.where(take_initial, h, out)

    x_c = x.to(torch.complex64)
    for t in range(L):
        xt = x_c[:, :, t].unsqueeze(-1)
        h = q_b * h + p_b * xt
        take = (capture_idx == t).view(B, 1, 1)
        out = torch.where(take, h, out)

    return out


def _fixed_cema(embed_dim: int, ndim: int) -> ComplexEMA:
    """Return a CEMA with deterministic, easy-to-reason-about coefficients."""
    cema = ComplexEMA(embed_dim=embed_dim, ndim=ndim)
    with torch.no_grad():
        cema.alpha.fill_(0.0)
        cema.delta.fill_(0.0)
        cema.theta.fill_(-100.0)
        cema.gamma_real.fill_(1.0)
        cema.gamma_imag.zero_()
        cema.omega.zero_()
    return cema


@dataclass(frozen=True)
class MaskCase:
    """Simple mask pattern container for parametrized tests."""

    name: str
    pattern: list[int] | None


MASK_CASES: list[MaskCase] = [
    MaskCase("none", None),
    MaskCase("all_valid", [1, 1, 1, 1, 1, 1]),
    MaskCase("right_pad", [1, 1, 1, 0, 0, 0]),
    MaskCase("left_pad", [0, 0, 1, 1, 1, 1]),
    MaskCase("internal_holes", [1, 0, 1, 1, 0, 1]),
    MaskCase("all_pad", [0, 0, 0, 0, 0, 0]),
]


@pytest.mark.parametrize("case", MASK_CASES, ids=lambda c: c.name)
@torch.no_grad()
def test_cema_returns_state_at_last_valid_token(case: MaskCase) -> None:
    """Returned state must correspond to the last *valid* token, not post-padding decay."""
    torch.manual_seed(0)

    B, D, N = 2, 3, 2
    L = 6
    cema = _fixed_cema(embed_dim=D, ndim=N).eval()

    x = torch.arange(B * D * L, dtype=torch.float32).view(B, D, L) / 100.0 + 1.0

    mask = _make_mask(case.pattern, batch=B)
    x_eff = _masked_x(x, mask)

    _y, h_last = cema(x, hx=None, compute_last_state=True, mask=mask)
    assert h_last is not None and torch.is_complex(h_last)

    if mask is None:
        capture = torch.full((B,), L - 1, device=x.device, dtype=torch.long)
    else:
        last_idx = _last_true_idx(mask)
        capture = last_idx

    ref = _manual_cema_state_at_indices(cema, x_eff, hx=None, capture_idx=capture)

    assert torch.allclose(h_last, ref, atol=1e-5, rtol=0.0)

    if mask is not None and mask.any() and (capture.max().item() < L - 1):
        end_state = _manual_cema_state_at_indices(
            cema,
            x_eff,
            hx=None,
            capture_idx=torch.full((B,), L - 1, device=x.device, dtype=torch.long),
        )
        assert not torch.allclose(h_last, end_state, atol=1e-6, rtol=0.0), (
            "Expected h_last (last valid) to differ from end_state (after padding decay)."
        )


@torch.no_grad()
def test_cema_all_masked_chunk_keeps_hx() -> None:
    """If a chunk contains no valid tokens, the returned state must not decay.

    This is critical for batched chunked prefill where some batch elements may
    be fully padded in later chunks.
    """
    torch.manual_seed(0)

    B, D, N = 2, 2, 2
    L = 5
    cema = _fixed_cema(embed_dim=D, ndim=N).eval()

    x = torch.randn(B, D, L)
    mask = torch.zeros(B, L, dtype=torch.bool)

    hx = torch.randn(B, D, N, dtype=torch.complex64)

    _y, h_last = cema(x, hx=hx, compute_last_state=True, mask=mask)
    assert h_last is not None

    assert torch.allclose(h_last, hx, atol=1e-6, rtol=0.0)


@pytest.mark.parametrize("split", [1, 2, 4], ids=lambda s: f"split={s}")
@pytest.mark.parametrize(
    "case",
    [
        MaskCase("none", None),
        MaskCase("all_valid", [1, 1, 1, 1, 1, 1]),
        MaskCase("right_pad_partial", [1, 1, 1, 1, 0, 0]),
    ],
    ids=lambda c: c.name,
)
@torch.no_grad()
def test_cema_streaming_matches_one_shot_when_each_chunk_has_valid(
    case: MaskCase, split: int
) -> None:
    """Splitting a sequence into chunks and using hx must match one-shot sequential."""
    torch.manual_seed(0)

    B, D, N = 2, 3, 2
    L = 6
    assert 0 < split < L

    cema = _fixed_cema(embed_dim=D, ndim=N).eval()
    x = torch.randn(B, D, L)
    mask = _make_mask(case.pattern, batch=B)

    y_full, h_full = cema(x, hx=None, compute_last_state=True, mask=mask)
    assert h_full is not None

    x1, x2 = x[:, :, :split], x[:, :, split:]
    m1 = None if mask is None else mask[:, :split]
    m2 = None if mask is None else mask[:, split:]

    y1, h1 = cema(x1, hx=None, compute_last_state=True, mask=m1)
    assert h1 is not None
    y2, h2 = cema(x2, hx=h1, compute_last_state=True, mask=m2)
    assert h2 is not None

    y_stream = torch.cat([y1, y2], dim=-1)

    assert torch.allclose(y_stream, y_full, atol=1e-5, rtol=0.0)
    assert torch.allclose(h2, h_full, atol=1e-5, rtol=0.0)


def test_cema_rejects_bad_mask_shape() -> None:
    """Mask shape must be (B, L)."""
    torch.manual_seed(0)
    cema = _fixed_cema(embed_dim=4, ndim=2)
    x = torch.randn(2, 4, 6)

    bad = torch.ones(2, 6, 1, dtype=torch.bool)
    with pytest.raises(ValueError, match=r"mask.*shape"):
        _ = cema(x, hx=None, compute_last_state=False, mask=bad)

    bad2 = torch.ones(3, 6, dtype=torch.bool)
    with pytest.raises(ValueError, match=r"mask.*shape"):
        _ = cema(x, hx=None, compute_last_state=False, mask=bad2)
