# coding=utf-8
"""Exhaustive path-matrix tests for :class:`~megalodon.modeling_megalodon.ComplexEMA`.

Why this file exists
--------------------
ComplexEMA is used in two very different regimes:

* **Training** (no cache): FFT path (fast)
* **Inference / streaming** (with cache): sequential recurrence (slow in pure torch)

On top of that, ComplexEMA has optional masking and accepts multiple hx encodings.
That combination is a classic source of "fixed one bug, reintroduced another".

These tests are intentionally *combinatorial* (not "representative") so that
mask/hx/dispatch regressions become immediately visible.
"""

from __future__ import annotations


import pytest
import torch

from megalodon.modeling_megalodon import ComplexEMA


TOL = 5e-4


def _make_mask(B: int, L: int) -> torch.Tensor:
    """Create a mask that mixes left-pad, right-pad, and an internal gap."""
    mask = torch.ones(B, L, dtype=torch.bool)
    if L >= 2:
        mask[0, :2] = False
    if B >= 2 and L >= 3:
        mask[1, -3:] = False
    if B >= 3 and L >= 5:
        mask[2, 2] = False
    return mask


def _masked_x_and_last_valid_idx(
    x: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replicate ComplexEMA.forward masking + last_valid_idx logic."""
    B, _, L = x.shape
    mask_bool = mask.to(dtype=torch.bool)
    x_masked = torch.where(mask_bool.unsqueeze(1), x, x.new_zeros(()))

    indices = torch.arange(L, device=x.device).expand(B, L)
    masked_indices = torch.where(mask_bool, indices, indices.new_full((), -1))
    last_valid_idx = masked_indices.max(dim=-1).values.clamp(min=0)
    return x_masked, last_valid_idx


@pytest.mark.parametrize("mask_present", [False, True])
@pytest.mark.parametrize("hx_present", [False, True])
@pytest.mark.parametrize("hx_encoding", ["complex", "realimag"])
@pytest.mark.parametrize("compute_last_state", [False, True])
@torch.no_grad()
def test_complex_ema_forward_path_matrix(
    mask_present: bool,
    hx_present: bool,
    hx_encoding: str,
    compute_last_state: bool,
) -> None:
    """Exhaustively validate forward() behavior across the key branches."""

    if not hx_present and hx_encoding != "complex":
        pytest.skip("hx_encoding only matters when hx is provided")

    torch.manual_seed(0)

    B, D, N, L = 3, 12, 4, 17
    cema = ComplexEMA(D, N).eval()

    x = torch.randn(B, D, L)

    mask = _make_mask(B, L) if mask_present else None

    hx = None
    hx_complex = None
    if hx_present:
        hx_complex = torch.randn(B, D, N, dtype=torch.complex64)
        if hx_encoding == "complex":
            hx = hx_complex
        elif hx_encoding == "realimag":
            hx = torch.stack([hx_complex.real, hx_complex.imag], dim=-1)
        else:
            raise AssertionError(f"unknown hx_encoding: {hx_encoding}")

    y, h = cema(x, hx=hx, compute_last_state=compute_last_state, mask=mask)

    assert y.shape == (B, D, L)
    if compute_last_state:
        assert h is not None
        assert h.shape == (B, D, N)
        assert torch.is_complex(h)
    else:
        assert h is None

    if mask is not None:
        x_eff, last_valid_idx = _masked_x_and_last_valid_idx(x, mask)
    else:
        x_eff = x
        last_valid_idx = None

    y_seq, h_seq = cema._forward_sequential(
        x_eff,
        hx=hx,
        last_valid_idx=last_valid_idx,
    )
    y_expected = y_seq + x_eff * cema.omega.view(1, -1, 1)

    assert torch.allclose(y, y_expected, atol=TOL, rtol=TOL), (
        "ComplexEMA.forward output mismatch for combo "
        f"mask_present={mask_present}, hx_present={hx_present}, "
        f"hx_encoding={hx_encoding}, compute_last_state={compute_last_state}. "
        f"max diff={(y - y_expected).abs().max().item():.6g}"
    )

    if compute_last_state:
        assert h_seq is not None
        assert torch.allclose(h, h_seq, atol=TOL, rtol=TOL), (
            "ComplexEMA.forward state mismatch for combo "
            f"mask_present={mask_present}, hx_present={hx_present}, "
            f"hx_encoding={hx_encoding}, compute_last_state={compute_last_state}. "
            f"max diff={(h - h_seq).abs().max().item():.6g}"
        )


@torch.no_grad()
def test_complex_ema_hx_realimag_matches_complex() -> None:
    """hx accepted as (real, imag) pair must behave identically to complex hx."""
    torch.manual_seed(0)
    B, D, N, L = 2, 8, 3, 9
    cema = ComplexEMA(D, N).eval()

    x = torch.randn(B, D, L)
    hx_complex = torch.randn(B, D, N, dtype=torch.complex64)
    hx_pair = torch.stack([hx_complex.real, hx_complex.imag], dim=-1)

    y_c, h_c = cema(x, hx=hx_complex, compute_last_state=True)
    y_p, h_p = cema(x, hx=hx_pair, compute_last_state=True)

    assert torch.allclose(y_c, y_p, atol=1e-6, rtol=1e-6)
    assert h_c is not None and h_p is not None
    assert torch.allclose(h_c, h_p, atol=1e-6, rtol=1e-6)
