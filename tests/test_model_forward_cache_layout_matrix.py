# coding=utf-8
"""HF cache layout and type/length validation.

Why this file exists
--------------------
The `past_key_values` contract is one of the highest-leverage stability points:

- HF generation utilities pass `past_key_values` back in with minimal checking.
- A subtle layout/type mismatch can silently corrupt caches (or masks), and the
  model will still run.

This file pins the expected cache layout:

1) Forward with `use_cache=True` returns `past_key_values`.
2) The returned object has a stable length and round-trips on the next step.
3) `past_key_values` inputs that are *shorter* than expected are padded with
   `None` (fail-soft), while malformed/too-long inputs fail-fast.

If you prefer a stricter contract (e.g. require exact length always), adjust the
assertions and keep the tests.
"""

from __future__ import annotations

import pytest
import torch

from megalodon import MegalodonConfig, MegalodonForCausalLM


def _tiny_cfg(num_layers: int = 2) -> MegalodonConfig:
    """Return a small config for cache-layout tests.

    :param int num_layers: Number of layers in the config.
    :return MegalodonConfig: Small configuration for CPU tests.
    """
    return MegalodonConfig(
        vocab_size=128,
        model_dim=64,
        num_layers=num_layers,
        num_heads=2,
        z_dim=64,
        value_dim=64,
        ffn_hidden_dim=128,
        cema_ndim=4,
        chunk_size=16,
        norm_num_groups=4,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        dropout=0.0,
    )


@torch.no_grad()
def test_forward_returns_cache_and_round_trips() -> None:
    """Forward pass should return cache that round-trips.

    :return None: This test returns ``None``.
    """
    torch.manual_seed(0)

    cfg = _tiny_cfg(num_layers=2)
    model = MegalodonForCausalLM(cfg).eval()

    input_ids = torch.randint(0, cfg.vocab_size, (1, 13))
    attention_mask = torch.ones_like(input_ids)

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )

    pkv = out.past_key_values
    assert pkv is not None

    assert len(pkv) == cfg.num_layers + 1, (
        f"Expected past_key_values length {cfg.num_layers + 1}, got {len(pkv)}. "
        "If you changed the cache layout, update this test and keep it strict."
    )

    next_token = torch.randint(0, cfg.vocab_size, (1, 1))
    out2 = model(
        input_ids=next_token,
        attention_mask=torch.ones_like(next_token),
        past_key_values=pkv,
        use_cache=True,
        return_dict=True,
    )
    assert out2.past_key_values is not None
    assert len(out2.past_key_values) == len(pkv)


@torch.no_grad()
def test_past_key_values_without_final_norm_state_is_accepted_and_padded() -> None:
    """Missing final NormState should be accepted and padded.

    :return None: This test returns ``None``.
    """
    torch.manual_seed(0)

    cfg = _tiny_cfg(num_layers=2)
    model = MegalodonForCausalLM(cfg).eval()

    input_ids = torch.randint(0, cfg.vocab_size, (1, 9))
    attention_mask = torch.ones_like(input_ids)
    pkv = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    ).past_key_values
    assert pkv is not None

    pkv_no_final = tuple(pkv[:-1])

    next_token = torch.randint(0, cfg.vocab_size, (1, 1))
    out2 = model(
        input_ids=next_token,
        attention_mask=torch.ones_like(next_token),
        past_key_values=pkv_no_final,
        use_cache=True,
        return_dict=True,
    )
    assert out2.past_key_values is not None
    assert len(out2.past_key_values) == cfg.num_layers + 1


@torch.no_grad()
def test_past_key_values_too_short_is_padded_with_none() -> None:
    """Short past_key_values should be padded to expected length.

    :return None: This test returns ``None``.
    """
    torch.manual_seed(0)

    cfg = _tiny_cfg(num_layers=3)
    model = MegalodonForCausalLM(cfg).eval()

    next_token = torch.randint(0, cfg.vocab_size, (1, 1))

    dummy_short = (None,)

    out = model(
        input_ids=next_token,
        attention_mask=torch.ones_like(next_token),
        past_key_values=dummy_short,
        use_cache=True,
        return_dict=True,
    )

    assert out.past_key_values is not None
    assert len(out.past_key_values) == cfg.num_layers + 1


def test_past_key_values_too_long_is_rejected() -> None:
    """Overlong past_key_values should raise.

    :return None: This test returns ``None``.
    """
    torch.manual_seed(0)

    cfg = _tiny_cfg(num_layers=2)
    model = MegalodonForCausalLM(cfg).eval()

    input_ids = torch.randint(0, cfg.vocab_size, (1, 5))
    attention_mask = torch.ones_like(input_ids)

    pkv = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    ).past_key_values
    assert pkv is not None

    bad = tuple(pkv) + (None,)

    next_token = torch.randint(0, cfg.vocab_size, (1, 1))
    with pytest.raises(
        (ValueError, TypeError), match=r"past_key_values|cache|length|layers"
    ):
        _ = model(
            input_ids=next_token,
            attention_mask=torch.ones_like(next_token),
            past_key_values=bad,
            use_cache=True,
            return_dict=True,
        )


def test_past_key_values_wrong_last_slot_type_is_rejected() -> None:
    """Wrong final slot type should raise.

    :return None: This test returns ``None``.
    """
    torch.manual_seed(0)

    cfg = _tiny_cfg(num_layers=2)
    model = MegalodonForCausalLM(cfg).eval()

    input_ids = torch.randint(0, cfg.vocab_size, (1, 7))
    attention_mask = torch.ones_like(input_ids)

    pkv = list(
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        ).past_key_values
    )
    assert pkv is not None

    pkv[-1] = 123

    next_token = torch.randint(0, cfg.vocab_size, (1, 1))
    with pytest.raises(
        (ValueError, TypeError), match=r"Norm|norm|final|state|past_key_values"
    ):
        _ = model(
            input_ids=next_token,
            attention_mask=torch.ones_like(next_token),
            past_key_values=tuple(pkv),
            use_cache=True,
            return_dict=True,
        )
