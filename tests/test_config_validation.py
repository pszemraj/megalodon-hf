# coding=utf-8
"""Config validation: fail fast on invalid shapes/values.

Why this file exists
--------------------
Invalid configs should raise immediately, not limp into a cryptic matmul error.

This file pins the expected error behavior for the most important invariants.
"""

from __future__ import annotations

import pytest

from megalodon import MegalodonConfig


def _base_kwargs() -> dict[str, object]:
    """Return a baseline config dict for validation tests.

    :return dict[str, object]: Base configuration keyword arguments.
    """
    return dict(
        vocab_size=128,
        model_dim=64,
        num_layers=2,
        num_heads=2,
        z_dim=64,
        value_dim=64,
        ffn_hidden_dim=128,
        cema_ndim=4,
        chunk_size=16,
        norm_num_groups=4,
        norm_eps=1e-5,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        dropout=0.0,
    )


def test_valid_config_constructs() -> None:
    """Valid configs should construct without error.

    :return None: This test returns ``None``.
    """
    _ = MegalodonConfig(**_base_kwargs())


def test_z_dim_must_divide_num_heads() -> None:
    """z_dim must be divisible by num_heads.

    :return None: This test returns ``None``.
    """
    kw = _base_kwargs()
    kw.update(z_dim=65)
    with pytest.raises((ValueError, AssertionError), match=r"z_dim|heads|div"):
        _ = MegalodonConfig(**kw)


def test_value_dim_must_divide_num_heads() -> None:
    """value_dim must be divisible by num_heads.

    :return None: This test returns ``None``.
    """
    kw = _base_kwargs()
    kw.update(value_dim=65)
    with pytest.raises((ValueError, AssertionError), match=r"value_dim|heads|div"):
        _ = MegalodonConfig(**kw)


def test_norm_num_groups_must_divide_model_dim() -> None:
    """norm_num_groups must divide model_dim.

    :return None: This test returns ``None``.
    """
    kw = _base_kwargs()
    kw.update(norm_num_groups=7)
    with pytest.raises((ValueError, AssertionError), match=r"groups|group|div"):
        _ = MegalodonConfig(**kw)


def test_norm_eps_must_be_positive() -> None:
    """norm_eps must be positive.

    :return None: This test returns ``None``.
    """
    kw = _base_kwargs()
    kw.update(norm_eps=0.0)
    with pytest.raises((ValueError, AssertionError), match=r"eps|epsilon|positive"):
        _ = MegalodonConfig(**kw)


def test_layerwise_ckpt_is_rejected_if_unsupported() -> None:
    """layerwise_ckpt should raise if unsupported.

    :return None: This test returns ``None``.
    """
    kw = _base_kwargs()
    kw.update(layerwise_ckpt=True)
    with pytest.raises(
        (ValueError, NotImplementedError, AssertionError), match=r"ckpt|checkpoint"
    ):
        _ = MegalodonConfig(**kw)
