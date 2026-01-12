# coding=utf-8
"""Model-level path coverage for cache/padding interactions.

This repo already has strong coverage of:

* tail-logits equivalence (full vs cached)
* right-padding masked batches
* attention cache mask preservation

There are still a couple of *high-risk integration paths* that are easy to miss:

1) **Training-mode cache opt-in** (`enable_training_cache=True`).
   The codebase intentionally disables caches during training by default.
   Without tests, this path can easily regress silently.

2) **Left-padding with attention_mask** when `use_cache=True`.
   In HF-style decoder-only models, left padding is common for batched generation.
   If left padding is unsupported, the model should fail fast.
   If it is supported, it must behave equivalently to the unpadded sequence.

These tests are designed to prevent "silent wrong" behavior.
"""

from __future__ import annotations

import torch

from megalodon import MegalodonConfig, MegalodonForCausalLM


TOL = 5e-4


def _tiny_cfg() -> MegalodonConfig:
    """Return a small config for cache/position tests.

    :return MegalodonConfig: Small configuration for CPU tests.
    """
    return MegalodonConfig(
        vocab_size=128,
        model_dim=64,
        num_layers=2,
        num_heads=2,
        z_dim=64,
        value_dim=64,
        ffn_hidden_dim=128,
        chunk_size=16,
        cema_ndim=4,
        norm_num_groups=4,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        dropout=0.0,
    )


def _disable_sdpa(model: MegalodonForCausalLM) -> None:
    """Disable SDPA to force the manual attention path.

    :param MegalodonForCausalLM model: Model to update.
    :return None: This helper returns ``None``.
    """
    for layer in model.model.layers:
        layer.attn.inner._sdpa_available = False


def _assert_cache_detached(pkv: tuple[object, ...]) -> None:
    """Assert that cached tensors are detached from autograd.

    :param tuple[object, ...] pkv: Cache tuple from the model output.
    :return None: This helper returns ``None``.
    """
    for layer_cache in pkv[:-1]:
        if layer_cache.attn is not None:
            assert not layer_cache.attn.k.requires_grad
            assert not layer_cache.attn.v.requires_grad
            if layer_cache.attn.mask is not None:
                assert not layer_cache.attn.mask.requires_grad
        if layer_cache.norm is not None:
            assert not layer_cache.norm.count.requires_grad
            assert not layer_cache.norm.mean.requires_grad
            assert not layer_cache.norm.var.requires_grad
        if layer_cache.ema is not None:
            assert not layer_cache.ema.requires_grad

    final_norm = pkv[-1]
    assert not final_norm.count.requires_grad
    assert not final_norm.mean.requires_grad
    assert not final_norm.var.requires_grad


def test_training_cache_opt_in_path_returns_cache_and_matches_eval() -> None:
    """`enable_training_cache=True` should return caches even in train() mode.

    Additionally, with all dropouts disabled, train()-mode and eval()-mode outputs
    should match closely.
    """
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = MegalodonForCausalLM(cfg)
    _disable_sdpa(model)

    B, L = 2, 23
    input_ids = torch.randint(0, cfg.vocab_size, (B, L))
    attn_mask = torch.ones(B, L, dtype=torch.long)

    model.eval()
    out_eval = model(
        input_ids=input_ids,
        attention_mask=attn_mask,
        use_cache=True,
        return_dict=True,
    )
    assert out_eval.past_key_values is not None

    model.train()
    out_train_default = model(
        input_ids=input_ids,
        attention_mask=attn_mask,
        use_cache=True,
        enable_training_cache=False,
        return_dict=True,
    )
    assert out_train_default.past_key_values is None

    out_train_cache = model(
        input_ids=input_ids,
        attention_mask=attn_mask,
        use_cache=True,
        enable_training_cache=True,
        return_dict=True,
    )
    assert out_train_cache.past_key_values is not None

    assert torch.allclose(
        out_train_cache.logits, out_eval.logits, atol=TOL, rtol=TOL
    ), (
        "Train-cache logits differ from eval-cache logits: "
        f"max diff={(out_train_cache.logits - out_eval.logits).abs().max().item():.6g}"
    )

    _assert_cache_detached(out_train_cache.past_key_values)


@torch.no_grad()
def test_left_padded_prompt_is_not_silently_wrong() -> None:
    """Left padding must either be supported (equivalent) or rejected explicitly.

    This test is intentionally written as a guardrail:

    * If the model raises a *clear* error for left-padding + mask, that's OK.
    * If it runs, logits for the valid suffix must match the unpadded sequence.

    Silent divergence is not acceptable, because it leads to extremely subtle
    bugs in batched generation.
    """
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = MegalodonForCausalLM(cfg).eval()
    _disable_sdpa(model)

    B = 2
    L_valid = 9
    L_total = 16

    ids_valid = torch.randint(0, cfg.vocab_size, (B, L_valid))
    mask_valid = torch.ones(B, L_valid, dtype=torch.long)

    pad_id = cfg.pad_token_id
    ids_left = torch.full((B, L_total), pad_id, dtype=torch.long)
    ids_left[:, -L_valid:] = ids_valid
    mask_left = torch.zeros(B, L_total, dtype=torch.long)
    mask_left[:, -L_valid:] = 1

    out_valid = model(
        input_ids=ids_valid,
        attention_mask=mask_valid,
        use_cache=False,
        return_dict=True,
    )

    try:
        out_left = model(
            input_ids=ids_left,
            attention_mask=mask_left,
            use_cache=False,
            return_dict=True,
        )
    except (ValueError, NotImplementedError) as e:
        msg = str(e).lower()
        assert "pad" in msg or "mask" in msg or "left" in msg or "position" in msg
        return

    logits_left_suffix = out_left.logits[:, -L_valid:, :]
    assert torch.allclose(logits_left_suffix, out_valid.logits, atol=TOL, rtol=TOL), (
        "Left-padding changed logits for valid positions (silent wrong). "
        f"max diff={(logits_left_suffix - out_valid.logits).abs().max().item():.6g}"
    )


@torch.no_grad()
def test_left_padded_prompt_cached_continuation_is_not_silently_wrong() -> None:
    """Left-padded cached decode must be equivalent to unpadded, or fail fast."""
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = MegalodonForCausalLM(cfg).eval()
    _disable_sdpa(model)

    B = 1
    L_valid = 7
    L_total = 16

    ids_valid = torch.randint(0, cfg.vocab_size, (B, L_valid))
    mask_valid = torch.ones(B, L_valid, dtype=torch.long)

    pad_id = cfg.pad_token_id
    ids_left = torch.full((B, L_total), pad_id, dtype=torch.long)
    ids_left[:, -L_valid:] = ids_valid
    mask_left = torch.zeros(B, L_total, dtype=torch.long)
    mask_left[:, -L_valid:] = 1

    out_valid = model(
        input_ids=ids_valid,
        attention_mask=mask_valid,
        use_cache=True,
        return_dict=True,
    )

    try:
        out_left = model(
            input_ids=ids_left,
            attention_mask=mask_left,
            use_cache=True,
            return_dict=True,
        )
    except (ValueError, NotImplementedError) as e:
        msg = str(e).lower()
        assert "pad" in msg or "mask" in msg or "left" in msg or "position" in msg
        return

    assert out_valid.past_key_values is not None
    assert out_left.past_key_values is not None

    next_token = torch.tensor([[1]], dtype=torch.long)
    next_mask = torch.ones_like(next_token)

    out_valid2 = model(
        input_ids=next_token,
        attention_mask=next_mask,
        past_key_values=out_valid.past_key_values,
        use_cache=True,
        return_dict=True,
    )
    out_left2 = model(
        input_ids=next_token,
        attention_mask=next_mask,
        past_key_values=out_left.past_key_values,
        use_cache=True,
        return_dict=True,
    )

    assert torch.allclose(out_left2.logits, out_valid2.logits, atol=TOL, rtol=TOL), (
        "Left-padding changed cached continuation logits (silent wrong). "
        f"max diff={(out_left2.logits - out_valid2.logits).abs().max().item():.6g}"
    )
