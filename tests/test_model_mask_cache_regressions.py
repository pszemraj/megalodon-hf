# coding=utf-8
"""End-to-end regressions for masking + caching in MegalodonForCausalLM.

Why this file exists
--------------------
Low-level attention tests catch a lot, but some masking/caching bugs only show up
once *the whole stack* is wired together (TimestepNorm -> CEMA -> Q/K/V ->
ChunkedSelfAttention -> LM head).

This file pins a couple of historically fragile behaviors:

1) **Cache mask must be respected even when the *current* attention_mask is
   ``None``.**
   HF generation sometimes feeds ``attention_mask=None`` for single-token
   decode steps. If the model ignores the prefix mask stored in the cache in
   that case, previously-masked prefix tokens leak into attention.

2) **max_cache_len < chunk_size must be rejected** (if that is the intended
   contract) at the model API level as well, so users don't get silent
   corruption.

All tests run on CPU with tiny configs.
"""

from __future__ import annotations

import pytest
import torch

from megalodon import MegalodonConfig, MegalodonForCausalLM


def _tiny_cfg(**kwargs) -> MegalodonConfig:
    defaults = dict(
        vocab_size=32,
        model_dim=32,
        num_layers=2,
        num_heads=2,
        z_dim=32,
        value_dim=32,
        ffn_hidden_dim=64,
        cema_ndim=4,
        chunk_size=8,
        norm_num_groups=4,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        dropout=0.0,
    )
    defaults.update(kwargs)
    return MegalodonConfig(**defaults)


def _disable_sdpa(model: MegalodonForCausalLM) -> None:
    for layer in model.model.layers:
        layer.attn.inner._sdpa_available = False


@torch.no_grad()
def test_cache_mask_respected_when_step_attention_mask_is_none() -> None:
    """Masked prefix tokens must not influence next-step logits, even if step mask is None."""
    torch.manual_seed(0)

    cfg = _tiny_cfg()
    model = MegalodonForCausalLM(cfg).eval()
    _disable_sdpa(model)

    with torch.no_grad():
        emb = (
            torch.arange(cfg.vocab_size, dtype=torch.float32)
            .unsqueeze(1)
            .repeat(1, cfg.model_dim)
        )
        model.model.embed.weight.copy_(emb)
        if getattr(model, "_tied_embeddings", False):
            model.lm_head.weight = model.model.embed.weight
        else:
            model.lm_head.weight.copy_(emb[: model.lm_head.weight.size(0)])

    prompt_a = torch.tensor([[5, 6, 7, 8, 9, 10]])
    prompt_b = torch.tensor([[5, 6, 21, 22, 23, 24]])
    mask = torch.tensor([[1, 1, 0, 0, 0, 0]], dtype=torch.long)

    out_a = model(
        input_ids=prompt_a,
        attention_mask=mask,
        use_cache=True,
        return_dict=True,
    )
    out_b = model(
        input_ids=prompt_b,
        attention_mask=mask,
        use_cache=True,
        return_dict=True,
    )

    pkv_a = out_a.past_key_values
    pkv_b = out_b.past_key_values
    assert pkv_a is not None and pkv_b is not None

    next_token = torch.tensor([[3]])

    logits_a_none = model(
        input_ids=next_token,
        attention_mask=None,
        past_key_values=pkv_a,
        use_cache=True,
        return_dict=True,
    ).logits[:, -1]
    logits_b_none = model(
        input_ids=next_token,
        attention_mask=None,
        past_key_values=pkv_b,
        use_cache=True,
        return_dict=True,
    ).logits[:, -1]

    logits_a_ones = model(
        input_ids=next_token,
        attention_mask=torch.ones_like(next_token),
        past_key_values=pkv_a,
        use_cache=True,
        return_dict=True,
    ).logits[:, -1]
    logits_b_ones = model(
        input_ids=next_token,
        attention_mask=torch.ones_like(next_token),
        past_key_values=pkv_b,
        use_cache=True,
        return_dict=True,
    ).logits[:, -1]

    assert torch.allclose(logits_a_ones, logits_b_ones, atol=1e-5, rtol=0.0)
    assert torch.allclose(logits_a_none, logits_b_none, atol=1e-5, rtol=0.0)

    assert torch.allclose(logits_a_none, logits_a_ones, atol=1e-5, rtol=0.0)
    assert torch.allclose(logits_b_none, logits_b_ones, atol=1e-5, rtol=0.0)


def test_model_rejects_max_cache_len_smaller_than_chunk_size() -> None:
    """If max_cache_len < chunk_size is invalid, enforce it at the top-level API."""
    torch.manual_seed(0)

    cfg = _tiny_cfg(chunk_size=8)
    model = MegalodonForCausalLM(cfg).eval()

    x = torch.randint(0, cfg.vocab_size, (1, 4))

    with pytest.raises(ValueError, match=r"max_cache_len.*chunk_size"):
        _ = model(
            input_ids=x,
            attention_mask=torch.ones_like(x),
            use_cache=True,
            max_cache_len=cfg.chunk_size - 1,
            return_dict=True,
        )


@torch.no_grad()
def test_cached_tail_logits_with_padding_mask() -> None:
    """Cached decoding with padding in the prefix must match one-shot decoding.

    This is an end-to-end guardrail for: mask -> cache.mask -> later step.
    """
    torch.manual_seed(0)

    cfg = _tiny_cfg(chunk_size=8)
    model = MegalodonForCausalLM(cfg).eval()
    _disable_sdpa(model)

    prefix = torch.tensor([[4, 5, 6, 7, 8, 9, 10, 11]])
    prefix_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.long)

    suffix = torch.tensor([[12, 13, 14]])

    full_ids = torch.cat([prefix, suffix], dim=1)
    full_mask = torch.cat([prefix_mask, torch.ones_like(suffix)], dim=1)

    ref_logits = model(
        input_ids=full_ids,
        attention_mask=full_mask,
        use_cache=True,
        return_dict=True,
    ).logits[:, -suffix.size(1) :]

    pref_out = model(
        input_ids=prefix,
        attention_mask=prefix_mask,
        use_cache=True,
        return_dict=True,
    )
    pkv = pref_out.past_key_values
    assert pkv is not None

    suf_out = model(
        input_ids=suffix,
        attention_mask=torch.ones_like(suffix),
        past_key_values=pkv,
        use_cache=True,
        return_dict=True,
    )

    test_logits = suf_out.logits

    assert torch.allclose(test_logits, ref_logits, atol=5e-3, rtol=5e-3)
