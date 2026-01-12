# coding=utf-8
"""Layer-cache mask alignment with input attention_mask.

Why this file exists
--------------------
A lot of masking bugs show up as *cache mask misalignment*:

- cache.mask dropped when current mask is None
- cache.mask constructed with the wrong slice direction
- prefix/current mask concatenation wrong when one side is None

Several other tests already cover end-to-end effects of these bugs. This test is
more direct: it asserts that when you pass an attention_mask with zeros on a
single forward, every layer that returns an attention cache also returns a mask
that:

- is boolean
- has length == cache length
- equals the input mask (when L <= chunk_size so there is no trimming)

This is a cheap, high-signal invariant test.
"""

from __future__ import annotations

import torch

from megalodon import MegalodonConfig, MegalodonForCausalLM


def _tiny_cfg() -> MegalodonConfig:
    """Return a small config for cache-mask tests.

    :return MegalodonConfig: Small configuration for CPU tests.
    """
    return MegalodonConfig(
        vocab_size=64,
        model_dim=64,
        num_layers=2,
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


def _disable_sdpa(model: MegalodonForCausalLM) -> None:
    """Disable SDPA to force the manual attention path.

    :param MegalodonForCausalLM model: Model to update.
    :return None: This helper returns ``None``.
    """
    for layer in model.model.layers:
        layer.attn.inner._sdpa_available = False


@torch.no_grad()
def test_layer_attn_cache_mask_matches_input_mask_when_no_trimming() -> None:
    """Layer cache masks should match input masks when no trimming occurs.

    :return None: This test returns ``None``.
    """
    torch.manual_seed(0)

    cfg = _tiny_cfg()
    model = MegalodonForCausalLM(cfg).eval()
    _disable_sdpa(model)

    B, L = 2, 11
    input_ids = torch.randint(0, cfg.vocab_size, (B, L))

    attention_mask = torch.tensor(
        [
            [1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1],
            [1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1],
        ],
        dtype=torch.long,
    )

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )

    pkv = out.past_key_values
    assert pkv is not None

    expected = attention_mask.to(dtype=torch.bool)

    for layer_cache in pkv[:-1]:
        if layer_cache.attn is None:
            continue
        cache = layer_cache.attn
        assert cache.mask is not None, (
            "Expected a cache mask when input attention_mask contains zeros."
        )
        assert cache.mask.dtype == torch.bool
        assert cache.mask.shape == expected.shape
        assert torch.equal(cache.mask, expected)
