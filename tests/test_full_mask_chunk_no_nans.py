# coding=utf-8
"""Tests that fully-masked chunks do not produce NaNs.

Why this matters
----------------
In padded batches (common during training), it is easy to end up with a full
*chunk* of padding for some batch elements (e.g. chunk_size=512, max_len=1024,
sequence_len=100). If attention is applied with a key mask that is all-false,
softmax can receive all -inf and produce NaNs.

NaNs at padded positions are not necessarily harmless in Megalodon, because
TimestepNorm uses cumulative sums; naive masking with `x * 0` does not clear NaNs
(`nan * 0 == nan`). So a single NaN at a padded position can corrupt later
normalization statistics and leak into valid tokens.

This test enforces that the full model remains finite under an all-zero chunk
mask.
"""

from __future__ import annotations

import torch

from megalodon import MegalodonConfig, MegalodonForCausalLM


def _disable_sdpa(model: MegalodonForCausalLM) -> None:
    for layer in model.model.layers:
        layer.attn.inner._sdpa_available = False


@torch.no_grad()
def test_model_forward_with_fully_padded_chunk_is_finite() -> None:
    torch.manual_seed(0)

    cfg = MegalodonConfig(
        vocab_size=64,
        model_dim=32,
        num_layers=2,
        num_heads=2,
        z_dim=32,
        value_dim=32,
        ffn_hidden_dim=64,
        cema_ndim=4,
        chunk_size=4,
        norm_num_groups=4,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        dropout=0.0,
    )
    model = MegalodonForCausalLM(cfg).eval()
    _disable_sdpa(model)

    input_ids = torch.randint(0, cfg.vocab_size, (1, cfg.chunk_size * 2))
    attention_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.long)

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )

    assert torch.isfinite(out.logits).all(), (
        "Model produced NaNs/inf with a fully padded chunk."
    )
