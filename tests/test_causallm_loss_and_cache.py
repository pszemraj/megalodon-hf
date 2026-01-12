# coding=utf-8
"""Causal LM loss shifting + cache passthrough.

Why this file exists
--------------------
Loss shifting is a classic off-by-one regression surface.

This test pins:
- Loss equals cross entropy of shifted logits/labels.
- ignore_index is honored.
- Cache passthrough behaves consistently when use_cache=True.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from megalodon import MegalodonConfig, MegalodonForCausalLM


def _tiny_cfg() -> MegalodonConfig:
    """Return a tiny config for loss/caching tests.

    :return MegalodonConfig: Small configuration for CPU unit tests.
    """
    return MegalodonConfig(
        vocab_size=97,
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


@torch.no_grad()
def test_labels_none_means_no_loss() -> None:
    """Labels=None should skip loss computation.

    :return None: This test returns ``None``.
    """
    torch.manual_seed(0)

    cfg = _tiny_cfg()
    model = MegalodonForCausalLM(cfg).eval()

    input_ids = torch.randint(0, cfg.vocab_size, (2, 11))
    out = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=None,
        use_cache=True,
        return_dict=True,
    )

    assert getattr(out, "loss", None) is None
    assert out.logits.shape[:2] == input_ids.shape
    assert out.past_key_values is not None


@torch.no_grad()
def test_causal_lm_loss_matches_shifted_cross_entropy() -> None:
    """Loss must match shifted cross-entropy.

    :return None: This test returns ``None``.
    """
    torch.manual_seed(0)

    cfg = _tiny_cfg()
    model = MegalodonForCausalLM(cfg).eval()

    B, L = 2, 13
    input_ids = torch.randint(0, cfg.vocab_size, (B, L))
    attention_mask = torch.ones_like(input_ids)

    labels = input_ids.clone()

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        use_cache=False,
        return_dict=True,
    )

    assert out.loss is not None
    logits = out.logits

    shift_logits = logits[:, :-1].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    ref = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )

    assert torch.allclose(out.loss, ref, atol=1e-6, rtol=0.0)


@torch.no_grad()
def test_ignore_index_excluded_from_loss() -> None:
    """ignore_index should be excluded from loss computation.

    :return None: This test returns ``None``.
    """
    torch.manual_seed(0)

    cfg = _tiny_cfg()
    model = MegalodonForCausalLM(cfg).eval()

    B, L = 2, 9
    input_ids = torch.randint(0, cfg.vocab_size, (B, L))
    attention_mask = torch.ones_like(input_ids)

    ignore_index = -123
    labels = input_ids.clone()
    labels[:, 3:6] = ignore_index

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        ignore_index=ignore_index,
        use_cache=False,
        return_dict=True,
    )

    logits = out.logits
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    ref = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )

    assert torch.allclose(out.loss, ref, atol=1e-6, rtol=0.0)
