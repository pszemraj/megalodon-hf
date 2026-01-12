# coding=utf-8
"""Two-hop residual wiring invariants.

Why this file exists
--------------------
MEGALODON's "two-hop residual" is not a stylistic preference; it is a stability
mechanism. The single most fragile point is *wiring*: the FFN residual must be
based on the **block input** (X), not the post-attention residual (Y-hat).

This test does not try to numerically re-derive the whole block; instead it
instruments the first block's FFN call and asserts that the tensor passed as
`residual_base` equals the block input.

This catches the classic regression:
  - `ffn(..., residual_base=attn_out)`  (WRONG)
  - `ffn(..., residual_base=x_in)`      (CORRECT)

The test relies only on class names (via introspection) so it is resilient to
minor attribute renames (`layers` vs `blocks`, etc.).
"""

from __future__ import annotations

import types

import torch

from megalodon import MegalodonConfig, MegalodonForCausalLM


def _tiny_cfg() -> MegalodonConfig:
    """Return a small config for residual wiring tests.

    :return MegalodonConfig: Small configuration for CPU tests.
    """
    return MegalodonConfig(
        vocab_size=128,
        model_dim=64,
        num_layers=1,
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


def _find_first_by_classname(
    root: torch.nn.Module, class_name: str
) -> torch.nn.Module | None:
    """Find the first module instance with a given class name.

    :param torch.nn.Module root: Root module to search.
    :param str class_name: Class name to match.
    :return Optional[torch.nn.Module]: Matching module or ``None`` if not found.
    """
    for m in root.modules():
        if m.__class__.__name__ == class_name:
            return m
    return None


@torch.no_grad()
def test_block_passes_original_input_as_ffn_residual_base() -> None:
    """FFN must receive the block input as residual_base.

    :return None: This test returns ``None``.
    """
    torch.manual_seed(0)

    cfg = _tiny_cfg()
    model = MegalodonForCausalLM(cfg).eval()

    block = _find_first_by_classname(model, "MegalodonBlock")
    assert block is not None, (
        "Could not find a module named MegalodonBlock. "
        "If your class is renamed, update the class-name lookup in this test."
    )

    ffn = _find_first_by_classname(block, "NormalizedFFN")
    assert ffn is not None, (
        "Could not find a module named NormalizedFFN inside MegalodonBlock. "
        "If your FFN class is renamed, update the lookup in this test."
    )

    captured: dict[str, torch.Tensor | None] = {
        "block_in": None,
        "ffn_residual_base": None,
    }

    def pre_hook(_module: torch.nn.Module, inputs: tuple[object, ...]) -> None:
        """Capture the block input before the forward pass.

        :param torch.nn.Module _module: Hooked module instance.
        :param tuple[object, ...] inputs: Forward inputs.
        :return None: This hook returns ``None``.
        """
        x = inputs[0]
        captured["block_in"] = x.detach().clone()

    hook_handle = block.register_forward_pre_hook(pre_hook)

    orig_forward = ffn.forward

    def wrapped_forward(
        self: torch.nn.Module, *args: object, **kwargs: object
    ) -> torch.Tensor:
        """Capture residual_base passed to the FFN.

        :param torch.nn.Module self: FFN module instance.
        :param object args: Positional arguments passed to FFN.
        :param object kwargs: Keyword arguments passed to FFN.
        :return torch.Tensor: FFN output tensor.
        """
        rb = kwargs.get("residual_base", None)
        if rb is None and len(args) >= 2 and torch.is_tensor(args[1]):
            rb = args[1]
        captured["ffn_residual_base"] = None if rb is None else rb.detach().clone()
        return orig_forward(*args, **kwargs)

    ffn.forward = types.MethodType(wrapped_forward, ffn)

    try:
        B, L = 2, 23
        input_ids = torch.randint(0, cfg.vocab_size, (B, L))
        attention_mask = torch.ones(B, L, dtype=torch.long)

        _ = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
    finally:
        hook_handle.remove()
        ffn.forward = orig_forward

    assert captured["block_in"] is not None
    assert captured["ffn_residual_base"] is not None, (
        "FFN was not called with residual_base=...; "
        "two-hop residual wiring requires passing the block input as residual_base."
    )

    assert torch.allclose(
        captured["ffn_residual_base"],
        captured["block_in"],
        atol=1e-6,
        rtol=0.0,
    )
