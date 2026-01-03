# coding=utf-8
# Copyright 2025 Peter Szemraj.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Training-focused smoke tests covering backward passes and device maps."""

import math

import pytest
import torch

from megalodon import MegalodonConfig, MegalodonForCausalLM


def _run_backward_step(
    model: MegalodonForCausalLM, device: str = "cpu", use_cache: bool = False
) -> None:
    """Run a single backward step and assert gradients look healthy."""
    torch.manual_seed(0)
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            pytest.skip("no CUDA available")
        free_mem, _ = torch.cuda.mem_get_info()
        if free_mem < 256 * 1024 * 1024:
            pytest.skip("insufficient CUDA memory for backward smoke test")
    model.to(device).train()
    cfg = model.config
    batch = 1
    seq = min(cfg.chunk_size, 512)
    inputs = torch.randint(0, cfg.vocab_size, (batch, seq), device=device)
    labels = torch.randint(0, cfg.vocab_size, (batch, seq), device=device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optim.zero_grad(set_to_none=True)

    outputs = model(
        input_ids=inputs,
        labels=labels,
        use_cache=use_cache,
        return_dict=True,
    )
    loss = outputs.loss
    logits = outputs.logits
    assert loss.requires_grad
    assert logits.shape == (batch, seq, cfg.vocab_size)

    loss.backward()

    grads = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        assert param.grad is not None, f"missing grad for {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite grad in {name}"
        grads.append(param.grad.detach())

    total_norm = torch.sqrt(
        torch.stack([g.abs().pow(2).sum() for g in grads]).sum()
    ).item()
    assert math.isfinite(total_norm) and total_norm > 0.0

    optim.step()
    optim.zero_grad(set_to_none=True)


def test_backward_cpu() -> None:
    """CPU backward pass should succeed with finite gradients."""
    cfg = MegalodonConfig()
    model = MegalodonForCausalLM(cfg)
    _run_backward_step(model, device="cpu")


@pytest.mark.cuda
def test_backward_cuda() -> None:
    """CUDA backward pass should succeed with finite gradients."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA available")
    cfg = MegalodonConfig()
    model = MegalodonForCausalLM(cfg)
    _run_backward_step(model, device="cuda")


def test_gradient_checkpointing_backward_cpu() -> None:
    """Checkpointed CPU training path should still propagate gradients."""
    cfg = MegalodonConfig()
    model = MegalodonForCausalLM(cfg)
    model.gradient_checkpointing_enable()
    assert model.model.gradient_checkpointing
    _run_backward_step(model, device="cpu", use_cache=True)


@pytest.mark.cuda
def test_gradient_checkpointing_backward_cuda() -> None:
    """Checkpointed CUDA path should backprop without NaNs."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA available")
    cfg = MegalodonConfig()
    model = MegalodonForCausalLM(cfg)
    model.gradient_checkpointing_enable()
    assert model.model.gradient_checkpointing
    _run_backward_step(model, device="cuda", use_cache=True)


def test_device_map_inference_cpu() -> None:
    """Device-map inference should place layers on CPU/disk under tight budget."""
    pytest.importorskip("accelerate")
    from accelerate.utils import infer_auto_device_map

    cfg = MegalodonConfig()
    model = MegalodonForCausalLM(cfg)
    device_map = infer_auto_device_map(
        model,
        max_memory={"cpu": "6GB"},
        no_split_module_classes=model._no_split_modules,
    )
    # Expect CPU or disk placement only given the CPU-only memory budget
    assert set(device_map.values()).issubset({"cpu", "disk"})
    assert "cpu" in device_map.values()


# -----------------------------------------------------------------------------
# Config variant tests: different model configurations
# -----------------------------------------------------------------------------


def _small_config(**kwargs) -> MegalodonConfig:
    """Return a small config for fast testing with custom overrides."""
    defaults = dict(
        vocab_size=1000,
        model_dim=128,
        num_layers=2,
        num_heads=2,
        z_dim=64,
        value_dim=128,
        ffn_hidden_dim=256,
        cema_ndim=8,
        chunk_size=64,
        norm_num_groups=8,
    )
    defaults.update(kwargs)
    return MegalodonConfig(**defaults)


def test_backward_swiglu_enabled() -> None:
    """SwiGLU FFN variant should train without issues."""
    cfg = _small_config(swiglu=True)
    model = MegalodonForCausalLM(cfg)
    _run_backward_step(model, device="cpu")


def test_backward_with_dropout() -> None:
    """Model with dropout > 0 should train without issues."""
    cfg = _small_config(dropout=0.1, attention_dropout=0.1, hidden_dropout=0.1)
    model = MegalodonForCausalLM(cfg)
    _run_backward_step(model, device="cpu")


def test_backward_rescale_nffn() -> None:
    """rescale_nffn=True variant should train without issues."""
    cfg = _small_config(rescale_nffn=True)
    model = MegalodonForCausalLM(cfg)
    _run_backward_step(model, device="cpu")


def test_backward_combined_variants() -> None:
    """Combined config variants: swiglu + dropout + rescale_nffn."""
    cfg = _small_config(
        swiglu=True,
        dropout=0.1,
        attention_dropout=0.1,
        hidden_dropout=0.1,
        rescale_nffn=True,
    )
    model = MegalodonForCausalLM(cfg)
    _run_backward_step(model, device="cpu")


# -----------------------------------------------------------------------------
# Padding-aware loss tests
# -----------------------------------------------------------------------------


def test_loss_excludes_padding_tokens() -> None:
    """Loss should ignore positions where labels == -100 (HuggingFace standard)."""
    torch.manual_seed(42)
    cfg = _small_config()
    model = MegalodonForCausalLM(cfg)
    model.eval()

    batch, seq = 2, 32
    inputs = torch.randint(0, cfg.vocab_size, (batch, seq))

    # Labels without padding
    labels_no_pad = torch.randint(0, cfg.vocab_size, (batch, seq))

    # Labels with padding: second half of each sequence is -100
    labels_with_pad = labels_no_pad.clone()
    labels_with_pad[:, seq // 2 :] = -100

    with torch.no_grad():
        out_no_pad = model(input_ids=inputs, labels=labels_no_pad, return_dict=True)
        out_with_pad = model(input_ids=inputs, labels=labels_with_pad, return_dict=True)

    # Both losses should be finite
    assert torch.isfinite(out_no_pad.loss), "Loss without padding should be finite"
    assert torch.isfinite(out_with_pad.loss), "Loss with padding should be finite"

    # Losses should differ because padding positions are excluded
    assert out_no_pad.loss != out_with_pad.loss, (
        "Loss should differ when padding is present"
    )


def test_all_padding_returns_nan_loss() -> None:
    """When all labels are -100, loss is NaN (no valid targets for mean reduction)."""
    torch.manual_seed(42)
    cfg = _small_config()
    model = MegalodonForCausalLM(cfg)
    model.eval()

    batch, seq = 2, 32
    inputs = torch.randint(0, cfg.vocab_size, (batch, seq))
    labels_all_pad = torch.full((batch, seq), -100, dtype=torch.long)

    with torch.no_grad():
        out = model(input_ids=inputs, labels=labels_all_pad, return_dict=True)

    # PyTorch cross_entropy with ignore_index returns NaN when all targets are ignored
    # (mean reduction divides by 0). This is expected - don't train with all-padding batches.
    assert torch.isnan(out.loss), "Loss should be NaN when all labels are padding"
