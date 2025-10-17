import math

import pytest
import torch

from megalodon import MegalodonConfig, MegalodonForCausalLM


def _run_backward_step(model, device="cpu", use_cache=False):
    torch.manual_seed(0)
    model.to(device).train()
    cfg = model.config
    batch = 1
    seq = min(cfg.chunk_size, 512)
    inputs = torch.randint(0, cfg.vocab_size, (batch, seq), device=device)
    labels = torch.randint(0, cfg.vocab_size, (batch, seq), device=device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optim.zero_grad(set_to_none=True)

    loss, logits = model(input_ids=inputs, labels=labels, use_cache=use_cache)[:2]
    assert loss.requires_grad
    assert logits.shape == (batch, seq, cfg.vocab_size)

    loss.backward()

    grads = []
    missing = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            missing.append(name)
            continue
        assert torch.isfinite(param.grad).all(), f"non-finite grad in {name}"
        grads.append(param.grad.detach())

    assert grads, "no parameters received gradients"

    total_norm = torch.sqrt(torch.stack([g.pow(2).sum() for g in grads]).sum()).item()
    assert math.isfinite(total_norm) and total_norm > 0.0

    optim.step()
    optim.zero_grad(set_to_none=True)


def test_backward_cpu():
    cfg = MegalodonConfig()
    model = MegalodonForCausalLM(cfg)
    _run_backward_step(model, device="cpu")


@pytest.mark.cuda
def test_backward_cuda():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA available")
    cfg = MegalodonConfig()
    model = MegalodonForCausalLM(cfg)
    _run_backward_step(model, device="cuda")


def test_gradient_checkpointing_backward_cpu():
    cfg = MegalodonConfig()
    model = MegalodonForCausalLM(cfg)
    model.gradient_checkpointing_enable()
    assert model.model.gradient_checkpointing
    _run_backward_step(model, device="cpu", use_cache=True)


@pytest.mark.cuda
def test_gradient_checkpointing_backward_cuda():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA available")
    cfg = MegalodonConfig()
    model = MegalodonForCausalLM(cfg)
    model.gradient_checkpointing_enable()
    assert model.model.gradient_checkpointing
    _run_backward_step(model, device="cuda", use_cache=True)


def test_device_map_inference_cpu():
    accelerate = pytest.importorskip("accelerate")
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
