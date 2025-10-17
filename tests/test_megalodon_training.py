import math

import pytest
import torch

from megalodon import MegalodonConfig, MegalodonForCausalLM


def tiny_cfg(chunk_size=4):
    return MegalodonConfig(
        vocab_size=101,
        model_dim=32,
        num_layers=1,
        num_heads=4,
        z_dim=32,
        value_dim=32,
        ffn_hidden_dim=64,
        cema_ndim=4,
        chunk_size=chunk_size,
        norm_num_groups=4,
        dropout=0.0,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        gradient_checkpointing=False,
    )


def _run_backward_step(model, device="cpu", use_cache=False):
    torch.manual_seed(0)
    model.to(device).train()
    cfg = model.config
    batch, seq = 2, cfg.chunk_size * 2  # ensure multiple of chunk size
    inputs = torch.randint(0, cfg.vocab_size, (batch, seq), device=device)
    labels = torch.randint(0, cfg.vocab_size, (batch, seq), device=device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optim.zero_grad(set_to_none=True)

    loss, logits = model(input_ids=inputs, labels=labels, use_cache=use_cache)[:2]
    assert loss.requires_grad
    assert logits.shape == (batch, seq, cfg.vocab_size)

    loss.backward()

    grads = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"missing grad for {name}"
            assert torch.isfinite(param.grad).all(), f"non-finite grad in {name}"
            grads.append(param.grad.detach())

    total_norm = torch.sqrt(torch.stack([g.pow(2).sum() for g in grads]).sum()).item()
    assert math.isfinite(total_norm) and total_norm > 0.0

    optim.step()
    optim.zero_grad(set_to_none=True)


def test_backward_cpu():
    cfg = tiny_cfg(chunk_size=4)
    model = MegalodonForCausalLM(cfg)
    _run_backward_step(model, device="cpu")


@pytest.mark.cuda
def test_backward_cuda():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA available")
    cfg = tiny_cfg(chunk_size=4)
    model = MegalodonForCausalLM(cfg)
    _run_backward_step(model, device="cuda")


def test_gradient_checkpointing_backward_cpu():
    cfg = tiny_cfg(chunk_size=4)
    model = MegalodonForCausalLM(cfg)
    model.gradient_checkpointing_enable()
    assert model.model.gradient_checkpointing
    _run_backward_step(model, device="cpu", use_cache=True)


@pytest.mark.cuda
def test_gradient_checkpointing_backward_cuda():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA available")
    cfg = tiny_cfg(chunk_size=4)
    model = MegalodonForCausalLM(cfg)
    model.gradient_checkpointing_enable()
    assert model.model.gradient_checkpointing
    _run_backward_step(model, device="cuda", use_cache=True)


def test_device_map_inference_cpu():
    accelerate = pytest.importorskip("accelerate")
    from accelerate.utils import infer_auto_device_map

    cfg = tiny_cfg(chunk_size=4)
    model = MegalodonForCausalLM(cfg)
    device_map = infer_auto_device_map(
        model,
        max_memory={"cpu": "1GB"},
        no_split_module_classes=model._no_split_modules,
    )
    # Expect a CPU-only map for small configs
    assert "" in device_map
    assert device_map[""] == "cpu"
