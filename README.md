# megalodon-hf

> Pure PyTorch + 🤗 Transformers reimplementation of the Megalodon language-model.

This repository provides a portable and inspectable version of the [Megalodon](https://arxiv.org/abs/2404.08801) decoder architecture. It runs on vanilla Torch tensors while preserving the streaming-attention semantics of the [original](https://github.com/XuezheMax/megalodon), CUDA-heavy project.

## Why this project exists

Megalodon is a fresh, exciting take on long-context modeling, but [the original repo](https://github.com/XuezheMax/megalodon) couples Python glue with large C++/CUDA extensions and never released trained weights[^1]. That makes it difficult to study the design, prototype, and/or compare vs. new ideas[^2], or integrate with modern HF tooling.

[^1]: at time of repo creation, October 2025. The original repo was released Apr 17, 2024 and does not have weights, [per this issue](https://github.com/XuezheMax/megalodon/issues/1) due to legal review limbo
[^2]: the complexity & lack of weights is a blocker for continued research/improvement on the concept and also leads to [improper comparisons of Megalodon](https://huggingface.co/papers/2510.03279#68ec662e8bfbf816c8335efa) to other techniques. It's hard to compare vs megalodon if you can't train/understand megalodon properly.

`megalodon-hf` focuses on:

- **Readability first:** everything lives in [src/megalodon](src/megalodon), implemented with standard PyTorch modules.
- **Feature parity where it matters:** complex EMA state, chunked rotary attention, streaming caches, and RMS/Timestep norms mirror the original behavior.
- **Modern Hugging Face support:** models subclass `PreTrainedModel`, support `gradient_checkpointing_enable()`, and are compatible with `device_map="auto"`.
- **Simple experimentation loop:** random-weight smoke tests, forward/backward coverage on CPU and GPU, and fixtures that exercise cache equivalence.

If you need the historical reference, a read-only copy of the upstream code sits in `third_party/upstream-megalodon`.

## Project layout

```
pyproject.toml
src/megalodon/
├── configuration_megalodon.py   # MegalodonConfig (HF-compatible)
├── modeling_megalodon.py        # MegalodonModel & MegalodonForCausalLM
└── __init__.py                  # convenience exports

tests/
├── test_megalodon_smoke.py      # forward/caching parity & CUDA smoke
└── test_megalodon_training.py   # backward passes, checkpointing, device maps
```

## Getting started

Clone the repository and install dependencies:

```bash
git clone https://github.com/pszemraj/megalodon-hf.git
cd megalodon-hf
# From a fresh environment with Python 3.9+
pip install -e .
```

The base install pulls in `torch>=2.6` and `transformers>=4.45`. Additional extras:

```bash
# Extras:
# - [tests] adds pytest + accelerate for the CI/test suite
# - [dev] adds [tests] plus ruff for local linting
# - [all] installs every extra in one go
pip install -e .[tests]
pip install -e .[dev]
pip install -e .[all]
```

### Optional: include the upstream reference

`third_party/upstream-megalodon` is populated from the original Megalodon repo via a git submodule. Initialize it if you want the read-only reference for comparisons:

```bash
git submodule update --init --recursive
# or clone with: git clone --recursive https://github.com/pszemraj/megalodon-hf.git
```

### Quick API demo

```python
from megalodon import MegalodonConfig, MegalodonForCausalLM
import torch

cfg = MegalodonConfig(
    vocab_size=32_000,
    model_dim=512,
    num_layers=8,
    num_heads=8,
    chunk_size=256,
    cema_ndim=16,
)

model = MegalodonForCausalLM(cfg).eval()
input_ids = torch.randint(0, cfg.vocab_size, (1, 128))
logits, cache = model(input_ids=input_ids, use_cache=True)
print(logits.shape)        # (1, 128, vocab_size)
print(len(cache))          # list of per-layer streaming caches
```

A copy of the tokenizer used in the Megalodon paper & default config (_llama-2, 32k vocab_) is available in [assets/tokenizer](assets/tokenizer) for convenience. You can load it with:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("assets/tokenizer")
print(tokenizer) # get a summary of the tokenizer
```

### Gradient checkpointing & device placement

Because the models inherit from `PreTrainedModel`, they plug directly into the HF ecosystem:

```python
model.gradient_checkpointing_enable()
outputs = model(
    input_ids=input_ids.cuda(),
    labels=input_ids.cuda(),
    use_cache=True,          # automatically disabled while checkpointing to keep outputs consistent
)
loss = outputs.loss
loss.backward()
```

You can generate a device map with 🤗 Accelerate:

```python
from accelerate.utils import infer_auto_device_map

device_map = infer_auto_device_map(
    model,
    max_memory={0: "12GiB", 1: "12GiB", "cpu": "48GiB"},
    no_split_module_classes=model._no_split_modules,
)
```

## Running the test suite

Tests rely on random-weight sanity checks and run quickly on CPU; CUDA-marked cases exercise the streaming caches on GPU.

```bash
pytest                    # CPU + optional accelerate device-map checks
pytest -m cuda            # CUDA smoke (skips if no GPU)
```

The training tests cover:

- Full forward/backward passes with AdamW on CPU & GPU.
- Gradient checkpointing compatibility.
- `infer_auto_device_map` integration (skips if `accelerate` is missing).

## Status & limitations

- Pure PyTorch implementation: no fused CUDA kernels or the paper's 4D chunk parallelism. Long-context training works but is markedly slower than the reference implementation.
- Complex EMA exposes both a sequential and FFT path; the FFT variant is automatically used during training when cache state is not requested.
- TimestepNorm keeps the numerically exact Welford update in PyTorch. A Triton/CUDA kernel would be required to match the paper's throughput.
- DropKey-style attention dropout and PyTorch's fused SDPA path are wired in, but FlashAttention-2 or other custom kernels are not bundled.

## Implementation notes

- **Complex EMA in pure Torch:** Rather than relying on fused kernels, the EMA recurrence is implemented directly. An FFT fast path kicks in when no cache state is requested, while the sequential recurrence maintains the streaming semantics used during generation.
- **Chunked rotary attention:** Rotary embeddings, block-diagonal attention, and cache updates follow the original semantics, including prefix handling when caches are supplied mid-sequence.
- **Z normalisation per head:** The shared Z projection is RMS-normalised per head (RMS over the head dimension) before forming Q/K via affine parameters. This matches the upstream reference implementation; the paper's equations suggest an L2 normalisation on the full Z vector, but per-head RMS brings the same stability in practice and integrates cleanly with the head-split path.
- **Test-first approach:** New features (e.g., HF compatibility, caching parity) land alongside targeted pytest coverage to prevent regressions.
- **HF alignment:** The models override input/output embedding accessors, tie weights, and advertise `_no_split_modules` so they behave well with `transformers` utilities, `Auto*` pipelines, and quantization/offloading workflows.

## Working with the upstream reference

`third_party/upstream-megalodon` contains a snapshot of the original repo for documentation, configuration defaults, and cross-referencing the CUDA kernels. The directory is populated through the `upstream-megalodon` git submodule, so re-run `git submodule update --init --recursive` (or clone with `--recursive`) if you ever need to refresh it. If you skip the optional submodule step, this directory will stay empty until you initialize it. Treat this directory as read-only-modifications should happen in `src/megalodon`.

## Contributing / hacking

1. Fork or clone the repo.
2. Create a new branch for your experiment.
3. Make changes under `src/megalodon` or `tests/`.
4. Run `pytest` (and `pytest -m cuda` if you touched device code) after `pip install -e .[tests]`.
5. Open a PR or share your diff.

Bug reports and feature proposals are welcome-file an issue describing the scenario, expected behavior, and repro script if possible.

## Citations

Original MEGA+Megalodon papers:

```bibtex
@misc{ma2024megalodon,
      title={Megalodon: Efficient LLM Pretraining and Inference with Unlimited Context Length},
      author={Xuezhe Ma and Xiaomeng Yang and Wenhan Xiong and Beidi Chen and Lili Yu and Hao Zhang and Jonathan May and Luke Zettlemoyer and Omer Levy and Chunting Zhou},
      year={2024},
      eprint={2404.08801},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

@inproceedings{
  ma2023mega,
  title={Mega: Moving Average Equipped Gated Attention},
  author={Xuezhe Ma and Chunting Zhou and Xiang Kong and Junxian He and Liangke Gui and Graham Neubig and Jonathan May and Luke Zettlemoyer},
  booktitle={The Eleventh International Conference on Learning Representations },
  year={2023},
}
```
