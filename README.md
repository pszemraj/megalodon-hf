# megalodon-hf

Pure PyTorch + 🤗 Transformers reimplementation of the Megalodon language-model architecture.
This repository provides a portable, inspectable version of the model that runs on vanilla Torch tensors while preserving the streaming-attention semantics of the original CUDA-heavy project.

## Why this project exists

The upstream Megalodon repo couples Python glue with large C++/CUDA extensions and never released trained weights. That makes it difficult to study the design, prototype new ideas, or integrate with modern HF tooling.
`megalodon-hf` focuses on:

- **Readability first:** everything lives in `src/megalodon`, implemented with standard PyTorch modules.
- **Feature parity where it matters:** complex EMA state, chunked rotary attention, streaming caches, and RMS/Timestep norms mirror the original behavior.
- **Modern Hugging Face support:** models subclass `PreTrainedModel`, support `gradient_checkpointing_enable()`, and are compatible with `device_map="auto"`.
- **Simple experimentation loop:** random-weight smoke tests, forward/backward coverage on CPU and GPU, and fixtures that exercise cache equivalence.

If you need the historical reference, a read-only copy of the upstream code sits in `third_party/upstream-megalodon`.

## Project layout

```
src/megalodon/
├── configuration_megalodon.py   # MegalodonConfig (HF-compatible)
├── modeling_megalodon.py        # MegalodonModel & MegalodonForCausalLM
└── __init__.py                  # convenience exports

tests/
├── test_megalodon_smoke.py      # forward/caching parity & CUDA smoke
└── test_megalodon_training.py   # backward passes, checkpointing, device maps
```

`pyproject.toml` wires everything up as a standard setuptools project with Torch/Transformers requirements.

## Getting started

```bash
# From a fresh environment with Python 3.9+
pip install -e .

# Base install pulls in torch>=2.6 and transformers>=4.45.

# Extras:
# - [tests] adds pytest + accelerate for the CI/test suite
# - [dev] adds [tests] plus ruff for local linting
# - [all] installs every extra in one go
pip install -e .[tests]
pip install -e .[dev]
pip install -e .[all]
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

## Implementation notes

- **Complex EMA in pure Torch:** Rather than relying on fused kernels, the EMA recurrence is implemented directly, maintaining carry-over state for streaming generation. The sequential formulation matches the fused CUDA behavior and is validated via cache-equivalence tests.
- **Chunked rotary attention:** Rotary embeddings, block-diagonal attention, and cache updates follow the original semantics, including prefix handling when caches are supplied mid-sequence.
- **Test-first approach:** New features (e.g., HF compatibility, caching parity) land alongside targeted pytest coverage to prevent regressions.
- **HF alignment:** The models override input/output embedding accessors, tie weights, and advertise `_no_split_modules` so they behave well with `transformers` utilities, `Auto*` pipelines, and quantization/offloading workflows.

## Working with the upstream reference

`third_party/upstream-megalodon` contains a snapshot of the original repo for documentation, configuration defaults, and cross-referencing the CUDA kernels. It is read-only-modifications should happen in `src/megalodon`.

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
