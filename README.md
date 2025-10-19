# megalodon-hf

A torch + 🤗 Transformers implementation of [Megalodon](https://arxiv.org/abs/2404.08801), grounded on [the original code](https://github.com/XuezheMax/megalodon).

## Features

- Readable, hackable code in [src/megalodon](src/megalodon), pure PyTorch
- Core architecture parity: complex EMA, chunked rotary attention, streaming cache, RMS/Timestep norms
- Hugging Face native: `PreTrainedModel`, `gradient_checkpointing_enable()`, `device_map="auto"`
- Simple experimentation: quick smoke tests on CPU/GPU, cache equivalence fixtures in [tests](tests)

## Installation

```bash
git clone https://github.com/pszemraj/megalodon-hf.git
cd megalodon-hf
# set up a virtualenv first, then:
pip install -e .
```

The base install pulls in `torch>=2.6` and `transformers>=4.45`. Extras: `[tests]`, `[dev]`, `[all]`.

### Upstream Reference

<details>
<summary><b>Click to Expand:</b> Instructions to add the original Megalodon repo as a submodule</summary>

The original CUDA-heavy reference can be added as a read-only submodule for comparison under [third_party/upstream-megalodon](third_party/upstream-megalodon):

```bash
git submodule update --init --recursive
# Or: git clone --recursive https://github.com/pszemraj/megalodon-hf.git
```

> [!NOTE]
> [third_party/upstream-megalodon](third_party/upstream-megalodon) stays empty until you initialize the submodule. Keep your modifications in [src/megalodon](src/megalodon) instead of editing the reference copy.

</details>

## Quick Start

Create a random-weights model and run a forward pass with dummy input:

```python
import torch
from megalodon import MegalodonConfig, MegalodonForCausalLM

cfg = MegalodonConfig(
    vocab_size=32_000,
    model_dim=512,
    num_layers=8,
    num_heads=8,
    chunk_size=256,
    cema_ndim=16,
) # 66M params
model = MegalodonForCausalLM(cfg).eval()
print(f"Model has {sum(p.numel() for p in model.parameters()):,} params")

# Dummy input and forward pass using random token ids
input_ids = torch.randint(0, cfg.vocab_size, (1, 128))
logits, caches = model(input_ids=input_ids, use_cache=True)
print(logits.shape)  # (1, 128, vocab_size)
print(len(caches))  # list of per-layer streaming caches
```

### Using a Tokenizer

A copy of the tokenizer lives in [assets/tokenizer](assets/tokenizer). To use the model with text inputs, load the tokenizer first and pass its config info when instantiating a new model.

Then encode text prompts as usual:

```python
import torch
from megalodon import MegalodonConfig, MegalodonForCausalLM
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("assets/tokenizer")
cfg = MegalodonConfig(
    vocab_size=tokenizer.vocab_size,
    pad_token_id=tokenizer.pad_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    model_dim=512,
    num_layers=8,
    num_heads=8,
    chunk_size=256,
    cema_ndim=16,
) # 66M params
model = MegalodonForCausalLM(cfg).eval()
print(f"Model has {sum(p.numel() for p in model.parameters()):,} params")

prompt = "Megalodon brings efficient long-context modeling to PyTorch."
encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    output = model(**encoded)

print(output.logits.shape)  # (1, sequence_length, vocab_size)
decoded = tokenizer.decode(output.logits.argmax(dim=-1)[0])
print(decoded) # random gibberish since model is untrained
```

## Advanced Usage

<details>
<summary><b>Click to Expand</b></summary>

### Gradient Checkpointing & Device Maps

Enabling gradient checkpointing for training works out of the box:

```python
model.gradient_checkpointing_enable()
outputs = model(
    input_ids=input_ids.cuda(),
    labels=input_ids.cuda(),
    use_cache=True,  # automatically disabled while checkpointing
)
loss = outputs.loss
loss.backward()
```

Automatic device mapping with `accelerate` is supported via the model's `_no_split_modules` attribute:

```python
from accelerate.utils import infer_auto_device_map

device_map = infer_auto_device_map(
    model,
    max_memory={0: "12GiB", 1: "12GiB", "cpu": "48GiB"},
    no_split_module_classes=model._no_split_modules,
)
```

### Cache Behavior During Gradient Checkpointing

When gradient checkpointing is enabled, `use_cache` is automatically disabled during
training to keep the autograd graph small:

```python
model.gradient_checkpointing_enable()
outputs = model(input_ids, use_cache=True)

# During training: cache is None (checkpointing wins over caching)
# During eval: cache is returned as normal
```

### Precision Requirements

The reference implementation targets float32 and bfloat16 dtypes. float16 is not
supported because the complex EMA, FFT path, and timestep statistics easily overflow.
If you need reduced precision, move the model to `torch.bfloat16` on Ampere+ GPUs or
modern CPUs.

</details>

## Architecture

### Why This Reimplementation Exists

Megalodon is a unique take on long-context modeling, but [the original repo](https://github.com/XuezheMax/megalodon) couples Python glue with large C++/CUDA extensions and never released trained weights[^1]. That makes it difficult to study the design, prototype, and/or compare vs. new ideas[^2], or integrate with modern HF tooling.

[^1]: at time of repo creation, October 2025. The original repo was released Apr 17, 2024 and does not have weights, [per this issue](https://github.com/XuezheMax/megalodon/issues/1) due to legal review limbo
[^2]: the complexity & lack of weights is a blocker for continued research/improvement on the concept and also leads to [improper comparisons of Megalodon](https://huggingface.co/papers/2510.03279#68ec662e8bfbf816c8335efa) to other techniques. It's hard to compare vs megalodon if you can't train/understand megalodon properly.

### Project Layout

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

<details>
<summary><b>Click to Expand:</b> Implenentation Details, Reference Parity Notes</summary>

### Implementation Details

- Complex EMA in pure Torch with FFT fast path (no cache) and sequential path (streaming)
- Chunked rotary attention with DropKey-style pre-softmax dropout and SDPA fallback
- Z normalisation uses full-vector L2 scaling (Equation 7) before the per-head affine that produces Q/K
- Test-first approach and HF alignment (`_no_split_modules`, weight tying, embeddings accessors)

### Reference Parity Notes

- Complex EMA learns the complex logarithm of the diagonal SSM eigenvalues (`log_q`) directly, matching the CUDA reference while keeping the PyTorch layer simple.
- Numerical precision follows vanilla PyTorch accumulation (no Kahan summation); monitor for instabilities only when pushing to extremely long sequences or very large batches.

</details>

### Limitations

> [!IMPORTANT]
> This repo is intentionally pure PyTorch. Expect slower throughput than the CUDA reference for long sequences, and plan for single-device (CPU or single GPU) workloads.

- PyTorch-focused implementation: no fused CUDA kernels[^3] or the paper's 4D chunk parallelism[^4].
- Complex EMA exposes both a sequential and FFT path; the FFT variant is automatically used during training when cache state is not requested[^5].
- TimestepNorm keeps the numerically exact Welford update in PyTorch. A Triton/CUDA kernel would be required to match the paper's throughput.
- DropKey-style attention dropout and PyTorch's fused SDPA path are wired in, but FlashAttention-2 or other custom kernels are not bundled.

[^3]: This repo does not and **will not** include custom CUDA kernels. The goal is to have a readable, hackable PyTorch implementation for experimentation and understanding. Triton kernels may be considered in the future if they can be made optional and do not complicate the codebase.
[^4]: _yet_.
[^5]: FFT convolution is O(L log L) and faster for training full sequences, but requires computing everything at once (no streaming). Sequential recurrence is O(L) and necessary for streaming inference where we maintain cache state across chunks. The implementation automatically uses FFT when `compute_last_state=False` (training) and sequential when maintaining state (inference).

## Contributing

<details>
<summary><b>Click to Expand:</b> How to Contribute, Run Tests</summary>

1. Fork or clone the repo
2. Create a new branch for your experiment
3. Make changes under [src/megalodon](src/megalodon) or [tests](tests)
4. Run `pytest` (and `pytest -m cuda` if you touched device code) after `pip install -e .[tests]`
5. Open a PR or share your diff

Bug reports and feature proposals are welcome-file an issue describing the scenario, expected behavior, and repro script if possible.

### Running Tests

Run tests after installing the `[tests]` extras:

```bash
pytest                    # CPU + optional accelerate device-map checks
pytest -m cuda            # CUDA smoke (skips if no GPU)
```

Training tests cover:

- Full forward/backward passes with AdamW on CPU & GPU
- Gradient checkpointing compatibility
- `infer_auto_device_map` integration (skips if `accelerate` is missing)

</details>

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
