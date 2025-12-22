# Megalodon Profiling Playbook

This guide explains how to profile Megalodon's PyTorch implementation for throughput and memory, interpret traces, and compare EMA paths (FFT vs sequential) and precision toggles.

## TL;DR

- Current nano_short timing check (RTX 4070 Laptop, bf16, chunk=512):
  - Megalodon (6L, d=384): ~8m09 for 300 steps; val loss 7.98→2.21→1.81 (steps 0/100/200).
  - Llama baseline (6L, d=384): ~3m46 for 300 steps; val loss 4.58→2.70→2.29.
  - Throughput: Megalodon ~2.2× slower but converges faster at the same step count.

- Use the provided script to capture Chrome traces and summaries:

```bash
conda run -n mega python scripts/profile_ops.py \
  --seq-lens 512 \
  --dtype bf16 \
  --schedule 1 1 2 1
```

- Outputs land under `profile/`:
  - `*/speed_step*.json`: open in Chrome at `chrome://tracing`
  - `*/reports/key_averages_*.txt`: top ops by CUDA time/memory
  - `*/reports/peak_mem_gb.txt`: peak allocated GPU memory
  - `*/reports/ms_per_step.txt`: average step time (ms)
  - `*/ema_*.json`: micro traces for EMA FFT vs sequential paths
  - `summary.csv`: consolidated runs (length, dtype, BF16 reduction, ms/step, peak GB)

## What's Instrumented

The profiling scripts tag a few coarse regions with `torch.profiler.record_function` (there are no in-model tags yet):

- `FORWARD` / `BACKWARD` / `OPTIMIZER`: training step phases in `scripts/profile_ops.py`
- `EMA_FFT_L{len}` / `EMA_SEQ_L{len}`: per-length EMA micro traces in `scripts/profile_ops.py`
- `FORWARD_INFER`: inference-only spans in `scripts/profile_forward.py`

These give step-level visibility; use `key_averages` for op-level detail.

## Usage Patterns

### Speed-focused schedule

Use a short schedule to sanity-check steady-state timing:

```bash
conda run -n mega python scripts/profile_ops.py \
  --seq-lens 512 2048 \
  --dtype bf16 \
  --bf16-sweep \
  --schedule 1 1 1 1
```

### Longer steady-state

To reduce variance in timing:

```bash
conda run -n mega python scripts/profile_ops.py \
  --seq-lens 4096 8192 \
  --dtype bf16 \
  --bf16-sweep \
  --schedule 2 3 3 2
```

### Sequential EMA vs FFT

Training defaults to FFT EMA (no cache) because the sequential recurrence is much slower in pure PyTorch. To profile the sequential path, enable caching in the training loop:

```bash
conda run -n mega python scripts/profile_ops.py \
  --seq-lens 2048 \
  --dtype fp32 \
  --train-use-cache
```

## Precision Knobs

Call this once before model creation if you want to control TF32 and BF16 reduction behavior:

```python
import megalodon

megalodon.configure_precision(
    allow_tf32=True,  # tf32 matmuls on Ampere+ for throughput
    # allow_bf16_reduced_precision_reduction=False,  # pin BF16 GEMMs to full-precision reductions
)
```

The profiler script exposes a BF16 sweep that compares reduced-precision reductions ON vs OFF in cuBLAS. In our runs:

- At L=512, BF16 reductions ON gave ~1.25x better ms/step vs OFF.
- At L=2048 with this config, BF16 reductions had negligible impact; the step was not GEMM-bound in our window.

## Findings and Recommendations

### 1. EMA path selection

- Upstream uses FFT for the forward output and a fused CUDA kernel for the last EMA state. That makes cache updates cheap.
- In pure PyTorch, computing the last EMA state via a sequential recurrence is much slower. We now disable caches during training (always FFT); sequential is reserved for streaming inference.

### 2. Stability and precision

- EMA eigenvalues are stable by construction (`|q| = 1 - alpha * delta`); EMA FFT/sequential computations accumulate in float32/complex64.
- Autocast is disabled inside EMA paths to avoid bf16 drift for complex ops.

### 3. Memory

- FFT zero-padding scales with sequence length; chunked attention keeps memory ~O(B*L), not O(L^2).
- Monitor `peak_mem_gb.txt` when increasing lengths-expect growth with L. No 2× VRAM spikes observed after the stability fixes.

## Interpreting Traces

- EMA path micro traces are stored separately as `ema_fft.json` / `ema_seq.json`. Compare `EMA_FFT_L{len}` vs `EMA_SEQ_L{len}` spans to see the sequential vs FFT path cost.
- In the main training traces, compare `FORWARD`/`BACKWARD`/`OPTIMIZER` durations to see where step time concentrates; use `key_averages` to identify kernel hotspots (attention, matmuls, etc).
- Long CPU-only spans in `key_averages` suggest Python overhead (e.g., Welford updates) and are candidates for kernel fusion.

## CSV Summary

Each run appends/rewrites `profile/summary.csv` with:

```csv
dtype, bf16_reduction, seq_len, batch_size, ms_per_step, peak_mem_gb
```

Use it to compare configurations quickly before diving into traces.
