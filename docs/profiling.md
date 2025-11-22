# Megalodon Profiling Playbook

This guide explains how to profile Megalodon's PyTorch implementation for throughput and memory, interpret traces, and compare EMA paths (FFT vs sequential) and precision toggles.

## TL;DR

- Current nano_short timing check (RTX 4070 Laptop, bf16, chunk=512):
  - Megalodon (6L, d=384): ~8m09 for 300 steps; val loss 7.98→2.21→1.81 (steps 0/100/200).
  - Llama baseline (6L, d=384): ~3m46 for 300 steps; val loss 4.58→2.70→2.29.
  - Throughput: Megalodon ~2.2× slower but converges faster at the same step count.

- Use the provided script to capture Chrome traces and summaries:

```bash
conda run -n dl python scripts/profile_ops.py \
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

The model tags key regions with `torch.profiler.record_function` so they show up as labeled blocks in traces:

- `TIMENORM`: streaming Welford stats + normalization
- `CEMA_FFT` / `CEMA_SEQ`: complex EMA convolution (FFT fast path) vs sequential recurrence (streaming)
- `RMSNORM`: RMSNorm + dropout
- `ATTN_PROJ`: Z projection + L2 norm + Q/K affine/split
- `INNER_ATTN`: chunked self-attention block
- `ATTN_GATE`: gating and output projections

This makes hotspots obvious within a few minutes of trace analysis.

## Usage Patterns

### Speed-focused schedule

Use a short schedule to sanity-check steady-state timing:

```bash
conda run -n inf python scripts/profile_ops.py \
  --seq-lens 512 2048 \
  --dtype bf16 \
  --bf16-sweep \
  --schedule 1 1 1 1
```

### Longer steady-state

To reduce variance in timing:

```bash
conda run -n inf python scripts/profile_ops.py \
  --seq-lens 4096 8192 \
  --dtype bf16 \
  --bf16-sweep \
  --schedule 2 3 3 2
```

### Sequential EMA vs FFT

Training defaults to FFT EMA (no cache) because the sequential recurrence is much slower in pure PyTorch. To profile the sequential path, enable caching in the training loop:

```bash
conda run -n inf python scripts/profile_ops.py \
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

1) EMA path selection

- Upstream uses FFT for the forward output and a fused CUDA kernel for the last EMA state. That makes cache updates cheap.
- In pure PyTorch, computing the last EMA state via a sequential recurrence is much slower. We now disable caches during training (always FFT); sequential is reserved for streaming inference.

2) Stability and precision

- EMA eigenvalues are projected to keep `|exp(log_q)| < 1`.
- EMA FFT/sequential computations accumulate in float32/complex64.
- Autocast is disabled inside EMA paths to avoid bf16 drift for complex ops.

3) Memory

- FFT zero-padding scales with sequence length; chunked attention keeps memory ~O(B*L), not O(L^2).
- Monitor `peak_mem_gb.txt` when increasing lengths-expect growth with L. No 2× VRAM spikes observed after the stability fixes.

## Interpreting Traces

- Look for thick `CEMA_FFT` or `CEMA_SEQ` blocks. If `CEMA_SEQ` dominates, training likely ran with cache ON. For speed, keep cache OFF during training.
- `INNER_ATTN` should be compute-bound at moderate L; if it's slow, check SDPA fallback and dropout overhead.
- Long CPU-only spans suggest Python overhead (e.g., Welford updates) and are candidates for kernel fusion.

## CSV Summary

Each run appends/rewrites `profile/summary.csv` with:

```csv
dtype, bf16_reduction, seq_len, batch_size, ms_per_step, peak_mem_gb
```

Use it to compare configurations quickly before diving into traces.
