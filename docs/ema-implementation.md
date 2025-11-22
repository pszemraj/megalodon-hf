# EMA Hidden State: Upstream vs Pure PyTorch

This note summarizes how the original Megalodon implementation computes the EMA hidden state and how this repo mirrors the behavior in pure PyTorch without custom kernels.

## Upstream (CUDA-heavy) Implementation

Source: third_party/upstream-megalodon (see `megalodon/csrc/ops/ema_hidden_kernel.cu`). Key points:

- The EMA hidden state `y[b,d,k] = p[d,k] * Σ_j x[b,d,j] * exp(log_q[d,k]*(L-j-1)) + exp(log_q[d,k]*L) * h[b,d,k]` is computed with specialized CUDA kernels.
- No FFT path is used. Instead, the implementation dispatches by batch size:
  - B=1: a dedicated kernel with shared-memory staging, block reductions, and compile-time unrolling for `N=16`.
  - 2 ≤ B ≤ 8: batched kernels specialized on B (and `N=16` when applicable).
  - B>8: a cuBLAS route that forms a weight matrix `v[d,k,j] = p[d,k] * exp(log_q[d,k]*(L-j-1))` then performs strided batched GEMM `y = x @ v^T`.
- Complex math accumulates in float (`at::acc_type<T, true>`), regardless of input type (bf16, fp16, fp32) for stability.
- Exponentials are precomputed via power tables (`q_pow`, `c_ptr`) to avoid repeated `exp()` calls.

## This Repo (Pure PyTorch)

We provide two equivalent numerical paths:

- FFT convolution (training): builds the EMA kernel over the current sequence and uses FFT-based convolution to compute the output when no cache is requested.
- Sequential recurrence (streaming): updates the EMA state `h_t = q ⊙ h_{t-1} + p ⊙ x_t` one step at a time when cache/streaming is needed.

Rationale for divergence:

- Upstream's "sequential" hidden-state path is fast because it relies on hand-tuned CUDA or cuBLAS. In pure PyTorch, a per-timestep recurrence is much slower. Using FFT for the no-cache case achieves competitive throughput while keeping correctness.
- When cache is needed (streaming inference), correctness requires the stepwise recurrence; we keep a vectorized recurrence with disabled autocast inside the block to protect stability.

## Stability Practices Kept

- Real(log_q) is clamped ≤ -1e-4 to keep |exp(log_q)| < 1.
- EMA accumulates in float32/complex64; autocast is disabled inside EMA paths.
- FFT constructs powers via cumprod rather than raw `exp(log_q * t)` to reduce error.

## Training Default

- Training now disables caches internally so the FFT path is always used in forward/backward. Sequential EMA remains for streaming inference and when callers explicitly request cache.

## Optional Speed-Up for Sequential Path

- `torch.compile` can reduce the Python overhead of the recurrence. Compile the model (or just the `ComplexEMA` module) yourself, for example:

```python
import torch
from megalodon import MegalodonForCausalLM, MegalodonConfig

model = MegalodonForCausalLM(MegalodonConfig()).cuda()
model = torch.compile(model, mode="default")  # PyTorch 2.1+
```

- Ensure you compile before the first forward pass. No additional flags or environment variables are required.
