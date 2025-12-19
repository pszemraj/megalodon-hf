# EMA Hidden State: Upstream vs Pure PyTorch

This note summarizes how the original Megalodon implementation computes the EMA hidden state and how this repo mirrors the behavior in pure PyTorch without custom kernels.

## Upstream (CUDA-heavy) Implementation

Source: third_party/upstream-megalodon. Key points:

- The output is computed as an FFT-based convolution: coefficients are produced by `ema_parameters(p, q, gamma, hx, length)`, then applied via `fftconv(x, k)` (with CUDA fast paths in a length window).
- When `hx` is provided, `ema_parameters` returns an additional bias term `b` that accounts for the contribution of the initial hidden state to the outputs.
- The final hidden state for streaming (`hx_next`) is produced by a fused kernel (`ema_hidden`), which avoids an explicit Python loop over timesteps.

## This Repo (Pure PyTorch)

We provide two equivalent numerical paths:

- FFT convolution (training): builds the EMA kernel over the current sequence and uses FFT-based convolution to compute the output when no cache is requested.
- Sequential recurrence (streaming): updates the EMA state `h_t = q ⊙ h_{t-1} + p ⊙ x_t` one step at a time when cache/streaming is needed.

Rationale for divergence:

- Upstream's "sequential" hidden-state path is fast because it relies on hand-tuned CUDA or cuBLAS. In pure PyTorch, a per-timestep recurrence is much slower. Using FFT for the no-cache case achieves competitive throughput while keeping correctness.
- When cache is needed (streaming inference), correctness requires the stepwise recurrence; we keep a vectorized recurrence with disabled autocast inside the block to protect stability.

## Stability Practices Kept

- Coefficients follow upstream alpha/delta/theta parameterization: `|q| = 1 - alpha * delta` stays inside the unit circle by construction.
- EMA accumulates in float32/complex64; autocast is disabled inside EMA paths.
- FFT constructs powers via magnitude/phase (`|q|^t * exp(i * phi * t)`) to avoid cumprod error accumulation and `log(0)` NaNs.

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
