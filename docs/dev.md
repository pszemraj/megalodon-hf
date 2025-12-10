# Dev Notes & Open Items

This PyTorch-first port intentionally mirrors the Megalodon architecture without the CUDA/Triton kernels from the reference. The biggest functional gap impacts the paper's "unlimited context" claim:

- **Streaming CEMA is slow in pure PyTorch.** When `use_cache=True`, the EMA falls back to the sequential path; the reference uses a fused CUDA kernel to return the last state cheaply. Long-context decoding works but is far slower than the paper. **TODO:** implement a fused/triton sequential CEMA kernel (or an equivalent optimized path) so cached inference scales to the advertised lengths.
- Training defaults to the FFT EMA path (no cache) to avoid the slow sequential recurrence. This is correct for full-sequence training but contributes to the above performance gap for streaming.
- Chunked attention and RoPE are present; the practical limit today is set by the sequential CEMA performance rather than a hard architectural cap.

If you pick up this TODO, document the kernel interface and update `MegalodonModel.use_cache` logic to re-enable cached paths during training-time profiling/benchmarks.

## Known gaps vs. paper/upstream

```
| Issue                                        | Impact (Train/Infer)                                       | Effort | Pure PyTorch Possible? |
| -------------------------------------------- | ---------------------------------------------------------- | ------ | ---------------------- |
| Single-chunk cache (drops KV beyond 1 chunk) | Train: OK. Infer: wrong for long prompts; only last chunk. | Medium | Yes                    |
| Cache disabled during training               | Train: seq CEMA/cache untested & slow path unused.         | Low-M  | Yes                    |
| Missing chunk-parallel axis                  | Train: no time-dim scaling across GPUs. Infer: unaffected. | High   | No (needs multi-GPU)   |
| No fused kernels/DropKey-before-softmax      | Both: perf/stability below paper (pure PyTorch paths).     | High   | Partially (slow)       |
| No multi-chunk attention at inference        | Train: N/A. Infer: can't attend across cached chunks.      | Medium | Yes                    |
```

Guardrails/notes:

- **Cache truncation:** Streaming decode only retains the most recent chunk of KV. To enable long-context inference, redesign the cache to store multiple chunks (or summaries) and extend causal masking/RoPE offsets accordingly.
- **Training cache path:** `use_cache` is disabled during training to avoid the slow sequential CEMA path. Add an opt-in flag and tests that exercise the cached path to catch regressions and measure the performance hit.
- **Chunk parallelism:** The 4D parallel axis from the paper is not implemented. Adding it requires process-group plumbing plus cross-rank exchange of TimestepNorm/CEMA state and sharded KV. Not needed for single-device learning runs.
- **Fused kernels:** Reference fused attention, DropKey-before-softmax, and sequential CEMA/TimestepNorm kernels are absent. Triton/CUDA implementations (with fallbacks) are needed to approach paper throughput/stability.
- **Inference multi-chunk attention:** Even with padding/trim support, decoding cannot attend beyond a single cached chunk. Fixing this is necessary to match the "unlimited context" behavior.

## Streaming semantics targets (multi-chunk branch)

Scope for the multi-chunk work on this branch (single GPU/CPU, pure Torch):

- **Attention layout:** Keep the block-diagonal chunked attention used in training. For streaming decode, allow attending over a configurable cache horizon composed of the most recent chunks (not just the tail of one chunk).
- **RoPE offsets:** Track absolute token positions in the cache so rotary phases advance monotonically even when KV is truncated. Offsets must survive cache eviction.
- **Stateful norms/EMA:** Continue streaming TimestepNorm and CEMA across segments; caches carry their running statistics/hidden state so chunked decoding matches full-sequence results.
- **Cache horizon knob:** Add a `max_cache_len` (or similar) to cap how many tokens of KV we retain; older KV are dropped but the absolute position counter is preserved.
- **Training path:** Keep FFT EMA for no-cache training. Provide an opt-in switch to exercise the sequential cached path during tests/benchmarks even if it is slower.
- **Performance caveat:** Without fused kernels, multi-chunk streaming will be correct but slower (2-5x) than the reference; Triton/CUDA kernels can be added later to close the gap.

### Upstream inference limitation (reference repo)

The original CUDA-heavy reference (`third_party/upstream-megalodon`) enforces a single-chunk inference window: in `megalodon/model/mega.py` the forward asserts `cache_len + seq_len <= chunk_size`, and `_InnerAttention` truncates cached KV to the remainder of one chunk. RoPE/masks are built for that one-chunk prefix. This means long prompts beyond one chunk are ignored in upstream streaming decode. Our goal on this branch is to exceed that limitation by supporting multi-chunk/windowed streaming with correct RoPE offsets and causal masking.

### Multi-chunk streaming status (this branch)

- Caches now carry an absolute `position` to keep RoPE offsets continuous across chunks; attention caches are clamped to `max_cache_len` to bound memory while preserving positions.
- Chunked attention remains block-diagonal (per paper); long-range context flows through EMA/TimestepNorm states and global positions rather than cross-chunk KV attention.
- Training still uses the block-diagonal path; streaming inference uses a sliding window cache with global positions. Performance is still limited by the pure-Torch sequential EMA (no fused kernels yet).

## Numerical alignment with reference

Recent changes to match paper/upstream numerics:

### Q/K normalization (Equations 6-8)

**Status: ALIGNED.** Q/K now use per-head RMSNorm (not L2 norm) before affine transform, matching the reference `FusedRMSNorm(z_head_dim, elementwise_affine=False)`. The "plus-one" gamma reparameterization is preserved.

### CEMA FFT kernel computation

**Status: ALIGNED.** The FFT path now computes `q^j = exp(log_q * j)` directly instead of using `torch.cumprod`. This avoids accumulated floating-point errors over long sequences, matching the reference kernel's approach.

### TimestepNorm streaming statistics

**Status: PARTIAL.** Uses Welford-style delta computation but with `torch.cumsum` instead of Kahan-compensated summation. Reference uses fused CUDA kernels with Kahan compensation (`welford.h` + `kahan.h`).

**If precision issues arise on very long sequences:**
1. Change `stats_dtype` from `float32` to `float64` in `TimestepNorm.forward()` (simple, ~2x memory for stats)
2. Implement a fused Triton/CUDA Kahan cumsum kernel (matches reference, no perf penalty)

A pure-Python Kahan cumsum was tested but is ~10x slower due to the loop; not viable without kernel fusion.

### EMA eigenvalue stability

**Status: STABLE.** Multiple measures prevent training collapse:
- Forward clamp: `log_q.real.clamp(max=-1e-6)` ensures `|q| < 1` (relaxed from -1e-4 since exp(log_q*j) doesn't accumulate errors)
- Post-optimizer projection: `model.project_ema_parameters()` prevents gradient drift
- Gamma soft clamp: `gamma / (1 + |gamma|/5)` bounds output magnitude
- Variance floor: `var_t.clamp_min(1e-6)` in TimestepNorm

### CEMA input phase (Equation 2)

**Status: ALIGNED.** The input coefficient now applies the same complex phase as the recurrence (`alpha * exp(i * theta)` derived from `log_q.imag)`), matching Eq. 2's rotated input term instead of injecting a purely real impulse.

### Attention value/gate path (Equations 16, 18, 20)

**Status: ALIGNED.** Values are computed from the block input `X` (pre-TimestepNorm residual path) per Eq. 16; the gate and candidate paths consume the raw CEMA output without an extra RMSNorm, and the candidate projection is wrapped in SiLU to follow Eq. 20.
