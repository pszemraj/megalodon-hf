# Dev Notes & Open Items

This PyTorch-first port intentionally mirrors the Megalodon architecture without the CUDA/Triton kernels from the reference. The biggest functional gap impacts the paper's "unlimited context" claim:

- **Streaming CEMA is slow in pure PyTorch.** When `use_cache=True`, the EMA falls back to the sequential path; the reference uses a fused CUDA kernel to return the last state cheaply. Long-context decoding works but is far slower than the paper. **TODO:** implement a fused/triton sequential CEMA kernel (or an equivalent optimized path) so cached inference scales to the advertised lengths.
- Training defaults to the FFT EMA path (no cache) to avoid the slow sequential recurrence. This is correct for full-sequence training but contributes to the above performance gap for streaming.
- Chunked attention and RoPE are present; the practical limit today is set by the sequential CEMA performance rather than a hard architectural cap.

If you pick up this TODO, document the kernel interface and update `MegalodonModel.use_cache` logic to re-enable cached paths during training-time profiling/benchmarks.

## Known gaps and extensions vs. paper/upstream

```
| Item                                    | Impact (Train/Infer)                                                                 | Effort | Pure PyTorch Possible? |
| --------------------------------------- | ------------------------------------------------------------------------------------ | ------ | ---------------------- |
| Extension: sliding KV horizon (opt-in)  | Train: OK. Infer: chunk-local by default; sliding/unbounded when explicitly enabled. | Done   | Yes                    |
| Cache disabled during training          | Train: seq CEMA/cache untested & slow path unused.                                   | Low-M  | Yes                    |
| Missing chunk-parallel axis             | Train: no time-dim scaling across GPUs. Infer: unaffected.                           | High   | No (needs multi-GPU)   |
| No fused kernels/DropKey-before-softmax | Both: perf/stability below paper (pure PyTorch paths).                               | High   | Partially (slow)       |
```

Guardrails/notes:

- **Training cache path:** `use_cache` is disabled during training to avoid the slow sequential CEMA path. Add an opt-in flag and tests that exercise the cached path to catch regressions and measure the performance hit.
- **Chunk parallelism:** The 4D parallel axis from the paper is not implemented. Adding it requires process-group plumbing plus cross-rank exchange of TimestepNorm/CEMA state and sharded KV. Not needed for single-device learning runs.
- **Fused kernels:** Reference fused attention, DropKey-before-softmax, and sequential CEMA/TimestepNorm kernels are absent. Triton/CUDA implementations (with fallbacks) are needed to approach paper throughput/stability.
- **Inference multi-chunk attention:** Chunk-local by default. Set `max_cache_len` above `chunk_size` for sliding-window attention or `cache_unbounded=True` to disable clamping when VRAM allows.

## Streaming semantics targets (multi-chunk branch)

Scope for the multi-chunk work on this branch (single GPU/CPU, pure Torch):

- **Attention layout:** Keep the block-diagonal chunked attention used in training. Streaming decode is chunk-local by default; optional sliding-window attention uses a configurable cache horizon.
- **RoPE offsets:** Track absolute token positions in the cache so rotary phases advance monotonically even when KV is truncated. Offsets must survive cache eviction.
- **Stateful norms/EMA:** Continue streaming TimestepNorm and CEMA across segments; caches carry their running statistics/hidden state so chunked decoding matches full-sequence results.
- **Cache horizon knob:** `max_cache_len` caps retained KV (defaults to `chunk_size`); set it above `chunk_size` for sliding-window attention or use `cache_unbounded=True` to disable clamping.
- **Training path:** Keep FFT EMA for no-cache training. Provide an opt-in switch to exercise the sequential cached path during tests/benchmarks even if it is slower.
- **Performance caveat:** Without fused kernels, multi-chunk streaming will be correct but slower (2-5x) than the reference; Triton/CUDA kernels can be added later to close the gap.

### Upstream inference limitation (reference repo)

The original CUDA-heavy reference (`third_party/upstream-megalodon`) enforces a single-chunk inference window: in `megalodon/model/mega.py` the forward asserts `cache_len + seq_len <= chunk_size`, and `_InnerAttention` truncates cached KV to the remainder of one chunk. RoPE/masks are built for that one-chunk prefix. This means long prompts beyond one chunk are ignored in upstream streaming decode. This repo keeps that chunk-local behavior by default and offers an opt-in sliding/unbounded KV window as an extension.

### Multi-chunk streaming status (this branch)

- Caches now carry an absolute `position` to keep RoPE offsets continuous across chunks; attention caches are clamped to `max_cache_len` (defaults to `chunk_size`) to bound memory while preserving positions.
- Chunked attention remains block-diagonal (per paper); long-range context flows through EMA/TimestepNorm states and global positions rather than cross-chunk KV attention.
- Training still uses the block-diagonal path; streaming inference is chunk-local by default with optional sliding/unbounded KV when configured. Performance is still limited by the pure-Torch sequential EMA (no fused kernels yet).

## Experimental: Cross-chunk KV attention (sliding window)

This opt-in extension lets attention span recent chunks via a sliding KV cache.

- Enable by setting `max_cache_len` above `chunk_size`, or set `cache_unbounded=True` to keep all KV (VRAM grows linearly).
- This adds cross-chunk attention edges and changes outputs vs. the paper/upstream defaults.

## Numerical alignment with reference

Recent changes to match paper/upstream numerics:

### Q/K normalization (Equations 6-8)

**Status: ALIGNED.** Q/K now use per-head RMSNorm (not L2 norm) before affine transform, matching the reference `FusedRMSNorm(z_head_dim, elementwise_affine=False)`. The "plus-one" gamma reparameterization is preserved.

### CEMA FFT kernel computation

**Status: ALIGNED.** The FFT path computes `q^j` via magnitude/phase powers (no cumprod). This avoids accumulated floating-point errors and stays stable when `|q|` collapses to zero.

### TimestepNorm streaming statistics

**Status: PARTIAL.** Uses Welford-style delta computation but with `torch.cumsum` instead of Kahan-compensated summation. Reference uses fused CUDA kernels with Kahan compensation (`welford.h` + `kahan.h`).

**If precision issues arise on very long sequences:**

1. Change `stats_dtype` from `float32` to `float64` in `TimestepNorm.forward()` (simple, ~2x memory for stats)
2. Implement a fused Triton/CUDA Kahan cumsum kernel (matches reference, no perf penalty)

A pure-Python Kahan cumsum was tested but is ~10x slower due to the loop; not viable without kernel fusion.

### EMA eigenvalue stability

**Status: STABLE.** EMA coefficients are stable by construction:

- `|q| = 1 - sigmoid(alpha) * sigmoid(delta)` (with the phase controlled by `theta`)
- `gamma` is scaled by `sqrt(1/ndim)` as in upstream
- Variance floor: `var_t.clamp_min(1e-6)` in TimestepNorm

### CEMA input phase (Equation 2)

**Status: ALIGNED (with upstream).** Coefficients follow the upstream alpha/delta/theta parameterization: `p = alpha` (real) and `q = (1 - alpha * delta) * exp(i * theta_k)` with uniformly spaced wavelets. The paper's Eq. (2) includes the same phase factor on the input term; this implementation follows upstream for reproducibility.

### Attention value/gate path (Equations 16, 18, 20)

**Status: ALIGNED.** Matches the reference forward pass:

- Values are computed from the attention input (post-TimestepNorm): `v = silu(wv(x_tn))`.
- CEMA output is RMSNormed to `mx = rmsnorm(out_cema)` before `wz`, `wr`, and `wh1`.
- Gate uses `r = silu(wr(mx))`.
- Candidate is the linear merge `h = wh1(mx) + wh2(attn)` (no extra SiLU on the candidate branch).

### Attention logit scaling (Equation 9)

**Status: ALIGNED.** Normalized attention uses `softmax(QK^T) V` (no `/sqrt(d_head)` scaling). SDPA is invoked with `scale=1.0`, and the manual path omits the division.

## Performance benchmarks

### enwik8 training (RTX 5090, bf16 autocast)

Benchmarks run on the `mega_multichunk_512_short.yaml` config (11.3M params, 6 layers, seq_len=512, chunk_size=256, 200 steps with grad_accum=16).

| Version            | Throughput | Final Loss | Notes                                 |
| ------------------ | ---------- | ---------- | ------------------------------------- |
| v0.1.2 (baseline)  | 2.22 it/s  | 1.5783     | Original implementation               |
| v0.2.0 (optimized) | 2.60 it/s  | 1.5789     | +17% throughput, identical loss curve |

Validation loss progression matches between versions:

| Step | v0.1.2 | v0.2.0 |
| ---- | ------ | ------ |
| 0    | 5.7233 | 5.7234 |
| 50   | 2.2120 | 2.2147 |
| 100  | 2.0255 | 2.0278 |
| 150  | 1.8172 | 1.8146 |

### Optimizations applied (v0.2.0)

1. **bf16-safe gamma storage**: ComplexEMA stores gamma as two fp32 tensors (`gamma_real`, `gamma_imag`) instead of complex64, preventing corruption on bf16 casts.

2. **SDPA fast path fix**: Removed incorrect condition that forced slow explicit-mask path during inference. SDPA's `is_causal=True` now used correctly.

3. **Vectorized multi-chunk attention**: Training attention uses a single SDPA call when no padding mask is present (reshapes chunks into batch dimension).

4. **FFT memory optimization**: Kernel construction now chunked (4096 default) to bound intermediate q^j buffers to O(D×N×chunk); the final (D×L) kernel is still materialized.

5. **Real FFT path**: Uses `rfft`/`irfft` instead of `fft`/`ifft` for ~2x memory/compute savings (since input is real and only real output is needed).
