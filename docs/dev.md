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