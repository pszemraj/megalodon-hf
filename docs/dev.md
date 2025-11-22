# Dev Notes & Open Items

This PyTorch-first port intentionally mirrors the Megalodon architecture without the CUDA/Triton kernels from the reference. The biggest functional gap impacts the paper's "unlimited context" claim:

- **Streaming CEMA is slow in pure PyTorch.** When `use_cache=True`, the EMA falls back to the sequential path; the reference uses a fused CUDA kernel to return the last state cheaply. Long-context decoding works but is far slower than the paper. **TODO:** implement a fused/triton sequential CEMA kernel (or an equivalent optimized path) so cached inference scales to the advertised lengths.
- Training defaults to the FFT EMA path (no cache) to avoid the slow sequential recurrence. This is correct for full-sequence training but contributes to the above performance gap for streaming.
- Chunked attention and RoPE are present; the practical limit today is set by the sequential CEMA performance rather than a hard architectural cap.

If you pick up this TODO, document the kernel interface and update `MegalodonModel.use_cache` logic to re-enable cached paths during training-time profiling/benchmarks.
