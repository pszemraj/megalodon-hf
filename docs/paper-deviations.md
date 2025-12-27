# Paper Deviations and Rationale

This repo aims to match the Megalodon paper architecture as closely as possible in pure PyTorch. The items below document the **intentional** or **pragmatic** deviations that remain after the parity work on the `parity-part2` branch.

## Architectural Deviations

- **Pure PyTorch (no fused kernels).** The reference relies on custom CUDA/Triton kernels for sequential CEMA, TimestepNorm, fused attention, and DropKey-before-softmax. This repo uses vanilla PyTorch (FFT EMA for training, sequential EMA for streaming; SDPA or matmul-softmax attention). Expect lower throughput and slightly different numerics on very long sequences.

- **No DropKey masking.** Attention dropout is standard post-softmax dropout. DropKey (pre-softmax masking) is not implemented without a fused kernel.

- **No 4D chunk-parallel axis.** The paper's time-parallel "chunk parallelism" is not implemented. Training is intended for a single device; multi-GPU scaling would require cross-rank exchange of EMA/Norm state and sharded KV.

- **Optional sliding/unbounded KV window (opt-in).** Training attention remains block-diagonal per chunk, as in the paper. Streaming inference is chunk-local by default (`max_cache_len = chunk_size`). Set `max_cache_len` above `chunk_size` or `cache_unbounded=True` to enable cross-chunk KV attention; long-range context is still primarily carried by EMA + TimestepNorm state.

## Parameterization / Stability Tweaks

- **TimestepNorm variance floor.** A small variance floor is enforced in TimestepNorm (`VARIANCE_FLOOR=1e-6`) to prevent division instability in early training steps.

- **Omega residual in CEMA.** The EMA block includes an `omega`-weighted skip connection from MEGA. This is not explicitly shown in Eq. 2 of the paper but is present in the upstream lineage and helps optimization.

- **CEMA input phase (paper vs upstream).** The paper's Eq. (2) applies the complex phase `(cos θ + i sin θ)` to both the input and recurrence terms. Upstream uses a real input coefficient `p = alpha` and encodes phase only in `q`; this implementation follows upstream for reproducibility.

- **RMS vs L2 normalization for Z.** The paper specifies L2 normalization of the shared `Z`. We use per-head RMS normalization followed by a `1/sqrt(d_head)` factor in the affine scale. This is mathematically equivalent to L2 normalization while matching the reference kernel.

If you spot any additional divergence, please open an issue or PR with the corresponding paper equation and code pointer.
