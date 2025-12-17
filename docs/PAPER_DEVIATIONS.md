# Paper Deviations and Rationale

This repo aims to match the Megalodon paper architecture as closely as possible in pure PyTorch. The items below document the **intentional** or **pragmatic** deviations that remain after the parity work on the `parity-part2` branch.

## Architectural Deviations

- **Pure PyTorch (no fused kernels).** The reference relies on custom CUDA/Triton kernels for sequential CEMA, TimestepNorm, fused attention, and DropKey-before-softmax. This repo uses vanilla PyTorch (FFT EMA for training, sequential EMA for streaming; SDPA or matmul-softmax attention). Expect lower throughput and slightly different numerics on very long sequences.

- **No DropKey masking.** Attention dropout is standard post-softmax dropout. DropKey (pre-softmax masking) is not implemented without a fused kernel.

- **No 4D chunk-parallel axis.** The paper's time-parallel "chunk parallelism" is not implemented. Training is intended for a single device; multi-GPU scaling would require cross-rank exchange of EMA/Norm state and sharded KV.

- **Multi-chunk streaming uses a sliding KV window.** Training attention remains block-diagonal per chunk, as in the paper. Streaming inference can attend over a configurable KV window across recent chunks via `max_cache_len` (default `chunk_size * 4`), or keep KV unbounded with `cache_unbounded=True`. Long-range context is still primarily carried by EMA + TimestepNorm state.

## Parameterization / Stability Tweaks

- **EMA stability clamps.** To prevent eigenvalue drift and blow-ups in pure PyTorch:
  - `log_q.real` is clamped to stay strictly negative (`|q| < 1`).
  - `gamma` is softly clamped to a finite magnitude (`GAMMA_CLAMP_MAX=5.0`).
  - A small variance floor is enforced in TimestepNorm (`VARIANCE_FLOOR=1e-6`).
  These do not change the target equations but improve training stability without fused kernels.

- **Omega residual in CEMA.** The EMA block includes an `omega`-weighted skip connection from MEGA. This is not explicitly shown in Eq. 2 of the paper but is present in the upstream lineage and helps optimization.

- **Learned `log_q` independent of `alpha`.** The paper describes `q = (1-)e^{i}`. This implementation follows the reference parameterization where `log_q` is learned directly (initialized from that form) and is not analytically tied to `` during training.

- **RMS vs L2 normalization for Z.** The paper specifies L2 normalization of the shared `Z`. We use per-head RMS normalization followed by a `1/d` factor in the affine scale. This is mathematically equivalent to L2 normalization while matching the reference kernel.

If you spot any additional divergence, please open an issue or PR with the corresponding paper equation and code pointer.
