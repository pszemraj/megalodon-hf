import argparse
import csv
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function, schedule

from megalodon import MegalodonConfig, MegalodonForCausalLM, configure_precision


def build_model(device: torch.device, dtype: torch.dtype) -> MegalodonForCausalLM:
    cfg = MegalodonConfig(
        vocab_size=256,
        model_dim=384,
        num_layers=6,
        num_heads=3,
        z_dim=192,
        value_dim=384,
        ffn_hidden_dim=1024,
        chunk_size=512,
        cema_ndim=8,
        dropout=0.0,
        hidden_dropout=0.0,
        norm_eps=1e-6,
        gradient_checkpointing=False,
    )
    model = MegalodonForCausalLM(cfg).to(device)
    if dtype == torch.bfloat16:
        # Manually cast only real-valued float params/buffers to bf16; keep complex in complex64
        for n, p in model.named_parameters(recurse=True):
            if p.dtype.is_complex:
                continue
            if p.is_floating_point():
                p.data = p.data.to(torch.bfloat16)
        for n, b in model.named_buffers(recurse=True):
            if b.dtype.is_complex:
                continue
            if b.is_floating_point():
                b.data = b.data.to(torch.bfloat16)
    model.train()
    return model


def train_step(model, batch, optimizer, *, use_cache: bool):
    with record_function("FORWARD"):
        out = model(input_ids=batch, labels=batch, use_cache=use_cache)
        loss = out.loss
    with record_function("BACKWARD"):
        loss.backward()
    with record_function("OPTIMIZER"):
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def benchmark_ms_per_step(
    model, batch, optimizer, warmup=5, active=10, *, use_cache: bool
) -> float:
    # Warmup
    for _ in range(warmup):
        out = model(input_ids=batch, labels=batch, use_cache=use_cache)
        out.loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(active):
        out = model(input_ids=batch, labels=batch, use_cache=use_cache)
        out.loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / float(active)


def run_profile_for_len(
    device: torch.device,
    dtype: torch.dtype,
    seq_len: int,
    batch_size: int,
    outdir: Path,
    sched_wait: int,
    sched_warmup: int,
    sched_active: int,
    sched_repeat: int,
    bf16_reduction: str = "auto",
    train_use_cache: bool = False,
):
    # Configure backends per run
    allow_bf16 = None
    if bf16_reduction == "on":
        allow_bf16 = True
    elif bf16_reduction == "off":
        allow_bf16 = False
    configure_precision(
        allow_tf32=True, allow_bf16_reduced_precision_reduction=allow_bf16
    )

    suffix = f"{dtype.__str__().split('.')[-1]}_{bf16_reduction}_L{seq_len}"
    run_dir = outdir / suffix
    ensure_dir(run_dir / "reports")

    torch.manual_seed(0)
    model = build_model(device, dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    batch = torch.randint(
        0, model.config.vocab_size, (batch_size, seq_len), device=device
    )

    # Reset memory stats per run
    torch.cuda.reset_peak_memory_stats(device)

    # Step-time benchmark
    ms = benchmark_ms_per_step(
        model, batch, optimizer, warmup=3, active=6, use_cache=train_use_cache
    )
    with open(run_dir / "reports" / "ms_per_step.txt", "w") as f:
        f.write(f"ms_per_step={ms:.3f}\n")

    # Scheduled profile
    def trace_handler(p):
        p.export_chrome_trace(str(run_dir / f"speed_step{p.step_num}.json"))

    prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(
            wait=sched_wait,
            warmup=sched_warmup,
            active=sched_active,
            repeat=sched_repeat,
        ),
        on_trace_ready=trace_handler,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    )

    total_steps = sched_wait + sched_warmup + (sched_active * sched_repeat)
    with prof:
        for _ in range(total_steps):
            train_step(model, batch, optimizer, use_cache=train_use_cache)
            prof.step()

    # Reports
    with open(run_dir / "reports" / "key_averages_cuda_time.txt", "w") as f:
        f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))
    with open(run_dir / "reports" / "key_averages_mem.txt", "w") as f:
        f.write(
            prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=100)
        )
    peak = torch.cuda.max_memory_allocated(device) / (1024**3)
    with open(run_dir / "reports" / "peak_mem_gb.txt", "w") as f:
        f.write(f"peak_mem_gb={peak:.3f}\n")

    # EMA micro traces (per length)
    for use_fft, tag in ((True, "fft"), (False, "seq")):
        ema_out = run_dir / f"ema_{tag}.json"
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as p2:
            with record_function(f"EMA_{tag.upper()}_L{seq_len}"):
                out = model(input_ids=batch, labels=batch, use_cache=not use_fft)
                out.loss.backward()
        p2.export_chrome_trace(str(ema_out))


def parse_args():
    p = argparse.ArgumentParser(description="Profile Megalodon ops on GPU")
    p.add_argument(
        "--seq-lens",
        nargs="+",
        type=int,
        default=[2048],
        help="Sequence lengths to profile",
    )
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    p.add_argument(
        "--bf16-sweep",
        action="store_true",
        help="If dtype=bf16, run with bf16 reductions on/off",
    )
    p.add_argument(
        "--schedule",
        nargs=4,
        type=int,
        metavar=("WAIT", "WARMUP", "ACTIVE", "REPEAT"),
        default=[1, 2, 2, 1],
    )
    p.add_argument(
        "--train-use-cache",
        action="store_true",
        help="Enable cache during training loop (sequential EMA); default uses FFT path",
    )
    return p.parse_args()


def main():
    assert torch.cuda.is_available(), "CUDA GPU is required for profiling"
    device = torch.device("cuda")

    args = parse_args()
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16

    outdir = Path("profile") / args.dtype
    ensure_dir(outdir)

    wait, warm, active, repeat = args.schedule

    # Runs
    if dtype == torch.bfloat16 and args.bf16_sweep:
        modes = ["on", "off"]
    else:
        modes = ["auto"]

    summary_rows = []
    for L in args.seq_lens:
        for mode in modes:
            run_profile_for_len(
                device=device,
                dtype=dtype,
                seq_len=L,
                batch_size=args.batch_size,
                outdir=outdir,
                sched_wait=wait,
                sched_warmup=warm,
                sched_active=active,
                sched_repeat=repeat,
                bf16_reduction=mode,
                train_use_cache=args.train_use_cache,
            )
            # Collect summary
            suffix = f"{dtype.__str__().split('.')[-1]}_{mode}_L{L}"
            run_dir = outdir / suffix / "reports"
            with open(run_dir / "ms_per_step.txt") as f:
                ms = float(f.read().strip().split("=")[1])
            with open(run_dir / "peak_mem_gb.txt") as f:
                gb = float(f.read().strip().split("=")[1])
            summary_rows.append(
                [
                    dtype.__str__().split(".")[-1],
                    mode,
                    L,
                    args.batch_size,
                    ms,
                    gb,
                ]
            )

    # Write CSV summary
    summary_path = Path("profile") / "summary.csv"
    with open(summary_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "dtype",
                "bf16_reduction",
                "seq_len",
                "batch_size",
                "ms_per_step",
                "peak_mem_gb",
            ]
        )
        writer.writerows(summary_rows)

    print("\nProfiling complete. Artifacts in ./profile:")
    print("- *_L*/speed_step*.json: Chrome traces (chrome://tracing)")
    print("- *_L*/reports/key_averages_*.txt: top ops by time/memory")
    print("- *_L*/reports/peak_mem_gb.txt: per-run peak CUDA memory")
    print("- *_L*/reports/ms_per_step.txt: avg step time (ms)")
    print("- *_L*/ema_*.json: EMA path micro traces")
    print("- summary.csv: consolidated results across runs")


if __name__ == "__main__":
    main()
