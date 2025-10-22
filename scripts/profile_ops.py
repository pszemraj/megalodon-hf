import os
import math
from pathlib import Path

import torch
from torch.profiler import profile, ProfilerActivity, schedule, record_function

from megalodon import MegalodonConfig, MegalodonForCausalLM, configure_precision


def build_model(device: torch.device) -> MegalodonForCausalLM:
    cfg = MegalodonConfig(
        vocab_size=32000,
        model_dim=512,
        num_layers=4,
        num_heads=8,
        z_dim=1024,
        value_dim=1024,
        ffn_hidden_dim=2048,
        chunk_size=256,
        cema_ndim=16,
        dropout=0.0,
        hidden_dropout=0.0,
        norm_eps=1e-6,
        gradient_checkpointing=False,
    )
    model = MegalodonForCausalLM(cfg).to(device)
    model.train()
    return model


def train_step(model, batch, optimizer):
    with record_function("FORWARD"):
        out = model(input_ids=batch, labels=batch)
        loss = out.loss
    with record_function("BACKWARD"):
        loss.backward()
    with record_function("OPTIMIZER"):
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def main():
    assert torch.cuda.is_available(), "CUDA GPU is required for profiling"
    device = torch.device("cuda")

    # Backend precision knobs
    configure_precision(allow_tf32=True)

    torch.manual_seed(0)
    model = build_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    B, L = 2, 2048
    batch = torch.randint(0, model.config.vocab_size, (B, L), device=device)

    outdir = Path("profile")
    ensure_dir(outdir)

    # Clean memory stats
    torch.cuda.reset_peak_memory_stats(device)

    # Speed-focused profile
    trace_handler = lambda p: p.export_chrome_trace(
        str(outdir / f"speed_step{p.step_num}.json")
    )

    prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=2, warmup=3, active=3, repeat=2),
        on_trace_ready=trace_handler,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    )

    steps_needed = 2 + 3 + (3 * 2)
    with prof:
        for step in range(steps_needed):
            train_step(model, batch, optimizer)
            prof.step()

    # Emit summary tables
    (outdir / "reports").mkdir(exist_ok=True)
    with open(outdir / "reports" / "key_averages_cuda_time.txt", "w") as f:
        f.write(
            prof.key_averages().table(sort_by="cuda_time_total", row_limit=100)
        )
    with open(outdir / "reports" / "key_averages_mem.txt", "w") as f:
        f.write(
            prof.key_averages().table(
                sort_by="self_cuda_memory_usage", row_limit=100
            )
        )

    # Peak memory
    peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    with open(outdir / "reports" / "peak_mem_gb.txt", "w") as f:
        f.write(f"peak_mem_gb={peak:.3f}\n")

    # Head-to-head EMA path microbench
    for use_fft, tag in ((True, "fft"), (False, "seq")):
        ema_out = outdir / f"ema_{tag}_L{L}.json"
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as p2:
            with record_function(f"EMA_{tag.upper()}_L{L}"):
                # Drive the path by toggling use_cache (seq path needs cache)
                out = model(input_ids=batch, labels=batch, use_cache=not use_fft)
                out.loss.backward()
        p2.export_chrome_trace(str(ema_out))

    print("\nProfiling complete. Artifacts in ./profile:")
    print("- speed_step*.json: Chrome traces (chrome://tracing)")
    print("- reports/key_averages_*.txt: top ops by time/memory")
    print("- peak_mem_gb.txt: peak CUDA memory during run")
    print("- ema_fft_*.json / ema_seq_*.json: EMA path micro traces")


if __name__ == "__main__":
    main()

