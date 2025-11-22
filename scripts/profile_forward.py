from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function, schedule

from megalodon import MegalodonConfig, MegalodonForCausalLM


def main() -> None:
    assert torch.cuda.is_available(), "CUDA GPU required"
    device = torch.device("cuda")

    cfg = MegalodonConfig(
        vocab_size=32_000,
        model_dim=512,
        num_layers=4,
        num_heads=8,
        z_dim=1_024,
        value_dim=1_024,
        ffn_hidden_dim=2_048,
        chunk_size=256,
        cema_ndim=16,
    )

    model = MegalodonForCausalLM(cfg).to(device)
    model.eval()

    batch, seq_len = 2, 4_096
    inputs = torch.randint(0, cfg.vocab_size, (batch, seq_len), device=device)

    outdir = Path("profile") / "forward"
    outdir.mkdir(parents=True, exist_ok=True)

    sched = schedule(wait=1, warmup=1, active=5, repeat=1)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=sched,
        record_shapes=True,
        profile_memory=True,
        on_trace_ready=lambda prof: prof.export_chrome_trace(
            str(outdir / f"forward_trace_step{prof.step_num}.json")
        ),
    ) as prof:
        for _ in range(7):
            with torch.no_grad():
                with record_function("FORWARD_INFER"):
                    model(input_ids=inputs, use_cache=True)
            prof.step()

    (outdir / "reports").mkdir(exist_ok=True)
    with open(outdir / "reports" / "key_averages.txt", "w") as fh:
        fh.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))

    with torch.no_grad():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        rounds = 6
        start.record()
        for _ in range(rounds):
            model(input_ids=inputs, use_cache=True)
        end.record()
        torch.cuda.synchronize()
        latency_ms = start.elapsed_time(end) / rounds

    with open(outdir / "reports" / "latency_ms.txt", "w") as fh:
        fh.write(f"avg_forward_ms={latency_ms:.3f}\n")


if __name__ == "__main__":
    main()
