"""Section 5.1.1: single-node FP32 all-reduce timing with NCCL (or Gloo on CPU).

The launcher runs one torchrun process group per world size, sequentially on the
same node. Each worker owns one GPU. Only all_reduce plus its completion wait is
timed; allocation, input reset, barriers, validation, and gathering are excluded.
"""

import argparse
import csv
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import statistics
import subprocess
import sys
import time


DEFAULT_DATA_SIZES = "1MB,10MB,100MB,1GB"
DEFAULT_WORLD_SIZES = "2,4,6"


def parse_data_sizes(value: str) -> list[tuple[str, int]]:
    """MB/GB are decimal bytes; explicit MiB/GiB are also accepted."""
    units = {"B": 1, "KB": 10**3, "MB": 10**6, "GB": 10**9, "KIB": 2**10, "MIB": 2**20, "GIB": 2**30}
    sizes = []
    for label in value.split(","):
        label = label.strip()
        match = re.fullmatch(r"([1-9][0-9]*)\s*(B|KB|MB|GB|KiB|MiB|GiB)", label, re.IGNORECASE)
        if match is None:
            raise ValueError("data-sizes must look like 1MB,10MB,100MB,1GB (or explicit MiB/GiB)")
        size = int(match[1]) * units[match[2].upper()]
        if size % 4:
            raise ValueError(f"{label}: byte count must be divisible by 4 for float32")
        sizes.append((label, size))
    if len({size for _, size in sizes}) != len(sizes):
        raise ValueError("data-sizes must not contain duplicate byte counts")
    return sizes


def parse_world_sizes(value: str) -> list[int]:
    sizes = [int(part.strip()) for part in value.split(",")]
    if not sizes or min(sizes) < 2 or max(sizes) > 8 or len(set(sizes)) != len(sizes):
        raise ValueError("world-sizes must be distinct integers between 2 and 8")
    return sizes


def summarize_samples(samples_by_rank: list[list[float]], size_bytes: int) -> dict:
    """Use the slowest rank in each iteration as the collective's latency."""
    if not samples_by_rank or not samples_by_rank[0] or len({len(row) for row in samples_by_rank}) != 1:
        raise ValueError("each rank must supply the same nonzero number of samples")
    samples = [max(step) for step in zip(*samples_by_rank, strict=True)]
    if any(not math.isfinite(value) or value <= 0 for row in samples_by_rank for value in row):
        raise ValueError("timing samples must be finite and positive")
    mean = statistics.mean(samples)
    world_size = len(samples_by_rank)
    algorithmic_gbps = size_bytes / (mean / 1000) / 10**9
    return {
        "mean_ms": mean,
        "std_ms": statistics.pstdev(samples),
        "min_ms": min(samples),
        "median_ms": statistics.median(samples),
        "max_ms": max(samples),
        "rank_mean_ms": statistics.mean(value for row in samples_by_rank for value in row),
        "algorithmic_GBps": algorithmic_gbps,
        # Conventional all-reduce bus bandwidth normalization, not a measurement
        # of traffic on a particular physical link or a claim about NCCL's algorithm.
        "bus_GBps": algorithmic_gbps * 2 * (world_size - 1) / world_size,
        "samples_ms": samples,
        "samples_by_rank_ms": samples_by_rank,
    }


def write_json(path: Path, value):
    # Replace atomically so completed sizes survive a later timeout/failure.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def run_worker(args):
    import torch
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if args.backend == "nccl":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    dist.init_process_group(backend=args.backend, timeout=timedelta(seconds=args.timeout_seconds))
    try:
        result = {
            "backend": args.backend,
            "dtype": "float32",
            "world_size": world_size,
            "hostname": socket.gethostname(),
            "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
            "nccl_version": list(torch.cuda.nccl.version()) if args.backend == "nccl" else None,
            "devices": [torch.cuda.get_device_name(i) for i in range(world_size)] if args.backend == "nccl" else ["cpu"] * world_size,
            "warmup_steps": args.warmup_steps,
            "steps": args.steps,
            "rows": [],
        }

        def synchronize():
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        for label, size_bytes in parse_data_sizes(args.data_sizes):
            # Preallocate once per size. In-place SUM mutates this buffer; reset
            # before every call to avoid exponential growth and eventual infs.
            tensor = torch.empty(size_bytes // 4, dtype=torch.float32, device=device)
            samples = []
            for step in range(args.warmup_steps + args.steps):
                tensor.fill_(rank + 1)
                synchronize()
                dist.barrier()
                synchronize()  # The NCCL barrier itself is also asynchronous.
                start = time.perf_counter()
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=False)
                # async_op=False does not imply CUDA work has finished.
                synchronize()
                elapsed_ms = (time.perf_counter() - start) * 1000
                if step >= args.warmup_steps:
                    samples.append(elapsed_ms)

            # Check the last result across the entire buffer, outside timing.
            expected = world_size * (world_size + 1) / 2
            low, high = torch.aminmax(tensor)
            if low.item() != expected or high.item() != expected:
                raise AssertionError(f"rank {rank}: incorrect all-reduce result for {label}; expected {expected}")
            del tensor

            # A small numeric all-gather avoids serializing GPU tensors as Python
            # objects. No reporting communication contributes to the timings.
            local_samples = torch.tensor(samples, dtype=torch.float64, device=device)
            gathered = [torch.empty_like(local_samples) for _ in range(world_size)]
            dist.all_gather(gathered, local_samples)
            if rank == 0:
                timing = summarize_samples([values.cpu().tolist() for values in gathered], size_bytes)
                row = {"world_size": world_size, "data_size": label, "size_bytes": size_bytes, "numel": size_bytes // 4, **timing}
                result["rows"].append(row)
                write_json(args.output_dir / "results.json", result)
                print(f"{world_size} ranks | {label:>6} | {timing['mean_ms']:.4f} +/- {timing['std_ms']:.4f} ms | bus {timing['bus_GBps']:.2f} GB/s", flush=True)
            dist.barrier()
    finally:
        dist.destroy_process_group()


def write_report(directory: Path, rows: list[dict]):
    fields = ["world_size", "data_size", "size_bytes", "numel", "mean_ms", "std_ms", "min_ms", "median_ms", "max_ms", "rank_mean_ms", "algorithmic_GBps", "bus_GBps"]
    with (directory / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "| Processes/GPUs | Data size per rank | Mean (ms) | Std (ms) | Algorithmic GB/s | Bus GB/s |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['world_size']} | {row['data_size']} | {row['mean_ms']:.4f} | {row['std_ms']:.4f} | {row['algorithmic_GBps']:.2f} | {row['bus_GBps']:.2f} |")
    lines += [
        "",
        "Each sample is the maximum latency across ranks for the same iteration; std is the population standard deviation across those samples.",
        "Algorithmic bandwidth = input bytes per rank / mean seconds. Bus bandwidth = algorithmic bandwidth * 2*(p-1)/p.",
        "MB/GB use decimal bytes. Bandwidth units are decimal GB/s. Bus bandwidth is a conventional normalization, not physical link telemetry.",
    ]
    # Factual observations from the measurements; leave causal interpretation to
    # the student rather than inventing explanations for a topology we didn't see.
    if len({row["size_bytes"] for row in rows}) > 1:
        p = min(row["world_size"] for row in rows)
        subset = sorted((row for row in rows if row["world_size"] == p), key=lambda row: row["size_bytes"])
        first, last = subset[0], subset[-1]
        lines += ["", f"At {p} processes, increasing the per-rank tensor from {first['data_size']} to {last['data_size']} changes mean latency from {first['mean_ms']:.4f} to {last['mean_ms']:.4f} ms."]
    if len({row["world_size"] for row in rows}) > 1:
        largest = max(row["size_bytes"] for row in rows)
        subset = sorted((row for row in rows if row["size_bytes"] == largest), key=lambda row: row["world_size"])
        first, last = subset[0], subset[-1]
        lines.append(f"For {first['data_size']} per rank, moving from {first['world_size']} to {last['world_size']} processes changes mean latency from {first['mean_ms']:.4f} to {last['mean_ms']:.4f} ms.")
    (directory / "comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    if not rows:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for metric, ylabel, filename in (("mean_ms", "All-reduce latency (ms)", "latency.png"), ("bus_GBps", "Normalized bus bandwidth (GB/s)", "bandwidth.png")):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for p in sorted({row["world_size"] for row in rows}):
            subset = sorted((row for row in rows if row["world_size"] == p), key=lambda row: row["size_bytes"])
            ax.errorbar(
                [row["size_bytes"] / 10**6 for row in subset],
                [row[metric] for row in subset],
                yerr=[row["std_ms"] for row in subset] if metric == "mean_ms" else None,
                marker="o", capsize=3, label=f"{p} processes",
            )
        ax.set_xscale("log")
        ax.set_xlabel("FP32 tensor size per rank (decimal MB)")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(directory / filename, dpi=160)
        plt.close(fig)


def launch_sweep(args, world_sizes: list[int]):
    if args.backend == "nccl":
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() < max(world_sizes):
            raise ValueError(f"NCCL needs at least {max(world_sizes)} visible GPUs; use --backend gloo for CPU debugging")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "sweep.json").exists():
        raise FileExistsError(f"Results already exist in {args.output_dir}; choose a new output-dir")
    records, rows = [], []
    for p in world_sizes:
        destination = args.output_dir / f"world-{p}"
        destination.mkdir(exist_ok=False)
        command = [
            sys.executable, "-u", "-m", "torch.distributed.run", "--standalone", "--nnodes=1", f"--nproc-per-node={p}",
            "-m", "cs336_systems.distributed_communication_benchmark", "--worker",
            "--backend", args.backend, "--data-sizes", args.data_sizes,
            "--warmup-steps", str(args.warmup_steps), "--steps", str(args.steps),
            "--timeout-seconds", str(args.timeout_seconds), "--output-dir", str(destination.resolve()),
        ]
        print(f"Starting {p} processes on this node; log: {destination / 'stdout.log'}", flush=True)
        start = time.monotonic()
        environment = os.environ.copy()
        environment["OMP_NUM_THREADS"] = "1"
        with (destination / "stdout.log").open("w") as log:
            with subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True) as process:
                try:
                    returncode = process.wait(timeout=args.case_timeout_seconds)
                    status = "success" if returncode == 0 else "failed"
                except subprocess.TimeoutExpired:
                    # Terminate the launcher AND its workers, avoiding stranded
                    # processes occupying GPUs during the next configuration.
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    returncode, status = process.returncode, "timeout"
        records.append({"world_size": p, "status": status, "returncode": returncode, "wall_seconds": time.monotonic() - start, "command": command})
        result_path = destination / "results.json"
        if result_path.exists():
            rows.extend(json.loads(result_path.read_text())["rows"])
        write_json(args.output_dir / "sweep.json", {"backend": args.backend, "dtype": "float32", "warmup_steps": args.warmup_steps, "steps": args.steps, "records": records})
        print((destination / "stdout.log").read_text(), flush=True)
    write_report(args.output_dir, rows)
    if any(record["status"] != "success" for record in records):
        raise RuntimeError("Some process counts failed; available measurements and per-world stdout.log files were preserved")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-sizes", default=DEFAULT_WORLD_SIZES, help="process counts, default: 2,4,6")
    parser.add_argument("--data-sizes", default=DEFAULT_DATA_SIZES, help="per-rank FP32 tensor bytes, default: 1MB,10MB,100MB,1GB")
    parser.add_argument("--backend", choices=("nccl", "gloo"), default="nccl")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--timeout-seconds", type=int, default=120, help="process-group collective timeout")
    parser.add_argument("--case-timeout-seconds", type=int, default=300, help="wall time limit per process count, including all tensor sizes")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        parse_data_sizes(args.data_sizes)
        world_sizes = parse_world_sizes(args.world_sizes)
        if min(args.steps, args.warmup_steps, args.timeout_seconds, args.case_timeout_seconds) < 1:
            raise ValueError("steps, warmup-steps, and timeouts must be positive")
    except ValueError as error:
        parser.error(str(error))
    if args.output_dir is None:
        from cs336_systems.modal_runner import run_name

        args.output_dir = Path("results/distributed_communication") / run_name()
    if args.worker:
        run_worker(args)
    else:
        launch_sweep(args, world_sizes)


if __name__ == "__main__":
    main()
