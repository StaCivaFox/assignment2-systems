"""Command planning and subprocess logging shared by the Modal entrypoints.

This module deliberately has no Modal or torch imports, so plans can be tested locally.
"""

from datetime import UTC, datetime
import csv
import itertools
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import traceback
import uuid
import zipfile


MODEL_SIZES = ("small", "medium", "large", "xl", "10B")
MODES = ("forward", "forward-backward", "train")
ATTENTION_D_MODELS = "16,32,64,128"
ATTENTION_SEQ_LENGTHS = "256,1024,4096,8192,16384"


def archive_source(source: Path, destination: Path):
    # Modal mounts may use Unix-epoch mtimes, earlier than ZIP's 1980 lower bound.
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, strict_timestamps=False) as archive:
        for path in source.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and ".pytest_cache" not in path.parts:
                archive.write(path, path.relative_to(source))


def run_name(value: str = "") -> str:
    value = value or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", value):
        raise ValueError("run-name must be 1–100 letters, digits, underscores, or hyphens")
    return value


def benchmark_jobs(sizes: str, modes: str, warmups: str, steps: int, extra_args: str = "") -> list[dict]:
    selected_sizes = sizes.split(",")
    selected_modes = modes.split(",")
    counts = [int(value.strip()) for value in warmups.split(",")]
    selected_sizes = [value.strip() for value in selected_sizes]
    selected_modes = [value.strip() for value in selected_modes]
    if not set(selected_sizes) <= set(MODEL_SIZES) or not set(selected_modes) <= set(MODES):
        raise ValueError(f"sizes must be drawn from {MODEL_SIZES}; modes from {MODES}")
    if steps <= 0 or min(counts) < 0:
        raise ValueError("steps must be positive and warmups nonnegative")
    if any(len(values) != len(set(values)) for values in (selected_sizes, selected_modes, counts)):
        raise ValueError("sweep lists must not contain duplicates")
    extra = shlex.split(extra_args)
    reserved = {"--model-size", "--mode", "--warmup-steps", "--steps", "--output", "--device"}
    if any(arg.split("=", 1)[0] in reserved for arg in extra):
        raise ValueError(f"extra-args must not override sweep options: {sorted(reserved)}")
    return [
        {
            "name": f"{size}-{mode}-w{warmup}",
            "argv": [
                "python",
                "-m",
                "cs336_systems.benchmark",
                "--model-size",
                size,
                "--mode",
                mode,
                "--warmup-steps",
                str(warmup),
                "--steps",
                str(steps),
                "--device",
                "cuda",
                *extra,
                "--output",
                "{output_dir}/timings.json",
            ],
        }
        for size, mode, warmup in itertools.product(selected_sizes, selected_modes, counts)
    ]


def command_argv(argv: list[str], output_dir: Path, profile: bool = False, nsys_args: str = "") -> list[str]:
    if not argv:
        raise ValueError("command cannot be empty")
    argv = [part.replace("{output_dir}", str(output_dir)) for part in argv]
    # Use the image's locked Python environment for these familiar CLI spellings.
    if argv[0] in ("python", "python3"):
        argv = [sys.executable, "-u", *argv[1:]]
    elif argv[0] in ("pytest", "torchrun"):
        module = "pytest" if argv[0] == "pytest" else "torch.distributed.run"
        argv = [sys.executable, "-u", "-m", module, *argv[1:]]
    if nsys_args and not profile:
        raise ValueError("nsys-args requires profile")
    if profile:
        argv = [
            "nsys",
            "profile",
            "--trace=cuda,nvtx,osrt,cublas,cudnn",
            "--sample=none",
            "--cpuctxsw=none",
            *shlex.split(nsys_args),
            "--output",
            str(output_dir / "trace"),
            *argv,
        ]
    return argv


def distributed_communication_jobs(
    world_sizes: str = "2,4,6",
    data_sizes: str = "1MB,10MB,100MB,1GB",
    warmup_steps: int = 5,
    steps: int = 20,
) -> list[dict]:
    """Keep all world sizes in one invocation so comparisons use the same node."""
    from cs336_systems.distributed_communication_benchmark import parse_data_sizes, parse_world_sizes

    parse_world_sizes(world_sizes)
    parse_data_sizes(data_sizes)
    if min(warmup_steps, steps) < 1:
        raise ValueError("warmup_steps and steps must be positive")
    return [{
        "name": "all-reduce",
        "argv": [
            "python", "-m", "cs336_systems.distributed_communication_benchmark",
            "--backend", "nccl", "--world-sizes", world_sizes, "--data-sizes", data_sizes,
            "--warmup-steps", str(warmup_steps), "--steps", str(steps),
            "--output-dir", "{output_dir}",
        ],
    }]


def ddp_jobs(mode: str = "benchmark", model_size: str = "xl", world_size: int = 2, steps: int = 10, warmup_steps: int = 5, extra_args: str = "", wrapper: str = "naive", allocator_conf: str = "") -> list[dict]:
    """One torchrun invocation containing all ranks on the allocated Modal node."""
    if mode not in ("train", "benchmark") or model_size not in MODEL_SIZES:
        raise ValueError("invalid DDP mode or model size")
    if not 1 <= world_size <= 8 or steps < 1 or warmup_steps < 0 or (mode == "benchmark" and warmup_steps == 0):
        raise ValueError("world_size must be 1–8; steps positive; benchmark requires warm-up")
    if mode == "train" and warmup_steps:
        raise ValueError("training does not discard warm-up updates")
    if wrapper not in ("naive", "flat", "both") or (mode == "train" and wrapper == "both"):
        raise ValueError("wrapper must be naive or flat (benchmark also accepts both)")
    extra = shlex.split(extra_args)
    reserved = {"--model-size", "--steps", "--warmup-steps", "--output-dir", "--backend", "--wrapper"}
    if any(arg.split("=", 1)[0] in reserved for arg in extra):
        raise ValueError(f"extra-args must not override DDP launch options: {sorted(reserved)}")
    wrappers = ("naive", "flat") if wrapper == "both" else (wrapper,)
    return [{"name": f"{variant}-ddp-{mode}", "allocator_conf": allocator_conf, "argv": [
        "torchrun", "--standalone", "--nnodes=1", f"--nproc-per-node={world_size}",
        "-m", f"cs336_systems.ddp_{mode}", "--backend", "nccl", "--model-size", model_size,
        "--wrapper", variant,
        "--steps", str(steps), "--warmup-steps", str(warmup_steps), *extra,
        "--output-dir", "{output_dir}",
    ]} for variant in wrappers]


def attention_jobs(
    d_models: str = ATTENTION_D_MODELS,
    seq_lengths: str = ATTENTION_SEQ_LENGTHS,
    implementations: str = "eager,compiled",
    warmup_steps: int = 5,
    steps: int = 100,
    extra_args: str = "",
) -> list[dict]:
    """One isolated process per attention shape/implementation, including OOM cases."""
    dims = [int(value.strip()) for value in d_models.split(",")]
    lengths = [int(value.strip()) for value in seq_lengths.split(",")]
    variants = [value.strip() for value in implementations.split(",")]
    if min(dims + lengths) <= 0 or steps <= 0 or warmup_steps < 1:
        raise ValueError("dimensions, lengths, steps, and warmup_steps must be positive")
    if not set(variants) <= {"eager", "compiled"}:
        raise ValueError("implementations must be eager, compiled, or eager,compiled")
    if any(len(values) != len(set(values)) for values in (dims, lengths, variants)):
        raise ValueError("sweep lists must not contain duplicates")
    extra = shlex.split(extra_args)
    reserved = {"--d-models", "--seq-lengths", "--implementations", "--warmup-steps", "--steps", "--single-case", "--output", "--output-dir"}
    if any(arg.split("=", 1)[0] in reserved for arg in extra):
        raise ValueError("extra-args must not override attention sweep options")
    return [
        {
            "name": f"attention-{variant}-d{dim}-s{length}",
            "argv": [
                "python",
                "-m",
                "cs336_systems.attention_benchmark",
                "--single-case",
                "--d-models",
                str(dim),
                "--seq-lengths",
                str(length),
                "--implementations",
                variant,
                "--warmup-steps",
                str(warmup_steps),
                "--steps",
                str(steps),
                *extra,
                "--output",
                "{output_dir}/timings.json",
            ],
        }
        for dim, length, variant in itertools.product(dims, lengths, variants)
    ]


def execute_job(job: dict, output_dir: Path, cwd: Path, profile: bool = False, nsys_args: str = "") -> dict:
    """Stream combined stdout/stderr and preserve failure metadata for each process."""
    output_dir.mkdir(parents=True, exist_ok=False)
    command = command_argv(job["argv"], output_dir, profile, nsys_args)
    record = {"name": job["name"], "command": command, "started_at": datetime.now(UTC).isoformat(), "status": "running"}
    metadata = output_dir / "job.json"
    metadata.write_text(json.dumps(record, indent=2) + "\n")
    environment = os.environ.copy()
    # Set allocator configuration before torchrun (and its workers) import torch.
    # Both variants receive identical settings when planned with wrapper=both.
    allocator_conf = job.get("allocator_conf", "")
    if allocator_conf:
        environment["PYTORCH_ALLOC_CONF"] = allocator_conf
        environment.pop("PYTORCH_CUDA_ALLOC_CONF", None)  # Avoid a conflicting legacy alias.
    record["allocator_conf"] = environment.get("PYTORCH_ALLOC_CONF", environment.get("PYTORCH_CUDA_ALLOC_CONF", ""))
    environment.update({"CS336_OUTPUT_DIR": str(output_dir), "PYTHONUNBUFFERED": "1"})
    # Fresh disk caches as well as a fresh process for every benchmark configuration.
    cache_dir = Path("/tmp/cs336-job-cache") / uuid.uuid4().hex
    environment.update({"TRITON_CACHE_DIR": str(cache_dir / "triton"), "TORCHINDUCTOR_CACHE_DIR": str(cache_dir / "inductor")})
    print(f"Running {shlex.join(command)}", flush=True)
    start = time.monotonic()
    try:
        with (output_dir / "stdout.log").open("w") as log:
            with subprocess.Popen(command, cwd=cwd, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
                assert process.stdout is not None
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                record["returncode"] = process.wait()
        record["status"] = "success" if record["returncode"] == 0 else "failed"
    except Exception:
        record.update(status="failed", returncode=-1, error=traceback.format_exc())
        print(record["error"], flush=True)
    finally:
        record["wall_seconds"] = time.monotonic() - start
        metadata.write_text(json.dumps(record, indent=2) + "\n")
    return record


def write_summary(directory: Path, records: list[dict]):
    """Include unsuccessful cases in the table instead of silently dropping them."""
    (directory / "summary.json").write_text(json.dumps(records, indent=2) + "\n")
    fields = [
        "job",
        "status",
        "returncode",
        "phase",
        "mean_ms",
        "std_ms",
        "steps",
        "warmup_steps",
        "device_name",
        "dtype",
        "mixed_precision",
        "context_length",
        "profile_memory",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "compile_model",
        "compile_backend",
        "implementation",
        "d_model",
        "seq_length",
        "batch_size",
        "causal",
        "pre_backward_allocated_bytes",
        "pre_backward_reserved_bytes",
        "forward_live_increment_bytes",
        "failure_phase",
        "world_size",
        "global_batch_size",
        "communication_percent",
        "allocator_conf",
    ]
    with (directory / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            base = {"job": record["name"], "status": record["status"], "returncode": record["returncode"]}
            timing_path = directory / record["name"] / "timings.json"
            if timing_path.exists():
                result = json.loads(timing_path.read_text())
                # Attention OOMs have a result file even when the subprocess failed.
                base.update(status=result.get("status", record["status"]), failure_phase=result.get("failure_phase", ""))
                for phase, timing in (result["timings"] or {"": {}}).items():
                    writer.writerow(
                        base
                        | {
                            "phase": phase,
                            "mean_ms": timing.get("mean_ms", ""),
                            "std_ms": timing.get("std_ms", ""),
                            "steps": result["config"]["steps"],
                            "warmup_steps": result["config"]["warmup_steps"],
                            "device_name": result["device_name"],
                            "world_size": result["config"].get("world_size", ""),
                            "global_batch_size": result["config"].get("global_batch_size", ""),
                            "communication_percent": result.get("communication_percent", ""),
                            "allocator_conf": result["config"].get("allocator_conf", ""),
                            **{
                                key: result["config"].get(key, "")
                                for key in (
                                    "dtype",
                                    "mixed_precision",
                                    "context_length",
                                    "profile_memory",
                                    "compile_model",
                                    "compile_backend",
                                    "implementation",
                                    "d_model",
                                    "seq_length",
                                    "batch_size",
                                    "causal",
                                )
                            },
                            **{
                                key: result.get("memory", {}).get(key, "")
                                for key in (
                                    "peak_allocated_bytes",
                                    "peak_reserved_bytes",
                                    "pre_backward_allocated_bytes",
                                    "pre_backward_reserved_bytes",
                                    "forward_live_increment_bytes",
                                )
                            },
                        }
                    )
            else:
                writer.writerow(base)
    write_attention_comparison(directory, records)


def write_attention_comparison(directory: Path, records: list[dict]):
    """Put eager and compiled measurements side by side; never speed up an OOM."""
    dimensions = ("d_model", "seq_length", "batch_size", "dtype", "causal")
    rows = {}
    for record in records:
        path = directory / record["name"] / "timings.json"
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        config = result["config"]
        variant = config.get("implementation")
        if variant not in ("eager", "compiled"):
            continue
        key = tuple(config.get(name, "") for name in dimensions)
        row = rows.setdefault(key, dict(zip(dimensions, key)))
        status = result.get("status", record["status"])
        row[f"{variant}_status"] = status
        row[f"{variant}_pre_backward_allocated_bytes"] = result.get("memory", {}).get("pre_backward_allocated_bytes", "")
        if status == "success":
            for phase in ("forward", "backward"):
                row[f"{variant}_{phase}_ms"] = result["timings"][phase]["mean_ms"]
    if not rows:
        return
    fields = [*dimensions]
    for variant in ("eager", "compiled"):
        fields.extend(f"{variant}_{suffix}" for suffix in ("status", "forward_ms", "backward_ms", "pre_backward_allocated_bytes"))
    fields.extend(("forward_speedup", "backward_speedup"))
    for row in rows.values():
        for phase in ("forward", "backward"):
            eager, compiled = row.get(f"eager_{phase}_ms"), row.get(f"compiled_{phase}_ms")
            if eager is not None and compiled is not None and compiled > 0:
                row[f"{phase}_speedup"] = eager / compiled
    with (directory / "attention_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows.values())
