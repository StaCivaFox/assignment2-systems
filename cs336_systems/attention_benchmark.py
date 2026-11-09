"""Sections 4.1.1 and 4.2(a): eager/compiled attention timing and memory sweeps.

Defaults: batch 8, no head dimension, FP32 causal attention, 100 forward/backward
pairs per shape. Each case runs in a fresh subprocess; OOMs do not stop the sweep.
"""

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shlex
import traceback
from timeit import default_timer

import torch

from cs336_basics.model import scaled_dot_product_attention
from cs336_systems.benchmark import summarize, synchronize
from cs336_systems.modal_runner import ATTENTION_D_MODELS, ATTENTION_SEQ_LENGTHS, attention_jobs, execute_job, run_name, write_summary


@dataclass
class AttentionConfig:
    d_model: int = 16
    seq_length: int = 256
    batch_size: int = 8
    implementation: str = "eager"
    warmup_steps: int = 5
    steps: int = 100
    device: str = "cuda"
    dtype: str = "float32"
    causal: bool = True
    seed: int = 0
    compile_backend: str = "inductor"

    def validate(self):
        for name in ("d_model", "seq_length", "batch_size", "warmup_steps", "steps"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive (warm up both forward and backward before timing)")
        if self.implementation not in ("eager", "compiled"):
            raise ValueError("implementation must be eager or compiled")
        if self.dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError("dtype must be float32, float16, or bfloat16")
        device = torch.device(self.device)
        if device.type not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; use --device cpu for a smoke test")


def is_oom(error: BaseException) -> bool:
    """Compilation may wrap the original CUDA OOM in a backend exception."""
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, torch.cuda.OutOfMemoryError):
            return True
        error = error.__cause__ or error.__context__
    return False


def benchmark_attention(config: AttentionConfig) -> dict:
    config.validate()
    device = torch.device(config.device)
    dtype = getattr(torch, config.dtype)
    element_bytes = torch.empty((), dtype=dtype).element_size()
    result = {
        "config": asdict(config),
        "status": "success",
        "torch_version": str(torch.__version__),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "timings": {},
        "memory": {},
        # Sizes of individual mathematical tensors, not an estimate of peak memory.
        "tensor_sizes_bytes": {
            "qkv": 3 * config.batch_size * config.seq_length * config.d_model * element_bytes,
            "one_attention_matrix": config.batch_size * config.seq_length**2 * element_bytes,
            "causal_mask": config.seq_length**2 if config.causal else 0,
        },
    }
    samples = {"forward": [], "backward": []}
    allocated_samples, reserved_samples = [], []
    phase = "initialization"
    try:
        torch.manual_seed(config.seed)
        shape = (config.batch_size, config.seq_length, config.d_model)
        q, k, v = [torch.randn(shape, device=device, dtype=dtype, requires_grad=True) for _ in range(3)]
        # Reuse a preallocated gradient, avoiding a reduction/loss in backward timing.
        upstream = torch.randn(shape, device=device, dtype=dtype)
        mask = torch.ones((config.seq_length, config.seq_length), device=device, dtype=torch.bool).tril() if config.causal else None
        attention = scaled_dot_product_attention
        if config.implementation == "compiled":
            attention = torch.compile(attention, backend=config.compile_backend, fullgraph=True, dynamic=False)

        def run_pair(measured: bool):
            nonlocal phase
            q.grad = k.grad = v.grad = None
            phase = "forward" if measured else "warmup_forward"
            synchronize(device)
            start = default_timer()
            # Autograd stays enabled: these are training forward passes. Each
            # backward consumes a fresh graph, with no retain_graph=True.
            output = attention(q, k, v, mask)
            synchronize(device)
            elapsed = (default_timer() - start) * 1000
            if measured:
                samples["forward"].append(elapsed)
                if device.type == "cuda":
                    allocated_samples.append(torch.cuda.memory_allocated(device))
                    reserved_samples.append(torch.cuda.memory_reserved(device))
            phase = "backward" if measured else "warmup_backward"
            start = default_timer()
            output.backward(upstream)
            synchronize(device)
            elapsed = (default_timer() - start) * 1000
            if measured:
                samples["backward"].append(elapsed)
            # output and its graph leave scope before the next forward is launched.

        for _ in range(config.warmup_steps):
            run_pair(measured=False)
        q.grad = k.grad = v.grad = None
        synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            result["memory"]["baseline_allocated_bytes"] = torch.cuda.memory_allocated(device)
        for _ in range(config.steps):
            run_pair(measured=True)
    except Exception as error:
        result.update(status="oom" if is_oom(error) else "error", failure_phase=phase, error=str(error), traceback=traceback.format_exc())
    finally:
        result["timings"] = {name: summarize(values) for name, values in samples.items() if values}
        if device.type == "cuda":
            result["memory"].update(
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                pre_backward_allocated_samples_bytes=allocated_samples,
                pre_backward_reserved_samples_bytes=reserved_samples,
            )
            if allocated_samples:
                result["memory"].update(
                    pre_backward_allocated_bytes=max(allocated_samples),
                    pre_backward_reserved_bytes=max(reserved_samples),
                    forward_live_increment_bytes=max(allocated_samples) - result["memory"]["baseline_allocated_bytes"],
                )
    return result


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-models", default=ATTENTION_D_MODELS)
    parser.add_argument("--seq-lengths", default=ATTENTION_SEQ_LENGTHS)
    parser.add_argument("--implementations", default="eager,compiled", help="comma-separated eager, compiled, or both")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--causal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compile-backend", default="inductor", help="keep inductor for performance results")
    parser.add_argument("--output-dir", type=Path, help="sweep directory; defaults to results/attention/<unique name>")
    parser.add_argument("--single-case", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    extra = shlex.join(
        [
            "--batch-size",
            str(args.batch_size),
            "--device",
            args.device,
            "--dtype",
            args.dtype,
            "--seed",
            str(args.seed),
            "--compile-backend",
            args.compile_backend,
            "--causal" if args.causal else "--no-causal",
        ]
    )
    try:
        jobs = attention_jobs(args.d_models, args.seq_lengths, args.implementations, args.warmup_steps, args.steps, extra)
        config = AttentionConfig(
            d_model=int(args.d_models.split(",")[0]),
            seq_length=int(args.seq_lengths.split(",")[0]),
            implementation=args.implementations.split(",")[0].strip(),
            batch_size=args.batch_size,
            steps=args.steps,
            warmup_steps=args.warmup_steps,
            device=args.device,
            dtype=args.dtype,
            causal=args.causal,
            seed=args.seed,
            compile_backend=args.compile_backend,
        )
        config.validate()
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    if args.single_case:
        if len(jobs) != 1 or args.output is None:
            parser.error("--single-case needs exactly one shape/implementation and --output")
        result = benchmark_attention(config)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"{config.implementation} | B={config.batch_size}, L={config.seq_length}, D={config.d_model} | {result['status']}")
        for name, timing in result["timings"].items():
            print(f"{name}: {timing['mean_ms']:.3f} ± {timing['std_ms']:.3f} ms ({len(timing['samples_ms'])} samples)")
        if "pre_backward_allocated_bytes" in result["memory"]:
            print(f"Allocated before backward (max): {result['memory']['pre_backward_allocated_bytes'] / 2**30:.3f} GiB")
        if result["status"] != "success":
            print(result["traceback"])
            raise SystemExit(1)
        return
    if args.output is not None:
        parser.error("use --output-dir for a sweep")
    destination = args.output_dir or Path(os.environ.get("CS336_OUTPUT_DIR", "results/attention")) / run_name()
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "plan.json").write_text(json.dumps(jobs, indent=2) + "\n")
    records = []
    for job in jobs:
        records.append(execute_job(job, destination / job["name"], Path(__file__).resolve().parents[1]))
        write_summary(destination, records)
    print(f"Saved sweep results to {destination.resolve()}")
    if any(record["status"] != "success" for record in records):
        raise SystemExit("Some cases failed (including OOMs); see summary.csv and per-case timings.json/stdout.log.")


if __name__ == "__main__":
    main()
