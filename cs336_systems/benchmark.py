"""Transformer timing, precision, memory profiling, and compilation (sections 2 and 4).

Run with ``uv run python -m cs336_systems.benchmark --help``.
"""

import argparse
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import statistics
import sys
from timeit import default_timer

import torch
import torch.cuda.nvtx as nvtx
from einops import einsum

import cs336_basics.model as basics_model
from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW


# (d_model, d_ff, num_layers, num_heads), from Table 1 of the handout.
MODEL_SIZES = {
    "small": (768, 3072, 12, 12),
    "medium": (1024, 4096, 24, 16),
    "large": (1280, 5120, 36, 20),
    "xl": (2560, 10240, 32, 32),
    "10B": (4608, 12288, 50, 36),
}
MODES = ("forward", "forward-backward", "train")


def nvtx_range(name: str, device: torch.device):
    """Keep CPU smoke tests usable even with a CPU-only PyTorch installation."""
    return nvtx.range(name) if device.type == "cuda" else nullcontext()


def annotated_scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None):
    """The basics attention computation with section 2.1.4's nested NVTX ranges.

    Ranges label CPU launches; they deliberately do not synchronize the GPU.
    The original softmax is reused so profiling preserves the benchmarked work.
    """
    with nvtx_range("scaled dot product attention", Q.device):
        with nvtx_range("computing attention scores", Q.device):
            d_k = K.shape[-1]
            attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)
            if mask is not None:
                attention_scores = attention_scores.masked_fill(~mask, float("-inf"))

        with nvtx_range("computing softmax", Q.device):
            attention_weights = basics_model.softmax(attention_scores, dim=-1)

        with nvtx_range("final matmul", Q.device):
            return einsum(attention_weights, V, "... query key, ... key d_v -> ... query d_v")


@contextmanager
def attention_annotations(enabled: bool):
    """Temporarily swap the module-global function used by the basics model."""
    original = basics_model.scaled_dot_product_attention
    if enabled:
        basics_model.scaled_dot_product_attention = annotated_scaled_dot_product_attention
    try:
        yield
    finally:
        if enabled:
            basics_model.scaled_dot_product_attention = original


@dataclass
class BenchmarkConfig:
    d_model: int = 768
    d_ff: int = 3072
    num_layers: int = 12
    num_heads: int = 12
    vocab_size: int = 10_000
    context_length: int = 512
    batch_size: int = 4
    rope_theta: float = 10_000.0
    mode: str = "forward"
    warmup_steps: int = 5
    steps: int = 10
    device: str = "cuda"
    dtype: str = "float32"
    seed: int = 0
    learning_rate: float = 1e-3
    time_components: bool = False
    annotate_attention: bool = False
    mixed_precision: bool = False
    profile_memory: bool = False
    memory_snapshot: str = "memory_snapshot.pickle"
    memory_max_entries: int = 1_000_000
    compile_model: bool = False
    compile_backend: str = "inductor"

    def validate(self):
        for name in ("d_model", "d_ff", "num_layers", "num_heads", "vocab_size", "context_length", "batch_size", "steps"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be nonnegative")
        if self.compile_model and self.warmup_steps == 0:
            raise ValueError("--compile requires at least one warm-up step to exclude forward/backward compilation")
        if self.d_model % self.num_heads or (self.d_model // self.num_heads) % 2:
            raise ValueError("d_model must be divisible by num_heads, with an even head dimension for RoPE")
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if self.dtype not in ("float32", "bfloat16", "float16"):
            raise ValueError("dtype must be float32, bfloat16, or float16")
        if self.mixed_precision and self.dtype != "float32":
            raise ValueError("--mixed-precision requires --dtype float32: parameters stay FP32, autocast uses BF16")
        if self.memory_max_entries <= 0:
            raise ValueError("memory_max_entries must be positive")
        if not self.rope_theta > 0 or not self.learning_rate > 0:
            raise ValueError("rope_theta and learning_rate must be positive")
        device = torch.device(self.device)
        if device.type not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda (optionally cuda:N)")
        if self.profile_memory and device.type != "cuda":
            raise ValueError("--profile-memory requires a CUDA device")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; use --device cpu for a CPU smoke test")


def summarize(samples: list[float]) -> dict:
    """Population standard deviation of the recorded samples; all times in ms."""
    return {
        "samples_ms": samples,
        "mean_ms": statistics.mean(samples),
        "std_ms": statistics.pstdev(samples),
    }


def synchronize(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@contextmanager
def memory_profile(config: BenchmarkConfig, device: torch.device):
    """Cover setup and warm-up OOMs, then reset history for measured steps."""
    stats = {}
    if not config.profile_memory:
        yield stats
        return
    snapshot = Path(config.memory_snapshot)
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    # Select the requested GPU for allocator APIs, including cuda:N runs.
    with torch.cuda.device(device):
        synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        stats.update(
            snapshot=str(snapshot),
            phase="initialization",
            baseline_allocated_bytes=torch.cuda.memory_allocated(device),
            baseline_reserved_bytes=torch.cuda.memory_reserved(device),
        )
        torch.cuda.memory._record_memory_history(max_entries=config.memory_max_entries)
        try:
            yield stats
        finally:
            original_error = sys.exception()
            stats["status"] = "success" if original_error is None else "oom" if isinstance(original_error, torch.cuda.OutOfMemoryError) else "error"
            if original_error is not None:
                stats["error"] = str(original_error)
            save_errors = []

            # Attempt each diagnostic independently, without synchronizing or
            # emptying the allocator cache after an OOM. Preserve the original
            # exception even if saving a snapshot or stopping history also fails.
            def attempt(action):
                try:
                    action()
                except Exception as error:
                    save_errors.append(error)
                    print(f"Memory profiling cleanup failed: {error}", file=sys.stderr, flush=True)

            try:
                attempt(
                    lambda: stats.update(
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                        end_allocated_bytes=torch.cuda.memory_allocated(device),
                        end_reserved_bytes=torch.cuda.memory_reserved(device),
                    )
                )
                attempt(lambda: torch.cuda.memory._dump_snapshot(str(snapshot)))
                # Also retain peak statistics if the benchmark fails before timings.json.
                attempt(lambda: snapshot.with_suffix(".json").write_text(json.dumps({"config": asdict(config), "memory": stats}, indent=2) + "\n"))
            finally:
                attempt(lambda: torch.cuda.memory._record_memory_history(enabled=None))
            if save_errors and original_error is None:
                raise save_errors[0]


def reset_memory_measurement(config: BenchmarkConfig, device: torch.device, stats: dict):
    """Discard successful setup/warm-up history and reset the measured baseline."""
    synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    stats.update(
        phase="measurement",
        baseline_allocated_bytes=torch.cuda.memory_allocated(device),
        baseline_reserved_bytes=torch.cuda.memory_reserved(device),
    )
    torch.cuda.memory._record_memory_history(max_entries=config.memory_max_entries, clear_history=True)


def benchmark(config: BenchmarkConfig) -> dict:
    config.validate()
    device = torch.device(config.device)
    with memory_profile(config, device) as memory:
        result = _benchmark(config, device, memory)
    return result


def _benchmark(config: BenchmarkConfig, device: torch.device, memory: dict) -> dict:
    torch.manual_seed(config.seed)
    model = TransformerLM(
        d_model=config.d_model,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        num_layers=config.num_layers,
        rope_theta=config.rope_theta,
    ).to(device=device, dtype=getattr(torch, config.dtype))
    needs_backward = config.mode != "forward"
    model.train(needs_backward)
    if config.compile_model:
        # Compile the model, including its autograd backward; loss and AdamW stay
        # outside the compiled region, as requested in torch_compile(b).
        model = torch.compile(model, backend=config.compile_backend, fullgraph=True, dynamic=False)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate) if config.mode == "train" else None

    # One synthetic next-token batch, allocated on the target device before timing.
    tokens = torch.randint(config.vocab_size, (config.batch_size, config.context_length + 1), device=device)
    inputs = tokens[:, :-1].contiguous()
    targets = tokens[:, 1:].contiguous()

    def run_step() -> dict[str, float]:
        # Clear gradients outside the timed region, including before warm-up steps.
        # Synchronizing here also excludes initialization and pending setup work.
        model.zero_grad(set_to_none=True)
        synchronize(device)
        component_times = {}

        def mark(name: str, start: float) -> float:
            synchronize(device)
            end = default_timer()
            component_times[name] = (end - start) * 1000
            return end

        with torch.set_grad_enabled(needs_backward):
            start = default_timer()
            with nvtx_range("forward", device):
                # Keep the context local to each forward; backward/AdamW run outside it.
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16) if config.mixed_precision else nullcontext():
                    logits = model(inputs)
            boundary = mark("forward", start) if config.time_components else start
            if needs_backward:
                with nvtx_range("loss", device):
                    # This custom loss is composed of primitive ops, so explicitly
                    # promote its reductions instead of relying on an AMP op policy.
                    loss = cross_entropy(logits.float() if config.mixed_precision else logits, targets)
                if config.time_components:
                    boundary = mark("loss", boundary)
                with nvtx_range("backward", device):
                    loss.backward()
                if config.time_components:
                    boundary = mark("backward", boundary)
                if optimizer is not None:
                    with nvtx_range("optimizer", device):
                        optimizer.step()
                    if config.time_components:
                        mark("optimizer", boundary)
            # CUDA launches are asynchronous: wait before stopping the host timer.
            synchronize(device)
            elapsed_ms = (default_timer() - start) * 1000
        return {"step": elapsed_ms, **component_times}

    samples: dict[str, list[float]] = {}
    with attention_annotations(config.annotate_attention):
        if config.profile_memory:
            memory["phase"] = "warmup"
        for _ in range(config.warmup_steps):
            run_step()

        if config.profile_memory:
            # Exclude leftover warm-up gradients from the measurement baseline.
            model.zero_grad(set_to_none=True)
            reset_memory_measurement(config, device, memory)
        with nvtx_range("measurement", device):
            for step in range(config.steps):
                with nvtx_range(f"step_{step}", device):
                    for name, elapsed_ms in run_step().items():
                        samples.setdefault(name, []).append(elapsed_ms)

    return {
        "config": asdict(config),
        "torch_version": str(torch.__version__),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "memory": memory,
        "timings": {name: summarize(values) for name, values in samples.items()},
    }


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-size", choices=MODEL_SIZES, default="small", help="Table 1 preset; explicit dimensions override it")
    for name in ("d-model", "d-ff", "num-layers", "num-heads"):
        parser.add_argument(f"--{name}", type=int)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    parser.add_argument("--mode", choices=MODES, default="forward", help="train includes forward, loss, backward, and AdamW")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10, help="number of measured steps")
    parser.add_argument("--device", default="cuda", help="cuda, cuda:N, or cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="float32", help="model dtype (not mixed precision)")
    parser.add_argument("--mixed-precision", action="store_true", help="BF16 autocast with FP32 parameters, gradients, optimizer state, and loss")
    parser.add_argument("--compile", dest="compile_model", action="store_true", help="compile the whole Transformer with torch.compile; requires warm-up")
    parser.add_argument("--compile-backend", default="inductor", help="torch.compile backend; keep inductor for performance measurements")
    parser.add_argument("--profile-memory", action="store_true", help="record CUDA history, including setup/warm-up OOMs; reset history and peaks after successful warm-up")
    parser.add_argument("--memory-snapshot", help="snapshot pickle path; defaults to memory_snapshot.pickle beside --output (or in the current directory)")
    parser.add_argument("--memory-max-entries", type=int, default=1_000_000, help="maximum allocation history entries to retain")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--time-components", action="store_true", help="also time each phase; extra synchronizations affect end-to-end timing")
    parser.add_argument("--annotate-attention", action="store_true", help="add section 2.1.4 NVTX ranges for attention scores, softmax, and final matmul")
    parser.add_argument("--output", type=Path, help="save configuration and all timing samples as JSON")
    args = parser.parse_args(argv)
    dimensions = dict(zip(("d_model", "d_ff", "num_layers", "num_heads"), MODEL_SIZES[args.model_size]))
    values = vars(args).copy()
    output = values.pop("output")
    if values["memory_snapshot"] is None:
        values["memory_snapshot"] = str((output.parent if output is not None else Path.cwd()) / "memory_snapshot.pickle")
    values.pop("model_size")
    for name, value in dimensions.items():
        if values[name] is None:
            values[name] = value
    config = BenchmarkConfig(**values)
    try:
        config.validate()
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    result = benchmark(config)
    precision = "BF16 mixed precision (FP32 parameters)" if config.mixed_precision else config.dtype
    print(f"{config.mode} | {result['device_name']} | {precision} | {result['parameter_count']:,} parameters")
    print(f"implementation={'compiled (' + config.compile_backend + ')' if config.compile_model else 'eager'}")
    print(f"batch={config.batch_size}, context={config.context_length}, warmup={config.warmup_steps}, measured={config.steps}")
    if config.time_components:
        print("Component timing enabled: extra synchronization is included in step timing.")
    if config.profile_memory:
        memory = result["memory"]
        print("Memory history enabled: profiling overhead is included in timing.")
        print(f"Peak allocated: {memory['peak_allocated_bytes'] / 2**30:.3f} GiB; peak reserved: {memory['peak_reserved_bytes'] / 2**30:.3f} GiB")
        print(f"Saved memory snapshot: {memory['snapshot']}")
    for name, timing in result["timings"].items():
        print(f"{name:>10}: {timing['mean_ms']:.3f} ms ± {timing['std_ms']:.3f} ms (mean ± population std)")
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
