"""Section 7 fsdp_accounting: steady-state timing, memory accounting, and NVTX.

Launch with torchrun or modal_app.py::fsdp_benchmark. The accounting update is
separate from measured updates because its synchronization changes scheduling.
"""

import argparse
from contextlib import nullcontext
import csv
from datetime import timedelta
import math
import os
from pathlib import Path
import statistics
import time
import traceback

import torch
import torch.distributed as dist

from cs336_basics.model import Embedding, Linear, TransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.benchmark import BenchmarkConfig, MODEL_SIZES, nvtx_range, summarize, synchronize
from cs336_systems.ddp import DDP
from cs336_systems.ddp_train import TokenBatches
from cs336_systems.distributed_communication_benchmark import write_json
from cs336_systems.fsdp import FSDP
from cs336_systems.sharded_optimizer import ShardedOptimizer


DTYPES = {"float32": None, "bfloat16": torch.bfloat16, "float16": torch.float16}
STRATEGIES = ("ddp", "optimizer_sharded", "fsdp")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--strategy", choices=STRATEGIES, default="fsdp")
    parser.add_argument("--model-size", choices=MODEL_SIZES, default="xl")
    for name in ("d-model", "d-ff", "num-layers", "num-heads"):
        parser.add_argument(f"--{name}", type=int)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--global-batch-size", type=int, default=4)
    parser.add_argument("--compute-dtype", choices=DTYPES, default="float32")
    parser.add_argument("--max-pending", type=int, default=2, help="FSDP reduction queue limit; ignored by other strategies")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--backend", choices=("nccl", "gloo"), default="nccl")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--profile-layers", action="store_true", help="per-layer NVTX and all-gather-path events; use in separate profiling runs")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    for name, value in zip(("d_model", "d_ff", "num_layers", "num_heads"), MODEL_SIZES[args.model_size], strict=True):
        if getattr(args, name) is None:
            setattr(args, name, value)
    if min(args.steps, args.warmup_steps, args.global_batch_size, args.cpu_threads, args.timeout_seconds) < 1:
        parser.error("steps, warmup-steps, global-batch-size, cpu-threads, and timeout-seconds must be positive")
    if args.seed < 0 or not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("seed and weight-decay must be nonnegative; weight-decay must be finite")
    if args.max_pending < 1:
        parser.error("max-pending must be positive")
    args.train_data = None  # Reuse the deterministic synthetic TokenBatches helper.
    return args


def expected_memory(module, world_size):
    """FP32 weights/grads and two FP32 Adam moments, ignoring activations/buffers.

    This is a balanced, unpadded prediction. Actual shard/owner sizes are counted
    separately by storage_breakdown, including replicated normalization states.
    """
    parameters = list(module.parameters())
    sharded_ids = {id(p) for m in module.modules() if isinstance(m, (Linear, Embedding)) for p in m.parameters(recurse=False)}
    total = sum(p.numel() for p in parameters)
    sharded = sum(p.numel() for p in parameters if id(p) in sharded_ids)
    replicated = total - sharded
    ddp = 16 * total
    optimizer_sharded = 8 * total + 8 * total / world_size
    fsdp = 16 * sharded / world_size + 16 * replicated
    return {
        "parameter_count": total, "sharded_parameter_count": sharded, "replicated_parameter_count": replicated,
        "world_size": world_size,
        "ideal_bytes_per_rank": {"ddp": ddp, "optimizer_sharded": optimizer_sharded, "fsdp": fsdp},
        "ideal_saving_vs_optimizer_sharded_bytes": optimizer_sharded - fsdp,
        "assumptions": "FP32 masters/gradients, two FP32 Adam moments, balanced ownership, no padding, activations, communication buffers or optimizer temporaries",
    }


def tensor_values(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensor_values(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from tensor_values(item)


def storage_breakdown(model, optimizer):
    """Count distinct tensor storages, so aliases/views are not counted twice."""
    states = getattr(model, "state_by_parameter", {})
    local_optimizer = optimizer.local_optimizer if isinstance(optimizer, ShardedOptimizer) else optimizer
    staged_gradients = [s.reduced_grad for s in states.values() if s.reduced_grad is not None]
    categories = {
        "master_weights": [states[p].local_shard if p in states else p for p in model.parameters()],
        "gathered_weights": [s.full_weight for s in states.values() if s.full_weight is not None],
        "gradients": [p.grad for p in model.parameters() if p.grad is not None] + staged_gradients,
        "optimizer_states": list(tensor_values(local_optimizer.state)) if local_optimizer is not None else [],
        # Includes retained reduce-scatter inputs/outputs. Replicated gradients
        # and DDP gradient references are already counted above.
        "communication_buffers": list(tensor_values(getattr(model, "pending", []))),
    }
    def count_storage(tensors, seen):
        size = 0
        for tensor in tensors:
            storage = tensor.untyped_storage()
            key = (str(tensor.device), storage.data_ptr())
            if storage.nbytes() and key not in seen:
                seen.add(key)
                size += storage.nbytes()
        return size

    seen = set()
    result = {category + "_bytes": count_storage(tensors, seen) for category, tensors in categories.items()}
    result["accounted_storage_bytes"] = sum(result.values())
    # This is a subset of gradients_bytes, not an additional storage category.
    result["staged_gradients_bytes"] = count_storage(staged_gradients, set())
    result["pending_reductions"] = len(getattr(model, "pending", []))
    return result


def memory_sample(device, model=None, optimizer=None):
    if device.type == "cuda":
        synchronize(device)
        sample = {
            "allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
    else:
        sample = dict.fromkeys(("allocated_bytes", "reserved_bytes", "peak_allocated_bytes", "peak_reserved_bytes"), None)
    if optimizer is not None:
        sample.update(storage_breakdown(model, optimizer))
        sample["unattributed_allocated_bytes"] = (
            max(0, sample["allocated_bytes"] - sample["accounted_storage_bytes"])
            if device.type == "cuda" else None
        )
    return sample


class ProfiledFSDP(FSDP):
    """Instrument forward/backward gathers without changing their scheduling."""

    def _before_forward(self, layer, inputs):
        return self._profile_gather("forward", super()._before_forward, layer, inputs)

    def _before_backward(self, layer, grad_output):
        return self._profile_gather("backward", super()._before_backward, layer, grad_output)

    def _after_forward(self, layer, inputs, output):
        if not self.measure_gathers:
            return super()._after_forward(layer, inputs, output)
        device = self.weight_states[layer].local_shard.device
        with nvtx_range(f"weight_reshard/forward/{self.layer_names[layer]}", device):
            return super()._after_forward(layer, inputs, output)

    def _after_backward(self, layer, grad_input, grad_output):
        if not self.measure_gathers:
            return super()._after_backward(layer, grad_input, grad_output)
        state = self.weight_states[layer]
        allocated = state.full_buffer.untyped_storage().nbytes() > 0
        with nvtx_range(f"backward_input_ready/{self.layer_names[layer]}", state.local_shard.device):
            result = super()._after_backward(layer, grad_input, grad_output)
        self._record_backward_release(state, allocated)
        return result

    def _gradient_ready(self, parameter):
        if not self.measure_gathers:
            return super()._gradient_ready(parameter)
        state = self.state_by_parameter.get(parameter)
        allocated = state is not None and state.full_buffer.untyped_storage().nbytes() > 0
        with nvtx_range(f"gradient_ready/{self.parameter_names[parameter]}", parameter.device):
            result = super()._gradient_ready(parameter)
        self.profile_stats["max_pending_observed"] = max(self.profile_stats["max_pending_observed"], len(self.pending))
        if state is not None:
            self._record_backward_release(state, allocated)
        return result

    def _record_backward_release(self, state, was_allocated):
        if was_allocated and state.full_buffer.untyped_storage().nbytes() == 0:
            name = self.parameter_layer_names[state.parameter]
            counts = self.profile_stats["backward_weight_releases"]
            counts[name] = counts.get(name, 0) + 1
            if state.local_shard.is_cuda:
                torch.cuda.nvtx.mark(f"weight_reshard/backward/{name}")

    def _complete_oldest(self):
        if not self.measure_gathers:
            return super()._complete_oldest()
        state, _, reduced_grad, _ = self.pending[0]
        kind = "all_reduce" if state is None else "reduce_scatter"
        # Includes the implementation's CUDA stream synchronize; do not add
        # synchronization or alter queue scheduling just for instrumentation.
        with nvtx_range(f"gradient_reduction_wait/{kind}", reduced_grad.device):
            result = super()._complete_oldest()
        self.profile_stats["completed_reductions"] += 1
        return result

    def _profile_gather(self, phase, hook, layer, hook_input):
        if not self.measure_gathers:
            return hook(layer, hook_input)
        name = f"{phase}/{self.layer_names[layer]}"
        device = self.weight_states[layer].local_shard.device
        events = [torch.cuda.Event(enable_timing=True) for _ in range(2)] if device.type == "cuda" else None
        start = time.perf_counter()
        if events:
            events[0].record()
        with nvtx_range(f"weight_all_gather/{name}", device):
            result = hook(layer, hook_input)
        if events:
            events[1].record()
        self.gather_records.append((name, events, (time.perf_counter() - start) * 1000))
        return result


def annotate_layers(module, device):
    """Register after FSDP's pre-hook: compute range starts after gather launch."""
    handles = []
    if device.type != "cuda":
        return handles
    for name, layer in module.named_modules():
        if isinstance(layer, (Linear, Embedding)):
            def before(m, inputs, label=name):
                torch.cuda.nvtx.range_push(f"layer_compute/{label}")

            def after(m, inputs, output):
                torch.cuda.nvtx.range_pop()

            # End the compute range before FSDP's after-forward storage release.
            handles.extend([layer.register_forward_pre_hook(before), layer.register_forward_hook(after, prepend=True)])
    return handles


def train_step(model, optimizer, batch, device, compute_dtype, snapshot=None):
    optimizer.zero_grad(set_to_none=True)
    with nvtx_range("forward_and_loss", device):
        with torch.autocast(device_type=device.type, dtype=compute_dtype) if compute_dtype else nullcontext():
            logits = model(batch[0])
        loss = cross_entropy(logits.float(), batch[1])
    if snapshot:
        snapshot("after_forward")
    with nvtx_range("backward", device):
        loss.backward()
    if snapshot:
        snapshot("after_backward_before_sync")
    with nvtx_range("finish_gradient_synchronization", device):
        model.finish_gradient_synchronization()
    if snapshot:
        snapshot("before_optimizer")
    with nvtx_range("optimizer", device):
        optimizer.step()
    if snapshot:
        snapshot("after_optimizer")
    return loss.detach()


def write_accounting(output, result):
    rows = []
    for rank in result["ranks"]:
        for phase, values in rank["memory_phases"].items():
            rows.append({"rank": rank["rank"], "phase": phase, **values})
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "memory.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    prediction = result["prediction"]
    lines = [
        f"{result['config']['strategy']} / {result['config']['compute_dtype']}: section 7 accounting\n",
        f"Mean max-rank step: {result['timings']['max_rank_step']['mean_ms']:.3f} ms.\n",
        "Ideal persistent FP32 weights + gradients + Adam moments per rank (buffers/activations excluded):\n",
    ]
    for name, size in prediction["ideal_bytes_per_rank"].items():
        lines.append(f"- {name}: {size / 2**30:.4f} GiB\n")
    peak = result["memory"]["peak_allocated_bytes"]
    if peak is not None:
        lines.append(f"Measured steady-state peak allocated memory (maximum across ranks): {peak / 2**30:.4f} GiB.\n")
    lines.extend([
        "Phase samples in GiB (CUDA allocated is unavailable on CPU):\n",
        "| Rank | Phase | Allocated | Masters | Gathered | Gradients | Adam states | Communication buffers | Staged gradients (subset) | Pending count |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    columns = ("allocated_bytes", "master_weights_bytes", "gathered_weights_bytes", "gradients_bytes",
               "optimizer_states_bytes", "communication_buffers_bytes", "staged_gradients_bytes")
    for row in rows:
        values = [f"{row[key] / 2**30:.4f}" if row.get(key) is not None else "n/a" for key in columns]
        lines.append(f"| {row['rank']} | {row['phase']} | " + " | ".join(values) + f" | {row.get('pending_reductions', 'n/a')} |")
    lines.append("")
    lines.extend([
        f"Ideal saving from optimizer-state sharding to FSDP: {prediction['ideal_saving_vs_optimizer_sharded_bytes'] / 2**30:.4f} GiB/rank.\n",
        "Measured CUDA peaks are in timings.json; per-rank phase samples and storage categories are in memory.csv. "
        "Phase peaks are cumulative across the separate accounting update. Initialization has a separate peak. "
        "CPU runs have no CUDA memory measurements.\n",
        "The current student FSDP releases full-weight storage after forward and after each layer's input and weight gradients are ready. "
        "Both after-forward and after-backward gathered storage should be zero. "
        "Completed local gradients in state.reduced_grad are included in gradients_bytes; staged_gradients_bytes is a subset, not extra memory. "
        "Communication buffers include the inputs/outputs of the bounded pending queue. "
        "finish_gradient_synchronization drains that queue and installs the staged gradients for the optimizer. "
        "Model construction initially allocates full FP32 weights.\n",
        "With --profile-layers, all_gather_path keys are forward/<layer> and backward/<layer>. "
        "Their timings include storage allocation, dtype conversion and gather dependencies, not isolated NCCL time. "
        "NVTX ranges describe CPU launches; inspect correlated CUDA streams to establish whether communication is hidden. "
        "The current gather hook waits just before each layer; it has no explicit forward prefetch.\n",
        "gradient_reduction_wait/{all_reduce,reduce_scatter} ranges include the queue-drain stream synchronization. "
        "These waits may occur inside backward as well as in the final synchronization phase. "
        "weight_reshard/backward/<layer> marks the CPU-side storage release request; profile_stats records release counts and queue occupancy.\n",
    ])
    (output / "report.md").write_text("\n".join(lines))


def run(args, device, rank, world_size):
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(world_size=world_size, local_batch_size=args.global_batch_size // world_size,
                  batch_size=args.global_batch_size // world_size, dtype="float32", mixed_precision=args.compute_dtype != "float32",
                  implementation=f"accounting_{args.strategy}", nsys_profile=os.environ.get("CS336_NSYS_PROFILE") == "1",
                  fsdp_weight_lifecycle="reshard_after_forward_and_backward" if args.strategy == "fsdp" else None,
                  fsdp_max_pending=args.max_pending if args.strategy == "fsdp" else None,
                  allocator_conf=os.environ.get("PYTORCH_ALLOC_CONF", os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")))
    if rank == 0:
        if (output / "config.json").exists():
            raise FileExistsError("Choose a fresh output-dir")
        write_json(output / "config.json", config)
    phase = "initialization"
    samples = []
    try:
        torch.manual_seed(args.seed)  # All ranks start from identical master weights.
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        module = TransformerLM(d_model=args.d_model, d_ff=args.d_ff, num_layers=args.num_layers, num_heads=args.num_heads,
                               vocab_size=args.vocab_size, context_length=args.context_length).to(device)
        prediction = expected_memory(module, world_size)
        phases = {"after_model_initialization": memory_sample(device)}
        compute_dtype = DTYPES[args.compute_dtype]
        if args.strategy == "fsdp":
            model = ProfiledFSDP(module, compute_dtype=compute_dtype)
            model.max_pending = args.max_pending
            model.layer_names = {layer: name for name, layer in module.named_modules()}
            model.parameter_names = {parameter: name for name, parameter in module.named_parameters()}
            model.parameter_layer_names = {state.parameter: model.layer_names[layer] for layer, state in model.weight_states.items()}
            model.measure_gathers = False
            model.gather_records = []
            model.profile_stats = {"max_pending_observed": 0, "completed_reductions": 0, "backward_weight_releases": {}}
        else:
            model = DDP(module)
        optimizer = (ShardedOptimizer(model.parameters(), AdamW, lr=args.learning_rate, weight_decay=args.weight_decay)
                     if args.strategy == "optimizer_sharded" else AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay))
        model.train()
        phases["after_initialization"] = memory_sample(device, model, optimizer)
        batch = TokenBatches(args, rank, world_size, device).get(0)
        phase = "warmup"
        for _ in range(args.warmup_steps):
            train_step(model, optimizer, batch, device, compute_dtype)
        synchronize(device)
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        phases["after_warmup_zero_grad"] = memory_sample(device, model, optimizer)
        handles = annotate_layers(module, device) if args.profile_layers else []
        if isinstance(model, ProfiledFSDP):
            model.measure_gathers = args.profile_layers
        phase = "measurement"
        with nvtx_range("fsdp_measurement" if rank == 0 else f"fsdp_worker_rank_{rank}", device):
            dist.barrier()
            for step in range(args.steps):
                dist.barrier()
                synchronize(device)
                with nvtx_range(f"step_{step + 1}/rank_{rank}/{args.strategy}", device):
                    start = time.perf_counter()
                    loss = train_step(model, optimizer, batch, device, compute_dtype)
                    synchronize(device)
                    samples.append((time.perf_counter() - start) * 1000)
            dist.barrier()
            synchronize(device)
        measured_memory = memory_sample(device, model, optimizer)
        for handle in handles:
            handle.remove()
        gathers = {}
        if isinstance(model, ProfiledFSDP):
            model.measure_gathers = False
            for name, events, host_ms in model.gather_records:
                gathers.setdefault(name, []).append(events[0].elapsed_time(events[1]) if events else host_ms)
        # The separately instrumented update exposes retained weights and buffers
        # without introducing intermediate device syncs into the timing samples.
        phase = "memory_accounting"
        dist.barrier()
        synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        def snapshot(name):
            phases[name] = memory_sample(device, model, optimizer)

        loss = train_step(model, optimizer, batch, device, compute_dtype, snapshot=snapshot)
        if not torch.isfinite(loss).item():
            raise FloatingPointError("Nonfinite loss; check precision and learning rate")
        rank_result = {"rank": rank, "samples_ms": samples, "memory_phases": phases,
                       "measured_memory": measured_memory, "final_local_loss": loss.item(),
                       "all_gather_path": {name: summarize(values) for name, values in gathers.items()},
                       "profile_stats": model.profile_stats if isinstance(model, ProfiledFSDP) and args.profile_layers else None}
        write_json(output / f"rank-{rank}.json", rank_result)
        ranks = [None] * world_size
        dist.all_gather_object(ranks, rank_result)
        if rank == 0:
            paired = list(zip(*(item["samples_ms"] for item in ranks), strict=True))
            timings = {"step": summarize([statistics.mean(values) for values in paired]),
                       "max_rank_step": summarize([max(values) for values in paired])}
            peak = {key: max(item["measured_memory"][key] for item in ranks) if device.type == "cuda" else None
                    for key in ("peak_allocated_bytes", "peak_reserved_bytes")}
            result = {"status": "success", "config": config, "prediction": prediction, "parameter_count": prediction["parameter_count"],
                      "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                      "timings": timings, "memory": peak, "ranks": ranks,
                      "global_tokens_per_second": args.global_batch_size * args.context_length / (timings["max_rank_step"]["mean_ms"] / 1000)}
            write_json(output / "timings.json", result)
            write_accounting(output, result)
            print((output / "report.md").read_text(), flush=True)
        dist.barrier()
    except Exception as error:
        failure = {"status": "oom" if isinstance(error, torch.cuda.OutOfMemoryError) else "error", "config": config,
                   "failure_phase": phase, "error": str(error), "traceback": traceback.format_exc(), "timings": {},
                   "device_name": str(device), "samples_ms": samples}
        if device.type == "cuda":
            try:
                failure["memory"] = {"allocated_bytes": torch.cuda.memory_allocated(device), "reserved_bytes": torch.cuda.memory_reserved(device),
                                     "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
                (output / f"rank-{rank}-memory-summary.txt").write_text(torch.cuda.memory_summary(device))
            except Exception as diagnostic_error:
                failure["memory_diagnostic_error"] = str(diagnostic_error)
        write_json(output / f"rank-{rank}-failure.json", failure)
        if rank == 0:
            write_json(output / "timings.json", failure)
        raise


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.cpu_threads)
    if "RANK" not in os.environ:
        raise RuntimeError("Launch with torchrun --standalone --nproc-per-node=2 -m cs336_systems.fsdp_benchmark")
    rank, local_rank, world_size = (int(os.environ[key]) for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    if args.global_batch_size % world_size:
        raise ValueError("global-batch-size must divide evenly across ranks")
    device = torch.device("cpu")
    if args.backend == "nccl":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    BenchmarkConfig(d_model=args.d_model, d_ff=args.d_ff, num_layers=args.num_layers, num_heads=args.num_heads,
                    vocab_size=args.vocab_size, context_length=args.context_length, batch_size=args.global_batch_size // world_size,
                    device=str(device), mode="train", steps=args.steps, learning_rate=args.learning_rate).validate()
    dist.init_process_group(args.backend, timeout=timedelta(seconds=args.timeout_seconds))
    try:
        run(args, device, rank, world_size)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
