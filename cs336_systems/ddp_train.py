"""Train the assignment-1 Transformer with the student's naive or flat DDP wrapper.

Launch with torchrun (one process per GPU), or use --backend gloo on CPU.
The companion ddp_benchmark module uses the same optimizer-step implementation.
"""

import argparse
from contextlib import nullcontext
import csv
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import statistics
import time
import traceback

import numpy as np
import torch
import torch.distributed as dist

from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.benchmark import BenchmarkConfig, MODEL_SIZES, nvtx_range, summarize, synchronize
from cs336_systems.ddp import FlatDDP, NaiveDDP
from cs336_systems.distributed_communication_benchmark import write_json
from cs336_systems.modal_runner import run_name


def parse_args(argv=None, *, benchmark=False):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-size", choices=MODEL_SIZES, default="xl" if benchmark else "small")
    parser.add_argument("--wrapper", choices=("naive", "flat"), default="naive", help="individual gradient all-reduces or one packed all-reduce")
    for name in ("d-model", "d-ff", "num-layers", "num-heads"):
        parser.add_argument(f"--{name}", type=int)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--global-batch-size", type=int, default=4, help="total examples across ALL ranks; must divide evenly")
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    parser.add_argument("--backend", choices=("nccl", "gloo"), default="nccl")
    parser.add_argument("--steps", type=int, default=10 if benchmark else 100, help="measured updates for benchmark; total target updates for training/resume")
    parser.add_argument("--warmup-steps", type=int, default=5 if benchmark else 0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.0 if benchmark else 1.0, help="clip AFTER averaging; 0 disables clipping")
    parser.add_argument("--mixed-precision", action="store_true", help="BF16 forward autocast, FP32 parameters/gradients/AdamW/loss")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-data", type=Path, help="1D integer .npy token array; otherwise generate synthetic next-token batches")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=0, help="training only: periodically replace checkpoint.pt; 0 means final only")
    parser.add_argument("--save-checkpoint", action=argparse.BooleanOptionalAction, default=not benchmark)
    parser.add_argument("--resume", type=Path, help="training only: resume this script's checkpoint on every rank")
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--cpu-threads", type=int, default=1, help="PyTorch CPU threads per worker")
    args = parser.parse_args(argv)
    args.benchmark = benchmark
    for name, value in zip(("d_model", "d_ff", "num_layers", "num_heads"), MODEL_SIZES[args.model_size], strict=True):
        if getattr(args, name) is None:
            setattr(args, name, value)
    if min(args.global_batch_size, args.steps, args.log_every, args.timeout_seconds, args.cpu_threads) < 1:
        parser.error("global-batch-size, steps, log-every and timeout-seconds must be positive")
    if args.warmup_steps < (1 if benchmark else 0) or args.checkpoint_every < 0:
        parser.error("benchmark requires at least one warm-up; counts must be nonnegative")
    if not benchmark and args.warmup_steps != 0:
        parser.error("training has no discarded warm-ups; use ddp_benchmark for timing")
    if benchmark and (args.resume or args.checkpoint_every):
        parser.error("resume and periodic checkpointing are training-only options")
    if args.checkpoint_every and not args.save_checkpoint:
        parser.error("checkpoint-every requires save-checkpoint")
    if any(not math.isfinite(value) or value < 0 for value in (args.weight_decay, args.max_grad_norm)):
        parser.error("weight-decay and max-grad-norm must be finite and nonnegative")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0 or args.seed < 0:
        parser.error("learning-rate must be finite and positive; seed must be nonnegative")
    return args


class TokenBatches:
    """Reproducible batches indexed by update, with disjoint corpus shards.

    Each rank samples windows only from its own contiguous shard of a memory-
    mapped .npy file. Synthetic streams also use rank-specific seeds. Indexing
    seeds by the absolute update makes checkpoint continuation reproducible.
    """

    def __init__(self, args, rank, world_size, device):
        self.args, self.rank, self.device = args, rank, device
        self.batch_size = args.global_batch_size // world_size
        self.tokens = None
        if args.train_data:
            tokens = np.load(args.train_data, mmap_mode="r", allow_pickle=False)
            if tokens.ndim != 1 or not np.issubdtype(tokens.dtype, np.integer):
                raise ValueError("train-data must be a 1D integer .npy array")
            self.tokens = tokens[len(tokens) * rank // world_size:len(tokens) * (rank + 1) // world_size]
            if len(self.tokens) <= args.context_length:
                raise ValueError("each rank's token shard must contain at least context-length + 1 tokens")

    def get(self, update):
        args = self.args
        generator = torch.Generator().manual_seed((args.seed + 1_000_003 * self.rank + 97 * update) % (2**63 - 1))
        if self.tokens is None:
            batch = torch.randint(args.vocab_size, (self.batch_size, args.context_length + 1), generator=generator)
        else:
            starts = torch.randint(len(self.tokens) - args.context_length, (self.batch_size,), generator=generator)
            batch = torch.from_numpy(np.stack([self.tokens[i:i + args.context_length + 1] for i in starts.tolist()]).astype(np.int64))
            if batch.min().item() < 0 or batch.max().item() >= args.vocab_size:
                raise ValueError("sampled token IDs are outside [0, vocab-size)")
        return batch[:, :-1].contiguous().to(self.device), batch[:, 1:].contiguous().to(self.device)


def train_step(model, optimizer, inputs, targets, device, *, mixed_precision=False, max_grad_norm=0.0, timed=False):
    """A full update. Synchronize timing boundaries only in benchmark mode."""
    if timed:
        synchronize(device)
    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    with nvtx_range("forward_and_loss", device):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16) if mixed_precision else nullcontext():
            logits = model(inputs)
        loss = cross_entropy(logits.float(), targets)
    with nvtx_range("backward", device):
        loss.backward()
    if timed:
        # Exclude unfinished backprop kernels from the communication interval.
        synchronize(device)
    communication_start = time.perf_counter()
    with nvtx_range("gradient_synchronization", device):
        model.synchronize_gradients()
    if timed:
        synchronize(device)
    communication_end = time.perf_counter()
    with nvtx_range("optimizer", device):
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
    if timed:
        synchronize(device)
    end = time.perf_counter()
    timing = {
        "step": (end - start) * 1000,
        "communication": (communication_end - communication_start) * 1000,
    } if timed else {}
    return loss.detach(), timing


def summarize_ranks(ranks):
    """Mean-rank step/communication times; paired, time-weighted fraction."""
    timings = {}
    for phase in ("step", "communication"):
        by_rank = [rank["samples_ms"][phase] for rank in ranks]
        timings[phase] = summarize([statistics.mean(values) for values in zip(*by_rank, strict=True)])
    timings["max_rank_step"] = summarize([max(values) for values in zip(*(rank["samples_ms"]["step"] for rank in ranks), strict=True)])
    fraction = timings["communication"]["mean_ms"] / timings["step"]["mean_ms"]
    return timings, fraction


def save_checkpoint(path, model, optimizer, completed_steps, config):
    temporary = path.with_suffix(".tmp")
    torch.save({"model": model.module.state_dict(), "optimizer": optimizer.state_dict(), "completed_steps": completed_steps, "config": config}, temporary)
    temporary.replace(path)


def restore_checkpoint(path, model, optimizer, config, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint["config"].get("wrapper", "naive") != config["wrapper"]:
        raise ValueError("resume configuration differs for wrapper")
    # Keep the data stream and optimizer trajectory comparable after resuming.
    for key in ("d_model", "d_ff", "num_layers", "num_heads", "vocab_size", "context_length", "rope_theta", "global_batch_size",
                "world_size", "seed", "train_data", "mixed_precision", "learning_rate", "weight_decay", "max_grad_norm"):
        if checkpoint["config"][key] != config[key]:
            raise ValueError(f"resume configuration differs for {key}")
    model.module.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["completed_steps"]


def run(args, device, rank, world_size):
    local_batch = args.global_batch_size // world_size
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(world_size=world_size, local_batch_size=local_batch, batch_size=local_batch, dtype="float32", implementation=f"{args.wrapper}_ddp")
    config["allocator_conf"] = os.environ.get("PYTORCH_ALLOC_CONF", os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""))
    config["train_data"] = str(args.train_data.resolve()) if args.train_data else None
    output = args.output_dir
    if rank == 0:
        if (output / "config.json").exists():
            raise FileExistsError(f"{output} already contains a run; choose a new output-dir")
        write_json(output / "config.json", config)
    samples = {"step": [], "communication": []}
    phase = "initialization"
    try:
        torch.manual_seed(args.seed + rank)
        model = TransformerLM(
            d_model=args.d_model, d_ff=args.d_ff, num_layers=args.num_layers, num_heads=args.num_heads,
            vocab_size=args.vocab_size, context_length=args.context_length, rope_theta=args.rope_theta,
        ).to(device)
        wrapper_class = {"naive": NaiveDDP, "flat": FlatDDP}[args.wrapper]
        model = wrapper_class(model)
        model.train()
        optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        start_update = restore_checkpoint(args.resume, model, optimizer, config, device) if args.resume else 0
        if start_update >= args.steps:
            raise ValueError("steps must exceed the checkpoint's completed_steps")
        batches = TokenBatches(args, rank, world_size, device)
        # A single preallocated local batch makes the timing independent of data IO.
        fixed_batch = batches.get(0) if args.benchmark else None
        if args.benchmark:
            phase = "warmup"
            for _ in range(args.warmup_steps):
                train_step(model, optimizer, *fixed_batch, device, mixed_precision=args.mixed_precision, max_grad_norm=args.max_grad_norm)
            synchronize(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
        phase = "measurement" if args.benchmark else "training"
        loss = None
        for update in range(start_update, args.steps):
            inputs, targets = fixed_batch if args.benchmark else batches.get(update)
            if args.benchmark:
                # Align the start of steps; don't insert a barrier before the
                # all-reduces, whose natural wait for slower ranks is real cost.
                dist.barrier()
                synchronize(device)
            loss, timing = train_step(
                model, optimizer, inputs, targets, device, mixed_precision=args.mixed_precision,
                max_grad_norm=args.max_grad_norm, timed=args.benchmark,
            )
            for name, value in timing.items():
                samples[name].append(value)
            # Reporting collectives and disk IO are outside benchmark timing.
            if (update + 1) % args.log_every == 0 or update + 1 == args.steps:
                mean_loss = loss.clone()
                dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
                mean_loss.div_(world_size)
                value = mean_loss.item()
                if not math.isfinite(value):
                    raise FloatingPointError(f"nonfinite loss at update {update + 1}")
                if rank == 0:
                    row = {"update": update + 1, "optimizer_updates_including_warmup": update + 1 + args.warmup_steps, "loss": value}
                    with (output / "train.jsonl").open("a") as stream:
                        stream.write(json.dumps(row) + "\n")
                    print(f"update={update + 1} global_mean_loss={value:.6f}", flush=True)
            if args.checkpoint_every and (update + 1) % args.checkpoint_every == 0:
                if rank == 0:
                    save_checkpoint(output / "checkpoint.pt", model, optimizer, update + 1, config)
                dist.barrier()

        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        rank_result = {
            "rank": rank, "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "samples_ms": samples, "final_local_loss": loss.item(), "status": "success",
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0,
        }
        write_json(output / f"rank-{rank}.json", rank_result)
        ranks = [None] * world_size
        dist.all_gather_object(ranks, rank_result)
        if rank == 0:
            result = {
                "status": "success", "config": config, "parameter_count": parameter_count,
                "gradient_tensor_count": sum(p.requires_grad for p in model.parameters()),
                "gradient_all_reduce_calls_per_step": 1 if args.wrapper == "flat" else sum(p.requires_grad for p in model.parameters()),
                "gradient_bytes_per_rank": sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad),
                "device_name": ranks[0]["device_name"], "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
                "ranks": ranks, "timings": {},
                "memory": {key: max(item[key] for item in ranks) for key in ("peak_allocated_bytes", "peak_reserved_bytes")},
            }
            if args.benchmark:
                result["timings"], result["communication_fraction"] = summarize_ranks(ranks)
                result["communication_percent"] = 100 * result["communication_fraction"]
                result["global_tokens_per_second"] = args.global_batch_size * args.context_length / (result["timings"]["max_rank_step"]["mean_ms"] / 1000)
                write_json(output / "timings.json", result)
                table = {
                    "wrapper": args.wrapper,
                    "allocator_conf": config["allocator_conf"],
                    "model_size": args.model_size, "world_size": world_size, "global_batch_size": args.global_batch_size,
                    "local_batch_size": local_batch, "context_length": args.context_length, "mixed_precision": args.mixed_precision,
                    "step_mean_ms": result["timings"]["step"]["mean_ms"], "step_std_ms": result["timings"]["step"]["std_ms"],
                    "communication_mean_ms": result["timings"]["communication"]["mean_ms"],
                    "communication_percent": result["communication_percent"], "global_tokens_per_second": result["global_tokens_per_second"],
                }
                with (output / "ddp_summary.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(table))
                    writer.writeheader()
                    writer.writerow(table)
                report = (
                    f"{args.wrapper.capitalize()} DDP: {world_size} processes, {args.model_size} preset, {parameter_count:,} parameters, {ranks[0]['device_name']}.\n"
                    f"Actual model: d_model={args.d_model}, d_ff={args.d_ff}, layers={args.num_layers}, heads={args.num_heads}.\n"
                    f"Allocator configuration: {config['allocator_conf'] or 'default'}.\n"
                    f"Global/local batch {args.global_batch_size}/{local_batch}, context {args.context_length}, FP32 parameters, BF16 autocast={args.mixed_precision}.\n"
                    f"Mean step: {table['step_mean_ms']:.3f} ms; gradient synchronization: {table['communication_mean_ms']:.3f} ms "
                    f"({table['communication_percent']:.2f}% of step time).\n"
                    "Times average ranks per iteration; std summarizes those iteration averages. Communication includes SUM all-reduces, "
                    "gradient division, packing/copyback for the flat wrapper, launch overhead, and waiting for peers. Step time includes zero_grad, forward/loss, backward, "
                    "synchronization, optional clipping, and AdamW; data preparation, barriers, logging and checkpoints are excluded.\n"
                )
                (output / "report.md").write_text(report)
                print(report, flush=True)
            else:
                result["completed_steps"] = args.steps
                write_json(output / "training_summary.json", result)
            if args.save_checkpoint:
                save_checkpoint(output / "checkpoint.pt", model, optimizer, args.steps + args.warmup_steps, config)
        dist.barrier()
    except Exception as error:
        failure = {"rank": rank, "status": "oom" if isinstance(error, torch.cuda.OutOfMemoryError) else "error", "failure_phase": phase,
                   "error": str(error), "traceback": traceback.format_exc(), "samples_ms": samples}
        if device.type == "cuda":
            try:
                # Inspect without synchronizing, emptying the cache, or retrying:
                # preserve the allocation state at failure and the original error.
                stats = torch.cuda.memory_stats(device)
                free, total = torch.cuda.mem_get_info(device)
                failure["memory"] = {
                    "allocated_bytes": torch.cuda.memory_allocated(device),
                    "reserved_bytes": torch.cuda.memory_reserved(device),
                    "inactive_split_bytes": stats.get("inactive_split_bytes.all.current", 0),
                    "driver_free_bytes": free, "device_total_bytes": total,
                }
                (output / f"rank-{rank}-memory-summary.txt").write_text(torch.cuda.memory_summary(device))
            except Exception as diagnostic_error:
                failure["memory_diagnostic_error"] = str(diagnostic_error)
        write_json(output / f"rank-{rank}-failure.json", failure)
        if rank == 0 and args.benchmark:
            write_json(output / "timings.json", {**failure, "config": config, "timings": {}, "device_name": str(device)})
        raise


def main(argv=None, *, benchmark=False):
    args = parse_args(argv, benchmark=benchmark)
    torch.set_num_threads(args.cpu_threads)
    if "RANK" not in os.environ:
        raise RuntimeError("Launch with torchrun --standalone --nproc-per-node=2 -m cs336_systems.ddp_train (or ddp_benchmark)")
    rank, local_rank, world_size = (int(os.environ[key]) for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    if args.global_batch_size % world_size:
        raise ValueError("global-batch-size must be divisible by WORLD_SIZE")
    device = torch.device("cpu")
    if args.backend == "nccl":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    BenchmarkConfig(
        d_model=args.d_model, d_ff=args.d_ff, num_layers=args.num_layers, num_heads=args.num_heads,
        vocab_size=args.vocab_size, context_length=args.context_length, batch_size=args.global_batch_size // world_size,
        rope_theta=args.rope_theta, device=str(device), mode="train", steps=args.steps, learning_rate=args.learning_rate,
    ).validate()
    dist.init_process_group(args.backend, timeout=timedelta(seconds=args.timeout_seconds))
    try:
        if args.output_dir is None:
            paths = [str(Path("results/ddp") / run_name()) if rank == 0 else None]
            dist.broadcast_object_list(paths, src=0)
            args.output_dir = Path(paths[0])
        args.output_dir.mkdir(parents=True, exist_ok=True)
        run(args, device, rank, world_size)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
