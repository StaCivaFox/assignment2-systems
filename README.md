# CS336 Spring 2026 Assignment 2: Systems

For a full description of the assignment, see the assignment handout at
[cs336_assignment2_systems.pdf](./cs336_assignment2_systems.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

This directory is organized as follows:

- [`./cs336-basics`](./cs336-basics): directory containing a module
  `cs336_basics` and its associated `pyproject.toml`. This module contains the staff 
  implementation of the language model from assignment 1. If you want to use your own 
  implementation, you can replace this directory with your own implementation.
- [`./cs336_systems`](./cs336_systems): This folder is basically empty! This is the
  module where you will implement your optimized Transformer language model. 
  Feel free to take whatever code you need from assignment 1 (in `cs336-basics`) and copy it 
  over as a starting point. In addition, you will implement distributed training and
  optimization in this module.

Visually, it should look something like:

``` sh
.
├── cs336_basics  # A python module named cs336_basics
│   ├── __init__.py
│   └── ... other files in the cs336_basics module, taken from assignment 1 ...
├── cs336_systems  # TODO(you): code that you'll write for assignment 2 
│   ├── __init__.py
│   └── ... TODO(you): any other files or folders you need for assignment 2 ...
├── README.md
├── pyproject.toml
└── ... TODO(you): other files or folders you need for assignment 2 ...
```

If you would like to use your own implementation of assignment 1, replace the `cs336-basics`
directory with your own implementation, or edit the outer `pyproject.toml` file to point to your
own implementation.

0. We use `uv` to manage dependencies. You can verify that the code from the `cs336-basics`
package is accessible by running:

```sh
$ uv run python
Using CPython 3.13.13
Creating virtual environment at: /path/to/uv/env/dir
      Built cs336-systems @ file:///path/to/systems/dir
      Built cs336-basics @ file:///path/to/basics/dir
Installed 78 packages in 168ms
Python 3.13.13 (main, Apr  7 2026, 20:49:46) [Clang 22.1.1 ] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import cs336_basics
...
```

`uv run` installs dependencies automatically as dictated in the `pyproject.toml` file.

## Benchmarking (section 2.1.3)

Run the benchmark with the assignment 1 Transformer, cross-entropy loss, and AdamW:

```sh
uv run python -m cs336_systems.benchmark --model-size small --mode train
```

Choose `--mode forward`, `--mode forward-backward`, or `--mode train`. Forward-only
runs with gradients disabled; the other modes include loss computation and backward,
and `train` also updates the weights. Defaults match the handout: FP32, batch size 4,
vocabulary size 10,000, context length 512, 5 warm-up steps, and 10 measured steps.
The device defaults to CUDA; select a specific GPU with `--device cuda:1`.

The script reports mean and population standard deviation in milliseconds. It
synchronizes the selected CUDA device before and after each step. Model/data setup
and gradient clearing are outside the measured region. Each step gets a fresh
autograd graph and cleared gradients. Warm-up executes the same work as measurement,
including optimizer updates in `train` mode.

Use `--time-components` to additionally measure forward, loss, backward, and optimizer
phases separately. This inserts extra CUDA synchronizations, affecting end-to-end
timing; omit it for the basic end-to-end benchmark. `--output` saves the configuration,
device, PyTorch version, parameter count, individual samples, and statistics as JSON.
For example, collect component timings for part (b) on a GPU with sufficient memory:

```sh
for size in small medium large xl 10B; do
    uv run python -m cs336_systems.benchmark --model-size "$size" --mode train \
        --time-components --output "results/${size}-components.json"
done
```

For part (c), launch a fresh process for each warm-up count so that previous runs do
not warm up the CUDA runtime or kernels:

```sh
for warmup in 0 1 2 5; do
    uv run python -m cs336_systems.benchmark --model-size small --mode train \
        --warmup-steps "$warmup" --output "results/small-warmup-${warmup}.json"
done
```

Explicit dimensions override a size preset. A small CPU smoke test is:

```sh
uv run python -m cs336_systems.benchmark --device cpu --d-model 32 --d-ff 64 \
    --num-layers 1 --num-heads 4 --vocab-size 128 --context-length 16 \
    --batch-size 2 --warmup-steps 1 --steps 3 --mode train
```

See `--help` for all options. `--dtype` changes parameter/activation precision;
it does not enable autocast mixed precision. For section 2.1.5, use
`--mixed-precision` with the default `--dtype float32`: forward operations use BF16
autocast, while parameters, gradients, and AdamW state stay FP32. The handwritten
cross-entropy runs in FP32, and backward/optimizer steps run outside autocast.
BF16 does not need the FP16 gradient-scaling machinery. CPU BF16 autocast is also
available for smoke tests; assignment measurements should use CUDA.

For section 2.1.6, `--profile-memory` starts CUDA allocation history before model
setup so OOMs during initialization or warm-up also produce a snapshot. After a
successful warm-up, it clears history and resets peak counters to isolate the
measured steps. It saves `memory_snapshot.pickle` beside `--output` (or in
the current directory). Override that path with `--memory-snapshot PATH` and the
history limit with `--memory-max-entries` (default 1,000,000). A companion
`memory_snapshot.json` stores peak/baseline/end statistics in bytes, also included
in `timings.json`; the terminal prints peaks in GiB. Snapshots and companion
statistics are also attempted if setup, warm-up, or a measured step fails,
including CUDA OOM. The JSON records the phase (`initialization`, `warmup`, or
`measurement`) and failure status. The original error is re-raised so Modal marks
the job failed while still preserving/downloading its diagnostic artifacts.
Snapshot export errors are logged without replacing the original OOM.

Peak allocated memory measures live PyTorch tensor allocations; reserved memory
includes the allocator's cached blocks. These are absolute peaks including model
parameters and any optimizer state initialized during warm-up. Warm-up gradients
are cleared before recording, and the allocator cache is retained. PyTorch's
allocator profiler does not include all memory allocated by CUDA libraries.
Open [PyTorch memory_viz](https://pytorch.org/memory_viz/) and drag the pickle onto
the page to inspect the Active Memory Timeline. The Allocator State History also
shows OOM events and the allocator state at the failed allocation. Use separate runs
without memory history for timing comparisons, since recording adds overhead.

## Running this assignment on Modal

[`modal_app.py`](./modal_app.py) runs the benchmark and later assignment scripts on
Modal. It uses the repository's `uv.lock`, uploads your current local code, and
includes CUDA development tools, Triton (via PyTorch), and Nsight Systems. Install
the client with `uv tool install modal==1.6.1` if needed, then authenticate once with
`modal setup`. Run the commands below from this repository's root.

The default is **one B200**, 8 CPU cores, 64 GiB host memory, and a one-hour timeout.
Set `CS336_MODAL_GPU` before invoking Modal to select another GPU or multiple GPUs
on one machine, for example `B200:2` or `H100:2`. Set `CS336_MODAL_TIMEOUT` (seconds)
for longer sweeps. Jobs run sequentially in one container, each in a fresh subprocess
with separate Triton/Inductor caches. A sweep's timeout covers the entire sweep.

Start with a tiny GPU smoke test:

```sh
modal run modal_app.py::benchmark --modes train --warmups 1 --steps 2 \
    --extra-args '--d-model 32 --d-ff 64 --num-layers 1 --num-heads 4 --vocab-size 128 --context-length 16 --batch-size 2'
```

For section 2.1.3, the `benchmark` entrypoint takes comma-separated model sizes,
modes, and warm-up counts and runs their Cartesian product:

```sh
# Part (b): all Table 1 models, all three end-to-end modes, 5 warm-ups / 10 samples.
modal run modal_app.py::benchmark --sizes small,medium,large,xl,10B --run-name section-213b

# Separate component measurements for forward, loss, backward, and optimizer.
modal run modal_app.py::benchmark --sizes small,medium,large,xl,10B --modes train \
    --extra-args '--time-components' --run-name section-213b-components

# Part (c): each case starts a fresh Python process.
modal run modal_app.py::benchmark --sizes small --warmups 0,1,2,5 --run-name section-213c
```

Large configurations may run out of GPU memory. Failed jobs retain their logs and
exit codes, subsequent cases still run, and the launcher reports an error after
downloading the results. Increasing GPU count does not shard the current benchmark;
distributed scripts must implement that behavior themselves.

For section 2.1.4, add `--nsys` to either entrypoint to collect a `.nsys-rep` trace:

```sh
modal run modal_app.py::benchmark --sizes small --modes train --nsys \
    --run-name small-nsys --nsys-args '--cuda-memory-usage=true'
```

The default trace includes CUDA, NVTX, OS runtime, cuBLAS, and cuDNN. CPU sampling
and context-switch tracing are disabled for container compatibility. Open downloaded
`.nsys-rep` files with the Nsight Systems desktop application.

Pass `--annotate-attention` to the benchmark (via Modal's `--extra-args`) to use the
handout's annotated attention implementation. It adds an outer `scaled dot product
attention` range with nested `computing attention scores` (including scaling and
masking), `computing softmax`, and `final matmul` ranges. The last range covers the
attention probabilities multiplied by V; the multihead output projection is outside
this function. These label forward operations; backward work appears under the
separate `backward` range. The original function is restored when the run finishes.
Annotations preserve the original softmax and add no GPU synchronizations.

The benchmark already labels `measurement`, individual steps, and the
forward/loss/backward/optimizer phases. Capture only measured steps, excluding
initialization and warm-up, with:

```sh
modal run modal_app.py::benchmark --sizes small --modes train --nsys \
    --extra-args '--annotate-attention' \
    --nsys-args '--capture-range=nvtx --nvtx-capture=measurement --capture-range-end=stop --env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0'
```

Use unprofiled runs without attention annotations or `--time-components` for timing
tables. All NVTX ranges are no-ops on CPU, so CPU smoke tests still work.

For section 2.1.5(c), compare FP32 with BF16 mixed precision across Table 1.
`--time-components` gives separate forward/loss/backward durations; it introduces
extra synchronization, so use the same setting in both runs:

```sh
modal run modal_app.py::benchmark --sizes small,medium,large,xl,10B \
    --modes forward,forward-backward --extra-args '--time-components' \
    --run-name section-215-fp32

modal run modal_app.py::benchmark --sizes small,medium,large,xl,10B \
    --modes forward,forward-backward --extra-args '--mixed-precision --time-components' \
    --run-name section-215-bf16
```

For section 2.1.6(a–c), collect XL inference and full-training snapshots at both
context lengths, in FP32 and BF16 mixed precision. The runner automatically saves
and downloads snapshots with each job's other artifacts:

```sh
for context in 128 2048; do
    modal run modal_app.py::benchmark --sizes xl --modes forward,train --steps 3 \
        --extra-args "--context-length $context --profile-memory" \
        --run-name "section-216-fp32-context-$context"
    modal run modal_app.py::benchmark --sizes xl --modes forward,train --steps 3 \
        --extra-args "--context-length $context --mixed-precision --profile-memory" \
        --run-name "section-216-bf16-context-$context"
done
```

The CSV summary includes precision, context length, whether memory history was
enabled, and peak allocated/reserved bytes. These commands launch GPU jobs; the
largest cases may exceed available GPU memory. With `--profile-memory`, an OOM
during setup or warm-up also triggers snapshot export; its statistics cover that
phase rather than a completed measurement window.

## Attention and compilation benchmarks (sections 4.1–4.2)

[`cs336_systems/attention_benchmark.py`](./cs336_systems/attention_benchmark.py)
benchmarks your `cs336_basics.model.scaled_dot_product_attention`, including your
handwritten softmax. Its defaults match the assignment: batch size 8, tensors of
shape `(batch, sequence, d_model)` without a head dimension, dimensions
`16,32,64,128`, lengths `256,1024,4096,8192,16384`, FP32, 5 warm-up pairs, and 100
measured forward/backward pairs. It uses causal masking by default; pass
`--no-causal` locally or `--extra-args '--no-causal'` on Modal for unmasked attention.

Forward runs with gradients enabled. Each backward consumes a fresh forward graph
with a preallocated random output gradient; forward and backward are timed
separately, with CUDA synchronization after each. Input gradient buffers are
cleared outside the timers. Both directions are warmed up, so forward and backward
compilation are excluded from the measured samples. `compiled` uses
`torch.compile(..., backend="inductor", fullgraph=True, dynamic=False)` on the same
attention function. A compiler error is reported rather than falling back silently.

Run both attention problems together on Modal (40 independent cases):

```sh
modal run modal_app.py::attention --run-name attention-eager-compiled

# Section 4.1 only, or restrict the sweep for a quick check.
modal run modal_app.py::attention --implementations eager --run-name attention-eager
modal run modal_app.py::attention --d-models 16 --seq-lengths 256 \
    --warmup-steps 2 --steps 3
```

Each shape/implementation runs in a fresh process, and the runner commits its files
before advancing. An OOM is recorded with its phase (`initialization`,
`warmup_forward`, `warmup_backward`, `forward`, or `backward`); the remaining cases
still run. Modal reports failed cases after downloading all available artifacts.
Keep the default `inductor` backend for performance results; `--compile-backend
aot_eager` is provided for correctness/debugging, not speed comparisons.

The downloaded `attention_comparison.csv` puts eager/compiled forward and backward
means side by side and reports speedups as eager time divided by compiled time;
OOMs remain labeled and get no speedup. `summary.csv` includes one row per case and
timing phase, including OOM rows. Each case's `timings.json` retains all samples and error details. Memory
fields are in bytes:

- `pre_backward_allocated_bytes`: maximum live PyTorch allocation observed
  immediately after a measured forward and before its backward; all per-step
  readings are also saved in JSON.
- `pre_backward_reserved_bytes`: corresponding maximum allocator reservation,
  which includes cached blocks.
- `baseline_allocated_bytes`: after warm-up with input gradients cleared; includes
  Q/K/V, the output-gradient tensor, mask, and any persistent compiler allocations.
- `forward_live_increment_bytes`: pre-backward allocated memory minus that
  baseline. This includes the output and saved tensors; it is not an exact count
  of saved tensors alone.
- `peak_allocated_bytes` / `peak_reserved_bytes`: peaks over the measured pairs
  (or through the failing warm-up if the case never reached measurement).

JSON also records the sizes of Q/K/V, a single `(B,L,L)` attention matrix, and the
mask to support your memory accounting. One FP32 attention matrix requires
`4 * B * L**2` bytes; this is one tensor, not the full peak. CPU smoke tests omit
CUDA memory metrics. For example:

```sh
uv run python -m cs336_systems.attention_benchmark --device cpu \
    --d-models 8 --seq-lengths 4 --batch-size 2 --warmup-steps 1 --steps 2 \
    --compile-backend aot_eager
```

That smoke test uses AOTAutograd without code generation. CPU Inductor additionally
needs a working C++ compiler and Python development headers; omit
`--compile-backend aot_eager` when testing actual generated kernels.

For section 4.2(b), the existing Transformer benchmark accepts `--compile`.
It compiles the complete model forward and associated backward; the loss and AdamW
remain outside that compiled region, while `train` still times the complete step.
At least one warm-up step is required. Compare otherwise identical runs:

```sh
modal run modal_app.py::benchmark --sizes small,medium,large,xl,10B \
    --modes forward,train --run-name transformer-eager

modal run modal_app.py::benchmark --sizes small,medium,large,xl,10B \
    --modes forward,train --extra-args '--compile' --run-name transformer-compiled
```

The Transformer CSV records `compile_model` and `compile_backend`. Use the `step`
mean for these end-to-end comparisons. Add `--time-components` to both runs if you
also need phase timings, accepting its extra synchronization overhead. Compilation
can make large sweeps slow; set `CS336_MODAL_TIMEOUT` before launching if needed.
These use PyTorch's [compilation API](https://docs.pytorch.org/docs/2.11/generated/torch.compile.html).

The `run` entrypoint accepts any executable and arguments, including Python modules,
root-level Python scripts, scripts under `scripts/`, pytest, and torchrun:

```sh
modal run modal_app.py::run --command 'pytest tests/test_benchmark.py -q'

# After implementing attention: run its GPU tests on the required B200.
modal run modal_app.py::run --command 'pytest tests/test_attention.py -v'

# Template for a future script that reads torchrun's RANK / LOCAL_RANK / WORLD_SIZE.
CS336_MODAL_GPU=B200:2 modal run modal_app.py::run \
    --command 'torchrun --standalone --nproc-per-node=2 -m cs336_systems.your_distributed_benchmark'

# For scripts/tests that already use multiprocessing.spawn, launch Python/pytest
# directly instead of wrapping them in torchrun.
CS336_MODAL_GPU=B200:2 modal run modal_app.py::run --command 'pytest tests/test_ddp.py -v'
```

This also launches later implementations of checkpointing, attention, DDP, and
sharding, which still need their own implementations. Source uploads include `cs336_systems/`,
`cs336-basics/cs336_basics/`, `tests/` (including fixtures), `scripts/` if present,
and root-level `.py` files. Add other input directories to the image if a future
script needs them.

Every invocation gets a unique run name unless you supply `--run-name`. Artifacts
persist in Modal Volume `cs336-a2-results` under `runs/<run-name>/` and download to
`results/modal/<run-name>/`. Each run saves `environment.json`, `plan.json`,
`source.zip`, `summary.json`, `summary.csv`, and per-job logs/status/timings/traces.
Choose a new run name for each invocation; existing results are never overwritten.

To persist custom script outputs, use the environment variable `CS336_OUTPUT_DIR`
inside the script, or pass the literal placeholder `{output_dir}` in its arguments:

```sh
modal run modal_app.py::run \
    --command 'python -m cs336_systems.benchmark --mode train --output {output_dir}/timings.json'
```

Arguments are parsed with `shlex`, without a shell: pipes, redirects, and `$VARIABLE`
expansion are not performed. `{output_dir}` is replaced by the job's Volume directory.
Files written elsewhere are not automatically persisted. Use `--no-download` for
large traces/checkpoints and fetch them later, or recover results after disconnection:

```sh
modal volume get cs336-a2-results runs/<run-name> ./downloaded-run
```

For long jobs, use `modal run --detach modal_app.py::benchmark ...`. Results are
committed after every job; a hard timeout can interrupt the current job. Override
the Volume name with `CS336_MODAL_VOLUME` when needed.

The runner follows Modal's [image/dependency setup](https://modal.com/docs/reference/modal.Image#uv_sync),
[GPU selection](https://modal.com/docs/guide/gpu), and
[Volume persistence](https://modal.com/docs/guide/volumes) APIs, with
[NVIDIA's Nsight installation guidance](https://docs.nvidia.com/nsight-systems/InstallationGuide/index.html).

## Single-node communication benchmark (section 5.1.1)

[`cs336_systems/distributed_communication_benchmark.py`](./cs336_systems/distributed_communication_benchmark.py)
implements `distributed_communication_single_node`. Run the complete assignment
sweep on Modal with:

```sh
modal run modal_app.py::distributed_communication --run-name all-reduce-b200
```

This entrypoint requests **six B200 GPUs in one container**, then runs `torchrun`
with 2, 4, and 6 processes sequentially on that same node. Each process uses its
own GPU and NCCL. Each process count tests FP32 tensors of 1 MB, 10 MB, 100 MB,
and 1 GB **per rank**, with 5 warm-ups and 20 measured iterations per size.
MB and GB mean decimal bytes: 1 MB is 1,000,000 bytes, or 250,000 FP32 elements.
The script also accepts explicit binary units such as `1MiB` and `1GiB`.

For a smaller GPU check, or a different GPU type:

```sh
modal run modal_app.py::distributed_communication --world-sizes 2 \
    --data-sizes 1MB,10MB --warmup-steps 2 --steps 5

modal run modal_app.py::distributed_communication --gpu H100 \
    --world-sizes 2,4,6 --steps 50
```

`--gpu` takes a type such as `B200` or `H100`; the entrypoint derives the GPU count
from the largest requested world size. It defaults to the type selected by
`CS336_MODAL_GPU`, or B200 when that variable is unset. GPU counts below the
allocated maximum use the first N visible devices. The whole allocation stays
reserved during the sweep. Modal's [GPU guide](https://modal.com/docs/guide/gpu)
documents single-node GPU count selection.

For each measurement, workers refill a preallocated buffer, synchronize, and
barrier before starting the timer. Only `dist.all_reduce(SUM)` and its completion
wait are timed; CUDA synchronization is necessary even with `async_op=False`.
Refilling outside timing prevents repeated in-place reductions from overflowing.
The final result is checked against the expected sum on every rank. Raw timing
samples are gathered outside timing. Each reported sample is the **maximum across
ranks for that iteration**; `mean_ms` and `std_ms` summarize those samples, while
`rank_mean_ms` averages all rank/iteration observations. This follows the handout's
synchronization guidance and PyTorch's
[collective completion semantics](https://docs.pytorch.org/docs/2.11/distributed.html#synchronous-and-asynchronous-collective-operations).

The run downloads to `results/modal/<run-name>/`. Inside `all-reduce/` you get:

- `comparison.csv` and `comparison.md`: the table and two observations derived
  from measured results, to help write your assignment commentary.
- `latency.png`: latency by tensor size with standard-deviation error bars.
- `bandwidth.png`: normalized all-reduce bus bandwidth by tensor size.
- `world-2/`, `world-4/`, and `world-6/`: raw per-rank samples, hardware/software
  metadata, and subprocess logs.
- `sweep.json`: exit status and wall time for each process count. Completed sizes
  remain available if a later size fails.

The top-level `environment.json` also captures `nvidia-smi topo -m`, which helps
explain how GPU connectivity affects the results. Algorithmic bandwidth is
`tensor_bytes / seconds`; normalized bus bandwidth multiplies that by `2*(p-1)/p`.
It is a conventional comparison metric, not measured traffic on a physical link.
Each process-count run has a 300-second wall timeout, and collectives have a
120-second timeout. The Modal container's existing timeout covers the entire sweep.

For CPU development, run the same launcher with Gloo and small tensors:

```sh
uv run python -m cs336_systems.distributed_communication_benchmark \
    --backend gloo --world-sizes 2 --data-sizes 1MB,10MB \
    --warmup-steps 2 --steps 5
```

On a local multi-GPU machine, omit `--backend gloo` and optionally pass
`--output-dir results/my-all-reduce-run`. Launch with Python directly: the script
starts `torchrun` itself. CPU measurements validate the script; use NCCL GPU
measurements for the assignment results.

## Naive and flat DDP training and benchmarking (sections 5.2 and 5.3.1)

[`cs336_systems/ddp_train.py`](./cs336_systems/ddp_train.py) provides a full
optimizer-step training loop using your `cs336_systems.ddp.NaiveDDP`, the
assignment-1 `TransformerLM`, cross-entropy loss, and your AdamW implementation.
It initializes the process group, assigns one GPU to each worker, splits the
global batch evenly, averages gradients before clipping/stepping, logs the global
mean loss, and saves model/optimizer checkpoints on rank 0. The learning rate is
constant. Synthetic next-token batches are the default; `--train-data` accepts a
one-dimensional integer `.npy` token array. Each worker samples exclusively from
its own contiguous corpus shard, so ranks don't train on overlapping windows.

[`cs336_systems/ddp_benchmark.py`](./cs336_systems/ddp_benchmark.py) runs the same
step implementation with measurement boundaries for `naive_ddp_benchmarking`.
Run the required **one node, two GPUs, xl** experiment on Modal:

```sh
modal run modal_app.py::ddp_benchmark --run-name naive-ddp-xl
```

Defaults: two B200 GPUs, `xl` (d_model=2560, d_ff=10240, layers=32, heads=32),
vocabulary 10,000, context length 512, **global batch 4 / local batch 2**, FP32
parameters and computation, five warm-up updates, and ten measured updates.
Warm-up updates initialize AdamW state and actually change the model; they are
excluded from timing statistics. The benchmark reuses a preallocated synthetic
batch on each rank. It performs one all-reduce per parameter gradient, after
backpropagation, through your wrapper.

Select the implementation with `--wrapper naive` (the default) or `--wrapper flat`.
The flat variant uses your `FlatDDP`: pack gradients, run one SUM all-reduce,
average, and copy the values back. Its entire packing/communication/copyback
sequence is included in the synchronization timer. Both variants record their
wrapper name in JSON configuration, CSV summaries, and the report.

For `minimal_ddp_flat_benchmarking`, run both sequentially on the **same allocated
node** with the Modal-only `--wrapper both` option:

```sh
modal run modal_app.py::ddp_benchmark --wrapper both --gpu H100 \
    --run-name ddp-naive-vs-flat

# Or run just the flat version.
modal run modal_app.py::ddp_benchmark --wrapper flat --gpu H100 \
    --run-name ddp-flat-xl
```

Both jobs use the same model/data seeds and settings, with fresh worker processes
and optimizer state for each variant. Artifacts appear under
`naive-ddp-benchmark/` and `flat-ddp-benchmark/`; the run's top-level `summary.csv`
contains both implementations. `--wrapper naive|flat` also works with the local
`ddp_benchmark`/`ddp_train` modules and the Modal `ddp_train` entrypoint. Pass it
directly to the Modal entrypoint, not inside `--extra-args`. The local modules
accept a single wrapper per invocation. Use the same wrapper when resuming a
checkpoint (older checkpoints without the field are treated as naive).

If the flat variant runs out of memory when allocating its packed gradient
buffer, try the allocator's expandable segments option on **both** variants:

```sh
modal run modal_app.py::ddp_benchmark --wrapper both --gpu H100 \
    --allocator-conf 'expandable_segments:True' --run-name ddp-expandable
```

The launcher sets `PYTORCH_ALLOC_CONF` in the remote subprocess environment
before torchrun or its workers import PyTorch; setting a variable only in your
local shell does not automatically forward it to Modal. The selected setting is
recorded in job metadata, benchmark configuration, and CSV reports. The same
option is available on `ddp_train`. For local torchrun, set `PYTORCH_ALLOC_CONF`
in its environment before launching.

Expandable segments can reduce fragmentation, but cannot remove the additional
full-gradient buffer or guarantee that the model fits. The FP32 `xl` model has
about 12.69 GiB of gradients. Parameters, gradients, AdamW's two moment tensors,
and the flat buffer alone require about 63.45 GiB per rank, before other live
tensors, optimizer temporaries, and communication resources. BF16 autocast in
this script retains FP32 gradients, so the packed buffer remains the same size.
For more memory headroom, rerun both variants with `--gpu H200` and a fresh run
name. On CUDA failure, `rank-N-failure.json` and `rank-N-memory-summary.txt`
preserve allocator diagnostics without emptying the cache or retrying the step.
See PyTorch's [allocator configuration documentation](https://docs.pytorch.org/docs/2.11/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf).

Start with a tiny GPU check or customize the experiment:

```sh
modal run modal_app.py::ddp_benchmark --warmup-steps 1 --steps 2 \
    --extra-args '--d-model 32 --d-ff 64 --num-layers 1 --num-heads 4 --vocab-size 128 --context-length 16 --global-batch-size 4'

modal run modal_app.py::ddp_benchmark --steps 20 \
    --extra-args '--context-length 1024 --global-batch-size 8 --mixed-precision' \
    --run-name naive-ddp-xl-bf16
```

`--mixed-precision` uses BF16 autocast for forward computation while preserving
FP32 parameters, communicated gradients, optimizer state, and loss computation.
Gradient clipping is disabled by default in the benchmark and defaults to norm
1.0 in the training script; set `--max-grad-norm 0` to disable it in training.
Record those choices when comparing runs. `--gpu H100` changes the Modal GPU
type; `--world-size` changes the number of processes and allocated GPUs, although
the section 5.2 deliverable uses two. Global batch size must divide evenly across
the processes.

Benchmark artifacts download to
`results/modal/<run-name>/naive-ddp-benchmark/`:

- `timings.json`: configuration, parameter/gradient sizes, per-rank raw samples,
  timings, memory peaks, communication fraction, and measured token throughput.
- `ddp_summary.csv` and `report.md`: the requested step time, synchronization time,
  synchronization percentage, and description of the measurement setup.
- `rank-N.json`: each worker's samples and device information.
- `train.jsonl`: global mean loss at reporting intervals.

The total-step interval includes gradient clearing, forward/loss, backward,
gradient synchronization, optional clipping, and AdamW. CUDA is synchronized
before measuring communication, after communication, and after the optimizer
step. The communication interval includes the individual SUM all-reduces,
gradient averaging, Python/launch overhead, and waiting for peer ranks. The
pre-step barrier, input preparation, loss logging, and checkpoint writes are
outside timing. These synchronization boundaries can themselves affect latency.
No extra barrier is inserted between backward and gradient synchronization.

For each iteration, the primary timings average durations across ranks. Their
means and population standard deviations summarize the measured iterations.
`communication_percent = 100 * mean_communication_ms / mean_step_ms`; it uses
the same ranks and iterations in both terms. `max_rank_step` also records the
slowest rank per iteration and determines global tokens/second. The top-level
Modal `summary.csv` includes the DDP world size, global batch, and communication
percentage. GPU performance measurements have to be collected on an actual run.

For training on Modal, use:

```sh
# Synthetic data, useful for checking the training loop. Saves a final checkpoint.
modal run modal_app.py::ddp_train --steps 100 --run-name ddp-training

# Upload a token array once, then train from the mounted Volume.
modal volume put cs336-a2-results /path/to/train.npy datasets/train.npy
modal run modal_app.py::ddp_train --steps 1000 \
    --extra-args '--train-data /artifacts/datasets/train.npy --vocab-size 10000 --checkpoint-every 100' \
    --run-name ddp-corpus-training
```

Set vocabulary size to match the tokenizer. The file path passed to a Modal
worker must exist inside the container (e.g. on the mounted Volume). Your local
dataset is not automatically uploaded. Training stores `checkpoint.pt`,
`config.json`, `training_summary.json`, and `train.jsonl` in its job directory.
`--no-save-checkpoint` disables checkpoint writing; Modal's `--no-download`
keeps artifacts in the Volume without downloading the potentially large checkpoint.

Both scripts can also run locally under `torchrun`. A complete CPU development run:

```sh
uv run torchrun --standalone --nproc-per-node=2 -m cs336_systems.ddp_train \
    --backend gloo --steps 3 --d-model 8 --d-ff 16 --num-layers 1 \
    --num-heads 2 --vocab-size 16 --context-length 4 --global-batch-size 4 \
    --output-dir results/ddp/tiny-train
```

To benchmark locally, replace `ddp_train` with `ddp_benchmark`, add
`--warmup-steps 1`, and select a new output directory. For local GPUs use
`--backend nccl`. Launch these scripts with `torchrun` directly; they do not
spawn a second launcher internally. Each worker defaults to one PyTorch CPU
thread (`--cpu-threads`).

Resume training by passing `--resume <previous-job>/checkpoint.pt`, keeping the
same model, data, world size, optimizer settings, and global batch size, and
choosing a **new output directory**. `--steps` is the total target number of
updates, including completed updates. Model and optimizer state are restored on
every rank; deterministic batch selection continues at the saved update number.
Periodic checkpoints replace `checkpoint.pt` atomically. Benchmark checkpointing
is off by default. These scripts use the PyTorch
[distributed communication](https://docs.pytorch.org/docs/2.11/distributed.html)
and [torchrun](https://docs.pytorch.org/docs/2.11/elastic/run.html) interfaces.

## Submitting

To submit, run `./test_and_make_submission.sh` . This script will install your
code's dependencies, run tests, and create a gzipped tarball with the output. We
should be able to unzip your submitted tarball and run
`./test_and_make_submission.sh` to verify your test results.
