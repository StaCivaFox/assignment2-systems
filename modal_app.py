"""Modal runners for assignment 2. See README.md for commands and GPU selection."""

import json
import os
from pathlib import Path
import shlex
import subprocess

import modal

from cs336_systems.modal_runner import (
    ATTENTION_D_MODELS,
    ATTENTION_SEQ_LENGTHS,
    DDP_NSYS_ARGS,
    FSDP_NSYS_ARGS,
    archive_source,
    attention_jobs,
    benchmark_jobs,
    command_argv,
    ddp_jobs,
    fsdp_jobs,
    distributed_communication_jobs,
    execute_job,
    run_name as make_run_name,
    write_summary,
)


ROOT = Path(__file__).resolve().parent
REMOTE_ROOT = Path("/workspace")
ARTIFACTS = Path("/artifacts")
GPU = os.environ.get("CS336_MODAL_GPU", "B200")
VOLUME_NAME = os.environ.get("CS336_MODAL_VOLUME", "cs336-a2-results")
TIMEOUT = int(os.environ.get("CS336_MODAL_TIMEOUT", "3600"))

# The CUDA development image also supplies a host compiler for Triton/torch.compile.
# uv_sync normally installs path dependencies, but cs336-basics is uploaded as source
# below: skip installing that package while retaining its locked third-party deps.
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.2-devel-ubuntu24.04", add_python="3.12")
    .apt_install("nsight-systems-2025.3.2")
    .uv_sync(str(ROOT), extra_options="--no-install-package cs336-basics")
    .workdir(str(REMOTE_ROOT))
    .env(
        {
            "PYTHONPATH": "/workspace:/workspace/cs336-basics",
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "8",
            "CS336_MODAL_GPU": GPU,
            "CS336_MODAL_VOLUME": VOLUME_NAME,
            "CS336_MODAL_TIMEOUT": str(TIMEOUT),
        }
    )
)
for local, remote in (
    ("cs336_systems", "cs336_systems"),
    ("cs336-basics/cs336_basics", "cs336-basics/cs336_basics"),
    ("tests", "tests"),
    ("scripts", "scripts"),
):
    if (ROOT / local).is_dir():
        image = image.add_local_dir(ROOT / local, str(REMOTE_ROOT / remote), ignore=["**/__pycache__/**", "**/*.pyc"])
for path in [ROOT / "pyproject.toml", ROOT / "uv.lock", *sorted(ROOT.glob("*.py"))]:
    image = image.add_local_file(path, str(REMOTE_ROOT / path.name))

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App("cs336-a2-systems")


@app.function(
    image=image,
    gpu=GPU,
    cpu=8,
    memory=65536,
    timeout=TIMEOUT,
    volumes={str(ARTIFACTS): volume},
    max_containers=1,
    scaledown_window=2,
)
def run_jobs(name: str, jobs: list[dict], profile: bool = False, nsys_args: str = "", gpu_request: str = "") -> dict:
    import torch

    volume.reload()
    destination = ARTIFACTS / "runs" / make_run_name(name)
    destination.mkdir(parents=True, exist_ok=False)
    records = []
    try:
        hardware = {
            "gpu_request": gpu_request or GPU,
            "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
            "gpus": [{"name": torch.cuda.get_device_name(i), "memory_bytes": torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())],
        }
        hardware["nvidia_smi"] = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False).stdout
        hardware["gpu_topology"] = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True, check=False).stdout
        hardware["nsys_version"] = subprocess.run(["nsys", "--version"], capture_output=True, text=True, check=False).stdout
        (destination / "environment.json").write_text(json.dumps(hardware, indent=2) + "\n")
        (destination / "plan.json").write_text(json.dumps(jobs, indent=2) + "\n")
        # Save exactly the uploaded source, including uncommitted implementations.
        archive_source(REMOTE_ROOT, destination / "source.zip")
        volume.commit()
        for job in jobs:
            job_dir = destination / make_run_name(job["name"])
            records.append(execute_job(job, job_dir, REMOTE_ROOT, profile, nsys_args))
            write_summary(destination, records)
            volume.commit()
    finally:
        volume.commit()
    files = sorted(str(path.relative_to(ARTIFACTS)) for path in destination.rglob("*") if path.is_file())
    return {"records": records, "files": files}


def submit(jobs: list[dict], run_name: str, output_dir: str, profile: bool, nsys_args: str, download: bool, gpu_request: str = ""):
    name = make_run_name(run_name)
    local = Path(output_dir) / name
    if download and local.exists():
        raise ValueError(f"Output directory already exists: {local}; choose a new run-name")
    print(f"Run {name}: {len(jobs)} job(s), GPU={gpu_request or GPU}, timeout={TIMEOUT}s")
    print(f"Artifacts: Modal Volume {VOLUME_NAME}/runs/{name}")
    for job in jobs:
        argv = command_argv(job["argv"], ARTIFACTS / "runs" / name / job["name"], profile, nsys_args)
        print(shlex.join(argv))
    runner = run_jobs.with_options(gpu=gpu_request) if gpu_request else run_jobs
    result = runner.remote(name, jobs, profile, nsys_args, gpu_request)
    if download:
        for filename in result["files"]:
            relative = Path(filename).relative_to(Path("runs") / name)
            target = local / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as stream:
                for chunk in volume.read_file(filename):
                    stream.write(chunk)
        print(f"Downloaded results to {local.resolve()}")
    failed = [record["name"] for record in result["records"] if record["status"] != "success"]
    if failed:
        raise RuntimeError(f"{len(failed)} job(s) failed: {', '.join(failed)}. See stdout.log and job.json in the run artifacts.")


@app.local_entrypoint()
def benchmark(
    sizes: str = "small",
    modes: str = "forward,forward-backward,train",
    warmups: str = "5",
    steps: int = 10,
    extra_args: str = "",
    run_name: str = "",
    output_dir: str = "results/modal",
    nsys: bool = False,
    nsys_args: str = "",
    download: bool = True,
):
    """Sweep sizes, modes, and warm-up counts, using a fresh process for each case."""
    jobs = benchmark_jobs(sizes, modes, warmups, steps, extra_args)
    submit(jobs, run_name, output_dir, nsys, nsys_args, download)


@app.local_entrypoint()
def attention(
    d_models: str = ATTENTION_D_MODELS,
    seq_lengths: str = ATTENTION_SEQ_LENGTHS,
    implementations: str = "eager,compiled",
    warmup_steps: int = 5,
    steps: int = 100,
    extra_args: str = "",
    run_name: str = "",
    output_dir: str = "results/modal",
    nsys: bool = False,
    nsys_args: str = "",
    download: bool = True,
):
    """Sections 4.1/4.2: attention forward/backward timings and memory by shape."""
    jobs = attention_jobs(d_models, seq_lengths, implementations, warmup_steps, steps, extra_args)
    submit(jobs, run_name, output_dir, nsys, nsys_args, download)


@app.local_entrypoint()
def distributed_communication(
    world_sizes: str = "2,4,6",
    data_sizes: str = "1MB,10MB,100MB,1GB",
    warmup_steps: int = 5,
    steps: int = 20,
    gpu: str = "",
    run_name: str = "",
    output_dir: str = "results/modal",
    download: bool = True,
):
    """Section 5.1.1: FP32 all-reduce across 2/4/6 GPUs on the same node."""
    jobs = distributed_communication_jobs(world_sizes, data_sizes, warmup_steps, steps)
    gpu_type = gpu or GPU.split(":", 1)[0]
    if ":" in gpu_type:
        raise ValueError("--gpu takes a type such as B200 or H100; the count comes from --world-sizes")
    count = max(int(value.strip()) for value in world_sizes.split(","))
    submit(jobs, run_name, output_dir, False, "", download, gpu_request=f"{gpu_type}:{count}")


@app.local_entrypoint()
def ddp_benchmark(
    model_size: str = "xl",
    wrapper: str = "naive",
    allocator_conf: str = "",
    world_size: int = 2,
    steps: int = 10,
    warmup_steps: int = 5,
    extra_args: str = "",
    gpu: str = "",
    run_name: str = "",
    output_dir: str = "results/modal",
    nsys: bool = False,
    nsys_args: str = "",
    download: bool = True,
):
    """Sections 5.2/5.3: naive, flat, overlap, both (naive/flat), or all; optional Nsight capture."""
    if nsys_args and not nsys:
        raise ValueError("--nsys-args requires --nsys")
    jobs = ddp_jobs("benchmark", model_size, world_size, steps, warmup_steps, extra_args, wrapper=wrapper, allocator_conf=allocator_conf)
    gpu_type = gpu or GPU.split(":", 1)[0]
    if ":" in gpu_type:
        raise ValueError("--gpu takes a GPU type; set the count with --world-size")
    profile_args = f"{DDP_NSYS_ARGS} {nsys_args}".strip() if nsys else ""
    submit(jobs, run_name, output_dir, nsys, profile_args, download, gpu_request=f"{gpu_type}:{world_size}")


@app.local_entrypoint()
def ddp_train(
    model_size: str = "small",
    wrapper: str = "naive",
    allocator_conf: str = "",
    world_size: int = 2,
    steps: int = 100,
    extra_args: str = "",
    gpu: str = "",
    run_name: str = "",
    output_dir: str = "results/modal",
    download: bool = True,
):
    """Train with naive, flat, or overlap DDP, optional .npy tokens, and rank-0 checkpoints."""
    jobs = ddp_jobs("train", model_size, world_size, steps, 0, extra_args, wrapper=wrapper, allocator_conf=allocator_conf)
    gpu_type = gpu or GPU.split(":", 1)[0]
    if ":" in gpu_type:
        raise ValueError("--gpu takes a GPU type; set the count with --world-size")
    submit(jobs, run_name, output_dir, False, "", download, gpu_request=f"{gpu_type}:{world_size}")


@app.local_entrypoint()
def fsdp_benchmark(
    strategies: str = "ddp,optimizer_sharded,fsdp",
    model_size: str = "xl",
    compute_dtype: str = "float32",
    world_size: int = 2,
    steps: int = 10,
    warmup_steps: int = 5,
    extra_args: str = "",
    allocator_conf: str = "",
    gpu: str = "",
    run_name: str = "",
    output_dir: str = "results/modal",
    nsys: bool = False,
    nsys_args: str = "",
    download: bool = True,
):
    """Section 7: memory accounting and layer all-gather profiles on one node."""
    if nsys_args and not nsys:
        raise ValueError("--nsys-args requires --nsys")
    jobs = fsdp_jobs(strategies, model_size, world_size, compute_dtype, steps, warmup_steps, extra_args,
                     allocator_conf=allocator_conf, profile_layers=nsys)
    gpu_type = gpu or GPU.split(":", 1)[0]
    if ":" in gpu_type:
        raise ValueError("--gpu takes a GPU type; use --world-size for the count")
    profile_args = f"{FSDP_NSYS_ARGS} {nsys_args}".strip() if nsys else ""
    submit(jobs, run_name, output_dir, nsys, profile_args, download, gpu_request=f"{gpu_type}:{world_size}")


@app.local_entrypoint()
def run(
    command: str,
    run_name: str = "",
    output_dir: str = "results/modal",
    nsys: bool = False,
    nsys_args: str = "",
    download: bool = True,
):
    """Run a Python script/module, pytest, torchrun, or another executable (no shell)."""
    jobs = [{"name": "command", "argv": shlex.split(command)}]
    submit(jobs, run_name, output_dir, nsys, nsys_args, download)
