import csv
import json
import os
from pathlib import Path
import sys
import zipfile

import pytest

from cs336_systems.modal_runner import archive_source, attention_jobs, benchmark_jobs, command_argv, execute_job, run_name, write_summary


def test_source_archive_handles_modal_epoch_timestamps(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    script = source / "example.py"
    script.write_text("print('example')\n")
    os.utime(script, (0, 0))
    (source / "__pycache__").mkdir()
    (source / "__pycache__" / "example.pyc").write_bytes(b"cache")
    output = tmp_path / "source.zip"
    archive_source(source, output)
    with zipfile.ZipFile(output) as archive:
        assert archive.namelist() == ["example.py"]
        assert archive.read("example.py").decode() == script.read_text()


def test_benchmark_sweep_covers_cartesian_product():
    jobs = benchmark_jobs("small,medium", "forward,train", "0,1,2,5", 10, "--context-length 256 --time-components")
    assert len(jobs) == len({job["name"] for job in jobs}) == 16
    assert jobs[0]["name"] == "small-forward-w0"
    assert jobs[-1]["name"] == "medium-train-w5"
    assert all(job["argv"][-2:] == ["--output", "{output_dir}/timings.json"] for job in jobs)


def test_attention_sweep_matches_handout():
    jobs = attention_jobs()
    assert len(jobs) == len({job["name"] for job in jobs}) == 40
    assert jobs[0]["name"] == "attention-eager-d16-s256"
    assert jobs[-1]["name"] == "attention-compiled-d128-s16384"
    assert all("--single-case" in job["argv"] for job in jobs)
    assert jobs[0]["argv"][jobs[0]["argv"].index("--steps") + 1] == "100"


@pytest.mark.parametrize("kwargs", [{"d_models": "0"}, {"seq_lengths": "16,16"}, {"implementations": "invalid"}, {"warmup_steps": 0}, {"extra_args": "--steps=2"}])
def test_attention_sweep_validation(kwargs):
    with pytest.raises(ValueError):
        attention_jobs(**kwargs)


def test_summary_retains_attention_oom(tmp_path):
    path = tmp_path / "oom"
    path.mkdir()
    result = {
        "config": {"steps": 100, "warmup_steps": 5, "implementation": "eager", "d_model": 16, "seq_length": 16384},
        "device_name": "test GPU",
        "status": "oom",
        "failure_phase": "warmup_forward",
        "timings": {},
        "memory": {},
    }
    (path / "timings.json").write_text(json.dumps(result))
    write_summary(tmp_path, [{"name": "oom", "status": "failed", "returncode": 1}])
    with (tmp_path / "summary.csv").open() as stream:
        row = next(csv.DictReader(stream))
    assert row["status"] == "oom"
    assert row["failure_phase"] == "warmup_forward"
    assert row["seq_length"] == "16384"
    assert row["mean_ms"] == ""


@pytest.mark.parametrize("compiled_status", ["success", "oom"])
def test_attention_comparison_speedups_require_success(tmp_path, compiled_status):
    records = []
    for variant, status, runtime in [("eager", "success", 6), ("compiled", compiled_status, 2)]:
        path = tmp_path / variant
        path.mkdir()
        result = {
            "config": {"steps": 100, "warmup_steps": 5, "implementation": variant, "d_model": 16, "seq_length": 256, "batch_size": 8, "dtype": "float32", "causal": True},
            "device_name": "test GPU",
            "status": status,
            "memory": {"pre_backward_allocated_bytes": 4096},
            "timings": {phase: {"mean_ms": runtime, "std_ms": 0} for phase in ("forward", "backward")},
        }
        (path / "timings.json").write_text(json.dumps(result))
        records.append({"name": variant, "status": "success" if status == "success" else "failed", "returncode": 0 if status == "success" else 1})
    write_summary(tmp_path, records)
    with (tmp_path / "attention_comparison.csv").open() as stream:
        row = next(csv.DictReader(stream))
    assert row["compiled_status"] == compiled_status
    assert row["forward_speedup"] == ("3.0" if compiled_status == "success" else "")
    assert row["backward_speedup"] == ("3.0" if compiled_status == "success" else "")


@pytest.mark.parametrize("extra", ["--output=elsewhere.json", "--device cpu", "--mode forward"])
def test_sweep_rejects_conflicting_arguments(extra):
    with pytest.raises(ValueError, match="override"):
        benchmark_jobs("small", "train", "5", 10, extra)


@pytest.mark.parametrize(
    "values",
    [
        ("unknown", "train", "5", 10),
        ("small", "invalid", "5", 10),
        ("small", "train", "-1", 10),
        ("small", "train", "5", 0),
        ("small,small", "train", "5", 10),
    ],
)
def test_invalid_sweep(values):
    with pytest.raises(ValueError):
        benchmark_jobs(*values)


def test_commands_preserve_arguments_and_use_image_python():
    command = command_argv(["python", "scripts/test.py", "a b", "$(echo unsafe)", "{output_dir}/result.json"], Path("/artifacts/my run"))
    assert command == [sys.executable, "-u", "scripts/test.py", "a b", "$(echo unsafe)", "/artifacts/my run/result.json"]
    distributed = command_argv(["torchrun", "--standalone", "--nproc-per-node=2", "-m", "cs336_systems.example"], Path("/tmp/result"))
    assert distributed[:4] == [sys.executable, "-u", "-m", "torch.distributed.run"]
    assert distributed[4:] == ["--standalone", "--nproc-per-node=2", "-m", "cs336_systems.example"]


def test_profile_wraps_command_and_preserves_extra_options():
    argv = command_argv(["pytest", "tests/test_attention.py"], Path("/tmp/run"), True, "--cuda-memory-usage=true")
    assert argv[:2] == ["nsys", "profile"]
    assert "--sample=none" in argv
    assert "--cuda-memory-usage=true" in argv
    assert argv[-7:] == ["--output", "/tmp/run/trace", sys.executable, "-u", "-m", "pytest", "tests/test_attention.py"]
    with pytest.raises(ValueError):
        command_argv(["python", "example.py"], Path("/tmp/run"), nsys_args="--cuda-memory-usage=true")


def test_subprocess_logging_environment_and_failure(tmp_path):
    script = "import os, sys; print(os.environ['CS336_OUTPUT_DIR']); print('stderr message', file=sys.stderr); sys.exit(3)"
    job = {"name": "failure", "argv": ["python", "-c", script]}
    destination = tmp_path / "failure"
    result = execute_job(job, destination, tmp_path)
    assert result["status"] == "failed"
    assert result["returncode"] == 3
    assert result == json.loads((destination / "job.json").read_text())
    log = (destination / "stdout.log").read_text()
    assert str(destination) in log
    assert "stderr message" in log
    with pytest.raises(FileExistsError):
        execute_job(job, destination, tmp_path)


@pytest.mark.parametrize("allocator_conf", ["", "expandable_segments:True"])
def test_allocator_setting_reaches_subprocess_and_is_recorded(tmp_path, monkeypatch, allocator_conf):
    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "backend:native")
    script = "import os, json; print(json.dumps({k: os.environ.get(k) for k in ('PYTORCH_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF')}))"
    destination = tmp_path / "allocator"
    result = execute_job({"name": "allocator", "argv": ["python", "-c", script], "allocator_conf": allocator_conf}, destination, tmp_path)
    values = json.loads((destination / "stdout.log").read_text())
    assert result["status"] == "success"
    assert result["allocator_conf"] == (allocator_conf or "backend:native")
    if allocator_conf:
        assert values == {"PYTORCH_ALLOC_CONF": allocator_conf, "PYTORCH_CUDA_ALLOC_CONF": None}
    else:
        assert values == {"PYTORCH_ALLOC_CONF": None, "PYTORCH_CUDA_ALLOC_CONF": "backend:native"}
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "backend:native"
    assert json.loads((destination / "job.json").read_text())["allocator_conf"] == result["allocator_conf"]


def test_missing_executable_still_records_failure(tmp_path):
    result = execute_job({"name": "missing", "argv": ["/nonexistent/cs336-test-program"]}, tmp_path / "missing", tmp_path)
    assert result["status"] == "failed"
    assert "FileNotFoundError" in result["error"]


def test_summary_keeps_success_and_failed_jobs(tmp_path):
    (tmp_path / "success").mkdir()
    (tmp_path / "success" / "timings.json").write_text(
        json.dumps(
            {
                "config": {"steps": 10, "warmup_steps": 5, "dtype": "float32", "mixed_precision": True, "context_length": 128, "profile_memory": True},
                "device_name": "test GPU",
                "timings": {"step": {"mean_ms": 1.0, "std_ms": 0.2}},
                "memory": {"peak_allocated_bytes": 1024, "peak_reserved_bytes": 2048},
            }
        )
    )
    records = [{"name": "success", "status": "success", "returncode": 0}, {"name": "failed", "status": "failed", "returncode": 1}]
    write_summary(tmp_path, records)
    with (tmp_path / "summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert rows[0]["mean_ms"] == "1.0"
    assert rows[0]["mixed_precision"] == "True"
    assert rows[0]["context_length"] == "128"
    assert rows[0]["peak_allocated_bytes"] == "1024"
    assert rows[0]["peak_reserved_bytes"] == "2048"
    assert rows[1]["status"] == "failed"
    assert rows[1]["mean_ms"] == ""


@pytest.mark.parametrize("name", ["../escape", "/absolute", "contains spaces", ""])
def test_run_names(name):
    if name:
        with pytest.raises(ValueError):
            run_name(name)
    else:
        assert run_name() != run_name()
