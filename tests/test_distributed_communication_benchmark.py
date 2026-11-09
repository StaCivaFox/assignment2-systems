import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from cs336_systems.distributed_communication_benchmark import parse_data_sizes, parse_world_sizes, summarize_samples
from cs336_systems.modal_runner import distributed_communication_jobs


def test_sizes_are_fp32_bytes_and_default_sweep_matches_handout():
    assert parse_data_sizes("1MB,10MB,100MB,1GB") == [("1MB", 10**6), ("10MB", 10**7), ("100MB", 10**8), ("1GB", 10**9)]
    assert parse_data_sizes("1MiB")[0][1] == 2**20
    jobs = distributed_communication_jobs()
    assert len(jobs) == 1  # One Modal invocation, not three different nodes.
    command = jobs[0]["argv"]
    assert command[command.index("--world-sizes") + 1] == "2,4,6"
    assert command[command.index("--backend") + 1] == "nccl"
    assert command[-1] == "{output_dir}"


@pytest.mark.parametrize("value", ["1B", "0MB", "1MB,1000KB", "1mbps", ""])
def test_reject_invalid_byte_counts(value):
    with pytest.raises(ValueError):
        parse_data_sizes(value)


@pytest.mark.parametrize("value", ["1", "2,2", "0,2", "2,9", ""])
def test_reject_invalid_world_sizes(value):
    with pytest.raises(ValueError):
        parse_world_sizes(value)


def test_aggregate_uses_slowest_rank_each_iteration_and_correct_bandwidth_units():
    result = summarize_samples([[1.0, 4.0], [3.0, 2.0]], size_bytes=10**6)
    assert result["samples_ms"] == [3.0, 4.0]
    assert result["mean_ms"] == 3.5
    assert result["std_ms"] == 0.5
    assert result["rank_mean_ms"] == 2.5
    assert result["algorithmic_GBps"] == pytest.approx(1 / 3.5)
    assert result["bus_GBps"] == result["algorithmic_GBps"]  # p=2 normalization.
    assert summarize_samples([[1.0]] * 4, 10**6)["bus_GBps"] == 1.5


def test_two_rank_gloo_end_to_end(tmp_path):
    """Exercise real collectives, rank aggregation, JSON/CSV and plot generation."""
    destination = tmp_path / "all-reduce"
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "1"
    completed = subprocess.run(
        [sys.executable, "-m", "cs336_systems.distributed_communication_benchmark",
         "--backend", "gloo", "--world-sizes", "2", "--data-sizes", "16B,1KB",
         "--warmup-steps", "1", "--steps", "3", "--output-dir", str(destination)],
        cwd=Path(__file__).resolve().parents[1], env=environment, capture_output=True, text=True, timeout=90,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    results = json.loads((destination / "world-2/results.json").read_text())
    assert results["dtype"] == "float32"
    assert results["backend"] == "gloo"
    assert len(results["rows"]) == 2
    for row in results["rows"]:
        assert row["numel"] * 4 == row["size_bytes"]
        assert [len(values) for values in row["samples_by_rank_ms"]] == [3, 3]
        assert row["mean_ms"] > 0
    with (destination / "comparison.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 2
    for name in ("latency.png", "bandwidth.png", "comparison.md", "sweep.json"):
        assert (destination / name).stat().st_size > 0
