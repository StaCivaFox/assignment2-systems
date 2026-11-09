import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from cs336_systems.ddp_train import TokenBatches, parse_args, summarize_ranks
from cs336_systems.modal_runner import ddp_jobs, write_summary


TINY = ["--d-model", "8", "--d-ff", "16", "--num-layers", "1", "--num-heads", "2",
        "--vocab-size", "16", "--context-length", "4", "--global-batch-size", "4"]


def launch(module, output, *extra):
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1")
    result = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc-per-node=2",
         "-m", f"cs336_systems.{module}", "--backend", "gloo", *TINY, "--output-dir", str(output), *extra],
        cwd=Path(__file__).resolve().parents[1], env=environment, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_paired_rank_aggregation_and_fraction():
    ranks = [{"samples_ms": {"step": [10, 20], "communication": [2, 4]}},
             {"samples_ms": {"step": [20, 30], "communication": [4, 6]}}]
    timings, fraction = summarize_ranks(ranks)
    assert timings["step"]["samples_ms"] == [15, 25]
    assert timings["step"]["std_ms"] == 5
    assert timings["max_rank_step"]["samples_ms"] == [20, 30]
    assert fraction == pytest.approx(0.2)


def test_corpus_shards_are_disjoint_and_next_token_aligned(tmp_path):
    path = tmp_path / "tokens.npy"
    np.save(path, np.arange(100, dtype=np.int32))
    args = parse_args([*TINY, "--vocab-size", "100", "--train-data", str(path)])
    batches = [TokenBatches(args, rank, 2, torch.device("cpu")) for rank in range(2)]
    for rank, source in enumerate(batches):
        x, y = source.get(5)
        assert x.shape == (2, 4)
        assert torch.equal(y, x + 1)
        assert x.min() >= rank * 50
        assert y.max() < (rank + 1) * 50
        assert torch.equal(source.get(5)[0], x)


def test_modal_ddp_plan_defaults_and_validation():
    command = ddp_jobs()[0]["argv"]
    assert "--nproc-per-node=2" in command
    assert "cs336_systems.ddp_benchmark" in command
    assert command[command.index("--model-size") + 1] == "xl"
    assert command[command.index("--backend") + 1] == "nccl"
    assert "cs336_systems.ddp_train" in ddp_jobs("train", warmup_steps=0)[0]["argv"]
    jobs = ddp_jobs(wrapper="both", allocator_conf="expandable_segments:True")
    assert all(job["allocator_conf"] == "expandable_segments:True" for job in jobs)
    assert [job["name"] for job in jobs] == ["naive-ddp-benchmark", "flat-ddp-benchmark"]
    assert [job["argv"][job["argv"].index("--wrapper") + 1] for job in jobs] == ["naive", "flat"]
    with pytest.raises(ValueError):
        ddp_jobs("train", warmup_steps=0, wrapper="both")
    with pytest.raises(ValueError):
        ddp_jobs(extra_args="--wrapper=flat")
    with pytest.raises(ValueError):
        ddp_jobs(extra_args="--backend=gloo")
    with pytest.raises(ValueError):
        ddp_jobs(warmup_steps=0)


def test_two_rank_benchmark_and_modal_summary(tmp_path):
    checkpoints = []
    records = []
    for wrapper in ("naive", "flat"):
        destination = tmp_path / wrapper
        launch("ddp_benchmark", destination, "--wrapper", wrapper, "--warmup-steps", "1", "--steps", "2", "--save-checkpoint")
        result = json.loads((destination / "timings.json").read_text())
        assert result["status"] == "success"
        assert result["config"]["wrapper"] == wrapper
        assert result["config"]["implementation"] == f"{wrapper}_ddp"
        assert result["config"]["world_size"] == 2
        assert result["config"]["global_batch_size"] == 4
        assert result["config"]["local_batch_size"] == 2
        assert result["gradient_all_reduce_calls_per_step"] == (1 if wrapper == "flat" else result["gradient_tensor_count"])
        for rank in result["ranks"]:
            assert len(rank["samples_ms"]["step"]) == 2
            assert all(total >= communication > 0 for total, communication in zip(rank["samples_ms"]["step"], rank["samples_ms"]["communication"]))
        assert 0 < result["communication_percent"] <= 100
        log = [json.loads(line) for line in (destination / "train.jsonl").read_text().splitlines()]
        assert log[-1]["optimizer_updates_including_warmup"] == 3
        checkpoints.append(torch.load(destination / "checkpoint.pt", weights_only=True))
        records.append({"name": wrapper, "status": "success", "returncode": 0})
        assert wrapper in (destination / "ddp_summary.csv").read_text()
        assert (destination / "report.md").read_text().startswith(f"{wrapper.capitalize()} DDP")
    # Both selectable wrappers must produce equivalent actual model updates.
    for name, expected in checkpoints[0]["model"].items():
        torch.testing.assert_close(checkpoints[1]["model"][name], expected, atol=1e-6, rtol=1e-5)
    write_summary(tmp_path, records)
    summary = (tmp_path / "summary.csv").read_text()
    assert "communication_percent" in summary
    assert "naive_ddp" in summary and "flat_ddp" in summary


def test_two_rank_training_resume_matches_uninterrupted_updates(tmp_path):
    path = tmp_path / "tokens.npy"
    np.save(path, np.arange(400, dtype=np.int32) % 16)
    original, partial, resumed = [tmp_path / name for name in ("original", "partial", "resumed")]
    common = ["--train-data", str(path), "--log-every", "1"]
    launch("ddp_train", original, *common, "--steps", "3")
    launch("ddp_train", partial, *common, "--steps", "2", "--checkpoint-every", "1")
    launch("ddp_train", resumed, *common, "--steps", "3", "--resume", str(partial / "checkpoint.pt"))
    expected = torch.load(original / "checkpoint.pt", weights_only=True)
    actual = torch.load(resumed / "checkpoint.pt", weights_only=True)
    before = torch.load(partial / "checkpoint.pt", weights_only=True)
    assert expected["completed_steps"] == actual["completed_steps"] == 3
    for name in expected["model"]:
        torch.testing.assert_close(actual["model"][name], expected["model"][name], atol=1e-7, rtol=1e-6)
    assert any(not torch.equal(before["model"][name], actual["model"][name]) for name in actual["model"])
    for param, state in expected["optimizer"]["state"].items():
        for key, value in state.items():
            torch.testing.assert_close(actual["optimizer"]["state"][param][key], value)
    assert len((resumed / "train.jsonl").read_text().splitlines()) == 1
