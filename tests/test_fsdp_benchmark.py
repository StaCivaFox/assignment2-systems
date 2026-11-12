import json
from pathlib import Path

import pytest
import torch

from cs336_basics.model import Embedding, Linear, RMSNorm
from cs336_systems.fsdp_benchmark import expected_memory, storage_breakdown
from cs336_systems.modal_runner import FSDP_NSYS_ARGS, command_argv, fsdp_jobs, write_summary
from .test_ddp_training import launch


def test_accounting_distinguishes_replicated_norm_states():
    module = torch.nn.Sequential(Embedding(5, 3), RMSNorm(3), Linear(3, 2))
    result = expected_memory(module, 2)
    assert result["parameter_count"] == 24
    assert result["sharded_parameter_count"] == 21
    assert result["replicated_parameter_count"] == 3
    assert result["ideal_bytes_per_rank"] == {"ddp": 384, "optimizer_sharded": 288, "fsdp": 216}
    assert result["ideal_saving_vs_optimizer_sharded_bytes"] == 72


def test_accounting_counts_aliased_gradients_once():
    module = torch.nn.Linear(2, 2, bias=False)
    module.weight.grad = torch.ones_like(module.weight)
    module.pending = [(None, None, module.weight.grad, module.weight.grad.view(-1))]
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    result = storage_breakdown(module, optimizer)
    assert result["master_weights_bytes"] == 16
    assert result["gradients_bytes"] == 16
    assert result["communication_buffers_bytes"] == 0
    assert result["accounted_storage_bytes"] == 32


def test_fsdp_modal_plan_and_capture():
    jobs = fsdp_jobs(profile_layers=True, allocator_conf="expandable_segments:True")
    assert len(jobs) == 3
    for strategy, job in zip(("ddp", "optimizer_sharded", "fsdp"), jobs, strict=True):
        assert job["name"] == f"{strategy}-float32-accounting"
        assert job["allocator_conf"] == "expandable_segments:True"
        command = command_argv(job["argv"], Path("/tmp/test"), True, FSDP_NSYS_ARGS)
        assert command[:2] == ["nsys", "profile"]
        assert "--nvtx-capture=fsdp_measurement" in command
        assert "--profile-layers" in command
        assert "--nproc-per-node=2" in command


@pytest.mark.parametrize("kwargs", [
    {"strategies": ""}, {"strategies": "fsdp,fsdp"}, {"strategies": "bogus"},
    {"compute_dtype": "fp8"}, {"world_size": 0}, {"warmup_steps": 0},
    {"extra_args": "--strategy=ddp"}, {"extra_args": "--compute-dtype float16"},
])
def test_fsdp_plan_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        fsdp_jobs(**kwargs)


def test_two_rank_accounting_strategies(tmp_path):
    results = {}
    records = []
    for strategy in ("ddp", "optimizer_sharded", "fsdp"):
        destination = tmp_path / strategy
        launch("fsdp_benchmark", destination, "--strategy", strategy, "--warmup-steps", "1", "--steps", "2", "--profile-layers")
        result = json.loads((destination / "timings.json").read_text())
        results[strategy] = result
        assert result["status"] == "success"
        assert result["memory"]["peak_allocated_bytes"] is None  # CPU is not a GPU memory measurement.
        for rank in result["ranks"]:
            phases = rank["memory_phases"]
            assert len(rank["samples_ms"]) == 2
            assert phases["after_initialization"]["optimizer_states_bytes"] == 0
            assert phases["before_optimizer"]["optimizer_states_bytes"] > 0
            assert phases["before_optimizer"]["communication_buffers_bytes"] == 0
            if strategy == "fsdp":
                assert result["config"]["fsdp_weight_lifecycle"] == "reshard_after_forward"
                assert phases["after_forward"]["gathered_weights_bytes"] == 0
                assert phases["after_backward_before_sync"]["gathered_weights_bytes"] > 0
                assert phases["after_backward_before_sync"]["communication_buffers_bytes"] > 0
                assert phases["before_optimizer"]["gathered_weights_bytes"] == 0
                forward_layers = {name.removeprefix("forward/") for name in rank["all_gather_path"] if name.startswith("forward/")}
                backward_layers = {name.removeprefix("backward/") for name in rank["all_gather_path"] if name.startswith("backward/")}
                assert forward_layers and forward_layers == backward_layers
                assert all(len(t["samples_ms"]) == 2 for t in rank["all_gather_path"].values())
        assert (destination / "memory.csv").is_file()
        assert (destination / "report.md").is_file()
        records.append({"name": strategy, "status": "success", "returncode": 0})
    # Same seeds, batches, optimizer, and four updates: validate actual training
    # trajectories in addition to report generation and tensor storage counts.
    for strategy in ("optimizer_sharded", "fsdp"):
        for baseline, candidate in zip(results["ddp"]["ranks"], results[strategy]["ranks"], strict=True):
            assert candidate["final_local_loss"] == pytest.approx(baseline["final_local_loss"], abs=1e-5)
    write_summary(tmp_path, records)
    summary = (tmp_path / "summary.csv").read_text()
    assert "compute_dtype" in summary
    assert all(f"accounting_{strategy}" in summary for strategy in results)


def test_two_rank_fsdp_bfloat16_accounting(tmp_path):
    launch("fsdp_benchmark", tmp_path, "--strategy", "fsdp", "--compute-dtype", "bfloat16", "--warmup-steps", "1", "--steps", "2")
    result = json.loads((tmp_path / "timings.json").read_text())
    assert result["config"]["mixed_precision"]
    for rank in result["ranks"]:
        phases = rank["memory_phases"]
        # Storage is released after forward and reallocated in BF16 for backward.
        assert phases["after_forward"]["gathered_weights_bytes"] == 0
        assert phases["after_backward_before_sync"]["gathered_weights_bytes"] == result["prediction"]["sharded_parameter_count"] * 2
        assert phases["before_optimizer"]["gradients_bytes"] == phases["before_optimizer"]["master_weights_bytes"]
