import itertools
import json
import weakref
from unittest.mock import Mock

import pytest
import torch

from cs336_systems import attention_benchmark as bench


def tiny_config(**overrides):
    return bench.AttentionConfig(**(dict(batch_size=2, seq_length=4, d_model=8, device="cpu", warmup_steps=2, steps=3) | overrides))


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_pairs_free_graphs_clear_gradients_and_exclude_warmup(monkeypatch):
    attention = bench.scaled_dot_product_attention
    outputs, gradients, shapes = [], [], []

    def run(q, k, v, mask):
        assert all(ref() is None for ref in outputs)
        assert all(t.grad is None for t in (q, k, v))
        assert all(t.requires_grad for t in (q, k, v))
        assert torch.is_grad_enabled()
        shapes.append(q.shape)
        if len(shapes) == 1:
            for tensor in (q, k, v):
                tensor.register_hook(lambda grad: gradients.append(grad.detach().clone()))
        output = attention(q, k, v, mask)
        outputs.append(weakref.ref(output))
        return output

    monkeypatch.setattr(bench, "scaled_dot_product_attention", run)
    result = bench.benchmark_attention(tiny_config())
    assert result["status"] == "success"
    assert shapes == [torch.Size([2, 4, 8])] * 5
    assert len(gradients) == 15
    assert all(torch.isfinite(grad).all() for grad in gradients)
    assert all(len(timing["samples_ms"]) == 3 for timing in result["timings"].values())
    assert result["memory"] == {}  # CPU measurements must not report fabricated GPU bytes.
    assert result["tensor_sizes_bytes"]["one_attention_matrix"] == 2 * 4 * 4 * 4


def test_synchronization_encloses_each_timing(monkeypatch):
    events = []
    clock = itertools.count()
    monkeypatch.setattr(bench, "synchronize", lambda device: events.append("sync"))

    def timer():
        events.append("timer")
        return next(clock)

    monkeypatch.setattr(bench, "default_timer", timer)
    result = bench.benchmark_attention(tiny_config(warmup_steps=1, steps=2))
    pair = ["sync", "timer", "sync", "timer", "timer", "sync", "timer"]
    assert events == pair + ["sync"] + pair * 2
    assert result["timings"]["forward"]["samples_ms"] == [1000.0, 1000.0]
    assert result["timings"]["backward"]["samples_ms"] == [1000.0, 1000.0]


@pytest.mark.parametrize("causal", [False, True])
def test_compilation_preserves_attention_outputs_and_gradients(causal):
    torch.manual_seed(7)
    q, k, v = [torch.randn(2, 4, 8, requires_grad=True) for _ in range(3)]
    mask = torch.ones(4, 4, dtype=torch.bool).tril() if causal else None
    upstream = torch.randn_like(q)
    compiled = torch.compile(bench.scaled_dot_product_attention, backend="aot_eager", fullgraph=True, dynamic=False)
    expected = bench.scaled_dot_product_attention(q, k, v, mask)
    actual = compiled(q, k, v, mask)
    torch.testing.assert_close(actual, expected)
    for got, want in zip(torch.autograd.grad(actual, (q, k, v), upstream), torch.autograd.grad(expected, (q, k, v), upstream)):
        torch.testing.assert_close(got, want)


def test_compiled_benchmark_warms_both_directions():
    result = bench.benchmark_attention(tiny_config(implementation="compiled", compile_backend="aot_eager"))
    assert result["status"] == "success", result.get("traceback")
    assert len(result["timings"]["forward"]["samples_ms"]) == 3
    assert len(result["timings"]["backward"]["samples_ms"]) == 3


@pytest.mark.parametrize("failure_phase", ["warmup_forward", "warmup_backward", "forward", "backward"])
def test_oom_preserves_failure_phase_and_partial_samples(monkeypatch, failure_phase):
    original = bench.scaled_dot_product_attention
    calls = 0

    def fail(*args):
        raise torch.cuda.OutOfMemoryError("simulated attention OOM")

    def attention(*args):
        nonlocal calls
        calls += 1
        failing_call = 1 if failure_phase.startswith("warmup") else 2
        if calls == failing_call and failure_phase.endswith("forward"):
            fail()
        output = original(*args)
        if calls == failing_call and failure_phase.endswith("backward"):
            output.register_hook(fail)
        return output

    monkeypatch.setattr(bench, "scaled_dot_product_attention", attention)
    result = bench.benchmark_attention(tiny_config(warmup_steps=1))
    assert result["status"] == "oom"
    assert result["failure_phase"] == failure_phase
    assert "simulated attention OOM" in result["error"]
    assert ("forward" in result["timings"]) == (failure_phase == "backward")


def test_wrapped_oom_is_distinguished_from_other_errors(monkeypatch):
    error = RuntimeError("compiler failed")
    error.__cause__ = torch.cuda.OutOfMemoryError("allocation failed")
    assert bench.is_oom(error)
    monkeypatch.setattr(bench, "scaled_dot_product_attention", Mock(side_effect=RuntimeError("unsupported operation")))
    assert bench.benchmark_attention(tiny_config())["status"] == "error"


@pytest.mark.parametrize("overrides", [{"steps": 0}, {"warmup_steps": 0}, {"seq_length": 0}, {"d_model": -1}, {"batch_size": 0}])
def test_invalid_config(overrides):
    with pytest.raises(ValueError):
        tiny_config(**overrides).validate()


def test_single_case_cli_writes_results(tmp_path):
    output = tmp_path / "timings.json"
    bench.main(
        [
            "--single-case",
            "--d-models",
            "8",
            "--seq-lengths",
            "4",
            "--implementations",
            "eager",
            "--device",
            "cpu",
            "--warmup-steps",
            "1",
            "--steps",
            "2",
            "--no-causal",
            "--output",
            str(output),
        ]
    )
    result = json.loads(output.read_text())
    assert result["config"]["batch_size"] == 8
    assert result["config"]["causal"] is False
    assert result["status"] == "success"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("implementation", ["eager", "compiled"])
def test_cuda_memory_before_backward(implementation):
    result = bench.benchmark_attention(tiny_config(device="cuda", implementation=implementation))
    assert result["status"] == "success", result.get("traceback")
    memory = result["memory"]
    assert len(memory["pre_backward_allocated_samples_bytes"]) == 3
    assert memory["pre_backward_allocated_bytes"] > memory["baseline_allocated_bytes"] > 0
    assert memory["peak_allocated_bytes"] >= memory["pre_backward_allocated_bytes"]
