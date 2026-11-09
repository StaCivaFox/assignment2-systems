import itertools
import json
from contextlib import nullcontext
from unittest.mock import Mock

import pytest
import torch

from cs336_systems import benchmark as bench


def tiny_config(**overrides):
    values = dict(d_model=8, d_ff=16, num_layers=1, num_heads=2, vocab_size=16, context_length=4, batch_size=1, device="cpu", warmup_steps=2, steps=3)
    return bench.BenchmarkConfig(**(values | overrides))


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("mode", bench.MODES)
@pytest.mark.parametrize("annotate_attention", [False, True])
def test_modes_run_real_model_and_clear_gradients(monkeypatch, mode, annotate_attention):
    model = bench.TransformerLM(d_model=8, num_heads=2, d_ff=16, vocab_size=16, context_length=4, num_layers=1)
    before = [p.detach().clone() for p in model.parameters()]
    grad_enabled = []

    def before_forward(module, args):
        assert all(p.grad is None for p in module.parameters())
        grad_enabled.append(torch.is_grad_enabled())
        assert args[0].shape == (1, 4)

    model.register_forward_pre_hook(before_forward)
    monkeypatch.setattr(bench, "TransformerLM", lambda **kwargs: model)
    original_attention = bench.basics_model.scaled_dot_product_attention
    annotated = Mock(wraps=bench.annotated_scaled_dot_product_attention)
    monkeypatch.setattr(bench, "annotated_scaled_dot_product_attention", annotated)
    result = bench.benchmark(tiny_config(mode=mode, annotate_attention=annotate_attention))
    assert annotated.call_count == (5 if annotate_attention else 0)
    assert bench.basics_model.scaled_dot_product_attention is original_attention
    assert grad_enabled == [mode != "forward"] * 5
    assert len(result["timings"]["step"]["samples_ms"]) == 3
    assert all(t > 0 for t in result["timings"]["step"]["samples_ms"])
    assert all((p.grad is not None) == (mode != "forward") for p in model.parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    changed = any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))
    assert changed == (mode == "train")


def test_warmup_excluded_and_timer_enclosed_by_synchronization(monkeypatch):
    events = []
    clock = itertools.count()

    def timer():
        events.append("timer")
        return next(clock)

    monkeypatch.setattr(bench, "default_timer", timer)
    monkeypatch.setattr(bench, "synchronize", lambda device: events.append("sync"))
    result = bench.benchmark(tiny_config())
    assert events == ["sync", "timer", "sync", "timer"] * 5
    assert result["timings"]["step"] == {"samples_ms": [1000.0] * 3, "mean_ms": 1000.0, "std_ms": 0.0}


def test_synchronize_uses_selected_cuda_device_only(monkeypatch):
    sync = Mock()
    monkeypatch.setattr(torch.cuda, "synchronize", sync)
    bench.synchronize(torch.device("cpu"))
    sync.assert_not_called()
    bench.synchronize(torch.device("cuda:1"))
    sync.assert_called_once_with(torch.device("cuda:1"))


def test_component_timings_and_single_step_without_warmup():
    result = bench.benchmark(tiny_config(mode="train", time_components=True, warmup_steps=0, steps=1))
    timings = result["timings"]
    assert set(timings) == {"step", "forward", "loss", "backward", "optimizer"}
    assert all(len(t["samples_ms"]) == 1 and t["std_ms"] == 0 for t in timings.values())
    assert timings["step"]["mean_ms"] >= sum(t["mean_ms"] for name, t in timings.items() if name != "step")


@pytest.mark.parametrize("overrides", [{"steps": 0}, {"warmup_steps": -1}, {"batch_size": 0}, {"d_model": 7}, {"d_model": 6}, {"rope_theta": 0}])
def test_invalid_config(overrides):
    with pytest.raises(ValueError):
        tiny_config(**overrides).validate()


@pytest.mark.parametrize("mode", bench.MODES)
def test_whole_model_compile_preserves_gradients_and_updates(monkeypatch, mode):
    factory = bench.TransformerLM
    models = []

    def make_model(**kwargs):
        model = factory(**kwargs)
        models.append(model)
        return model

    monkeypatch.setattr(bench, "TransformerLM", make_model)
    for compiled in (False, True):
        result = bench.benchmark(tiny_config(mode=mode, warmup_steps=1, steps=2, compile_model=compiled, compile_backend="aot_eager"))
        assert len(result["timings"]["step"]["samples_ms"]) == 2
    for expected, actual in zip(models[0].parameters(), models[1].parameters()):
        torch.testing.assert_close(actual, expected)
        if mode != "forward":
            torch.testing.assert_close(actual.grad, expected.grad)


def test_compilation_requires_warmup():
    with pytest.raises(ValueError, match="warm-up"):
        tiny_config(compile_model=True, warmup_steps=0).validate()


def test_cli_overrides_preset_and_writes_json(tmp_path):
    output = tmp_path / "results" / "run.json"
    bench.main(
        [
            "--model-size",
            "10B",
            "--d-model",
            "8",
            "--d-ff",
            "16",
            "--num-layers",
            "1",
            "--num-heads",
            "2",
            "--vocab-size",
            "16",
            "--context-length",
            "4",
            "--batch-size",
            "1",
            "--device",
            "cpu",
            "--annotate-attention",
            "--mixed-precision",
            "--warmup-steps",
            "0",
            "--steps",
            "1",
            "--output",
            str(output),
        ]
    )
    result = json.loads(output.read_text())
    assert result["config"]["d_model"] == 8
    assert result["config"]["warmup_steps"] == 0
    assert result["config"]["annotate_attention"] is True
    assert result["config"]["mixed_precision"] is True
    assert result["config"]["memory_snapshot"] == str(output.parent / "memory_snapshot.pickle")
    assert len(result["timings"]["step"]["samples_ms"]) == 1


@pytest.mark.parametrize("masked", [False, True])
def test_annotated_attention_matches_outputs_and_gradients(masked):
    generator = torch.Generator().manual_seed(42)
    q = torch.randn(2, 3, 5, 4, dtype=torch.float64, generator=generator, requires_grad=True)
    k = torch.randn(2, 3, 7, 4, dtype=torch.float64, generator=generator, requires_grad=True)
    v = torch.randn(2, 3, 7, 6, dtype=torch.float64, generator=generator, requires_grad=True)
    mask = torch.ones(5, 7, dtype=torch.bool).tril() if masked else None
    expected = bench.basics_model.scaled_dot_product_attention(q, k, v, mask)
    actual = bench.annotated_scaled_dot_product_attention(q, k, v, mask)
    upstream = torch.randn(expected.shape, dtype=torch.float64, generator=generator)
    expected_grads = torch.autograd.grad(expected, (q, k, v), upstream)
    actual_grads = torch.autograd.grad(actual, (q, k, v), upstream)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


def test_attention_function_restored_after_failure(monkeypatch):
    original_attention = bench.basics_model.scaled_dot_product_attention
    monkeypatch.setattr(bench, "annotated_scaled_dot_product_attention", Mock(side_effect=RuntimeError("attention failed")))
    with pytest.raises(RuntimeError, match="attention failed"):
        bench.benchmark(tiny_config(annotate_attention=True))
    assert bench.basics_model.scaled_dot_product_attention is original_attention


def test_cpu_benchmark_never_calls_cuda_nvtx(monkeypatch):
    monkeypatch.setattr(bench.nvtx, "range", Mock(side_effect=AssertionError("CPU run called CUDA NVTX")))
    bench.benchmark(tiny_config(mode="train", annotate_attention=True, warmup_steps=0, steps=1))


@pytest.mark.parametrize("mode", bench.MODES)
def test_bf16_autocast_preserves_fp32_training_state(monkeypatch, mode):
    model = bench.TransformerLM(d_model=8, num_heads=2, d_ff=16, vocab_size=16, context_length=4, num_layers=1)
    before = [p.detach().clone() for p in model.parameters()]
    observed_logits = []
    observed_losses = []
    optimizers = []
    original_loss, original_optimizer = bench.cross_entropy, bench.AdamW

    def check_loss(logits, targets):
        assert logits.dtype == torch.float32
        assert not torch.is_autocast_enabled("cpu")
        loss = original_loss(logits, targets)
        observed_losses.append(loss.detach())
        return loss

    def make_optimizer(*args, **kwargs):
        optimizer = original_optimizer(*args, **kwargs)
        optimizers.append(optimizer)
        return optimizer

    def check_forward(module, args, output):
        assert torch.is_autocast_enabled("cpu")
        observed_logits.append(output.dtype)

    model.register_forward_hook(check_forward)
    monkeypatch.setattr(bench, "TransformerLM", lambda **kwargs: model)
    monkeypatch.setattr(bench, "cross_entropy", check_loss)
    monkeypatch.setattr(bench, "AdamW", make_optimizer)
    result = bench.benchmark(tiny_config(mode=mode, mixed_precision=True, annotate_attention=True))
    assert result["config"]["mixed_precision"] is True
    assert observed_logits == [torch.bfloat16] * 5
    assert len(observed_losses) == (0 if mode == "forward" else 5)
    assert all(loss.dtype == torch.float32 and torch.isfinite(loss) for loss in observed_losses)
    assert all(p.dtype == torch.float32 for p in model.parameters())
    assert all(p.grad.dtype == torch.float32 and torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters())) == (mode == "train")
    for optimizer in optimizers:
        assert all(value.dtype == torch.float32 for state in optimizer.state.values() for value in state.values() if isinstance(value, torch.Tensor))


@pytest.mark.parametrize("overrides", [{"mixed_precision": True, "dtype": "bfloat16"}, {"profile_memory": True}, {"memory_max_entries": 0}])
def test_invalid_precision_and_memory_config(overrides):
    with pytest.raises(ValueError):
        tiny_config(**overrides).validate()


@pytest.mark.parametrize("fail", [False, True])
def test_memory_history_lifecycle_and_failure_snapshot(monkeypatch, tmp_path, fail):
    events = []
    device = torch.device("cuda:1")
    snapshot = tmp_path / "nested" / "memory.pickle"
    config = tiny_config(profile_memory=True, memory_snapshot=str(snapshot), memory_max_entries=123)
    monkeypatch.setattr(torch.cuda, "device", lambda selected: nullcontext())
    monkeypatch.setattr(bench, "synchronize", lambda selected: events.append(("sync", selected)))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda selected: events.append(("reset", selected)))
    for name, value in [("memory_allocated", 10), ("memory_reserved", 20), ("max_memory_allocated", 30), ("max_memory_reserved", 40)]:
        monkeypatch.setattr(torch.cuda, name, lambda selected, value=value: value)
    monkeypatch.setattr(torch.cuda.memory, "_record_memory_history", lambda **kwargs: events.append(kwargs))
    dump = Mock(side_effect=lambda path: events.append(("dump", path)))
    monkeypatch.setattr(torch.cuda.memory, "_dump_snapshot", dump)
    with pytest.raises(RuntimeError, match="step failed") if fail else nullcontext():
        with bench.memory_profile(config, device) as stats:
            events.append("step")
            if fail:
                raise RuntimeError("step failed")
    assert events == [("sync", device), ("reset", device), {"max_entries": 123}, "step", ("dump", str(snapshot)), {"enabled": None}]
    assert stats["peak_allocated_bytes"] == 30
    assert stats["peak_reserved_bytes"] == 40
    assert json.loads(snapshot.with_suffix(".json").read_text())["memory"] == stats


def test_memory_recording_resets_after_warmup(monkeypatch):
    from contextlib import contextmanager

    events = []
    model = bench.TransformerLM(d_model=8, num_heads=2, d_ff=16, vocab_size=16, context_length=4, num_layers=1)
    model.register_forward_pre_hook(lambda module, args: events.append("forward"))
    monkeypatch.setattr(bench, "TransformerLM", lambda **kwargs: model)

    @contextmanager
    def recorder(config, device):
        assert all(p.grad is None for p in model.parameters())
        events.append("record")
        yield {"peak_allocated_bytes": 123}
        events.append("stop")

    monkeypatch.setattr(bench, "memory_profile", recorder)
    monkeypatch.setattr(bench, "reset_memory_measurement", lambda config, device, stats: events.append("reset"))
    config = tiny_config(mode="train", profile_memory=True)
    # Exercise orchestration on CPU; CUDA allocator behavior is tested separately.
    monkeypatch.setattr(config, "validate", lambda: None)
    result = bench.benchmark(config)
    assert events == ["record", "forward", "forward", "reset", "forward", "forward", "forward", "stop"]
    assert result["memory"]["peak_allocated_bytes"] == 123


@pytest.mark.parametrize("phase", ["initialization", "warmup", "measurement"])
@pytest.mark.parametrize("dump_fails", [False, True])
def test_oom_snapshot_covers_entire_benchmark(monkeypatch, tmp_path, phase, dump_fails):
    snapshot = tmp_path / "memory.pickle"
    config = tiny_config(mode="train", profile_memory=True, memory_snapshot=str(snapshot))
    # Use the real CPU model to test control flow with a simulated CUDA allocator.
    monkeypatch.setattr(config, "validate", lambda: None)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", Mock())
    for name in ("memory_allocated", "memory_reserved", "max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, lambda device: 1024)
    history = Mock()
    monkeypatch.setattr(torch.cuda.memory, "_record_memory_history", history)
    oom = torch.cuda.OutOfMemoryError(f"simulated {phase} OOM")
    model_factory = bench.TransformerLM

    def make_model(**kwargs):
        # Recording must be enabled before even constructing the model.
        assert history.call_count == 1
        if phase == "initialization":
            raise oom
        model = model_factory(**kwargs)
        calls = 0

        def before_forward(module, args):
            nonlocal calls
            calls += 1
            if calls == (1 if phase == "warmup" else config.warmup_steps + 1):
                raise oom

        model.register_forward_pre_hook(before_forward)
        return model

    def dump(path):
        if dump_fails:
            raise OSError("snapshot write failed")
        snapshot.write_bytes(b"simulated snapshot")

    monkeypatch.setattr(bench, "TransformerLM", make_model)
    dumper = Mock(side_effect=dump)
    monkeypatch.setattr(torch.cuda.memory, "_dump_snapshot", dumper)
    with pytest.raises(torch.cuda.OutOfMemoryError) as raised:
        bench.benchmark(config)
    assert raised.value is oom
    dumper.assert_called_once_with(str(snapshot))
    assert snapshot.exists() == (not dump_fails)
    memory = json.loads(snapshot.with_suffix(".json").read_text())["memory"]
    assert memory["phase"] == phase
    assert memory["status"] == "oom"
    assert memory["error"] == str(oom)
    assert history.call_args.kwargs == {"enabled": None}
    assert any(call.kwargs.get("clear_history") for call in history.call_args_list) == (phase == "measurement")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
@pytest.mark.parametrize("mode", ["forward", "train"])
@pytest.mark.parametrize("mixed_precision", [False, True])
def test_cuda_memory_snapshot(tmp_path, mode, mixed_precision):
    import pickle

    snapshot = tmp_path / "memory.pickle"
    result = bench.benchmark(tiny_config(device="cuda", mode=mode, mixed_precision=mixed_precision, profile_memory=True, memory_snapshot=str(snapshot), warmup_steps=1, steps=2))
    memory = result["memory"]
    assert memory["peak_allocated_bytes"] > memory["baseline_allocated_bytes"] > 0
    assert memory["peak_reserved_bytes"] >= memory["peak_allocated_bytes"]
    with snapshot.open("rb") as stream:
        data = pickle.load(stream)
    assert data["segments"]
    assert any(data["device_traces"])
