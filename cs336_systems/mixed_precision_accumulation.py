"""Section 2.1.5: accumulation experiments and ToyModel autocast dtype checks.

Run on CUDA for the handout's results; --device cpu is a local smoke test.
"""

import argparse

import torch
from torch import nn
from torch.nn import functional as F


def accumulation_experiment():
    """The handout's original four accumulation experiments (on CPU)."""
    s = torch.tensor(0, dtype=torch.float32)
    for _ in range(1000):
        s += torch.tensor(0.01, dtype=torch.float32)
    print("FP32 accumulator, FP32 increments:", s)

    s = torch.tensor(0, dtype=torch.float16)
    for _ in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    print("FP16 accumulator, FP16 increments:", s)

    s = torch.tensor(0, dtype=torch.float32)
    for _ in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    print("FP32 accumulator, FP16 increments:", s)

    s = torch.tensor(0, dtype=torch.float32)
    for _ in range(1000):
        x = torch.tensor(0.01, dtype=torch.float16)
        s += x.type(torch.float32)
    print("FP32 accumulator, FP16 increments explicitly cast to FP32:", s)


class ToyModel(nn.Module):
    """The model from benchmarking_mixed_precision(a), unchanged."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x


def check_dtype(name: str, tensor: torch.Tensor, expected: torch.dtype | None = None):
    print(f"{name:<45} {str(tensor.dtype):<16} shape={tuple(tensor.shape)}")
    if expected is not None and tensor.dtype != expected:
        raise AssertionError(f"{name}: expected {expected}, got {tensor.dtype}")


def toy_model_experiment(device: torch.device, dtype: torch.dtype):
    """Inspect actual tensors; use built-in cross-entropy as the example loss.

    CUDA autocast policies: https://docs.pytorch.org/docs/2.11/amp.html
    This is one diagnostic backward pass, not an FP16 training loop; no optimizer
    or loss scaler is needed to inspect the parameter gradient storage dtype.
    """
    torch.manual_seed(0)
    model = ToyModel(in_features=16, out_features=5).to(device=device, dtype=torch.float32)
    inputs = torch.randn(4, 16, device=device, dtype=torch.float32)
    targets = torch.randint(5, (4,), device=device)
    print(f"\nToyModel: device={device}, autocast={dtype}, torch={torch.__version__}")
    if device.type == "cpu":
        print("CPU smoke test: LayerNorm autocast policy differs from CUDA; use CUDA for the assignment.")

    # Hooks inspect the real forward pass without reimplementing or rerunning it.
    def output_hook(name, expected):
        def hook(module, args, output):
            check_dtype(name, output, expected)

        return hook

    handles = [
        model.fc1.register_forward_hook(output_hook("fc1 output", dtype)),
        model.relu.register_forward_hook(output_hook("ReLU output", dtype)),
        model.ln.register_forward_hook(output_hook("LayerNorm output", torch.float32 if device.type == "cuda" else None)),
        model.fc2.register_forward_hook(output_hook("fc2 output / predicted logits", dtype)),
    ]
    try:
        # Do not call model.half(): autocast selects operation dtypes while keeping
        # parameter storage in FP32. Compute the loss inside the same context.
        with torch.autocast(device_type=device.type, dtype=dtype):
            for name, parameter in model.named_parameters():
                check_dtype(f"parameter inside autocast: {name}", parameter, torch.float32)
            check_dtype("input", inputs, torch.float32)
            check_dtype("class-index targets", targets, torch.int64)
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)
            check_dtype("cross-entropy loss", loss, torch.float32)

        # Backward runs outside autocast. Gradients accumulate into FP32 parameters.
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                raise AssertionError(f"Missing gradient for {name}")
            check_dtype(f"parameter gradient: {name}", parameter.grad, torch.float32)
            if not torch.isfinite(parameter.grad).all().item():
                raise AssertionError(f"Nonfinite gradient for {name}")
        print(f"Loss value: {loss.item():.6f}; dtype checks passed.")
    finally:
        for handle in handles:
            handle.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("all", "accumulation", "toy"), default="all")
    parser.add_argument("--device", default="cuda", help="cuda, cuda:N, or cpu (smoke test only)")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16", help="autocast dtype for ToyModel; parameters remain FP32")
    args = parser.parse_args()
    device = torch.device(args.device)
    if args.experiment in ("all", "toy"):
        if device.type not in ("cpu", "cuda"):
            parser.error("ToyModel supports cpu or cuda")
        if device.type == "cuda" and not torch.cuda.is_available():
            parser.error("CUDA is unavailable; run on Modal or use --device cpu for a smoke test")
    if args.experiment in ("all", "accumulation"):
        accumulation_experiment()
    if args.experiment in ("all", "toy"):
        toy_model_experiment(device, getattr(torch, args.dtype))


if __name__ == "__main__":
    main()
