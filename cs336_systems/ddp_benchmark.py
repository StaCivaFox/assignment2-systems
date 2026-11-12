"""Sections 5.2/5.3: naive/flat/overlap DDP updates and gradient sync timings.

Defaults: naive wrapper, xl, global batch 4, context 512, FP32, 5 warm-ups, 10 samples.
Launch with torchrun or modal run modal_app.py::ddp_benchmark.
"""

from cs336_systems.ddp_train import main


if __name__ == "__main__":
    main(benchmark=True)
