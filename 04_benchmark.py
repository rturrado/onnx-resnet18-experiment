"""Step 4: benchmark PyTorch eager vs ONNX Runtime on CPU.

Having confirmed in step 3 that both engines compute the same thing,
measure how fast each one does it.
"Eager" means PyTorch's default mode where ops run one by one as Python calls
them; it is the fairest baseline for a model that was written and trained in
PyTorch.

What this script does:
1. Builds one random input and feeds the same bytes to both engines.
2. For each engine: runs WARMUP untimed iterations (to let allocators and
   thread pools settle), then ITERS timed iterations.
3. Reports mean / p50 (median) / p95 / min in milliseconds.
4. Prints the ORT-over-PyTorch speedup at the p50.

The number this produces -- "ORT is Nx faster than PyTorch eager on CPU for
ResNet18 batch 1" -- is one of the honest, concrete observations the whole
experiment is trying to generate.
"""

import time

import numpy as np
import onnxruntime as ort
import torch
from torchvision import models

MODEL_PATH = "resnet18.onnx"
WEIGHTS_PATH = "resnet18.pt"
INPUT_SHAPE = (1, 3, 224, 224)
WARMUP = 20  # untimed iterations: let allocators, thread pools, kernels settle
ITERS = 200  # timed iterations


def bench_pytorch(input_torch: torch.Tensor) -> list[int]:
    # Rebuild the exact same model exported in step 2, in inference mode.
    model = models.resnet18(weights=None).eval()
    model.load_state_dict(torch.load(WEIGHTS_PATH, weights_only=True))

    # no_grad() off throughout: no autograd bookkeeping during timing.
    with torch.no_grad():
        for _ in range(WARMUP):
            model(input_torch)
        times_ns: list[int] = []
        for _ in range(ITERS):
            t0 = time.perf_counter_ns()
            model(input_torch)
            times_ns.append(time.perf_counter_ns() - t0)
    return times_ns


def bench_ort(input_np: np.ndarray) -> list[int]:
    # Create the ORT session once; expensive setup (graph load + optimization) is paid here.
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

    for _ in range(WARMUP):
        session.run(["logits"], {"input": input_np})
    times_ns: list[int] = []
    for _ in range(ITERS):
        t0 = time.perf_counter_ns()
        session.run(["logits"], {"input": input_np})
        times_ns.append(time.perf_counter_ns() - t0)
    return times_ns


def summarize(name: str, times_ns: list[int]) -> dict[str, float]:
    arr_ms = np.array(times_ns, dtype=np.float64) / 1e6
    stats = {
        "mean": float(arr_ms.mean()),
        "p50": float(np.percentile(arr_ms, 50)),
        "p95": float(np.percentile(arr_ms, 95)),
        "min": float(arr_ms.min()),
    }
    print(
        f"{name:9} "
        f"mean={stats['mean']:6.2f} ms, "
        f"p50={stats['p50']:6.2f} ms, "
        f"p95={stats['p95']:6.2f} ms, "
        f"min={stats['min']:6.2f} ms"
    )
    return stats


def main() -> None:
    rng = np.random.default_rng(0)
    input_np = rng.standard_normal(INPUT_SHAPE).astype(np.float32)
    input_torch = torch.from_numpy(input_np)

    print(f"warmup={WARMUP}  iters={ITERS}")
    print(f"torch threads = {torch.get_num_threads()}")
    print("ort default intra-op threads = (library default, uses all CPU cores)")
    print()

    pt_stats = summarize("pytorch", bench_pytorch(input_torch))
    ort_stats = summarize("ort", bench_ort(input_np))

    speedup = pt_stats["p50"] / ort_stats["p50"]
    print()
    print(f"speedup (p50): ort is {speedup:.2f}x pytorch eager")


if __name__ == "__main__":
    main()
