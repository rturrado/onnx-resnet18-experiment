"""Step 7: dynamic int8 quantization. Compare size, latency, and output drift.

"Quantization" means replacing fp32 (4-byte floating-point) weights and/or
activations with int8 (1-byte integer) values plus a scale factor.
The promise: 4x less memory, and ideally faster math on hardware that has
native int8 instructions.

Two main flavors:
- Dynamic quantization (what this script does): weights are quantized ahead
  of time; activations are quantized on the fly inside each kernel call.
  Cheap and simple -- no calibration dataset required.
- Static quantization: both weights and activations are quantized ahead of
  time, using a small calibration dataset to pick per-tensor scale factors.
  More work, but the runtime can fuse quantize/dequantize steps into the
  compute kernels, so the latency win is real.

What this script does:
1. Pre-processes the ONNX model (shape inference + constant folding, which
   the quantizer requires).
2. Runs quantize_dynamic to produce resnet18_int8.onnx with int8 weights.
3. Compares fp32 vs int8 on three dimensions: file size, latency (p50/p95),
   and output drift (max absolute difference in logits, top-1 class match).

Expected result for ResNet18 on x86 CPU: ~4x smaller on disk, roughly the
same latency (dynamic quantization is a bad fit for CNNs on CPU without
VNNI instructions; the quantize/dequantize overhead eats the int8 win),
and the predicted class still matches fp32.
"""

import os
import time
from collections import defaultdict

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process

MODEL_PATH = "resnet18.onnx"
PREPROC_MODEL_PATH = "resnet18_preproc.onnx"
QUANT_MODEL_PATH = "resnet18_int8.onnx"
INPUT_SHAPE = (1, 3, 224, 224)
WARMUP = 20
ITERS = 200


def total_size_mb(path: str) -> float:
    # ONNX stores big weights in a sibling *.data file; include it in the size.
    size = os.path.getsize(path)
    data_path = path + ".data"
    if os.path.exists(data_path):
        size += os.path.getsize(data_path)
    return size / (1024 * 1024)


def op_counts(path: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for node in onnx.load(path).graph.node:
        counts[node.op_type] += 1
    return dict(counts)


def bench_and_run(path: str) -> dict:
    """Benchmark a session and also return one output for correctness comparison."""
    rng = np.random.default_rng(0)
    input_np = rng.standard_normal(INPUT_SHAPE).astype(np.float32)

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(path, sess_opts, providers=["CPUExecutionProvider"])

    for _ in range(WARMUP):
        session.run(["logits"], {"input": input_np})
    times_ns: list[int] = []
    for _ in range(ITERS):
        t0 = time.perf_counter_ns()
        session.run(["logits"], {"input": input_np})
        times_ns.append(time.perf_counter_ns() - t0)

    (output,) = session.run(["logits"], {"input": input_np})
    arr_ms = np.array(times_ns, dtype=np.float64) / 1e6
    return {
        "p50": float(np.percentile(arr_ms, 50)),
        "p95": float(np.percentile(arr_ms, 95)),
        "output": output,
    }


def main() -> None:
    # Pre-process: shape inference + constant folding + symbolic shape resolution.
    # Required for the quantizer to reason about tensor shapes consistently.
    quant_pre_process(MODEL_PATH, PREPROC_MODEL_PATH)

    # Dynamic quantization: weights are quantized to int8 ahead of time,
    # activations are quantized on the fly inside the kernel.
    # No calibration dataset is needed.
    quantize_dynamic(PREPROC_MODEL_PATH, QUANT_MODEL_PATH, weight_type=QuantType.QInt8)

    size_fp32 = total_size_mb(MODEL_PATH)
    size_int8 = total_size_mb(QUANT_MODEL_PATH)
    print(f"size fp32 : {size_fp32:6.2f} MB")
    print(f"size int8 : {size_int8:6.2f} MB  (ratio {size_fp32 / size_int8:.1f}x smaller)")

    print("\nop counts (int8 model):")
    for op, n in sorted(op_counts(QUANT_MODEL_PATH).items(), key=lambda kv: -kv[1]):
        print(f"  {op:24} {n}")

    fp32 = bench_and_run(MODEL_PATH)
    int8 = bench_and_run(QUANT_MODEL_PATH)

    print(f"\nlatency fp32: p50={fp32['p50']:5.2f} ms  p95={fp32['p95']:5.2f} ms")
    print(f"latency int8: p50={int8['p50']:5.2f} ms  p95={int8['p95']:5.2f} ms")
    print(f"speedup p50 : {fp32['p50'] / int8['p50']:.2f}x")

    max_abs = float(np.max(np.abs(fp32["output"] - int8["output"])))
    top1_fp32 = int(np.argmax(fp32["output"]))
    top1_int8 = int(np.argmax(int8["output"]))
    print(f"\nmax abs diff (logits): {max_abs:.3e}")
    print(f"top-1 class  : fp32={top1_fp32}  int8={top1_int8}  match={top1_fp32 == top1_int8}")


if __name__ == "__main__":
    main()
