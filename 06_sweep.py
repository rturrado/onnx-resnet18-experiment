"""Step 6: sweep ORT optimization level and thread count; also export a 'raw' model.

Two kinds of knobs affect inference speed on CPU:
- Graph-level: fuse operators, fold constants, change tensor layouts.
  These happen once at session load.
  ORT exposes four cumulative levels (DISABLE_ALL / BASIC / EXTENDED / ALL).
- Runtime-level: how many CPU threads a single inference call may use
  (intra_op_num_threads).

In parallel, two layers of fusion can hide what the model really looks like:
- Export-time (PyTorch): BatchNorm gets folded into the preceding Conv while
  writing the .onnx.
  Once that happens, no runtime option brings it back.
- Load-time (ORT): Conv+Add+Relu become FusedConv, etc.
  This we can disable.

What this script produces:
- resnet18_raw.onnx: re-export of ResNet18 skipping PyTorch's optimization
  pass, so BatchNorm/Add/Relu survive as separate nodes.
- resnet18_opt.onnx: ORT's own optimized form of resnet18.onnx (dumped so we
  can see which fusions ORT applies when loading).
- A sweep table of p50 latency across all opt-level x thread-count pairs,
  measured on resnet18.onnx (the production model).
- A profile of resnet18_raw.onnx under ORT_DISABLE_ALL, showing the ops that
  are normally hidden by fusions and how much time each one costs.
"""

import json
import os
import time
from collections import defaultdict

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torchvision import models

MODEL_PATH = "resnet18.onnx"
RAW_MODEL_PATH = "resnet18_raw.onnx"
OPT_MODEL_PATH = "resnet18_opt.onnx"
WEIGHTS_PATH = "resnet18.pt"
INPUT_SHAPE = (1, 3, 224, 224)
OPSET = 18

WARMUP = 10
ITERS = 100

# Cumulative: each level adds to the previous one.
#   DISABLE_ALL: run the graph exactly as loaded.
#   BASIC      : hardware-agnostic cleanup (constant folding, dead-node elimination, CSE).
#   EXTENDED   : operator fusions (Conv+Add+Relu -> FusedConv, MatMul+Add -> Gemm, ...).
#   ALL        : provider/hardware-specific transforms (CPU: NCHWc layout for AVX kernels).
#                Serialized models under ALL are no longer portable across hardware.
OPT_LEVELS = {
    "DISABLE_ALL": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    "BASIC": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    "EXTENDED": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    "ALL": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
}
THREAD_COUNTS = [1, 2, 4, 8]


def op_counts(path: str) -> dict[str, int]:
    """Count how many nodes of each op type a saved ONNX graph has."""
    graph = onnx.load(path).graph
    counts: dict[str, int] = defaultdict(int)
    for node in graph.node:
        counts[node.op_type] += 1
    return dict(counts)


def from_torch_to_raw_onnx() -> None:
    """Re-export without PyTorch's graph-optimization pass (keeps BN visible)."""
    model = models.resnet18(weights=None).eval()
    model.load_state_dict(torch.load(WEIGHTS_PATH, weights_only=True))
    dummy_input = torch.randn(*INPUT_SHAPE)  # same convention as 02_export.py
    torch.onnx.export(
        model,
        (dummy_input,),
        RAW_MODEL_PATH,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=OPSET,
        # optimize=False disables the dynamo exporter's post-trace graph-cleanup pass.
        # Without this flag, BatchNorm would be folded into Conv and never appear as a node.
        optimize=False,
    )


def from_onnx_to_opt_onnx(model_path: str, out_path: str) -> None:
    """Serialize ORT's internal post-optimization graph (fusions + layout transforms) to out_path."""
    sess_opts = ort.SessionOptions()
    # How aggressively ORT rewrites the graph at session load.
    # ORT_ENABLE_ALL enables operator fusion, constant folding, and provider-specific layout transforms
    # (e.g. NCHWc for CPU with AVX kernels).
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # Path where ORT dumps the post-optimization graph as ONNX.
    # Set once; the dump happens as a side effect of session construction.
    sess_opts.optimized_model_filepath = out_path
    # Constructing the session triggers optimization and the dump; no run needed.
    ort.InferenceSession(model_path, sess_opts, providers=["CPUExecutionProvider"])


def bench(model_path: str, opt_level, threads: int) -> dict:
    """Build an ORT session with the given settings and time ITERS runs."""
    rng = np.random.default_rng(0)
    input_np = rng.standard_normal(INPUT_SHAPE).astype(np.float32)

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = opt_level
    # How many CPU threads a single session.run call may use
    # (e.g. to parallelize matmul rows inside one Conv). 0 means "pick a default".
    sess_opts.intra_op_num_threads = threads
    session = ort.InferenceSession(model_path, sess_opts, providers=["CPUExecutionProvider"])

    for _ in range(WARMUP):
        session.run(["logits"], {"input": input_np})
    times_ns: list[int] = []
    for _ in range(ITERS):
        t0 = time.perf_counter_ns()
        session.run(["logits"], {"input": input_np})
        times_ns.append(time.perf_counter_ns() - t0)

    arr_ms = np.array(times_ns, dtype=np.float64) / 1e6
    return {
        "p50": float(np.percentile(arr_ms, 50)),
        "p95": float(np.percentile(arr_ms, 95)),
    }


def profile(model_path: str, opt_level, iters: int = 50) -> list[tuple[str, int, float]]:
    """Run with profiling on and return (op_name, count, total_ms) sorted by total time."""
    rng = np.random.default_rng(0)
    input_np = rng.standard_normal(INPUT_SHAPE).astype(np.float32)
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = opt_level
    sess_opts.enable_profiling = True
    session = ort.InferenceSession(model_path, sess_opts, providers=["CPUExecutionProvider"])
    for _ in range(iters):
        session.run(["logits"], {"input": input_np})
    profile_path = session.end_profiling()
    with open(profile_path) as f:
        events = json.load(f)
    os.remove(profile_path)

    totals_us: dict[str, int] = defaultdict(int)
    counts: dict[str, int] = defaultdict(int)
    for ev in events:
        if ev.get("cat") != "Node":
            continue
        op = ev.get("args", {}).get("op_name")
        if op is None:
            continue
        totals_us[op] += ev["dur"]
        counts[op] += 1
    rows = [(op, counts[op], totals_us[op] / 1000) for op in totals_us]
    rows.sort(key=lambda r: -r[2])
    return rows


def main() -> None:
    print("== A. Re-export ONNX file without PyTorch graph optimization ==")
    from_torch_to_raw_onnx()
    cnt_fused = op_counts(MODEL_PATH)
    cnt_raw = op_counts(RAW_MODEL_PATH)
    all_ops = sorted(set(cnt_fused) | set(cnt_raw))
    print(f"{'op':24} {'resnet18.onnx':>15} {'resnet18_raw.onnx':>20}")
    for op in all_ops:
        print(f"{op:24} {cnt_fused.get(op, 0):>15} {cnt_raw.get(op, 0):>20}")

    print("\n== B. Dump the ORT-optimized form of resnet18.onnx ==")
    from_onnx_to_opt_onnx(MODEL_PATH, OPT_MODEL_PATH)
    cnt_opt = op_counts(OPT_MODEL_PATH)
    for op, n in sorted(cnt_opt.items(), key=lambda kv: -kv[1]):
        print(f"  {op:24} {n}")

    print("\n== C. Benchmark: graph optimization level x intra-op threads (p50 ms) ==")
    header = f"{'opt_level':<12} " + " ".join(f"th={t:<2}" for t in THREAD_COUNTS)
    print(header)
    for name, level in OPT_LEVELS.items():
        row = [f"{name:<12}"]
        for threads in THREAD_COUNTS:
            res = bench(MODEL_PATH, level, threads)
            row.append(f"{res['p50']:5.2f} ")
        print(" ".join(row))

    print("\n== D. Profile raw model with ORT_DISABLE_ALL (ops stay unfused) ==")
    rows = profile(RAW_MODEL_PATH, ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
    print(f"{'op':24} {'count':>8} {'total ms':>10}")
    for op, n, total_ms in rows:
        print(f"{op:24} {n:>8} {total_ms:>10.2f}")


if __name__ == "__main__":
    main()
