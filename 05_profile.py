"""Step 5: profile ONNX Runtime and list the top operators by total time.

Knowing the whole model takes X ms is useful; knowing *where inside* those
X ms are spent is more useful. ORT has a built-in profiler: enable it on
the session and ORT writes one JSON event per op invocation in a format
compatible with Chrome's trace viewer. Aggregating by op type reveals the
real cost centers.

What this script does:
1. Builds an ORT session with profiling enabled.
2. Runs the model N times.
3. Closes profiling (ORT returns the path to the generated JSON file).
4. Parses the JSON, groups durations by op_name, and prints the ops sorted
   by total time.

Expected result for ResNet18 CPU batch 1: Conv dominates (~95%+). Any op
you might have expected (BatchNorm, Add, Relu) that does not appear in the
list has probably been fused into Conv (either at export time by PyTorch
or at session-load time by ORT). Step 6 explores how to re-expose them.
"""

import json
import os
from collections import defaultdict

import numpy as np
import onnxruntime as ort

MODEL_PATH = "resnet18.onnx"
INPUT_SHAPE = (1, 3, 224, 224)
WARMUP = 20
ITERS = 100


def main() -> None:
    rng = np.random.default_rng(0)
    input_np = rng.standard_normal(INPUT_SHAPE).astype(np.float32)

    # Enable profiling: when true, ORT emits one trace event per op invocation to a JSON file in cwd using the Chrome
    # tracing schema.
    # Can be opened in chrome://tracing or parsed as plain JSON; here we parse it to aggregate durations by op type.
    sess_opts = ort.SessionOptions()
    sess_opts.enable_profiling = True
    session = ort.InferenceSession(
        MODEL_PATH, sess_opts, providers=["CPUExecutionProvider"]
    )

    for _ in range(WARMUP):
        session.run(["logits"], {"input": input_np})
    for _ in range(ITERS):
        session.run(["logits"], {"input": input_np})

    # Close the profiling stream; returns the path of the JSON file written to cwd.
    profile_path = session.end_profiling()
    with open(profile_path) as f:
        events = json.load(f)

    # Each "Node" event is one op invocation. Group durations by op_type.
    totals_us: dict[str, int] = defaultdict(int)
    counts: dict[str, int] = defaultdict(int)
    total_runs_us = 0
    for ev in events:
        if ev.get("cat") != "Node":
            continue
        args = ev.get("args", {})
        op = args.get("op_name")
        if op is None:
            continue
        dur = ev["dur"]
        totals_us[op] += dur
        counts[op] += 1
        total_runs_us += dur

    print(f"profile file : {profile_path}")
    print(f"iters        : {ITERS} (plus {WARMUP} warmup)")
    print(f"total op time: {total_runs_us / 1000:.1f} ms "
          f"(summed over all iters)")
    print()
    print(f"{'op':20} {'count':>8} {'total ms':>10} {'avg us':>10} {'%':>6}")
    for op, total in sorted(totals_us.items(), key=lambda kv: -kv[1]):
        n = counts[op]
        pct = 100 * total / total_runs_us
        print(f"{op:20} {n:>8} {total/1000:>10.2f} {total/n:>10.1f} {pct:>5.1f}%")

    # Clean up the profile file so repeated runs don't pile up.
    os.remove(profile_path)


if __name__ == "__main__":
    main()
