# ONNX Runtime quick experiment

A short hands-on experiment: export a torchvision model to ONNX, run it with ONNX Runtime, and learn where the time goes.
Six scripts, roughly two hours of work.

## Setup

Dependencies are declared in `pyproject.toml` and pinned in `uv.lock`.
With [uv](https://docs.astral.sh/uv/) installed:

```
uv sync
```

That creates a `.venv/` and installs the full locked set.
Then run any script with `uv run python 02_export.py` (or activate the venv).

Everything runs on CPU. No GPU required.

## Scripts

Run in order; each writes files the next step reads.

| script | what it does |
|---|---|
| `02_export.py` | Build ResNet18 (random weights), export to `resnet18.onnx` with a dynamic batch axis, validate with `onnx.checker`. |
| `03_verify.py` | Run the same input through PyTorch and ORT, confirm the outputs match (max abs diff ~1e-6). |
| `04_benchmark.py` | Time PyTorch eager vs ORT: warmup + 200 timed iterations, report mean / p50 / p95. |
| `05_profile.py` | Enable ORT's profiler, group op durations by type, list the top cost centers. |
| `06_sweep.py` | Sweep ORT graph-optimization level and thread count; also re-export a "raw" model (no fusions) and profile it to expose BatchNorm / Add / Relu individually. |
| `07_quantize.py` | Produce an int8 version of the model with `quantize_dynamic`; compare size, latency, and output drift against fp32. |

Each script has a module docstring explaining the step and expected result.

## Results

Numbers below are from one representative run on a 16-logical-core x86 Linux machine.
Latencies are per inference, batch=1, 224x224 input.

### PyTorch eager vs ONNX Runtime (step 4)

| engine | p50 | p95 |
|---|---:|---:|
| PyTorch eager | 11.46 ms | 14.17 ms |
| ONNX Runtime | 7.33 ms | 13.97 ms |

ORT is ~1.5x faster out of the box.

### Top ops by total time (step 5, 100 iters, ORT_ENABLE_ALL)

| op | % of time |
|---|---:|
| Conv | 96.4% |
| Gemm | 2.3% |
| MaxPool | 0.6% |
| other | <0.5% each |

`Conv` dominates.
`BatchNorm`, `Add`, `Relu` do not appear here: PyTorch's exporter folded BN into Conv, and ORT fused `Conv+Add+Relu` into a single `FusedConv` at session load.

### Graph optimization level x threads (step 6, p50 ms)

| opt_level | th=1 | th=2 | th=4 | th=8 |
|---|---:|---:|---:|---:|
| DISABLE_ALL | 35.51 | 18.74 | 11.65 | 10.03 |
| BASIC | 35.42 | 18.90 | 12.74 | 9.38 |
| EXTENDED | 35.61 | 18.79 | 12.16 | 9.41 |
| **ALL** | **29.24** | **14.88** | **8.00** | **5.97** |

The real win in ORT's optimization pipeline is `ORT_ENABLE_ALL`, which activates `NchwcTransformer` and reorders Conv weights into a blocked layout that the AVX kernels consume efficiently.
`BASIC` and `EXTENDED` give essentially no improvement over `DISABLE_ALL` on this model.

Thread scaling: nearly linear 1->2, decent 2->4, modest 4->8.
Beyond 8 would likely hurt for batch 1.

### Raw-model profile with ORT_DISABLE_ALL (step 6D)

With all fusions disabled, BatchNorm reappears as a first-class op:

| op | % of total op time |
|---|---:|
| Conv | ~85% |
| BatchNormalization | ~6% |
| Expand | ~3% |
| Relu | ~2% |
| other | <2% each |

That ~6% BN plus ~2% Relu is what the fusion saves at zero cost.

### Dynamic int8 quantization (step 7)

| | fp32 | int8 |
|---|---:|---:|
| size on disk | 44.7 MB | 11.2 MB (4.0x smaller) |
| p50 latency | ~30 ms | ~31 ms (no speedup) |
| top-1 class | 377 | 377 (matches) |
| max abs diff (logits) | - | 1.5e-1 |

4x smaller on disk, same latency on CPU.
Dynamic quantization is the wrong tool for CNNs on x86 without VNNI: the per-inference `DynamicQuantizeLinear` + `Mul` + `Cast` overhead eats the int8 win.
It is the right tool for transformers (MatMul-heavy, few quantize steps amortize).

## Takeaways

1. **For ResNet18 CPU batch 1, ORT with default settings is ~1.5x faster than PyTorch eager.**
   The gap grows to ~2x when ORT is configured well (ORT_ENABLE_ALL + the right thread count).
2. **Conv is where the time is** (96% in the production model, 85% with fusions off).
   Any work on this model's inference speed is really work on Conv.
3. **In ORT's four optimization levels, `ORT_ENABLE_ALL` is the only one that matters on CPU for this model.**
   BASIC and EXTENDED are near-no-ops because the exported graph is already clean and fusions on CNNs only pay once the data layout matches what the kernel wants.
4. **Dynamic int8 quantization is a size optimization here, not a latency one.**
   For latency gains on CNN inference you need static (calibrated) quantization, hardware with int8 instructions, or both.

## Open questions

Things I would want to understand better:

- How ORT picks the NCHWc block size (8 vs 16 channels) and whether this is driven by CPU feature detection at session-load time.
- The `Expand` / `Shape` / `Cast` scaffolding around BN in the raw export: what does the dynamo exporter decompose BN into, and why those specific ops.
- How much of ORT's advantage over PyTorch eager is MLAS kernels vs the graph layout.
  Running the ORT-optimized `.onnx` back through PyTorch's `torch.compile` would isolate this.
- What static quantization actually gives on the same model with a small ImageNet calibration set.
- Whether the dynamo exporter's `optimize=False` flag disables all exporter-side passes or only some.
