"""Step 2: export torchvision ResNet18 to ONNX and validate the file.

ONNX (Open Neural Network Exchange) is a portable file format for ML models.
PyTorch (and TensorFlow, JAX, ...) can export a trained model to .onnx;
many different runtimes (ONNX Runtime, TensorRT, OpenVINO, custom-chip SDKs) can
then load that same file to run inference.
It is the handoff format between "training-side" and "inference-side" tooling.

What this script does:
1. Builds ResNet18 with random (untrained) weights -- fine for benchmarks since
   the amount of math is the same whether the weights are random or trained.
2. Saves those weights to resnet18.pt so later steps can rebuild the exact
   same PyTorch model for apples-to-apples comparison.
3. Traces the model once with a dummy 224x224 RGB image and writes the result
   to resnet18.onnx (graph) + resnet18.onnx.data (weights, stored externally
   because they are large).
4. Runs onnx.checker to validate the file structure.
5. Prints a quick summary of the exported graph.
"""

import os

import onnx
import torch
from torchvision import models

MODEL_PATH = "resnet18.onnx"
WEIGHTS_PATH = "resnet18.pt"
INPUT_SHAPE = (1, 3, 224, 224)  # (batch, channels, height, width)
OPSET = 18  # ONNX operator-set version; dynamo exporter emits >= 18


def main() -> None:
    # Build ResNet18 with random (untrained) weights; eval() freezes BN stats and disables Dropout.
    model = models.resnet18(weights=None).eval()

    # Persist the weights so step 3 can rebuild the exact same PyTorch model and compare.
    torch.save(model.state_dict(), WEIGHTS_PATH)

    # One concrete input is required so the exporter can trace the computation.
    # Values do not matter; only shape and dtype do.
    dummy_input = torch.randn(*INPUT_SHAPE)

    # Trace the model and write out (graph + weights) as an ONNX file.
    # dynamic_axes marks axis 0 of input/output as "any size at runtime" (batch dimension).
    torch.onnx.export(
        model,
        dummy_input,
        MODEL_PATH,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=OPSET,
    )

    # Load the file back as a Python object (protobuf tree) for inspection.
    onnx_model = onnx.load(MODEL_PATH)

    # Validate graph structure against the ONNX spec. Raises on malformed files; does not run the model.
    onnx.checker.check_model(onnx_model)

    # Pretty-print an input/output shape as a list, keeping symbolic names like "batch".
    def shape_of(value_info) -> list:
        return [
            d.dim_param if d.HasField("dim_param") else d.dim_value
            for d in value_info.type.tensor_type.shape.dim
        ]

    # Count how many nodes of each op type the exported graph has (gives a feel for the model).
    size_kb = os.path.getsize(MODEL_PATH) / 1024
    graph = onnx_model.graph
    op_counts: dict[str, int] = {}
    for node in graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1

    print(f"wrote {MODEL_PATH}  ({size_kb:.1f} KB, {len(graph.node)} nodes)")
    print(f"opset: {onnx_model.opset_import[0].version}")
    print(f"input  : {graph.input[0].name:7} shape={shape_of(graph.input[0])}")
    print(f"output : {graph.output[0].name:7} shape={shape_of(graph.output[0])}")
    print("op counts:")
    for op, n in sorted(op_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {op:24} {n}")


if __name__ == "__main__":
    main()
