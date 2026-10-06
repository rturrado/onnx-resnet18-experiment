"""Step 3: run the ONNX model with ONNX Runtime and compare against PyTorch.

Export (step 2) is a place where silent bugs live: a wrong opset version,
a model layer that behaves differently in training vs eval mode,
control flow that only partially gets traced.
Before measuring anything, confirm the exported ONNX file actually computes
the same function as the original.

What this script does:
1. Rebuilds the exact PyTorch model from step 2 (same random weights,
   via the saved .pt file).
2. Feeds the same random input through both engines: PyTorch (eager) and
   ONNX Runtime (loading the .onnx file).
3. Compares the two 1000-logit outputs: max/mean absolute difference, and
   whether the top-1 predicted class matches.

Expected result: differences ~1e-6 (floating-point noise from a different
accumulation order), top-1 class matches. If the numbers were much bigger,
the export would be semantically broken and later measurements worthless.
"""

import numpy as np
import onnxruntime as ort
import torch
from torchvision import models

MODEL_PATH = "resnet18.onnx"
WEIGHTS_PATH = "resnet18.pt"
INPUT_SHAPE = (1, 3, 224, 224)
SEED = 0


def main() -> None:
    # Rebuild the same PyTorch model that step 2 exported, by loading its saved weights.
    # weights_only=True is the safe loader (no arbitrary code execution from the pickle).
    pytorch_model = models.resnet18(weights=None).eval()
    pytorch_model.load_state_dict(torch.load(WEIGHTS_PATH, weights_only=True))

    # Build one random input in NumPy; both engines will receive the exact same bytes.
    rng = np.random.default_rng(SEED)
    input_np = rng.standard_normal(INPUT_SHAPE).astype(np.float32)
    input_torch = torch.from_numpy(input_np)  # zero-copy view, same underlying data

    # Run PyTorch inference. no_grad() disables autograd bookkeeping (faster, less memory).
    with torch.no_grad():
        logits_pt = pytorch_model(input_torch).numpy()

    # Create an ORT session: loads the ONNX file and prepares it for execution on CPU.
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

    # Run ORT inference. Inputs are a dict keyed by the ONNX input names declared in step 2.
    # session.run returns a list (one array per requested output); unpack the single element.
    (logits_ort,) = session.run(["logits"], {"input": input_np})

    # Compare the two outputs numerically and semantically.
    assert logits_pt.shape == logits_ort.shape, (logits_pt.shape, logits_ort.shape)
    max_abs = float(np.max(np.abs(logits_pt - logits_ort)))
    mean_abs = float(np.mean(np.abs(logits_pt - logits_ort)))
    top1_pt = int(np.argmax(logits_pt))     # predicted class index from PyTorch
    top1_ort = int(np.argmax(logits_ort))   # predicted class index from ORT

    print(f"output shape : {logits_pt.shape}")
    print(f"max abs diff : {max_abs:.3e}")
    print(f"mean abs diff: {mean_abs:.3e}")
    print(f"top-1 class  : pytorch={top1_pt}  ort={top1_ort}  match={top1_pt == top1_ort}")


if __name__ == "__main__":
    main()
