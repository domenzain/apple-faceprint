# Findings

- **Model:** `VisionCore.framework/Resources/facerec_fp4.0_fa1.4_md2_2_fp16.espresso.net`. The tested Vision request selects this RGB model. The grayscale `Vision.framework/Resources/faceprint_ek_fp16.bundle` is a different asset, not supported by this exporter.
- **Input/output:** RGB 128×128 → 128-dimensional embedding plus sigmoid confidence. The exported embedding/confidence branches have 4,460,033 parameters. Other attribute heads are pruned.
- **Container:** the installed `.net` is a single-chunk PBZE/LZFSE JSON graph. Its weight file has a little-endian blob index, float16 convolution/linear weights, and float32 biases. `export_model.py` reads the weight filename from the graph.
- **Vision descriptor:** run the crop and its horizontal mirror; subtract each feature vector's mean, L2-normalize each, then average. Average the two confidences too. Vision does not normalize the final average; `VisionFaceprint.faceprint` does so for cosine comparisons.
- **Crop:** for the tested revision-3 detector, expand the original **unaligned** normalized box by float32(1.765), then round pixel bounds outward. The crop does not use the rotated box or landmark shape. `alignment.py` preserves that geometry and approximates the power-of-two pyramid/resampling. Detection, exact edge handling, and quarter-turn faces are outside the port.

Validated on macOS 15.8.2 (Intel) and Linux x86_64 float32. Other builds may change assets or private APIs. CUDA and mixed precision are untested.

## Use in a loss

```python
import torch.nn.functional as F
from faceprint import VisionFaceprint

model = VisionFaceprint("model").to(device)
query = model(generated_face_crops)["faceprint"]
loss = (1 - F.cosine_similarity(query, reference_faceprints)).mean()
loss.backward()  # weights stay frozen; pixels receive gradients
```

Keep preprocessing consistent between reference and generated crops. Do not detach generated pixels or wrap the model call in `no_grad` when training. For CUDA, use the training job's PyTorch environment; `uv sync` defaults to CPU on Linux.

## Reproduce the illustrated demo

On macOS, after exporting weights:

```sh
uv run tools/prepare_demo.py
```

This compiles the private-API capture tool, detects the largest face in each public example, saves the actual native 128×128 crops, and writes comparison data under ignored `private/demo/`. The capture refuses to overwrite an existing `native.jsonl`.

Copy the checkout, `model/`, and `private/demo/expected.npz` to Linux, then:

```sh
uv sync
uv run tools/demo.py --native private/demo/expected.npz
uv run python -m unittest discover -s tests -v
```

The tests use a tiny synthetic graph and invented weights; no downloaded or system models are needed. The demo compares actual native and PyTorch outputs on the public portraits.
