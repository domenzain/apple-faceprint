# apple-faceprint

Run the face-recognition model already on your Mac in PyTorch on Linux. Frozen weights; gradients to pixels.

```sh
uv run export_model.py                         # on macOS
# Copy this checkout and the exported model/ directory to Linux.
uv sync
uv run example.py examples/crops/reference.png
```

![The actual pipeline and comparisons](docs/pipeline.png)

[Demo and scores](docs/demo.md) · [Findings](docs/findings.md)

Currently tested: `facerec_fp4.0_fa1.4_md2_2_fp16`. Input: 128×128 RGB face crops. `VisionFaceprint` reproduces Vision's descriptor and confidence; `Faceprint` exposes the raw network heads.

Code and public demo images only. Export your own system weights. Face detection isn't included; the optional crop sampler approximates Apple's resampling. [MIT](LICENSE).
