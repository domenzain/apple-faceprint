# apple-faceprint

Run the face-recognition model already on your Mac in PyTorch on Linux. Frozen weights; gradients to pixels.

```sh
uv run export_model.py                  # on macOS
# Copy this checkout and model/ to Linux.
uv sync
uv run example.py face-crop.png
```

Currently tested: `facerec_fp4.0_fa1.4_md2_2_fp16`. Input: 128×128 RGB face crops.

`VisionFaceprint` reproduces Vision's descriptor and confidence; `Faceprint` exposes the raw heads. Face detection isn't included; the optional crop sampler approximates Apple's resampling.

Code only. Export your own system weights. [MIT](LICENSE).
