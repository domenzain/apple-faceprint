"""Run one or more 128x128 RGB face crops; print descriptors and pairwise cosines."""

from pathlib import Path
import argparse, json
import numpy as np
from PIL import Image, ImageOps
import torch
from faceprint import VisionFaceprint

p = argparse.ArgumentParser()
p.add_argument("images", nargs="+", type=Path)
p.add_argument("--model", type=Path, default=Path(__file__).parent / "model")
p.add_argument("--device", default="cpu")
a = p.parse_args()
images = []
for path in a.images:
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if im.size != (128, 128):
        p.error(f"{path}: expected a 128x128 face crop")
    images.append(np.array(im).transpose(2, 0, 1))
torch.set_num_threads(4)
x = torch.from_numpy(np.stack(images)).float().to(a.device) / 255
model = VisionFaceprint(a.model).to(a.device).eval()
with torch.no_grad():
    out = model(x)
print(
    json.dumps(
        {
            "files": [p.name for p in a.images],
            "confidence": out["confidence"].cpu().tolist(),
            "faceprint": out["faceprint"].cpu().tolist(),
            "cosine_similarity": (out["faceprint"] @ out["faceprint"].T).cpu().tolist(),
        },
        indent=2,
    )
)
