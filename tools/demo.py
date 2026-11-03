"""Run the public portraits through PyTorch and render measured pipeline steps."""

from pathlib import Path
import argparse, json, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from faceprint import Faceprint, VisionFaceprint
from alignment import vision_crop_rects

p = argparse.ArgumentParser()
p.add_argument(
    "--native", required=True, type=Path, help="expected.npz from tools/prepare_demo.py"
)
a = p.parse_args()
names = ["reference", "match", "person-b", "person-c"]
labels = ["Reference", "Matching variation", "Fictional person B", "Fictional person C"]
native = np.load(a.native, allow_pickle=False)
assert native["files"].tolist() == names
crops = [Image.open(root / "examples/crops" / f"{n}.png").convert("RGB") for n in names]
x = (
    torch.from_numpy(
        np.stack([np.array(im).transpose(2, 0, 1) for im in crops])
    ).float()
    / 255
)
torch.set_num_threads(4)
with torch.no_grad():
    result = VisionFaceprint(root / "model")(x)
    raw = Faceprint(root / "model")(x)
features = result["faceprint"].numpy()
confidence = result["confidence"].numpy()
cosine = features @ features.T
error = float(abs(result["descriptor"].numpy() - native["descriptors"]).max())
confidence_error = float(abs(confidence - native["confidence"]).max())
assert error < 2e-6, error
assert confidence_error < 1e-4, confidence_error
assert abs(raw["features"].numpy() - native["raw_features"]).max() < 1e-4
report = {
    "descriptor_max_abs_error": error,
    "confidence_max_abs_error": confidence_error,
    "torch": torch.__version__,
    "examples": [
        {
            "name": n,
            "cosine_vs_reference": float(cosine[0, i]),
            "confidence": float(confidence[i]),
        }
        for i, n in enumerate(names)
    ],
}
(root / "private/demo").mkdir(parents=True, exist_ok=True)
(root / "private/demo/port-results.json").write_text(
    json.dumps(report, indent=2) + "\n"
)

font = lambda size: ImageFont.load_default(size=size)
canvas = Image.new("RGB", (1280, 1240), "#f6f6f2")
d = ImageDraw.Draw(canvas)
d.text((24, 18), "PORTRAIT TO FACEPRINT", font=font(32), fill="#182521")
d.text(
    (24, 62),
    "Synthetic portraits. Actual Apple model outputs, reproduced by PyTorch on Linux.",
    font=font(20),
    fill="#48564f",
)
columns = [24, 300, 500, 720, 1000]
headers = [
    "Mac: detect + crop",
    "128 x 128 RGB",
    "+ horizontal flip",
    "128-D descriptor",
    "Against reference",
]
for col, text in zip(columns, headers):
    d.text((col, 104), text, font=font(19), fill="#31483e")
limit = float(abs(features).max())
for i, (name, label, crop) in enumerate(zip(names, labels, crops)):
    top = 144 + 260 * i
    source = Image.open(root / "examples/assets" / f"{name}.png").convert("RGB")
    thumb = source.resize((216, 216), Image.Resampling.LANCZOS)
    td = ImageDraw.Draw(thumb)
    rx, ry, rw, rh = vision_crop_rects(native["boxes"][i : i + 1], *source.size)[
        0
    ].tolist()
    sx, sy = 216 / source.width, 216 / source.height
    td.rectangle(
        (
            max(1, rx * sx),
            max(1, (source.height - ry - rh) * sy),
            min(214, (rx + rw) * sx),
            min(214, (source.height - ry) * sy),
        ),
        outline="#19bfa4",
        width=3,
    )
    canvas.paste(thumb, (columns[0], top))
    canvas.paste(
        crop.resize((176, 176), Image.Resampling.NEAREST), (columns[1], top + 20)
    )
    canvas.paste(
        ImageOps.mirror(crop).resize((176, 176), Image.Resampling.NEAREST),
        (columns[2], top + 20),
    )
    for j, value in enumerate(features[i]):
        strength = min(1, abs(float(value)) / limit)
        base = (30, 108, 160) if value >= 0 else (213, 111, 43)
        color = tuple(round(246 + (v - 246) * strength) for v in base)
        cx = columns[3] + (j % 16) * 14
        cy = top + 40 + (j // 16) * 14
        d.rectangle((cx, cy, cx + 12, cy + 12), fill=color)
    d.text(
        (columns[3], top + 168),
        "center / normalize / mean",
        font=font(16),
        fill="#48564f",
    )
    d.text(
        (columns[4], top + 42),
        f"{cosine[0, i]:.4f}",
        font=font(40),
        fill="#17705b" if i < 2 else "#5c625f",
    )
    d.text((columns[4], top + 92), "cosine similarity", font=font(18), fill="#48564f")
    d.text(
        (columns[4], top + 146),
        f"confidence {confidence[i]:.4f}",
        font=font(18),
        fill="#48564f",
    )
    d.text((columns[0], top + 224), label, font=font(20), fill="#182521")
    if i < 3:
        d.line((24, top + 250, 1256, top + 250), fill="#d9ddd6", width=1)
d.text(
    (24, 1200),
    "Confidence is a separate head. Cosine similarity is not a match probability.",
    font=font(18),
    fill="#48564f",
)
canvas.save(root / "docs/pipeline.png", optimize=True)
rows = "\n".join(
    f"| {label} | {cosine[0, i]:.4f} | {confidence[i]:.4f} |"
    for i, label in enumerate(labels)
)
(root / "docs/demo.md").write_text(f"""# From portrait to faceprint

![Actual pipeline inputs and outputs](pipeline.png)

1. Native Vision detects the face and creates the RGB crop shown here.
2. PyTorch runs the crop and its horizontal mirror through the exported network.
3. Each embedding is mean-centered and normalized; the two are averaged. Confidence is averaged separately.
4. Normalize the final descriptors and take their dot product to compare faces.

| Portrait | Cosine vs reference | Confidence |
|---|---:|---:|
{rows}

The matching variation was generated from the supplied reference; B and C are new fictional people. These are examples, not a recognition benchmark. Confidence is not identity similarity, and cosine is not a probability.

On these four inputs, the Linux port's maximum error against native CPU Vision was **{error:.2g}** for the descriptor and **{confidence_error:.2g}** for confidence.

[Reproduce it](findings.md#reproduce-the-illustrated-demo) · [Image provenance and prompts](prompts.md)
""")
print(json.dumps(report, indent=2))
