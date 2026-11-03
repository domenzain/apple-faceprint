# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26", "pillow>=10.2"]
# ///
"""macOS: capture native inputs/outputs for the four explicitly public demo images."""

from pathlib import Path
import json, subprocess
import numpy as np
from PIL import Image

root = Path(__file__).resolve().parents[1]
work = root / "private/demo"
work.mkdir(parents=True, exist_ok=True)
crops = work / "crops"
crops.mkdir(exist_ok=True)
binary = work / "native_capture"
subprocess.run(
    [
        "clang",
        "-w",
        "-fobjc-arc",
        str(root / "tools/native_capture.m"),
        "-framework",
        "Foundation",
        "-framework",
        "Vision",
        "-framework",
        "CoreVideo",
        "-o",
        str(binary),
    ],
    check=True,
)
names = ["reference", "match", "person-b", "person-c"]
manifest = [
    {"file": name, "path": str(root / "examples/assets" / f"{name}.png")}
    for name in names
]
manifest_path = work / "manifest.json"
manifest_path.write_text(json.dumps(manifest))
result = work / "native.jsonl"
# The native runner refuses to overwrite prior results. Keep captures reproducible.
subprocess.run([str(binary), str(manifest_path), str(result), str(crops)], check=True)
rows = [json.loads(line) for line in result.read_text().splitlines()]
features = []
confidence = []
public_crops = root / "examples/crops"
public_crops.mkdir(exist_ok=True)
for i, row in enumerate(rows, 1):
    if "error" in row:
        raise RuntimeError(f"{row['file']}: {row['error']}")
    assert len(row["native_crops"]) == 1
    crop = row["native_crops"][0]
    assert (crop["width"], crop["height"], crop["format"]) == (128, 128, 1111970369)
    bgra = np.fromfile(crops / crop["file"], np.uint8).reshape(
        128, crop["stride"] // 4, 4
    )[:, :128]
    Image.fromarray(bgra[:, :, [2, 1, 0]]).save(public_crops / f"{row['file']}.png")
    features.append(np.fromfile(crops / f"{i:04d}-flatten_output.raw", np.float32))
    confidence.append(np.fromfile(crops / f"{i:04d}-confidence.raw", np.float32)[0])
np.savez_compressed(
    work / "expected.npz",
    files=np.array(names),
    descriptors=np.array([r["embedding"] for r in rows], np.float32),
    confidence=np.array([r["faceprint_confidence"] for r in rows]),
    raw_features=np.stack(features),
    raw_confidence=np.array(confidence),
    boxes=np.array([r["detector_box"] for r in rows]),
)
print(f"Public face crops: {public_crops}; native validation data: {work}")
