"""Portable Apple Espresso network; frozen weights, differentiable image inputs.

Faceprint exposes raw heads; VisionFaceprint reproduces the original/mirror ensemble.
"""

from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Faceprint(nn.Module):
    """Input: aligned RGB [N,3,128,128] floats in [0,1]. No face detection/alignment."""

    def __init__(self, model_dir=None):
        super().__init__()
        root = (
            Path(model_dir) if model_dir else Path(__file__).resolve().parent / "model"
        )
        graph = json.loads((root / "graph.json").read_text())
        self.layers = graph["layers"]
        transform = graph["transform_params"]["data"]
        assert transform["is_network_bgr"] == 0 and not transform["center_mean"]
        assert all(transform[k] == 0 for k in ["bias_a", "bias_r", "bias_g", "bias_b"])
        self.input_scale = 255 * transform["scale"]
        self.ops = nn.ModuleDict()
        with np.load(root / "weights.npz", allow_pickle=False) as weights:
            for i, l in enumerate(self.layers):
                kind = l["type"]
                if kind == "convolution":
                    m = nn.Conv2d(
                        l["K"],
                        l["C"],
                        (l["Ny"], l["Nx"]),
                        stride=(l.get("stride_y", 1), l.get("stride_x", 1)),
                        groups=l["n_groups"],
                        bias=bool(l.get("has_biases")),
                    )
                elif kind == "inner_product":
                    m = nn.Linear(l["nB"], l["nC"], bias=bool(l.get("has_biases")))
                elif kind == "activation":
                    assert l["mode"] == 3
                    continue
                elif kind == "elementwise":
                    assert (
                        l["operation"] in (0, 1) and l["alpha"] == 1 and l["beta"] == 0
                    )
                    continue
                elif kind == "pool":
                    assert l["is_global"] == 1 and l["avg_or_max"] == 0
                    continue
                else:
                    raise ValueError(f"Unsupported operator {kind}")
                with torch.no_grad():
                    m.weight.copy_(
                        torch.from_numpy(weights[f"l{i}.weight"].astype(np.float32))
                    )
                    if m.bias is not None:
                        m.bias.copy_(torch.from_numpy(weights[f"l{i}.bias"].copy()))
                self.ops[f"l{i}"] = m
        self.requires_grad_(False)
        self.eval()

    def forward(self, aligned_rgb):
        if aligned_rgb.ndim != 4 or tuple(aligned_rgb.shape[1:]) != (3, 128, 128):
            raise ValueError("Expected aligned RGB tensor [N,3,128,128] in [0,1]")
        if not aligned_rgb.is_floating_point():
            raise TypeError(
                "Input must be floating point; convert uint8 images and divide by 255"
            )
        v = {"data": aligned_rgb * self.input_scale}
        for i, l in enumerate(self.layers):
            inputs = [v[name] for name in l["bottom"].split(",")]
            x = inputs[0]
            kind = l["type"]
            if kind == "convolution":
                pad = (l["pad_l"], l["pad_r"], l["pad_t"], l["pad_b"])
                if any(pad):
                    x = F.pad(x, pad)
                y = self.ops[f"l{i}"](x)
                if l.get("fused_relu"):
                    y = F.relu(y)
            elif kind == "inner_product":
                y = self.ops[f"l{i}"](x.flatten(1))
                if l.get("has_relu"):
                    y = F.relu(y)
            elif kind == "pool":
                y = x.mean(dim=(2, 3), keepdim=True)
            elif kind == "activation":
                y = torch.sigmoid(x)
            elif kind == "elementwise":
                y = x + inputs[1] if l["operation"] == 0 else x * inputs[1]
            else:
                raise RuntimeError(kind)
            v[l["top"]] = y
        features = v["flatten_output"].flatten(1)
        return {
            "features": features,
            "faceprint": F.normalize(features, dim=1, eps=1e-12),
            "confidence": v["confidence"].reshape(-1),
        }


class VisionFaceprint(Faceprint):
    """Vision's original/mirror ensemble on an already aligned RGB crop.

    descriptor matches Apple's unrenormalized average; faceprint is its unit
    vector for cosine losses. Confidence is averaged across the two passes.
    """

    def forward(self, aligned_rgb):
        count = len(aligned_rgb)
        heads = super().forward(torch.cat((aligned_rgb, aligned_rgb.flip(-1)), dim=0))
        features = heads["features"]
        centered = features - features.mean(dim=1, keepdim=True)
        unit = F.normalize(centered, dim=1, eps=1e-12)
        descriptor = (unit[:count] + unit[count:]) * 0.5
        confidence = (heads["confidence"][:count] + heads["confidence"][count:]) * 0.5
        return {
            "descriptor": descriptor,
            "faceprint": F.normalize(descriptor, dim=1, eps=1e-12),
            "confidence": confidence,
        }
