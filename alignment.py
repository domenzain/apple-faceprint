"""Recovered crop geometry plus an approximate differentiable native sampler.

Inputs must be upright RGB and Apple's original unaligned revision-3 detector
boxes in normalized bottom-left xywh coordinates. This does not detect faces.
"""

import math
import torch
from torch.nn import functional as F


def vision_crop_rects(boxes, width, height):
    """Return integer bottom-left xywh crops, exactly matching the traced geometry."""
    b = torch.as_tensor(boxes, dtype=torch.float64)
    if (
        b.ndim != 2
        or b.shape[1] != 4
        or not torch.isfinite(b).all()
        or (b[:, 2:] <= 0).any()
    ):
        raise ValueError("Expected finite [N,4] xywh boxes with positive sizes")
    pixels = b * b.new_tensor([width, height, width, height])
    # The native revision-3 scaling factor is stored as float32.
    pad = (1.7649999856948853 - 1) * pixels[:, 2:] * 0.5
    low = torch.floor(pixels[:, :2] - pad)
    high = torch.ceil(pixels[:, :2] + pixels[:, 2:] + pad)
    return torch.cat((low, high - low), dim=1).to(torch.int64)


def crop_faces(images, boxes):
    """Differentiable approximate sampler; box estimation/rounding has no gradient.

    The exact native rectangle and observed pyramid quantization are preserved.
    shortcut: bicubic antialiased pyramid substitutes for Apple's image decoding
    and resampling; use cached native crops when exact pixel parity is required.
    """
    if images.ndim != 4 or images.shape[1] != 3 or not images.is_floating_point():
        raise ValueError("Expected float upright RGB images [N,3,H,W] in [0,1]")
    height, width = images.shape[-2:]
    rects = vision_crop_rects(boxes, width, height)
    if len(rects) != len(images):
        raise ValueError("One detector box per image required")
    result = []
    for image, rect in zip(images, rects.detach().cpu().tolist()):
        x, y, w, h = rect
        scale = 2 ** max(0, math.floor(math.log2(min(w, h) / 128)))
        small = image.unsqueeze(0)
        sh, sw = max(1, height // scale), max(1, width // scale)
        if scale > 1:
            small = F.interpolate(
                small,
                size=(sh, sw),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            ).clamp(0, 1)
        x, y, w, h = (math.trunc(v / scale) for v in (x, y, w, h))
        top = sh - y - h
        left_pad, right_pad = max(-x, 0), max(x + w - sw, 0)
        top_pad, bottom_pad = max(-top, 0), max(top + h - sh, 0)
        small = F.pad(small, (left_pad, right_pad, top_pad, bottom_pad))
        x += left_pad
        top += top_pad
        crop = small[:, :, top : top + h, x : x + w]
        result.append(
            F.interpolate(
                crop,
                size=(128, 128),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        )
    return torch.cat(result, dim=0)
