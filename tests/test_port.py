"""Synthetic fixtures only: no system weights, photos, or biometric data."""

from pathlib import Path
import json, struct, tempfile, unittest
import numpy as np
import torch
from export_model import export
from faceprint import Faceprint, VisionFaceprint
from alignment import crop_faces, vision_crop_rects


def fixture(root):
    rng = np.random.default_rng(17)
    weight = (rng.integers(-8, 9, (128, 3, 1, 1)) / 32).astype("<f2")
    bias = rng.normal(0, 0.1, 128).astype("<f4")
    head = (rng.integers(-4, 5, (1, 128)) / 64).astype("<f2")
    hb = np.array([0.2], dtype="<f4")
    blobs = {
        9: weight.tobytes(),
        2: bias.tobytes(),
        31: head.tobytes(),
        16: hb.tobytes(),
    }
    packed = (
        struct.pack("<Q", len(blobs))
        + b"".join(struct.pack("<QQ", k, len(v)) for k, v in blobs.items())
        + b"".join(blobs.values())
    )
    (root / "test.weights").write_bytes(packed)
    layers = [
        dict(
            type="convolution",
            name="conv",
            bottom="data",
            top="conv",
            K=3,
            C=128,
            Nx=1,
            Ny=1,
            n_groups=1,
            stride_x=2,
            stride_y=2,
            pad_l=0,
            pad_r=0,
            pad_t=0,
            pad_b=0,
            has_biases=1,
            blob_weights_f16=9,
            blob_biases=2,
        ),
        dict(
            type="pool",
            name="pool",
            bottom="conv",
            top="flatten_output",
            is_global=1,
            avg_or_max=0,
        ),
        dict(
            type="inner_product",
            name="fc",
            bottom="flatten_output",
            top="logit",
            nB=128,
            nC=1,
            has_biases=1,
            blob_weights_f16=31,
            blob_biases=16,
        ),
        dict(
            type="activation", name="sigmoid", bottom="logit", top="confidence", mode=3
        ),
        dict(type="unused_operator", name="unused", bottom="conv", top="unused"),
    ]
    graph = {
        "layers": layers,
        "storage": "test.weights",
        "transform_params": {
            "data": dict(
                scale=1 / 255,
                is_network_bgr=0,
                center_mean=0,
                bias_a=0,
                bias_r=0,
                bias_g=0,
                bias_b=0,
            )
        },
    }
    (root / "synthetic.espresso.net").write_text(json.dumps(graph))
    export(root, root / "model", "synthetic.espresso")
    return weight.astype("float32")[:, :, 0, 0], bias, head.astype("float32"), hb


class PortTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_export_heads_ensemble_and_gradients(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            w, b, h, hb = fixture(root)
            self.assertEqual(
                len(json.loads((root / "model/graph.json").read_text())["layers"]), 4
            )
            rng = np.random.default_rng(23)
            pixels = rng.uniform(0.1, 0.9, (2, 3, 128, 128)).astype("float32")

            def expected(image):
                features = image[:, :, ::2, ::2].mean((2, 3)) @ w.T + b
                centered = features - features.mean(1, keepdims=True)
                unit = centered / np.linalg.norm(centered, axis=1, keepdims=True)
                confidence = 1 / (1 + np.exp(-(features @ h.T + hb)))
                return features, unit, confidence[:, 0]

            raw, unit, c = expected(pixels)
            _, flipped, fc = expected(pixels[:, :, :, ::-1])
            x = torch.tensor(pixels, requires_grad=True)
            single = Faceprint(root / "model")
            out = single(x)
            np.testing.assert_allclose(out["features"].detach(), raw, atol=2e-6)
            np.testing.assert_allclose(out["confidence"].detach(), c, atol=2e-6)
            model = VisionFaceprint(root / "model")
            out = model(x)
            np.testing.assert_allclose(
                out["descriptor"].detach(), (unit + flipped) / 2, atol=2e-6
            )
            np.testing.assert_allclose(
                out["confidence"].detach(), (c + fc) / 2, atol=2e-6
            )
            torch.testing.assert_close(
                out["descriptor"], model(x.flip(-1))["descriptor"]
            )
            loss = (out["faceprint"][0] * torch.linspace(-1, 1, 128)).sum()
            loss.backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertGreater(x.grad.norm().item(), 0)
            self.assertTrue(
                all(not p.requires_grad and p.grad is None for p in model.parameters())
            )
            direction = x.grad / x.grad.norm()
            eps = 1e-3
            with torch.no_grad():

                def value(z):
                    return (model(z)["faceprint"][0] * torch.linspace(-1, 1, 128)).sum()

                finite = (value(x + eps * direction) - value(x - eps * direction)) / (
                    2 * eps
                )
            self.assertAlmostEqual(float(finite / x.grad.norm()), 1, delta=0.02)
            with self.assertRaises(ValueError):
                model(torch.zeros(1, 3, 64, 64))
            with self.assertRaises(TypeError):
                model(torch.zeros(1, 3, 128, 128, dtype=torch.uint8))

    def test_corrupt_weight_table_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            p = root / "test.weights"
            p.write_bytes(p.read_bytes()[:-1])
            with self.assertRaises(AssertionError):
                export(root, root / "invalid", "synthetic.espresso")

    def test_crop_geometry_and_image_gradient(self):
        rect = vision_crop_rects([[0.25, 0.25, 0.5, 0.5]], 100, 100)
        self.assertEqual(rect.tolist(), [[5, 5, 90, 90]])
        with self.assertRaises(ValueError):
            vision_crop_rects([[0, 0, 0, 1]], 100, 100)
        x = torch.linspace(0, 1, 3 * 256 * 256).reshape(1, 3, 256, 256).requires_grad_()
        crop = crop_faces(x, [[0.01, 0.2, 0.7, 0.7]])
        self.assertEqual(tuple(crop.shape), (1, 3, 128, 128))
        crop.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad.norm().item(), 0)


if __name__ == "__main__":
    unittest.main()
