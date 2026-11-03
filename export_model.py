# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26"]
# ///
"""Extract the installed macOS Espresso faceprint/confidence graph into portable files."""

from pathlib import Path
import argparse, ctypes, hashlib, json, struct
import numpy as np

MODEL = "facerec_fp4.0_fa1.4_md2_2_fp16.espresso"
SOURCE = Path(
    "/System/Library/PrivateFrameworks/VisionCore.framework/Versions/A/Resources"
)


def decompress(data):
    if not data.startswith(b"pbze"):
        return data
    # This installed asset has one LZFSE chunk; fail on another container layout.
    raw_size, packed_size = struct.unpack_from(">QQ", data, 12)
    assert 28 + packed_size == len(data) and raw_size < 10_000_000
    lib = ctypes.CDLL("/usr/lib/libcompression.dylib")
    decode = lib.compression_decode_buffer
    decode.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    decode.restype = ctypes.c_size_t
    buf = ctypes.create_string_buffer(raw_size)
    size = decode(buf, raw_size, data[28:], packed_size, None, 0x801)
    assert size == raw_size
    return buf.raw


def export(source, destination, model=MODEL):
    net_bytes = (source / (model + ".net")).read_bytes()
    net = json.loads(decompress(net_bytes))
    weights_bytes = (source / net["storage"]).read_bytes()
    count = struct.unpack_from("<Q", weights_bytes)[0]
    assert count < 10000
    offset = 8 + 16 * count
    blobs = {}
    for i in range(count):
        key, length = struct.unpack_from("<QQ", weights_bytes, 8 + 16 * i)
        assert key not in blobs and offset + length <= len(weights_bytes)
        blobs[key] = weights_bytes[offset : offset + length]
        offset += length
    assert offset == len(weights_bytes)
    needed = {"flatten_output", "confidence"}
    layers = []
    for layer in reversed(net["layers"]):
        if layer["top"] in needed:
            layers.append(layer)
            needed.update(layer["bottom"].split(","))
    layers.reverse()
    assert needed - {l["top"] for l in layers} == {"data"}
    arrays = {}
    for i, l in enumerate(layers):
        if l["type"] == "convolution":
            assert not l.get("has_batch_norm") and not l.get("fused_tanh")
            assert l.get("pad_fill_mode", 0) == 0 and l.get("pad_mode", 0) == 0
            assert l.get("pad_value", 0) == 0 and l.get("n_parallel", 1) == 1
            shape = (l["C"], l["K"] // l["n_groups"], l["Ny"], l["Nx"])
        elif l["type"] == "inner_product":
            assert not l.get("has_tanh") and not l.get("has_prelu")
            shape = (l["nC"], l["nB"])
        else:
            continue
        w = np.frombuffer(blobs[l["blob_weights_f16"]], dtype="<f2")
        assert w.size == np.prod(shape)
        arrays[f"l{i}.weight"] = w.reshape(shape).copy()
        if l.get("has_biases"):
            b = np.frombuffer(blobs[l["blob_biases"]], dtype="<f4")
            assert b.size == shape[0]
            arrays[f"l{i}.bias"] = b.copy()
    destination.mkdir(parents=True, exist_ok=True)
    graph = {
        "model": model,
        "layers": layers,
        "transform_params": net["transform_params"],
        "input": "Aligned RGB float tensor, NCHW, [N,3,128,128], range [0,1]",
        "source_sha256": {
            "net": hashlib.sha256(net_bytes).hexdigest(),
            "weights": hashlib.sha256(weights_bytes).hexdigest(),
        },
    }
    (destination / "graph.json").write_text(json.dumps(graph, indent=2))
    np.savez_compressed(destination / "weights.npz", **arrays)
    print(
        f"Exported {len(layers)} layers, {sum(a.size for a in arrays.values()):,} weight/bias values"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=SOURCE)
    p.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parent / "model"
    )
    p.add_argument(
        "--model",
        default=MODEL,
        help="Espresso basename; compatible embedding/confidence graphs only",
    )
    args = p.parse_args()
    export(args.source, args.output, args.model)
