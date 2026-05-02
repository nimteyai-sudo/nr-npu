#!/usr/bin/env python3
"""Measure NPU INT8 output ranges for setting LIF thresholds in the INT8 domain."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import numpy as np
from rknnlite.api import RKNNLite
from spike_encoder import encode_image_pixels, encode_spikes

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')


def load_mnist_quick(n=200):
    import gzip
    import struct
    import urllib.request
    from pathlib import Path

    data_dir = Path(MODEL_DIR) / "mnist_raw"
    img_path = data_dir / "t10k-images-idx3-ubyte.gz"
    if not img_path.exists():
        url = "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz"
        urllib.request.urlretrieve(url, str(img_path))

    with gzip.open(str(img_path), "rb") as f:
        magic, nimg, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(n * rows * cols), dtype=np.uint8).reshape(n, rows * cols)
        images = images.astype(np.float32) / 255.0
    return images


def measure_layer_output(rknn_path, inputs, layer_name=""):
    rknn = RKNNLite()
    ret = rknn.load_rknn(rknn_path)
    if ret != 0:
        raise RuntimeError(f"Failed to load {rknn_path}")

    ret = rknn.init_runtime(target=None, core_mask=RKNNLite.NPU_CORE_AUTO)
    if ret != 0:
        raise RuntimeError(f"Failed to init runtime for {rknn_path}")

    all_outputs = []
    for inp in inputs:
        outputs = rknn.inference(inputs=[inp])
        out = outputs[0] if isinstance(outputs, list) else outputs
        all_outputs.append(out.flatten())

    rknn.release()

    all_outputs = np.array(all_outputs)
    stats = {
        "min": float(np.min(all_outputs)),
        "max": float(np.max(all_outputs)),
        "mean": float(np.mean(all_outputs)),
        "std": float(np.std(all_outputs)),
        "p01": float(np.percentile(all_outputs, 1)),
        "p99": float(np.percentile(all_outputs, 99)),
        "abs_max": float(np.max(np.abs(all_outputs))),
        "shape": str(all_outputs.shape[1:]),
    }

    print(f"\n  {layer_name} output statistics ({len(inputs)} samples):")
    for k, v in stats.items():
        print(f"    {k}: {v}")

    nonzero = all_outputs[all_outputs != 0]
    if len(nonzero) > 0:
        print(f"    nonzero count: {len(nonzero)} / {all_outputs.size} ({100*len(nonzero)/all_outputs.size:.1f}%)")
        print(f"    nonzero range: [{np.min(nonzero):.4f}, {np.max(nonzero):.4f}]")

    return stats, all_outputs


def main():
    print("=" * 60)
    print("NR-NPU: INT8 Output Range Measurement")
    print("=" * 60)

    images = load_mnist_quick(200)

    layer1_inputs = []
    for i in range(200):
        tensor = encode_image_pixels(images[i], 784)
        layer1_inputs.append(tensor)

    rknn1_path = os.path.join(MODEL_DIR, "snn_layer1.rknn")
    print("\n[1/2] Measuring Layer 1 output range...")
    l1_stats, l1_outputs = measure_layer_output(rknn1_path, layer1_inputs, "Layer 1 (Conv2d 784->128)")

    # Use actual L1 outputs as spike patterns for realistic L2 input
    layer2_inputs = []
    for i in range(200):
        l1_out = l1_outputs[i][:128]
        spike_indices = [j for j, v in enumerate(l1_out) if v > 0]
        tensor = encode_spikes(spike_indices, 128)
        layer2_inputs.append(tensor)

    rknn2_path = os.path.join(MODEL_DIR, "snn_layer2.rknn")
    print("\n[2/2] Measuring Layer 2 output range...")
    l2_stats, l2_outputs = measure_layer_output(rknn2_path, layer2_inputs, "Layer 2 (Conv2d 128->16)")

    print("\n" + "=" * 60)
    print("INT8 QUANTIZATION ANALYSIS")
    print("=" * 60)

    for name, stats in [("Layer 1", l1_stats), ("Layer 2", l2_stats)]:
        abs_max = stats["abs_max"]
        if abs_max > 0:
            scale = 127.0 / abs_max
            print(f"\n  {name}:")
            print(f"    Float range: [{stats['min']:.4f}, {stats['max']:.4f}]")
            print(f"    Abs max: {abs_max:.4f}")
            print(f"    INT8 scale: {scale:.2f} (1 int8 unit = {1/scale:.4f} float)")
            print(f"    INT8 range: [{stats['min']*scale:.0f}, {stats['max']*scale:.0f}]")
            print(f"    LIF threshold (float 1.0) = {1.0 * scale:.0f} in INT8")
            print(f"    LIF leak (float 0.1) = {0.1 * scale:.0f} in INT8")


if __name__ == "__main__":
    main()