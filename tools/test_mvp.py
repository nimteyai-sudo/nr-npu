#!/usr/bin/env python3
"""End-to-end SNN inference on RK3588 NPU: train ANN, convert to RKNN, run SNN."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import numpy as np
import torch
import torch.nn as nn
import time
import struct
import gzip
import urllib.request
from pathlib import Path

from spike_encoder import encode_spikes, encode_image_pixels, decode_output
from lif_neuron import LIFNeuron
from ann_to_snn import (
    linear_to_conv2d,
    ann_to_snn_model,
    export_conv_to_onnx,
    compute_snn_threshold,
    compute_required_timesteps,
)
from snn_executor import SNNLayer, SNNExecutor

from rknn.api import RKNN
from rknnlite.api import RKNNLite

PROJECT_DIR = Path(__file__).parent / '..'
MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

MNIST_URLS = {
    "train_images": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
}


def _download(url, path):
    if not path.exists():
        print(f"    Downloading {url.split('/')[-1]}...")
        urllib.request.urlretrieve(url, str(path))


def load_mnist(train=True):
    """Load MNIST as (N, 784) float32 in [0,1] and (N,) int64 labels."""
    prefix = "train" if train else "t10k"
    data_dir = MODEL_DIR / "mnist_raw"
    data_dir.mkdir(exist_ok=True)

    img_path = data_dir / f"{prefix}-images-idx3-ubyte.gz"
    lbl_path = data_dir / f"{prefix}-labels-idx1-ubyte.gz"

    _download(MNIST_URLS[f"{'train' if train else 'test'}_images"], img_path)
    _download(MNIST_URLS[f"{'train' if train else 'test'}_labels"], lbl_path)

    with gzip.open(str(img_path), "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows * cols)
        images = images.astype(np.float32) / 255.0

    with gzip.open(str(lbl_path), "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return images, labels


class SimpleANN(nn.Module):
    """784 -> 128 (ReLU) -> 10"""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_ann(epochs=3, batch_size=64):
    images, labels = load_mnist(train=True)

    model = SimpleANN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    n = images.shape[0]
    model.train()
    for epoch in range(epochs):
        perm = np.random.permutation(n)
        total_loss = 0
        correct = 0
        total = 0

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = perm[start:end]

            data = torch.tensor(images[idx], dtype=torch.float32)
            target = torch.tensor(labels[idx], dtype=torch.long)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

        acc = 100.0 * correct / total
        print(f"  Epoch {epoch+1}/{epochs}: loss={total_loss/(n//batch_size):.4f}, acc={acc:.1f}%")

    return model


def evaluate_ann(model):
    images, labels = load_mnist(train=False)
    model.eval()
    correct = 0
    total = images.shape[0]

    with torch.no_grad():
        batch_size = 100
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            data = torch.tensor(images[start:end], dtype=torch.float32)
            target = torch.tensor(labels[start:end], dtype=torch.long)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    return 100.0 * correct / total


def convert_ann_to_rknn(model):
    """Convert ANN layers to RKNN models with INT8 quantization."""
    conv1 = linear_to_conv2d(model.fc1, align_channels=True)
    conv2 = linear_to_conv2d(model.fc2, align_channels=True)

    in1 = ((784 + 7) // 8) * 8
    out1 = ((128 + 7) // 8) * 8
    in2 = out1
    out2 = ((10 + 7) // 8) * 8

    print(f"  Channel alignment: fc1 ({784}->{in1} in, {128}->{out1} out), fc2 ({128}->{in2} in, {10}->{out2} out)")

    onnx1_path = str(MODEL_DIR / "snn_layer1.onnx")
    onnx2_path = str(MODEL_DIR / "snn_layer2.onnx")

    export_conv_to_onnx(conv1, in1, onnx1_path)
    export_conv_to_onnx(conv2, in2, onnx2_path)
    print(f"  Exported ONNX: {onnx1_path}, {onnx2_path}")

    rknn1_path = str(MODEL_DIR / "snn_layer1.rknn")
    rknn2_path = str(MODEL_DIR / "snn_layer2.rknn")

    for onnx_path, rknn_path, in_ch, prefix in [
        (onnx1_path, rknn1_path, in1, "calib_l1"),
        (onnx2_path, rknn2_path, in2, "calib_l2"),
    ]:
        # RKNN requires a .txt file listing .npy calibration paths
        calib_dir = MODEL_DIR / prefix
        calib_dir.mkdir(exist_ok=True)
        calib_lines = []
        for i in range(10):
            data = np.random.randn(1, in_ch, 1, 1).astype(np.float32)
            npy_path = str(calib_dir / f"s{i}.npy")
            np.save(npy_path, data)
            calib_lines.append(npy_path)
        calib_txt = str(MODEL_DIR / f"{prefix}.txt")
        with open(calib_txt, "w") as f:
            f.write("\n".join(calib_lines))

        rknn = RKNN()
        ret = rknn.config(
            mean_values=[[0] * in_ch],
            std_values=[[1] * in_ch],
            target_platform="rk3588",
        )
        if ret != 0:
            raise RuntimeError(f"RKNN config failed for {onnx_path}")

        ret = rknn.load_onnx(model=onnx_path)
        if ret != 0:
            raise RuntimeError(f"RKNN load_onnx failed for {onnx_path}")

        ret = rknn.build(do_quantization=True, dataset=calib_txt)
        if ret != 0:
            raise RuntimeError(f"RKNN build failed for {onnx_path}")

        ret = rknn.export_rknn(rknn_path)
        if ret != 0:
            raise RuntimeError(f"RKNN export failed for {rknn_path}")

        rknn.release()
        print(f"  Built RKNN: {rknn_path}")

    return rknn1_path, rknn2_path


def run_snn_inference(rknn1_path, rknn2_path, num_timesteps=20):
    images, labels = load_mnist(train=False)

    layer1 = SNNLayer(
        rknn_model_path=rknn1_path,
        num_in_neurons=784,
        num_out_neurons=128,
        threshold=1.0,
        leak_rate=0.1,
        core_id=0,
    )
    layer2 = SNNLayer(
        rknn_model_path=rknn2_path,
        num_in_neurons=128,
        num_out_neurons=10,
        threshold=1.0,
        leak_rate=0.1,
        core_id=0,
    )

    executor = SNNExecutor(layers=[layer1, layer2], num_timesteps=num_timesteps)
    executor.init_all()

    num_test = 100
    correct = 0
    total_time = 0

    print(f"\n  Running SNN inference on {num_test} samples (T={num_timesteps} timesteps)...")
    for i in range(num_test):
        img = images[i]
        label = int(labels[i])

        spike_rate, timings = executor.infer_timed(img)
        pred = np.argmax(spike_rate[:10])

        if pred == label:
            correct += 1
        total_time += timings["total_ms"]

        if (i + 1) % 10 == 0:
            print(f"    [{i+1}/{num_test}] acc={100*correct/(i+1):.1f}%, avg={total_time/(i+1):.1f}ms/inf")

    accuracy = 100.0 * correct / num_test
    avg_latency = total_time / num_test

    executor.release_all()
    return accuracy, avg_latency


def main():
    print("=" * 60)
    print("NR-NPU: Neuromorphic Runtime for RK3588 NPU -- MVP Test")
    print("=" * 60)

    print("\n[1/4] Training ANN on MNIST...")
    ann_model = train_ann(epochs=3)
    ann_acc = evaluate_ann(ann_model)
    print(f"  ANN test accuracy: {ann_acc:.1f}%")

    print("\n[2/4] Converting ANN -> SNN Conv2D -> RKNN...")
    rknn1_path, rknn2_path = convert_ann_to_rknn(ann_model)

    print("\n[3/4] Computing SNN parameters...")
    threshold1 = compute_snn_threshold(ann_model.fc1, method="max")
    threshold2 = compute_snn_threshold(ann_model.fc2, method="max")
    required_T = compute_required_timesteps([ann_model.fc1, ann_model.fc2], method="max")
    print(f"  Layer 1 threshold: {threshold1:.4f}")
    print(f"  Layer 2 threshold: {threshold2:.4f}")
    print(f"  Recommended timesteps (max method): {required_T}")
    print(f"  Using T=20 for MVP (balance of speed/accuracy)")

    print("\n[4/4] Running SNN on NPU...")
    snn_acc, snn_latency = run_snn_inference(rknn1_path, rknn2_path, num_timesteps=20)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  ANN accuracy:   {ann_acc:.1f}%")
    print(f"  SNN accuracy:    {snn_acc:.1f}%")
    print(f"  SNN latency:    {snn_latency:.1f} ms/inference (T=20)")
    print(f"  Per timestep:    {snn_latency/20:.2f} ms")
    print("=" * 60)


if __name__ == "__main__":
    main()