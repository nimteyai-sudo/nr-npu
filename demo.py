#!/usr/bin/env python3
"""Real-time MNIST classifier with terminal visualization on RK3588 NPU."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np
import time
from pathlib import Path
from rknnlite.api import RKNNLite

from spike_encoder import encode_image_pixels, encode_spikes
from lif_neuron import LIFNeuron
from snn_executor import SNNLayer, SNNExecutor

MODEL_DIR = Path(__file__).parent / "models"


def load_mnist(n=50):
    import gzip, struct, urllib.request
    data_dir = MODEL_DIR / "mnist_raw"
    data_dir.mkdir(exist_ok=True)
    img_path = data_dir / "t10k-images-idx3-ubyte.gz"
    lbl_path = data_dir / "t10k-labels-idx1-ubyte.gz"
    if not img_path.exists():
        urllib.request.urlretrieve("https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz", str(img_path))
    if not lbl_path.exists():
        urllib.request.urlretrieve("https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz", str(lbl_path))
    with gzip.open(str(img_path), "rb") as f:
        magic, nimg, rows, cols = struct.unpack(">IIII", f.read(16))
        n = min(n, nimg)
        images = np.frombuffer(f.read(n * rows * cols), dtype=np.uint8).reshape(n, 28, 28)
    with gzip.open(str(lbl_path), "rb") as f:
        magic, nimg2 = struct.unpack(">II", f.read(8))
        n2 = min(n, nimg2)
        labels = np.frombuffer(f.read(n2), dtype=np.uint8)
    return images, labels


def draw_digit(image, width=28, height=14):
    chars = " .:-=+*#%@"
    img = image.astype(np.int32)
    lines = []
    for y in range(0, 28, 2):
        line = ""
        for x in range(28):
            if y + 1 < 28:
                val = (img[y, x] + img[y + 1, x]) / 2
            else:
                val = img[y, x]
            idx = min(int(val / 255.0 * len(chars)), len(chars) - 1)
            line += chars[idx]
        lines.append(line)
    return lines


def draw_spike_bar(spikes, width=50, label=""):
    n = len(spikes)
    fired = int(spikes.sum())
    bar_len = int(fired / n * width) if n > 0 else 0
    bar = "#" * bar_len + "." * (width - bar_len)
    return f"{label}: [{bar}] {fired}/{n} fired"


def draw_membrane_bar(membrane, width=50, label="", threshold=1.0):
    max_val = max(np.max(np.abs(membrane)), threshold)
    norm = membrane / max_val
    bar = ""
    for v in norm[:min(len(norm), width)]:
        if v > threshold / max_val:
            bar += "!"
        elif v > 0.5 * threshold / max_val:
            bar += "+"
        elif v > 0:
            bar += "-"
        else:
            bar += " "
    return f"{label}: [{bar}] max={np.max(membrane):.2f}"


def draw_output_rates(rates, width=30, actual=10):
    lines = []
    for i in range(actual):
        bar_len = int(rates[i] * width)
        bar = "=" * bar_len + "." * (width - bar_len)
        marker = " <---" if i == np.argmax(rates[:actual]) else ""
        lines.append(f"  [{i}] [{bar}] {rates[i]:.2f}{marker}")
    return "\n".join(lines)


def run_demo(num_samples=20, T=5, delay=0.3):
    print("=" * 60, flush=True)
    print("  NR-NPU: Neuromorphic SNN on RK3588 NPU", flush=True)
    print("  Real-time MNIST Classification Demo", flush=True)
    print("=" * 60, flush=True)

    images, labels = load_mnist(50)

    rknn1 = str(MODEL_DIR / "snn_layer1.rknn")
    rknn2 = str(MODEL_DIR / "snn_layer2.rknn")
    l1 = SNNLayer(rknn1, 784, 128, threshold=1.0, leak_rate=0.1, core_id=0)
    l2 = SNNLayer(rknn2, 128, 10, threshold=1.0, leak_rate=0.1, core_id=0)
    executor = SNNExecutor(layers=[l1, l2], num_timesteps=T)
    executor.init_all()

    print(f"\n  Engine: 2-layer SNN (784->128->10) on RK3588 NPU", flush=True)
    print(f"  Timesteps: T={T}", flush=True)
    print(f"  LIF: threshold=1.0, leak=0.1\n", flush=True)

    executor.infer(images[0].flatten().astype(np.float32) / 255.0)

    correct = 0
    total_time = 0

    for idx in range(num_samples):
        img = images[idx].flatten().astype(np.float32) / 255.0
        label = labels[idx]

        for layer in executor.layers:
            layer.lif.reset_state()

        digit_art = draw_digit(images[idx])
        print(f"\n{'─' * 60}", flush=True)
        print(f"  Sample #{idx + 1}  |  True label: {label}", flush=True)
        print(f"{'─' * 60}", flush=True)
        for line in digit_art:
            print(f"  {line}", flush=True)

        input_tensor = encode_image_pixels(img, 784)

        t0 = time.perf_counter()
        step_data = []
        for t in range(T):
            current_input = input_tensor
            for i, layer in enumerate(executor.layers):
                outputs = layer._rknn.inference(inputs=[current_input])
                output = outputs[0] if isinstance(outputs, list) else outputs
                spikes = layer.lif.step(output)
                if i == 0:
                    spike1 = spikes
                    membrane1 = layer.lif.membrane.copy()
                else:
                    spike2 = spikes
                    membrane2 = layer.lif.membrane.copy()

                if i < len(executor.layers) - 1:
                    spike_indices = [j for j, s in enumerate(spikes) if s > 0.5]
                    current_input = encode_spikes(spike_indices, executor.layers[i + 1].num_in_neurons)

            step_data.append({
                "l1_spikes": spike1,
                "l1_membrane": membrane1,
                "l2_spikes": spike2 if 'spike2' in dir() else np.zeros(16),
                "l2_membrane": membrane2 if 'membrane2' in dir() else np.zeros(16),
            })

        total_ms = (time.perf_counter() - t0) * 1000
        total_time += total_ms

        print(f"\n  Timestep activity (T={T}):", flush=True)
        for t, sd in enumerate(step_data):
            l1_fired = int(sd["l1_spikes"][:128].sum())
            l2_fired = int(sd["l2_spikes"][:10].sum())
            l1_bar = "#" * min(l1_fired, 40) + "." * max(0, 40 - l1_fired)
            l2_bar = "#" * min(l2_fired * 4, 40) + "." * max(0, 40 - l2_fired * 4)
            print(f"    t={t}: L1[{l1_bar}] L2[{l2_bar}]", flush=True)

        rate = executor.layers[-1].lif.get_spike_rate()
        pred = np.argmax(rate[:10])
        is_correct = pred == label
        correct += is_correct

        print(f"\n  Output spike rates:", flush=True)
        print(draw_output_rates(rate, width=25, actual=10), flush=True)

        result = "CORRECT" if is_correct else "WRONG"
        print(f"\n  Prediction: {pred}  |  {result}", flush=True)
        print(f"  Latency: {total_ms:.1f} ms ({total_ms/T:.2f} ms/step)", flush=True)

        if delay > 0:
            time.sleep(delay)

    avg_latency = total_time / num_samples
    accuracy = 100.0 * correct / num_samples

    executor.release_all()

    print(f"\n{'=' * 60}", flush=True)
    print(f"  DEMO RESULTS", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"  Samples:     {num_samples}", flush=True)
    print(f"  Accuracy:    {accuracy:.1f}%", flush=True)
    print(f"  Avg latency: {avg_latency:.1f} ms", flush=True)
    print(f"  Timesteps:   T={T}", flush=True)
    print(f"  Throughput:   {1000/avg_latency:.1f} FPS", flush=True)
    print(f"{'=' * 60}", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NR-NPU Demo")
    parser.add_argument("-n", "--samples", type=int, default=20, help="Number of samples")
    parser.add_argument("-t", "--timesteps", type=int, default=5, help="Timesteps T")
    parser.add_argument("-d", "--delay", type=float, default=0.3, help="Delay between samples (s)")
    args = parser.parse_args()

    run_demo(num_samples=args.samples, T=args.timesteps, delay=args.delay)