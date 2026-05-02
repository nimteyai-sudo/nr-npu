#!/usr/bin/env python3
"""Phase 2 benchmarks: T sweep, threshold sweep, INT8 LIF, multi-core throughput."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import numpy as np
import time
import json
import threading
from pathlib import Path
from rknnlite.api import RKNNLite

from spike_encoder import encode_image_pixels, encode_spikes
from lif_neuron import LIFNeuron, LIFNeuronInt8
from snn_executor import SNNLayer, SNNExecutor

MODEL_DIR = Path(__file__).parent / ".." / "models"


def load_mnist(n=200):
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
        images = np.frombuffer(f.read(n * rows * cols), dtype=np.uint8).reshape(n, rows * cols).astype(np.float32) / 255.0
    with gzip.open(str(lbl_path), "rb") as f:
        magic, nimg2 = struct.unpack(">II", f.read(8))
        n = min(n, nimg2)
        labels = np.frombuffer(f.read(n), dtype=np.uint8)
    return images, labels


def run_snn(images, labels, n, T, threshold, leak=0.1):
    rknn1 = str(MODEL_DIR / "snn_layer1.rknn")
    rknn2 = str(MODEL_DIR / "snn_layer2.rknn")
    l1 = SNNLayer(rknn1, 784, 128, threshold=threshold, leak_rate=leak, core_id=0)
    l2 = SNNLayer(rknn2, 128, 10, threshold=threshold, leak_rate=leak, core_id=0)
    ex = SNNExecutor(layers=[l1, l2], num_timesteps=T)
    ex.init_all()

    correct = 0
    total_ms = 0
    for i in range(n):
        rate, timings = ex.infer_timed(images[i])
        if np.argmax(rate[:10]) == labels[i]:
            correct += 1
        total_ms += timings["total_ms"]

    ex.release_all()
    return 100.0 * correct / n, total_ms / n


def run_snn_int8(images, labels, n, T):
    rknn1 = str(MODEL_DIR / "snn_layer1.rknn")
    rknn2 = str(MODEL_DIR / "snn_layer2.rknn")
    rknn_l1 = RKNNLite()
    rknn_l1.load_rknn(rknn1)
    rknn_l1.init_runtime(target=None, core_mask=RKNNLite.NPU_CORE_AUTO)
    rknn_l2 = RKNNLite()
    rknn_l2.load_rknn(rknn2)
    rknn_l2.init_runtime(target=None, core_mask=RKNNLite.NPU_CORE_AUTO)

    lif1 = LIFNeuronInt8(128, threshold_int=15, leak_int=2, scale=15.0)
    lif2 = LIFNeuronInt8(10, threshold_int=31, leak_int=3, scale=30.6)

    correct = 0
    total_ms = 0
    for i in range(n):
        lif1.reset_state()
        lif2.reset_state()
        inp = encode_image_pixels(images[i], 784)
        t0 = time.perf_counter()
        for t in range(T):
            o1 = rknn_l1.inference(inputs=[inp])
            o1 = o1[0] if isinstance(o1, list) else o1
            s1 = lif1.step(o1)
            si = [j for j, s in enumerate(s1) if s > 0.5]
            inp2 = encode_spikes(si, 128)
            o2 = rknn_l2.inference(inputs=[inp2])
            o2 = o2[0] if isinstance(o2, list) else o2
            lif2.step(o2)
        rate = lif2.get_spike_rate()
        if np.argmax(rate[:10]) == labels[i]:
            correct += 1
        total_ms += (time.perf_counter() - t0) * 1000

    rknn_l1.release()
    rknn_l2.release()
    return 100.0 * correct / n, total_ms / n


def test_throughput_parallel(images, labels, n=30, T=20):
    """Multi-core throughput: process multiple images in parallel across cores."""
    rknn1 = str(MODEL_DIR / "snn_layer1.rknn")
    rknn2 = str(MODEL_DIR / "snn_layer2.rknn")

    l1 = SNNLayer(rknn1, 784, 128, threshold=1.0, leak_rate=0.1, core_id=0)
    l2 = SNNLayer(rknn2, 128, 10, threshold=1.0, leak_rate=0.1, core_id=0)
    ex = SNNExecutor(layers=[l1, l2], num_timesteps=T)
    ex.init_all()

    t0 = time.perf_counter()
    for i in range(n):
        ex.infer(images[i])
    single_time = (time.perf_counter() - t0) * 1000
    single_fps = n / (single_time / 1000)
    ex.release_all()

    executors = []
    for core in [RKNNLite.NPU_CORE_0, RKNNLite.NPU_CORE_1, RKNNLite.NPU_CORE_2]:
        l1 = SNNLayer(rknn1, 784, 128, threshold=1.0, leak_rate=0.1, core_id=0)
        l2 = SNNLayer(rknn2, 128, 10, threshold=1.0, leak_rate=0.1, core_id=0)
        ex = SNNExecutor(layers=[l1, l2], num_timesteps=T)
        ex.init_all()
        executors.append(ex)

    barrier = threading.Barrier(3)
    results = [None] * 3

    def worker(idx, img_indices):
        try:
            barrier.wait(timeout=5)
        except:
            pass
        t0 = time.perf_counter()
        for i in img_indices:
            executors[idx].infer(images[i])
        results[idx] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    batch_size = 3
    for start in range(0, n, batch_size):
        indices = [min(start + j, n - 1) for j in range(batch_size)]
        threads = [threading.Thread(target=worker, args=(j, [indices[j]])) for j in range(batch_size)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
    triple_time = (time.perf_counter() - t0) * 1000
    triple_fps = n / (triple_time / 1000)

    for ex in executors:
        ex.release_all()

    return {
        "single_core_ms": single_time / n,
        "single_fps": single_fps,
        "triple_core_ms": triple_time / n,
        "triple_fps": triple_fps,
        "throughput_speedup": triple_fps / single_fps if single_fps > 0 else 0,
    }


def main():
    print("=" * 60, flush=True)
    print("NR-NPU Phase 2: Corrected Benchmarks", flush=True)
    print("=" * 60, flush=True)

    images, labels = load_mnist(200)
    print(f"Loaded {len(images)} samples\n", flush=True)

    print("[1/5] T sweep (threshold=1.0)...", flush=True)
    t_sweep = []
    for T in [3, 5, 7, 10, 15, 20, 30, 50]:
        acc, lat = run_snn(images, labels, 100, T, threshold=1.0)
        t_sweep.append({"T": T, "threshold": 1.0, "acc": acc, "latency_ms": lat, "per_step_ms": lat / T})
        print(f"  T={T:2d}: {acc:.1f}%, {lat:.1f}ms ({lat/T:.2f}ms/step)", flush=True)

    print("\n[2/5] Threshold sweep for T=5,10...", flush=True)
    thresh_sweep = []
    for T in [5, 10]:
        for th in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
            acc, lat = run_snn(images, labels, 100, T, threshold=th)
            thresh_sweep.append({"T": T, "threshold": th, "acc": acc, "latency_ms": lat})
            print(f"  T={T:2d}, th={th:.1f}: {acc:.1f}%, {lat:.1f}ms", flush=True)

    print("\n[3/5] INT8 LIF vs Float LIF (T=20)...", flush=True)
    float_acc, float_lat = run_snn(images, labels, 50, T=20, threshold=1.0)
    int8_acc, int8_lat = run_snn_int8(images, labels, 50, T=20)
    print(f"  Float: {float_acc:.1f}%, {float_lat:.1f}ms", flush=True)
    print(f"  INT8:  {int8_acc:.1f}%, {int8_lat:.1f}ms", flush=True)

    print("\n[4/5] Multi-core throughput (3 cores)...", flush=True)
    throughput = test_throughput_parallel(images, labels, n=30, T=20)
    print(f"  Single core: {throughput['single_core_ms']:.1f}ms/inf, {throughput['single_fps']:.1f} FPS", flush=True)
    print(f"  Triple core: {throughput['triple_core_ms']:.1f}ms/inf, {throughput['triple_fps']:.1f} FPS", flush=True)
    print(f"  Throughput speedup: {throughput['throughput_speedup']:.2f}x", flush=True)

    print("\n[5/5] Best combined (lowest T with >=98% accuracy)...", flush=True)
    best = min(t_sweep, key=lambda r: r["latency_ms"] if r["acc"] >= 98.0 else 9999)
    if best["acc"] < 98.0:
        best = max(t_sweep, key=lambda r: r["acc"])
    print(f"  Best: T={best['T']}, {best['acc']:.1f}%, {best['latency_ms']:.1f}ms", flush=True)

    best_th = min(thresh_sweep, key=lambda r: r["latency_ms"] if r["acc"] >= 98.0 else 9999)
    if best_th["acc"] >= 98.0:
        print(f"  With threshold tuning: T={best_th['T']}, th={best_th['threshold']}, {best_th['acc']:.1f}%, {best_th['latency_ms']:.1f}ms", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("PHASE 2 SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"  MVP baseline: T=20, 15.3ms/inf, ~99% accuracy", flush=True)
    print(f"  Best single-core: T={best['T']}, {best['latency_ms']:.1f}ms/inf, {best['acc']:.1f}%", flush=True)
    print(f"  Speedup from T reduction: {15.3 / best['latency_ms']:.2f}x" if best['latency_ms'] > 0 else "", flush=True)
    print(f"  INT8 LIF: {int8_acc:.1f}% (vs {float_acc:.1f}% float)", flush=True)
    print(f"  Multi-core throughput: {throughput['throughput_speedup']:.2f}x", flush=True)
    # RKNN serializes inference within process, so no per-input pipelining
    print(f"  NOTE: RKNN serializes inference within process -- no per-input pipelining possible", flush=True)
    print("=" * 60, flush=True)

    all_results = {
        "t_sweep": t_sweep,
        "threshold_sweep": thresh_sweep,
        "int8_lif": {"float_acc": float_acc, "float_lat": float_lat, "int8_acc": int8_acc, "int8_lat": int8_lat},
        "throughput": throughput,
        "best": best,
    }
    with open(str(Path(__file__).parent / "phase2_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to phase2_results.json", flush=True)


if __name__ == "__main__":
    main()