#!/usr/bin/env python3
"""Energy consumption measurement using thermal proxies and TDP estimates."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import numpy as np
import time
import json
from pathlib import Path
from rknnlite.api import RKNNLite

from spike_encoder import encode_image_pixels, encode_spikes
from lif_neuron import LIFNeuron
from snn_executor import SNNLayer, SNNExecutor

MODEL_DIR = Path(__file__).parent / ".." / "models"


def load_mnist(n=100):
    import gzip, struct, urllib.request
    data_dir = MODEL_DIR / "mnist_raw"
    data_dir.mkdir(exist_ok=True)
    img_path = data_dir / "t10k-images-idx3-ubyte.gz"
    if not img_path.exists():
        urllib.request.urlretrieve("https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz", str(img_path))
    with gzip.open(str(img_path), "rb") as f:
        magic, nimg, rows, cols = struct.unpack(">IIII", f.read(16))
        n = min(n, nimg)
        images = np.frombuffer(f.read(n * rows * cols), dtype=np.uint8).reshape(n, rows * cols).astype(np.float32) / 255.0
    return images


def read_thermal():
    """Read thermal zone temperatures (millidegrees C)."""
    temps = {}
    thermal_dir = Path("/sys/class/thermal")
    if thermal_dir.exists():
        for tz in sorted(thermal_dir.glob("thermal_zone*")):
            try:
                temp = int((tz / "temp").read_text().strip())
                typ = (tz / "type").read_text().strip() if (tz / "type").exists() else "unknown"
                temps[f"zone{tz.name[-1]}_{typ}"] = temp
            except:
                pass
    return temps


def read_cpu_freq():
    freqs = {}
    for i in range(8):
        path = Path(f"/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_cur_freq")
        try:
            freqs[f"cpu{i}"] = int(path.read_text().strip())
        except:
            pass
    return freqs


def read_npu_load():
    """Read NPU load (needs sudo)."""
    try:
        load = Path("/sys/kernel/debug/rknpu/load").read_text().strip()
        return load
    except:
        return "N/A (needs sudo)"


def measure_snn_energy(images, T=5, n=200):
    rknn1 = str(MODEL_DIR / "snn_layer1.rknn")
    rknn2 = str(MODEL_DIR / "snn_layer2.rknn")
    l1 = SNNLayer(rknn1, 784, 128, threshold=1.0, leak_rate=0.1, core_id=0)
    l2 = SNNLayer(rknn2, 128, 10, threshold=1.0, leak_rate=0.1, core_id=0)
    ex = SNNExecutor(layers=[l1, l2], num_timesteps=T)
    ex.init_all()

    for i in range(5):
        ex.infer(images[i])

    temps_idle = read_thermal()
    npu_idle = read_npu_load()
    time.sleep(0.5)
    temps_idle2 = read_thermal()

    print(f"  Running {n} SNN inferences (T={T})...", flush=True)
    t0 = time.perf_counter()
    correct = 0
    for i in range(n):
        rate = ex.infer(images[i])
        if np.argmax(rate[:10]) != -1:
            correct += 1
    total_time = time.perf_counter() - t0

    temps_load = read_thermal()
    npu_load = read_npu_load()
    freqs = read_cpu_freq()

    ex.release_all()

    return {
        "total_time_s": total_time,
        "per_inf_ms": total_time / n * 1000,
        "throughput_fps": n / total_time,
        "temps_idle": temps_idle,
        "temps_load": temps_load,
        "temp_delta": {k: temps_load.get(k, 0) - temps_idle.get(k, 0) for k in temps_idle},
        "npu_idle": npu_idle,
        "npu_load": npu_load,
        "cpu_freqs": freqs,
    }


def estimate_energy(thermal_results, latency_ms, mode="snn"):
    """Estimate energy per classification using TDP model."""
    # RK3588 power estimates (conservative)
    P_IDLE = 3.0
    P_NPU_ACTIVE = 3.0
    P_CPU_ACTIVE = 2.0
    P_MEMORY = 0.5

    if mode == "snn":
        p_total = P_IDLE + P_NPU_ACTIVE + P_CPU_ACTIVE + P_MEMORY
    elif mode == "cpu_ann":
        # Pure CPU: NPU idle, CPU fully active
        p_total = P_IDLE + 4.0 + P_MEMORY
    else:
        p_total = P_IDLE + 1.0

    latency_s = latency_ms / 1000.0
    energy_j = p_total * latency_s

    return {
        "mode": mode,
        "estimated_power_w": p_total,
        "latency_ms": latency_ms,
        "energy_per_class_j": energy_j,
    }


def main():
    print("=" * 60, flush=True)
    print("NR-NPU: Energy Consumption Measurement", flush=True)
    print("=" * 60, flush=True)

    images = load_mnist(200)

    print("\n[1/4] Thermal baseline...", flush=True)
    temps = read_thermal()
    for k, v in temps.items():
        print(f"  {k}: {v/1000:.1f}C", flush=True)
    npu = read_npu_load()
    print(f"  NPU load: {npu}", flush=True)

    print("\n[2/4] SNN energy (different T)...", flush=True)
    snn_results = {}
    for T in [3, 5, 10, 20]:
        print(f"  T={T}...", flush=True)
        thermal = measure_snn_energy(images, T=T, n=100)
        energy = estimate_energy(thermal, thermal["per_inf_ms"], mode="snn")
        snn_results[T] = {**thermal, "energy": energy}
        print(f"    {thermal['per_inf_ms']:.1f}ms/inf, {thermal['throughput_fps']:.1f} FPS, "
              f"~{energy['energy_per_class_j']:.3f} J/inf", flush=True)

    print("\n[3/4] CPU ANN baseline (ONNX Runtime)...", flush=True)
    try:
        import onnxruntime as ort
        onnx_path = str(MODEL_DIR / "snn_layer1.onnx")
        if not Path(onnx_path).exists():
            print("  ONNX model not found, skipping", flush=True)
            cpu_energy = None
        else:
            sess = ort.InferenceSession(onnx_path)
            input_name = sess.get_inputs()[0].name

            t0 = time.perf_counter()
            for i in range(200):
                inp = np.random.randn(1, 784, 1, 1).astype(np.float32)
                sess.run(None, {input_name: inp})
            cpu_time = (time.perf_counter() - t0) / 200 * 1000
            cpu_energy = estimate_energy({}, cpu_time, mode="cpu_ann")
            print(f"  CPU ANN: {cpu_time:.1f}ms/inf, ~{cpu_energy['energy_per_class_j']:.3f} J/inf", flush=True)
    except Exception as e:
        print(f"  CPU ANN measurement failed: {e}", flush=True)
        cpu_energy = None

    print("\n[4/4] Energy comparison...", flush=True)
    print("\n" + "=" * 60, flush=True)
    print("ENERGY EFFICIENCY SUMMARY", flush=True)
    print("=" * 60, flush=True)

    best_T = 5
    if best_T in snn_results:
        snn_e = snn_results[best_T]["energy"]
        print(f"  SNN (T={best_T}): {snn_e['energy_per_class_j']:.3f} J/classification", flush=True)
        if cpu_energy:
            ratio = cpu_energy['energy_per_class_j'] / snn_e['energy_per_class_j']
            print(f"  CPU ANN:      {cpu_energy['energy_per_class_j']:.3f} J/classification", flush=True)
            print(f"  SNN advantage: {ratio:.1f}x more energy-efficient", flush=True)

    print("\n  Per-T breakdown:", flush=True)
    for T, r in snn_results.items():
        e = r["energy"]
        print(f"    T={T:2d}: {e['energy_per_class_j']:.3f} J/inf ({e['estimated_power_w']:.1f}W x {e['latency_ms']:.1f}ms)", flush=True)

    # Power values are estimated from TDP specs, not measured
    print("\n  NOTE: Power values are estimated from RK3588 TDP specs.", flush=True)
    print("  For precise measurement, use USB power meter (e.g., Power-Z KM002C).", flush=True)
    print("=" * 60, flush=True)

    results = {
        "snn": {str(k): v for k, v in snn_results.items()},
        "cpu_ann": cpu_energy,
        "thermal": temps,
    }
    with open(str(Path(__file__).parent / "energy_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to energy_results.json", flush=True)


if __name__ == "__main__":
    main()