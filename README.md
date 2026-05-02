# NR-NPU: Spiking Neural Networks on RK3588 NPU

**First-ever working SNN inference engine for Rockchip RK3588 NPU**  
*Yes, it really runs. And yes, it beats ANN on MNIST (99.0% vs 96.7%).*

## What is this?

This is a Python runtime that lets you run **spiking neural networks** on the 3‑core NPU of RK3588 boards (Orange Pi 5 Ultra, etc.).  
No special hardware. No external accelerators. Just the NPU that sits on the SoC.

I built this because everyone said “NPU is only for 2D convs, forget about SNNs, forget about transformers”. Well, that’s not entirely true. SNNs work. And they work **fast**.

## Key results (real hardware, no simulation)

| Metric | Value |
|--------|-------|
| MNIST accuracy (SNN) | **99.0%** |
| MNIST accuracy (ANN baseline) | 96.7% |
| Latency, T=20 timesteps | 15.8 ms |
| Latency, T=5 timesteps | 3.9 ms |
| Energy per inference (T=5) | ~0.051 J |
| Per‑timestep overhead | 0.79 ms |

**What does T=5 mean?**  
Only 5 timesteps per classification. That’s enough for 99% accuracy.  
And 3.9 ms is **faster than many ANNs on CPU**.

## How it works (the short version)

1. Train a small ANN (or generate random weights).  
2. Convert linear layers to `Conv2D(1x1)` – NPU loves 1x1 convs.  
3. Encode input spikes as dense tensors aligned to channels=8 (NPU’s native SIMD width).  
4. Run forward on NPU, one timestep at a time.  
5. Update membrane potentials (LIF) on CPU – simple, fast enough.  

**Why not pure NPU?**  
Because RKNN runtime doesn’t expose LIF. But CPU update takes <0.1 ms per timestep. Not a problem.

## What I learned (and you should know)

- **Multi‑core parallelism for a single model?**  
  RKNN serializes inference calls even with different core masks. Threading doesn’t help.  
  I got **0.16× speedup** (i.e., slower). Don’t waste time.  
  *But* three independent models on three cores work great – see Discovery 1 in my research.

- **INT8 LIF gives zero gain**  
  NPU already dequantizes outputs to float32. Converting back to int doesn’t help.  
  Leave LIF in plain float.

- **Batch >1 still segfaults**  
  RKNN toolkit 2.3.2 crashes at `init_runtime` for batch size >1.  
  So we loop over timesteps. With T=5 and 0.79 ms per step it’s fine.

- **Optimal configuration for MNIST**  
  T=5, threshold=1.0, leak=0.1  
  → 99% accuracy, 3.9 ms, 0.051 J

## Repository structure

```
nrnpu/
├── spike_encoder.py      # Dense encoding, 8‑channel aligned
├── lif_neuron.py         # CPU LIF (integrate, leak, threshold, reset)
├── ann_to_snn.py         # Linear → Conv2D converter
├── snn_executor.py       # SNNLayer + SNNExecutor runtime
├── test_mvp.py           # Full MNIST pipeline
└── results/              # Benchmark tables (CSV, JSON)
```

## Requirements

- RK3588 board (Orange Pi 5 Ultra, Rock 5B, etc.)
- RKNN Toolkit 2.3.2 + RKNNLite2 1.5.2 (or newer)
- Python 3.11
- NumPy, ONNX Runtime (for CPU baseline)

## Quick start

```bash
git clone https://github.com/nimteyai-sudo/nr-npu
cd nr-npu
python test_mvp.py --timesteps 5 --threshold 1.0
```

Expected output:
```
ANN accuracy: 96.7%
SNN accuracy: 99.0%
SNN latency: 3.9 ms (T=5)
Energy: ~0.051 J/inference
```

## Real‑world applications (where this makes sense)

- **Low‑power gesture recognition** (accelerometer data → spikes → classify in <5 ms)
- **Anomaly detection in vibration** (predict bearing failure before it happens)
- **Keyword spotting on microphone** (always‑on voice trigger with ~0.05 J per inference)

The NPU draws very little power when active. Combined with SNN’s event‑driven nature, you can run for days on a battery.

## Limitations (honest ones)

- MNIST only for now. You’ll need to adapt to your own dataset.
- T=5 works for MNIST; other problems may need more timesteps.
- LIF runs on CPU – not a big deal, but it’s there.
- No batching (RKNN segfault). Accept it.

## Citation

If you use this in your research, please cite:

```
@misc{nr-npu2025,
  author = {nimteyai-sudo},
  title = {NR-NPU: First Spiking Neural Network Runtime for RK3588 NPU},
  year = {2026},
  ORCID: {0009-0009-5906-2867},
  howpublished = {\url[{https://github.com/nimteyai-sudo/nr-npu)],
  note = {Achieves 99.0% MNIST accuracy at 3.9 ms inference latency}
}
```

## License

MIT – do whatever you want, but mention where you got it.

## Contact

Open an issue on GitHub. I usually reply within a day.

---

*Made on Orange Pi 5 Ultra, Debian 13, with a lot of coffee and cursing at RKNN documentation.*




