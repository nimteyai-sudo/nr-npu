"""Pipelined SNN executor: Layer1 on Core0, Layer2 on Core1, overlapping timesteps."""

import numpy as np
import time
import threading
from queue import Queue
from rknnlite.api import RKNNLite

from spike_encoder import encode_spikes, encode_image_pixels
from lif_neuron import LIFNeuron


class PipelinedSNNExecutor:
    """Two-layer SNN with pipelined multi-core NPU execution."""

    def __init__(self, rknn1_path, rknn2_path, num_in_neurons, num_mid_neurons,
                 num_out_neurons, num_timesteps=20, threshold1=1.0, threshold2=1.0,
                 leak_rate=0.1):
        self.rknn1_path = rknn1_path
        self.rknn2_path = rknn2_path
        self.num_in_neurons = num_in_neurons
        self.num_mid_neurons = num_mid_neurons
        self.num_out_neurons = num_out_neurons
        self.num_timesteps = num_timesteps

        self.lif1 = LIFNeuron(num_mid_neurons, threshold=threshold1, leak_rate=leak_rate)
        self.lif2 = LIFNeuron(num_out_neurons, threshold=threshold2, leak_rate=leak_rate)

        self._rknn1 = None
        self._rknn2 = None
        self._l1_to_l2 = Queue(maxsize=2)

    def init_runtime(self):
        """Load both RKNN models onto separate NPU cores."""
        self._rknn1 = RKNNLite()
        ret = self._rknn1.load_rknn(self.rknn1_path)
        if ret != 0:
            raise RuntimeError(f"Failed to load RKNN model 1: {self.rknn1_path}")
        ret = self._rknn1.init_runtime(target=None, core_mask=RKNNLite.NPU_CORE_0)
        if ret != 0:
            raise RuntimeError("Failed to init RKNN runtime on Core 0")

        self._rknn2 = RKNNLite()
        ret = self._rknn2.load_rknn(self.rknn2_path)
        if ret != 0:
            raise RuntimeError(f"Failed to load RKNN model 2: {self.rknn2_path}")
        ret = self._rknn2.init_runtime(target=None, core_mask=RKNNLite.NPU_CORE_1)
        if ret != 0:
            raise RuntimeError("Failed to init RKNN runtime on Core 1")

    def release(self):
        if self._rknn1:
            self._rknn1.release()
        if self._rknn2:
            self._rknn2.release()

    def _layer1_step(self, input_tensor):
        outputs = self._rknn1.inference(inputs=[input_tensor])
        output = outputs[0] if isinstance(outputs, list) else outputs
        spikes = self.lif1.step(output)
        return spikes

    def _layer2_step(self, spike_indices):
        spike_tensor = encode_spikes(spike_indices, self.num_mid_neurons)
        outputs = self._rknn2.inference(inputs=[spike_tensor])
        output = outputs[0] if isinstance(outputs, list) else outputs
        spikes = self.lif2.step(output)
        return spikes

    def infer_serial(self, input_tensor):
        """Serial execution baseline (both layers on Core 0)."""
        self.lif1.reset_state()
        self.lif2.reset_state()

        if input_tensor.ndim == 1:
            input_tensor = encode_image_pixels(input_tensor, self.num_in_neurons)

        for t in range(self.num_timesteps):
            spikes1 = self._layer1_step(input_tensor)
            spike_indices = [j for j, s in enumerate(spikes1) if s > 0.5]
            self._layer2_step(spike_indices)

        return self.lif2.get_spike_rate()

    def infer_pipelined(self, input_tensor):
        """Pipelined execution: Layer1 on Core0, Layer2 on Core1 via queue."""
        self.lif1.reset_state()
        self.lif2.reset_state()

        if input_tensor.ndim == 1:
            input_tensor = encode_image_pixels(input_tensor, self.num_in_neurons)

        l2_done = threading.Event()
        l2_errors = []

        def layer2_worker():
            try:
                for t in range(self.num_timesteps):
                    spike_indices = self._l1_to_l2.get()
                    if spike_indices is None:
                        break
                    self._layer2_step(spike_indices)
            except Exception as e:
                l2_errors.append(e)
            finally:
                l2_done.set()

        l2_thread = threading.Thread(target=layer2_worker, daemon=True)
        l2_thread.start()

        for t in range(self.num_timesteps):
            spikes1 = self._layer1_step(input_tensor)
            spike_indices = [j for j, s in enumerate(spikes1) if s > 0.5]
            self._l1_to_l2.put(spike_indices)

        self._l1_to_l2.put(None)
        l2_thread.join(timeout=5.0)

        if l2_errors:
            raise l2_errors[0]

        return self.lif2.get_spike_rate()

    def infer_pipelined_timed(self, input_tensor):
        """Pipelined execution with timing."""
        self.lif1.reset_state()
        self.lif2.reset_state()

        if input_tensor.ndim == 1:
            input_tensor = encode_image_pixels(input_tensor, self.num_in_neurons)

        l2_done = threading.Event()

        def layer2_worker():
            for t in range(self.num_timesteps):
                spike_indices = self._l1_to_l2.get()
                if spike_indices is None:
                    break
                self._layer2_step(spike_indices)
            l2_done.set()

        total_start = time.perf_counter()
        l2_thread = threading.Thread(target=layer2_worker, daemon=True)
        l2_thread.start()

        step_times_l1 = []
        for t in range(self.num_timesteps):
            step_start = time.perf_counter()
            spikes1 = self._layer1_step(input_tensor)
            spike_indices = [j for j, s in enumerate(spikes1) if s > 0.5]
            self._l1_to_l2.put(spike_indices)
            step_times_l1.append((time.perf_counter() - step_start) * 1000)

        self._l1_to_l2.put(None)
        l2_thread.join(timeout=5.0)
        total_ms = (time.perf_counter() - total_start) * 1000

        timings = {
            "total_ms": total_ms,
            "l1_avg_ms": np.mean(step_times_l1),
            "l1_max_ms": np.max(step_times_l1),
            "per_timestep_ms": total_ms / self.num_timesteps,
        }

        return self.lif2.get_spike_rate(), timings

    def infer_serial_timed(self, input_tensor):
        """Serial execution with timing."""
        self.lif1.reset_state()
        self.lif2.reset_state()

        if input_tensor.ndim == 1:
            input_tensor = encode_image_pixels(input_tensor, self.num_in_neurons)

        step_times = []
        total_start = time.perf_counter()

        for t in range(self.num_timesteps):
            step_start = time.perf_counter()
            spikes1 = self._layer1_step(input_tensor)
            spike_indices = [j for j, s in enumerate(spikes1) if s > 0.5]
            self._layer2_step(spike_indices)
            step_times.append((time.perf_counter() - step_start) * 1000)

        total_ms = (time.perf_counter() - total_start) * 1000

        timings = {
            "total_ms": total_ms,
            "avg_step_ms": np.mean(step_times),
            "max_step_ms": np.max(step_times),
            "per_timestep_ms": total_ms / self.num_timesteps,
        }

        return self.lif2.get_spike_rate(), timings