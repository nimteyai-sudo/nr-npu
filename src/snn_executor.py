"""SNN runtime for RK3588 NPU. Main loop: encode spikes -> NPU Conv2D -> LIF update -> spike detection."""

import numpy as np
import time
from pathlib import Path
from rknnlite.api import RKNNLite

from spike_encoder import encode_spikes, encode_image_pixels, decode_output
from lif_neuron import LIFNeuron


class SNNLayer:
    """One SNN layer backed by one RKNN model on NPU."""

    def __init__(self, rknn_model_path, num_in_neurons, num_out_neurons,
                 threshold=1.0, leak_rate=0.1, core_id=0):
        self.num_in_neurons = num_in_neurons
        self.num_out_neurons = num_out_neurons
        self.lif = LIFNeuron(num_out_neurons, threshold=threshold, leak_rate=leak_rate)
        self.core_id = core_id
        self.rknn_model_path = rknn_model_path
        self._rknn = None
        self._input_shape = None

    def init_runtime(self):
        self._rknn = RKNNLite()
        ret = self._rknn.load_rknn(self.rknn_model_path)
        if ret != 0:
            raise RuntimeError(f"Failed to load RKNN model: {self.rknn_model_path}")

        ret = self._rknn.init_runtime(target=None, core_mask=RKNNLite.NPU_CORE_AUTO)
        if ret != 0:
            raise RuntimeError(f"Failed to init RKNN runtime on core {self.core_id}")

        # NC1HWC2 alignment: input channels must be 8x
        self._input_shape = ((self.num_in_neurons + 7) // 8) * 8
        return True

    def step(self, spike_indices):
        """One timestep: encode spikes -> NPU inference -> LIF update -> spike indices."""
        spike_tensor = encode_spikes(spike_indices, self.num_in_neurons)
        outputs = self._rknn.inference(inputs=[spike_tensor])
        output = outputs[0] if isinstance(outputs, list) else outputs
        new_spikes = self.lif.step(output)
        return [i for i, s in enumerate(new_spikes) if s > 0.5]

    def step_tensor(self, spike_tensor):
        """One timestep with pre-encoded input tensor."""
        outputs = self._rknn.inference(inputs=[spike_tensor])
        output = outputs[0] if isinstance(outputs, list) else outputs
        return self.lif.step(output)

    def release(self):
        if self._rknn:
            self._rknn.release()
            self._rknn = None


class SNNExecutor:
    """Multi-layer SNN runtime on RK3588 NPU."""

    def __init__(self, layers, num_timesteps=20):
        self.layers = layers
        self.num_timesteps = num_timesteps

    def init_all(self):
        for layer in self.layers:
            layer.init_runtime()

    def release_all(self):
        for layer in self.layers:
            layer.release()

    def infer(self, input_tensor):
        """Run full SNN inference over T timesteps. Returns spike rate from last layer."""
        for layer in self.layers:
            layer.lif.reset_state()

        if input_tensor.ndim == 1:
            num_in = self.layers[0].num_in_neurons
            input_tensor = encode_image_pixels(input_tensor, num_in)

        for t in range(self.num_timesteps):
            current_input = input_tensor
            for i, layer in enumerate(self.layers):
                outputs = layer._rknn.inference(inputs=[current_input])
                output = outputs[0] if isinstance(outputs, list) else outputs
                spikes = layer.lif.step(output)

                if i < len(self.layers) - 1:
                    spike_indices = [j for j, s in enumerate(spikes) if s > 0.5]
                    current_input = encode_spikes(
                        spike_indices, self.layers[i + 1].num_in_neurons
                    )

        return self.layers[-1].lif.get_spike_rate()

    def infer_timed(self, input_tensor):
        """Run inference with per-step timing measurements."""
        for layer in self.layers:
            layer.lif.reset_state()

        if input_tensor.ndim == 1:
            num_in = self.layers[0].num_in_neurons
            input_tensor = encode_image_pixels(input_tensor, num_in)

        timings = {"timesteps": [], "total_ms": 0}
        total_start = time.perf_counter()

        for t in range(self.num_timesteps):
            step_start = time.perf_counter()

            current_input = input_tensor
            for i, layer in enumerate(self.layers):
                outputs = layer._rknn.inference(inputs=[current_input])
                output = outputs[0] if isinstance(outputs, list) else outputs
                spikes = layer.lif.step(output)

                if i < len(self.layers) - 1:
                    spike_indices = [j for j, s in enumerate(spikes) if s > 0.5]
                    current_input = encode_spikes(
                        spike_indices, self.layers[i + 1].num_in_neurons
                    )

            step_ms = (time.perf_counter() - step_start) * 1000
            timings["timesteps"].append(step_ms)

        timings["total_ms"] = (time.perf_counter() - total_start) * 1000
        timings["avg_step_ms"] = np.mean(timings["timesteps"])

        return self.layers[-1].lif.get_spike_rate(), timings