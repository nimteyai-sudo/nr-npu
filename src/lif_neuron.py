"""LIF and IF neurons for NR-NPU.

CPU-side membrane tracking. NPU does Conv2D for synaptic input,
this handles integrate/leak/threshold/reset that NPU can't.
"""

import numpy as np


class LIFNeuron:
    """Float32 LIF neuron group."""

    def __init__(self, num_neurons, threshold=1.0, leak_rate=0.1, reset_value=0.0):
        self.num_neurons = ((num_neurons + 7) // 8) * 8
        self.actual_neurons = num_neurons
        self.threshold = threshold
        self.leak_rate = leak_rate
        self.decay = 1.0 - leak_rate
        self.reset_value = reset_value
        self.membrane = np.zeros(self.num_neurons, dtype=np.float32)
        self.spike_count = np.zeros(self.num_neurons, dtype=np.int32)

    def reset_state(self):
        self.membrane[:] = 0.0
        self.spike_count[:] = 0

    def step(self, delta_potential):
        if delta_potential.ndim > 1:
            delta_potential = delta_potential.flatten()[:self.num_neurons]
        self.membrane += delta_potential
        self.membrane *= self.decay
        fired = self.membrane > self.threshold
        self.membrane[fired] = self.reset_value
        self.spike_count[fired] += 1
        spikes = np.zeros(self.num_neurons, dtype=np.float32)
        spikes[fired] = 1.0
        return spikes

    def get_spike_rate(self):
        rates = self.spike_count[:self.actual_neurons].astype(np.float32)
        mx = rates.max()
        if mx > 0:
            rates /= mx
        return rates


class LIFNeuronInt8:
    """Integer-membrane LIF for reduced float overhead.

    Uses int16 saturating arithmetic. Subtractive leak instead of multiplicative.
    Scale factor maps NPU float32 output back to int16 domain.
    Measured scale: L1=15.0, L2=30.6 for MNIST Conv2D models.
    """

    def __init__(self, num_neurons, threshold_int=15, leak_int=2, scale=15.0):
        self.num_neurons = ((num_neurons + 7) // 8) * 8
        self.actual_neurons = num_neurons
        self.threshold_int = threshold_int
        self.leak_int = leak_int
        self.scale = scale
        self.membrane = np.zeros(self.num_neurons, dtype=np.int16)
        self.spike_count = np.zeros(self.num_neurons, dtype=np.int32)

    def reset_state(self):
        self.membrane[:] = 0
        self.spike_count[:] = 0

    def step(self, delta_potential):
        if delta_potential.ndim > 1:
            delta_potential = delta_potential.flatten()[:self.num_neurons]
        delta_int = (delta_potential * self.scale).astype(np.int16)
        self.membrane = np.clip(
            self.membrane.astype(np.int32) + delta_int.astype(np.int32),
            -32768, 32767).astype(np.int16)
        self.membrane = np.clip(
            self.membrane.astype(np.int32) - np.sign(self.membrane) * self.leak_int,
            -32768, 32767).astype(np.int16)
        fired = self.membrane > self.threshold_int
        self.membrane[fired] = 0
        self.spike_count[fired] += 1
        spikes = np.zeros(self.num_neurons, dtype=np.float32)
        spikes[fired] = 1.0
        return spikes

    def get_spike_rate(self):
        rates = self.spike_count[:self.actual_neurons].astype(np.float32)
        mx = rates.max()
        if mx > 0:
            rates /= mx
        return rates


class IFNeuron(LIFNeuron):
    """Integrate-and-Fire without leak."""
    def __init__(self, num_neurons, threshold=1.0, reset_value=0.0):
        super().__init__(num_neurons, threshold, leak_rate=0.0, reset_value=reset_value)