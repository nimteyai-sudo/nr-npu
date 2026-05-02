"""Spike encoding utilities for NR-NPU.

Converts sparse spike events into dense tensors that RK3588 NPU can process
as standard Conv2D inputs. Channel counts are padded to multiples of 8
for NC1HWC2 alignment.
"""

import numpy as np


def encode_spikes(spike_indices, num_neurons):
    """Convert spike indices to dense (1, C, 1, 1) tensor. C padded to mult of 8."""
    aligned = ((num_neurons + 7) // 8) * 8
    tensor = np.zeros((1, aligned, 1, 1), dtype=np.float32)
    for idx in spike_indices:
        if 0 <= idx < num_neurons:
            tensor[0, idx, 0, 0] = 1.0
    return tensor


def encode_rate(input_value, num_neurons):
    """Deterministic rate coding: neuron i fires if i/num_neurons < input_value."""
    return [i for i in range(num_neurons) if (i / num_neurons) < input_value]


def encode_image_pixels(image, num_neurons):
    """Map flattened image pixels to spike tensor (1, C, 1, 1)."""
    if image.ndim > 1:
        image = image.flatten()
    num_pixels = image.shape[0]
    aligned = ((num_neurons + 7) // 8) * 8
    tensor = np.zeros((1, aligned, 1, 1), dtype=np.float32)
    tensor[0, :num_pixels, 0, 0] = image.astype(np.float32)
    return tensor


def decode_output(output_tensor):
    """Extract spike indices from NPU output (non-zero channels)."""
    flat = output_tensor.flatten()
    return [i for i, v in enumerate(flat) if v > 1e-6]