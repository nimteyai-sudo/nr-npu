"""ANN to SNN converter for RK3588 NPU. Converts Linear layers to channel-aligned Conv2D."""

import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxoptimizer
from onnx import helper, TensorProto
import tempfile
import os


def linear_to_conv2d(linear, align_channels=True):
    """Convert Linear layer to equivalent 1x1 Conv2d, padding channels to 8x."""
    in_features = linear.in_features
    out_features = linear.out_features

    if align_channels:
        in_aligned = ((in_features + 7) // 8) * 8
        out_aligned = ((out_features + 7) // 8) * 8
    else:
        in_aligned = in_features
        out_aligned = out_features

    conv = nn.Conv2d(in_aligned, out_aligned, kernel_size=1, bias=True)

    with torch.no_grad():
        conv.weight[:] = 0.0
        conv.weight[:out_features, :in_features, 0, 0] = linear.weight.data
        conv.bias[:] = 0.0
        conv.bias[:out_features] = linear.bias.data

    return conv


def ann_to_snn_model(layers, align_channels=True):
    """Convert a list of Linear layers to Sequential of channel-aligned Conv2d layers."""
    conv_layers = []
    for linear in layers:
        conv = linear_to_conv2d(linear, align_channels)
        conv_layers.append(conv)
    return nn.Sequential(*conv_layers)


def export_conv_to_onnx(conv, input_channels, path):
    dummy_input = torch.randn(1, input_channels, 1, 1)
    torch.onnx.export(
        conv,
        dummy_input,
        path,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,
    )


def export_snn_model_to_onnx(model, input_shape, path):
    dummy_input = torch.randn(1, *input_shape)
    torch.onnx.export(
        model,
        dummy_input,
        path,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
    )


def compute_snn_threshold(linear, method="max"):
    """Compute firing threshold for ANN→SNN conversion."""
    w = linear.weight.data.abs()
    if method == "max":
        return w.max().item()
    elif method == "percentile":
        return np.percentile(w.numpy(), 99)
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_required_timesteps(layers, method="max"):
    """Estimate minimum timesteps from bias/threshold ratio across layers."""
    max_ratio = 0
    for layer in layers:
        threshold = compute_snn_threshold(layer, method)
        bias = layer.bias.data.abs().max().item()
        if threshold > 0:
            ratio = bias / threshold
            max_ratio = max(max_ratio, ratio)
    return int(max_ratio) + 1