import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import onnxscript
import torch
import typer
from kokoro.model import KModel, KModelForONNX
from onnxruntime.quantization import shape_inference
from rich import print
from torch.nn import utils

from kokoro_onnx.quantize import get_onnx_inputs

from .cli import app
from .util import execution_providers, load_vocab, mel_spectrogram_distance
from .verify import verify


@app.command()
def export(
    output_path: str = typer.Option("kokoro.onnx", help="Path to save the ONNX model"),
    remove_weight_norm: bool = typer.Option(
        True, help="Remove weight norm from the model"
    ),
    quant_preprocess: bool = typer.Option(
        True, help="Preprocess the model for quantization after exporting"
    ),
    score_difference: bool = typer.Option(
        True,
        help="Score the difference between the Torch and ONNX model for a test input after exporting",
    ),
) -> None:
    """
    Export the Kokoro model to ONNX format.

    Args:
        output_path: Path where the ONNX model will be saved
    """
    output_path = Path(output_path)
    batch_size: int = 1
    dummy_seq_length: int = 12
    style_dim: int = 256
    dummy_speed: float = 0.95
    opset_version: int = 20

    # Initialize model
    model = KModelForONNX(KModel(disable_complex=True)).eval()

    # Create dummy inputs
    input_ids = torch.zeros((batch_size, dummy_seq_length), dtype=torch.long)
    input_ids[0, :] = torch.LongTensor([0] + [1] * (dummy_seq_length - 2) + [0])

    # Style reference tensor
    style = torch.randn(batch_size, style_dim)

    def remove_weight_norm_recursive(module):
        for child in module.children():
            if hasattr(child, "weight_v"):
                # This module has weight norm
                utils.remove_weight_norm(child)
            else:
                # Recursively check this module's children
                remove_weight_norm_recursive(child)

    # Use it on your whole model
    if remove_weight_norm:
        remove_weight_norm_recursive(model)

    # Define dynamic axes
    dynamic_axes = {
        "input_ids": {1: "sequence_length"},
        "waveform": {0: "num_samples"},
    }

    print("Starting ONNX export...")

    torch.onnx.export(
        model,
        (input_ids, style, torch.tensor([dummy_speed], dtype=torch.float32)),
        output_path,
        input_names=["input_ids", "style", "speed"],
        output_names=["waveform"],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        export_params=True,
        do_constant_folding=True,
    )

    onnx_model = onnx.load(output_path)
    if quant_preprocess:
        print("Pre-processing model for quantization...")
        shape_inference.quant_pre_process(
            onnx_model, output_model_path=output_path, skip_symbolic_shape=True
        )
        onnx_model = onnx.load(output_path)

    # validate the model
    onnx.checker.check_model(onnx_model)

    print("Model was successfully exported to ONNX")

    if score_difference:
        verify(
            onnx_path=output_path,
            text="Despite its lightweight architecture, it delivers comparable quality to larger models",
            voice="af_heart",
            output_dir=None,
            profile=False,
        )
