import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import typer
from kokoro.model import KModel, KModelForONNX
from loguru import logger

from .cli import app


@app.command()
def export(
    output_path: str = typer.Option("kokoro.onnx", help="Path to save the ONNX model"),
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

    # Validate the inputs against the model first
    start_time = time.time()
    output = model(input_ids=input_ids, ref_s=style, speed=dummy_speed)
    logger.info(f"Time for dummy inputs: {time.time() - start_time}")

    # Define dynamic axes
    dynamic_axes = {
        "input_ids": {1: "sequence_length"},
        "waveform": {0: "num_samples"},
    }

    logger.info("Starting ONNX export...")
    try:
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

        # Verify the model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        logger.info("Model was successfully exported to ONNX")

        # Additional check: Run a simple inference to validate the exported model
        ort_session = ort.InferenceSession(str(output_path))
        ort_inputs = {
            "input_ids": input_ids.numpy(),
            "style": style.numpy(),
            "speed": np.array([dummy_speed], dtype=np.float32),
        }
        ort_outputs = ort_session.run(None, ort_inputs)

        # Compare audio output
        torch_audio = output
        onnx_audio = torch.tensor(ort_outputs[0])
        audio_mse = (torch_audio - onnx_audio).pow(2).mean().item()
        logger.info(f"MSE for audio waveform: {audio_mse:.5f}")

    except Exception as e:
        logger.error(f"Export failed: {str(e)}")
        raise
