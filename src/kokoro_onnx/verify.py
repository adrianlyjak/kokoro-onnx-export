import json
import os
from typing import Optional

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
import typer
from huggingface_hub import hf_hub_download
from kokoro.model import KModel, KModelForONNX
from kokoro.pipeline import KPipeline
from loguru import logger
from torch.nn.functional import mse_loss

from .cli import app


@app.command()
def verify(
    onnx_path: str = typer.Option("kokoro.onnx", help="Path to the ONNX model file"),
    text: str = typer.Option(
        "How could I know? It's an unanswerable question.",
        help="Input text to synthesize",
    ),
    voice: str = typer.Option("af_heart", help="Voice ID to use"),
    output_dir: Optional[str] = typer.Option(
        None, help="Directory to save audio files. If None, uses current directory"
    ),
) -> float:
    """
    Verify ONNX model output against PyTorch model output.

    Args:
        onnx_path: Path to the ONNX model file
        text: Input text to synthesize
        voice: Voice ID to use
        output_dir: Directory to save audio files. If None, uses current directory

    Returns:
        float: Mean squared error between PyTorch and ONNX outputs
    """
    # Initialize the pipeline
    pipeline = KPipeline(lang_code="a", model=False)

    # Load vocabulary from Hugging Face
    repo_id = "hexgrad/Kokoro-82M"
    config_filename = "config.json"
    config_path = hf_hub_download(repo_id=repo_id, filename=config_filename)
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    vocab = config["vocab"]

    # Initialize the KModel
    torch_model = KModelForONNX(KModel()).eval()

    # Tokenize and phonemize
    _, tokens = pipeline.g2p(text)

    # Process the first token sequence (for simplicity)
    graphemes, phonemes, token_list = next(pipeline.en_tokenize(tokens))

    with torch.no_grad():
        # Convert phonemes to input_ids
        input_ids = torch.LongTensor([[0, *map(lambda p: vocab.get(p), phonemes), 0]])

        # Load and process the style vector
        ref_s = pipeline.load_voice(voice)
        ref_s = ref_s[input_ids.shape[1] - 1]  # Select the appropriate style vector

        # Run the PyTorch model
        torch_output = torch_model(input_ids=input_ids, ref_s=ref_s, speed=1.0)

        # Run the ONNX model
        ort_inputs = {
            "input_ids": input_ids.numpy(),
            "style": ref_s.numpy(),
            "speed": np.array([1.0], dtype=np.float32),
        }
        session_options = ort.SessionOptions()
        session_options.enable_profiling = True
        session = ort.InferenceSession(onnx_path, session_options)
        ort_outputs = session.run(None, ort_inputs)

        # Export the profile data
        profile_file = session.end_profiling()
        logger.info(f"ONNX model profiling data saved to: {profile_file}")

        # Get audio outputs
        torch_audio = torch_output.cpu().numpy()
        onnx_audio = ort_outputs[0]

        # Calculate MSE for audio outputs
        audio_mse = mse_loss(
            torch.tensor(torch_audio).flatten(), torch.tensor(onnx_audio).flatten()
        ).item()
        logger.info(f"MSE for audio output: {audio_mse:.5f}")

        # Save audio files
        output_dir = output_dir or "."
        torch_path = os.path.join(output_dir, "torch_output.wav")
        onnx_path = os.path.join(output_dir, "onnx_output.wav")

        sf.write(torch_path, torch_audio, 24000)
        sf.write(onnx_path, onnx_audio, 24000)

        logger.info(
            f"Audio comparison complete. Files written: '{torch_path}', '{onnx_path}'."
        )

        logger.info(
            f"Mean squared error between PyTorch and ONNX outputs: {audio_mse:.5f}"
        )
