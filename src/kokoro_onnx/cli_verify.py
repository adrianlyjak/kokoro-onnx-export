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

from .cli import app
from .util import execution_providers, mel_spectrogram_distance


@app.command()
def verify(
    onnx_path: str = typer.Option("kokoro.onnx", help="Path to the ONNX model file"),
    text: str = typer.Option(
        "Our quantization tool supports three calibration methods: MinMax, Entropy and Percentile. Please refer to calibrate.py for details.",
        help="Input text to synthesize",
    ),
    voice: str = typer.Option("af_heart", help="Voice ID to use"),
    output_dir: Optional[str] = typer.Option(
        None, help="Directory to save audio files. If None, uses current directory"
    ),
    profile: bool = typer.Option(False, help="Whether to profile the ONNX model"),
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
    _, phonemes, _ = next(pipeline.en_tokenize(tokens))

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
        if profile:
            session_options.enable_profiling = True
        session = ort.InferenceSession(
            onnx_path,
            session_options,
            providers=execution_providers,
        )
        ort_outputs = session.run(None, ort_inputs)

        # Export the profile data
        if profile:
            profile_file = session.end_profiling()
            print(f"ONNX model profiling data saved to: {profile_file}")

        # Get audio outputs
        torch_audio = torch_output.cpu().numpy()
        onnx_audio = ort_outputs[0]
        # Unwrap extra dimensions if needed
        if onnx_audio.ndim == 2:
            onnx_audio = onnx_audio[0]

        # Calculate MSE for audio outputs
        audio_mse = mel_spectrogram_distance(
            torch_audio, onnx_audio, distance_type="L2"
        )
        print(f"loss between PyTorch and ONNX outputs: {audio_mse:.5f}")

        # Save audio files
        output_dir = output_dir or "."
        torch_path = os.path.join(output_dir, "torch_output.wav")
        onnx_path = os.path.join(output_dir, "onnx_output.wav")

        sf.write(torch_path, torch_audio, 24000)
        sf.write(onnx_path, onnx_audio, 24000)

        print(
            f"Audio comparison complete. Files written: '{torch_path}', '{onnx_path}'."
        )
