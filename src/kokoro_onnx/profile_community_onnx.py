import os
from typing import List, Optional

import numpy as np
import onnxruntime as ort
import soundfile
from huggingface_hub import hf_hub_download
from onnxruntime import InferenceSession

from .cli import app

example_tokens = "[50, 157, 43, 135, 16, 53, 135, 46, 16, 43, 102, 16, 56, 156, 57, 135, 6, 16, 102, 62, 61, 16, 70, 56, 16, 138, 56, 156, 72, 56, 61, 85, 123, 83, 44, 83, 54, 16, 53, 65, 156, 86, 61, 62, 131, 83, 56, 4, 16, 54, 156, 43, 102, 53, 16, 156, 72, 61, 53, 102, 112, 16, 70, 56, 16, 138, 56, 44, 156, 76, 158, 123, 56, 16, 62, 131, 156, 43, 102, 54, 46, 16, 102, 48, 16, 81, 47, 102, 54, 16, 54, 156, 51, 158, 46, 16, 70, 16, 92, 156, 135, 46, 16, 54, 156, 43, 102, 48, 4, 16, 81, 47, 102, 16, 50, 156, 72, 64, 83, 56, 62, 16, 156, 51, 158, 64, 83, 56, 16, 44, 157, 102, 56, 16, 44, 156, 76, 158, 123, 56, 4]"


@app.command()
def prof_community(
    tokens: str = example_tokens,
    voice_file: str = "af.bin",
    model_name: str = "unquant.onnx",
    output_file: str = "community_onnx.wav",
    speed: float = 1.0,
) -> None:
    """
    Run profiled inferrence against competing onnx-community/Kokoro-82M-ONNX, to compare performance.

    Args:
        tokens: python List of input tokens, formatted as a string
        voice_file: Name of the voice file to use
        model_name: Name of the ONNX model file
        output_file: Path to save the output audio
        speed: Speed factor for audio generation
        enable_profiling: Whether to enable ONNX profiling
    """
    enable_profiling: bool = True
    repo_id: str = "onnx-community/Kokoro-82M-ONNX"
    voices_dir: str = "./voices"

    # Convert tokens string to list of ints
    tokens = eval(tokens)

    # Context length is 512, but leave room for the pad token 0 at the start & end
    assert len(tokens) <= 510, f"Token length {len(tokens)} exceeds maximum of 510"

    # Ensure voices directory exists
    if not os.path.exists(voices_dir):
        os.makedirs(voices_dir)

    # Download voice file if needed
    voice_path = os.path.join(voices_dir, voice_file)
    if not os.path.exists(voice_path):
        hf_hub_download(
            repo_id=repo_id,
            filename=f"voices/{voice_file}",
            local_dir=".",
            local_dir_use_symlinks=False,
        )

    # Style vector based on len(tokens), ref_s has shape (1, 256)
    voices = np.fromfile(voice_path, dtype=np.float32).reshape(-1, 1, 256)
    ref_s = voices[len(tokens)]

    # Add the pad ids, and reshape tokens, should now have shape (1, <=512)
    tokens_input = [[0, *tokens, 0]]

    # Set up ONNX session
    session_options = ort.SessionOptions()
    session_options.enable_profiling = enable_profiling
    sess = InferenceSession(model_name, session_options)

    # Run inference
    audio = sess.run(
        None,
        dict(
            input_ids=tokens_input,
            style=ref_s,
            speed=np.ones(1, dtype=np.float32) * speed,
        ),
    )

    if enable_profiling:
        sess.end_profiling()

    # Save audio output
    soundfile.write(output_file, audio[0][0], 24000)
