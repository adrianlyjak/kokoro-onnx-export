import json
from typing import Union

import numpy as np
import torch
from huggingface_hub import hf_hub_download

execution_providers = [
    "CUDAExecutionProvider",
    "CoreMLExecutionProvider",
    "CPUExecutionProvider",
]


def mse_output_score(
    a: Union[torch.Tensor, np.ndarray],
    b: Union[torch.Tensor, np.ndarray],
) -> float:
    """Compare two audio outputs, handling different lengths.

    Args:
        torch_audio: Audio output from PyTorch model (torch.Tensor or np.ndarray)
        onnx_audio: Audio output from ONNX model (torch.Tensor or np.ndarray)

    Returns:
        MSE score between the outputs. If lengths differ, shorter output is zero-padded,
        which naturally increases the score.
    """
    # Convert to numpy if needed
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    if isinstance(b, torch.Tensor):
        b = b.detach().cpu().numpy()

    # Ensure arrays are 2D (batch_size, sequence_length)
    if a.ndim == 1:
        a = a[np.newaxis, :]
    if b.ndim == 1:
        b = b[np.newaxis, :]

    length_diff = abs(a.shape[-1] - b.shape[-1])

    # Pad shorter array to match longer one
    if a.shape[-1] > b.shape[-1]:
        pad_width = ((0, 0), (0, length_diff))
        b = np.pad(b, pad_width, mode="constant")
    elif b.shape[-1] > a.shape[-1]:
        pad_width = ((0, 0), (0, length_diff))
        a = np.pad(a, pad_width, mode="constant")

    return np.mean(np.square(a - b))


def load_vocab(
    repo_id: str = "hexgrad/Kokoro-82M", config_filename: str = "config.json"
) -> dict[str, int]:
    # Load vocabulary from Hugging Face
    config_path = hf_hub_download(repo_id=repo_id, filename=config_filename)
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    vocab = config["vocab"]
    return vocab
