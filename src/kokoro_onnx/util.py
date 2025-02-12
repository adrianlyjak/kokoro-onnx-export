from typing import Union

import numpy as np
import torch

execution_providers = [
    "CUDAExecutionProvider",
    "CoreMLExecutionProvider",
    "CPUExecutionProvider",
]


def mse_output_score(
    torch_audio: Union[torch.Tensor, np.ndarray],
    onnx_audio: Union[torch.Tensor, np.ndarray],
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
    if isinstance(torch_audio, torch.Tensor):
        torch_audio = torch_audio.detach().cpu().numpy()
    if isinstance(onnx_audio, torch.Tensor):
        onnx_audio = onnx_audio.detach().cpu().numpy()

    # Ensure arrays are 2D (batch_size, sequence_length)
    if torch_audio.ndim == 1:
        torch_audio = torch_audio[np.newaxis, :]
    if onnx_audio.ndim == 1:
        onnx_audio = onnx_audio[np.newaxis, :]

    length_diff = abs(torch_audio.shape[-1] - onnx_audio.shape[-1])

    # Pad shorter array to match longer one
    if torch_audio.shape[-1] > onnx_audio.shape[-1]:
        pad_width = ((0, 0), (0, length_diff))
        onnx_audio = np.pad(onnx_audio, pad_width, mode="constant")
    elif onnx_audio.shape[-1] > torch_audio.shape[-1]:
        pad_width = ((0, 0), (0, length_diff))
        torch_audio = np.pad(torch_audio, pad_width, mode="constant")

    return np.mean(np.square(torch_audio - onnx_audio))
