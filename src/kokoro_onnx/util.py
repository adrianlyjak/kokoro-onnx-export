import json
from typing import Union

import numpy as np
import torch
import torchaudio
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


def mel_spectrogram_distance(
    ref_waveform: np.ndarray,
    test_waveform: np.ndarray,
    sample_rate: int = 24000,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 80,
    window_fn=torch.hann_window,
    distance_type: str = "L2",
) -> float:
    """
    Compute a perceptual distance between two audio signals by comparing
    their log-mel spectrograms.

    Args:
        ref_waveform (np.ndarray): Reference audio (1D or 2D: channels x time).
        test_waveform (np.ndarray): Test audio (1D or 2D: channels x time).
        sample_rate (int): Sample rate of the waveforms.
        n_fft (int): FFT size.
        hop_length (int): Number of samples between frames.
        n_mels (int): Number of mel-frequency bins.
        window_fn (callable): Window function for the STFT.
        distance_type (str): "L1" or "L2" distance.

    Returns:
        float: The average distance between the log-mel spectrograms.
    """

    # Convert inputs to torch.Tensor (mono or first channel if multi-channel)
    if ref_waveform.ndim > 1:
        ref_waveform = ref_waveform[0]  # pick first channel
    if test_waveform.ndim > 1:
        test_waveform = test_waveform[0]

    ref_waveform_t = torch.from_numpy(ref_waveform).float().unsqueeze(0)
    test_waveform_t = torch.from_numpy(test_waveform).float().unsqueeze(0)

    # If lengths differ, pad the shorter one
    len_ref = ref_waveform_t.shape[-1]
    len_test = test_waveform_t.shape[-1]
    if len_ref > len_test:
        pad_amount = len_ref - len_test
        test_waveform_t = torch.nn.functional.pad(test_waveform_t, (0, pad_amount))
    elif len_test > len_ref:
        pad_amount = len_test - len_ref
        ref_waveform_t = torch.nn.functional.pad(ref_waveform_t, (0, pad_amount))

    # Create a mel spectrogram transform
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        window_fn=window_fn,
    )

    # Compute mel spectrograms
    ref_mel = mel_transform(ref_waveform_t)  # shape: (1, n_mels, time_frames)
    test_mel = mel_transform(test_waveform_t)

    # Convert to log-mel
    ref_log_mel = torch.log(ref_mel + 1e-8)
    test_log_mel = torch.log(test_mel + 1e-8)

    # Compute distance
    if distance_type == "L2":
        dist = (ref_log_mel - test_log_mel).pow(2).mean().sqrt()
    else:  # "L1" by default
        dist = (ref_log_mel - test_log_mel).abs().mean()

    return dist.item()
