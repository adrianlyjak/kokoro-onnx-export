import json
from typing import Union

import numpy as np
import onnx
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from kokoro.pipeline import KPipeline
from onnx import TensorProto

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


def get_onnx_inputs(
    voice: str, text: str, vocab: dict[str, int]
) -> dict[str, np.ndarray]:
    """Process text into corresponding ONNX inputs."""

    lang_code = voice[0]
    pipeline = KPipeline(lang_code=lang_code, model=False)

    # Get tokens from pipeline
    for result in pipeline(text):
        phoneme_output = result[1]
        break

    # Convert phonemes to input_ids
    tokens = [x for x in map(lambda p: vocab.get(p), phoneme_output) if x is not None]
    input_ids = torch.LongTensor([[0, *tokens, 0]])

    # Load and process the style vector
    ref_s = pipeline.load_voice(voice)
    ref_s = ref_s[input_ids.shape[1] - 1]

    return {
        "input_ids": input_ids.numpy(),
        "style": ref_s.numpy(),
        "speed": np.array([1.0], dtype=np.float32),
    }


def count_embedded_tensor_params(node: onnx.NodeProto) -> tuple[int, int]:
    """Count the number of parameters and estimated size in bits"""
    # Handle Constant nodes which contain embedded tensor data
    params = 0
    size = 0
    if node.op_type == "Constant":
        for attr in node.attribute:
            if attr.name == "value" and attr.t:
                tensor = attr.t
                num_params = np.prod(tensor.dims)
                params += num_params
                size += num_params * (TENSOR_TYPE_TO_SIZE.get(tensor.data_type, 4))

    for attr in node.attribute:
        if attr.type == onnx.AttributeProto.TENSOR:
            tensor = attr.t
            cnt = np.prod(tensor.dims)
            params += cnt
            size += cnt * (TENSOR_TYPE_TO_SIZE.get(tensor.data_type, 4))
        elif attr.type == onnx.AttributeProto.TENSORS:
            for tensor in attr.tensors:
                cnt = np.prod(tensor.dims)
                params += cnt
                size += cnt * (TENSOR_TYPE_TO_SIZE.get(tensor.data_type, 4))

    return params, size


def count_params_with_initializers(
    node: onnx.NodeProto, graph: onnx.GraphProto
) -> tuple[int, int]:
    """Count the number of parameters and estimated size in bits"""
    # Count parameters from node attributes
    initializers = build_initializer_lookup(graph)
    return count_params_with_initializers_lookup(node, initializers)


def build_initializer_lookup(graph: onnx.GraphProto) -> dict[str, TensorProto]:
    return {initializer.name: initializer for initializer in graph.initializer}


def count_params_with_initializers_lookup(
    node: onnx.NodeProto, initializers: dict[str, TensorProto]
) -> tuple[int, int]:
    """Count the number of parameters and estimated size in bits"""

    # Handle Constant nodes which contain embedded tensor data
    params, size = count_embedded_tensor_params(node)
    # Count parameters from initializers connected to this node's inputs
    for input_name in node.input:
        initializer = initializers.get(input_name)
        if initializer:
            cnt = np.prod(initializer.dims)
            params += cnt
            size += cnt * (TENSOR_TYPE_TO_SIZE.get(initializer.data_type, 4))

    return params, size


# Move type mapping to module level
TENSOR_TYPE_TO_NAME = {
    TensorProto.FLOAT: "FP32",
    TensorProto.FLOAT16: "FP16",
    TensorProto.BFLOAT16: "BFLOAT16",
    TensorProto.DOUBLE: "FP64",
    TensorProto.INT8: "INT8",
    TensorProto.UINT8: "UINT8",
    TensorProto.INT16: "INT16",
    TensorProto.UINT16: "UINT16",
    TensorProto.INT32: "INT32",
    TensorProto.UINT32: "UINT32",
    TensorProto.INT64: "INT64",
    TensorProto.UINT64: "UINT64",
    TensorProto.BOOL: "BOOL",
}

TENSOR_NAME_TO_TYPE = {name: dtype for dtype, name in TENSOR_TYPE_TO_NAME.items()}

TENSOR_TYPE_TO_SIZE = {
    TensorProto.FLOAT: 4,  # FP32
    TensorProto.FLOAT16: 2,  # FP16
    TensorProto.BFLOAT16: 2,  # BFLOAT16
    TensorProto.DOUBLE: 8,  # FP64
    TensorProto.INT8: 1,  # INT8
    TensorProto.UINT8: 1,  # UINT8
    TensorProto.INT16: 2,  # INT16
    TensorProto.UINT16: 2,  # UINT16
    TensorProto.INT32: 4,  # INT32
    TensorProto.UINT32: 4,  # UINT32
    TensorProto.INT64: 8,  # INT64
    TensorProto.UINT64: 8,  # UINT64
    TensorProto.BOOL: 1,  # BOOL
}
