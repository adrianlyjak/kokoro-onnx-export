import csv
import json
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort
import torch
import typer
from huggingface_hub import hf_hub_download
from kokoro.pipeline import KPipeline
from onnxruntime.quantization import (
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.calibrate import CalibrationDataReader
from rich import print
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from .cli import app


def process_calibration_row(
    row: dict[str, str], vocab: dict[str, int], pipeline: KPipeline
):
    """Process a single row of calibration data."""
    text = row["Text"]
    voice = row["Voice"]

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


@app.command()
def quantize(
    model_path: str = typer.Option(
        "kokoro.onnx", help="Path to the ONNX model to quantize"
    ),
    output_path: str = typer.Option(
        "kokoro_quantized.onnx", help="Path to save the quantized model"
    ),
    calibration_data: str = typer.Option(
        "data/quant-calibration.csv", help="Path to calibration data CSV"
    ),
    samples: Optional[int] = typer.Option(
        None,
        help="Maximum callibration samples to use. Uses all if not provided.",
    ),
) -> None:
    """
    Quantize an ONNX model using static quantization with calibration data.

    Args:
        model_path: Path to the input ONNX model
        output_path: Path to save the quantized model
        calibration_data: Path to CSV containing calibration data
    """
    print("Starting quantization process...")

    # Load calibration data using csv module
    calibration_rows = []
    with open(calibration_data, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        calibration_rows = list(reader)
    if samples is not None:
        calibration_rows = calibration_rows[:samples]
    print(f"Loaded {len(calibration_rows)} calibration samples")

    # Load vocabulary from Hugging Face
    repo_id = "hexgrad/Kokoro-82M"
    config_filename = "config.json"
    config_path = hf_hub_download(repo_id=repo_id, filename=config_filename)
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    vocab = config["vocab"]

    # Create calibration data reader
    data_reader = CsvCalibrationDataReader(calibration_rows, vocab)

    print("Starting calibration and quantization...")
    quantize_static(
        model_input=model_path,
        model_output=output_path,
        calibration_data_reader=data_reader,
        quant_format=QuantFormat.QDQ,
        op_types_to_quantize=["Conv"],  # Only quantize convolutions
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
    )

    print("Quantization complete!")


class CsvCalibrationDataReader(CalibrationDataReader):
    def __init__(self, rows, vocab):
        self.vocab = vocab
        self.rows = rows
        self.enum_data = iter(rows)
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            auto_refresh=False,
        )
        self.task = self.progress.add_task("Calibrating", total=len(rows))
        self.progress.start()

    def get_next(self):
        try:
            row = next(self.enum_data)
            # Extract language code from first character of voice name
            lang_code = row["Voice"][0]
            # Create pipeline for this language
            pipeline = KPipeline(lang_code=lang_code, model=False)
            processed = process_calibration_row(row, self.vocab, pipeline)
            self.progress.advance(self.task)
            self.progress.refresh()
            return processed
        except StopIteration:
            self.progress.stop()
            return None

    def rewind(self):
        self.enum_data = iter(self.rows)
        self.progress.reset(self.task)


class KokoroCalibrationDataReader(CalibrationDataReader):
    def __init__(self, calibration_data):
        """Initialize calibration data reader.

        Args:
            calibration_data: List of numpy arrays or path to .npz file containing calibration data
        """
        self.data = []
        if isinstance(calibration_data, (str, Path)):
            # Load from .npz file
            data = np.load(calibration_data)
            self.data = [data[key] for key in data.files]
        else:
            # Assume list of numpy arrays
            self.data = calibration_data

        self.data_counter = 0
        self.enum_data = iter([])

    def get_next(self):
        """Get next batch of calibration data."""
        try:
            item = next(self.enum_data)
            return item
        except StopIteration:
            return None

    def rewind(self):
        """Reset data iterator"""
        self.enum_data = iter([{f"input_{i}": arr} for i, arr in enumerate(self.data)])


def quantize_model(
    model_path: Path,
    output_path: Path,
    calibration_data,
    op_types_to_quantize=["Conv"],
    calibration_method=CalibrationMethod.MinMax,
):
    """Quantize an ONNX model using static quantization.

    Args:
        model_path: Path to input ONNX model
        output_path: Path to save quantized model
        calibration_data: List of numpy arrays or path to .npz file containing calibration data
        op_types_to_quantize: List of operator types to quantize (default: ['Conv'])
        calibration_method: Calibration method to use (default: MinMax)
    """
    # Create calibration data reader
    data_reader = KokoroCalibrationDataReader(calibration_data)

    # Quantize model
    quantize_static(
        model_input=model_path,
        model_output=output_path,
        calibration_data_reader=data_reader,
        quant_format=QuantFormat.QDQ,
        op_types_to_quantize=op_types_to_quantize,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=calibration_method,
    )
