import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generator, Literal, Optional

import numpy as np
import onnx
import onnxruntime as ort
import soundfile as sf
import torch
import typer
from neural_compressor import PostTrainingQuantConfig, quantization
from neural_compressor.config import AccuracyCriterion, TuningCriterion
from onnxruntime.quantization import (
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.calibrate import CalibrationDataReader

# from rich import print
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from kokoro_onnx.quantize_trial import (
    QUANT_RULES,
    DataType,
    NodeToReduce,
    QuantizationSelection,
    estimate_quantized_size,
    run_float16_trials,
    run_quantization_trials,
    select_node_datatypes,
)
from kokoro_onnx.util import load_vocab

from .cli import app
from .convert_float_to_float16 import convert_float_to_float16
from .util import get_onnx_inputs, mel_spectrogram_distance


@app.command()
def quantize_neural_compressor(
    onnx_path: str = typer.Option(
        "kokoro.onnx", help="Path to the ONNX model to quantize"
    ),
    output_path: str = typer.Option(
        "kokoro_quantized.onnx", help="Path to save the quantized model"
    ),
    calibration_data: str = typer.Option(
        "data/quant-calibration.csv", help="Path to calibration data CSV"
    ),
    samples: Optional[int] = typer.Option(
        None, help="Number of samples to use for calibration"
    ),
    trial_path: str = typer.Option(
        "quantization_trials.csv", help="Path to load trial results from"
    ),
    quant_threshold: float = typer.Option(0.25, help="Threshold for trial results"),
    fp16_threshold: float = typer.Option(0.25, help="Threshold for trial results"),
    eval_text: str = typer.Option(
        "Despite its lightweight architecture, it delivers comparable quality to larger models",
        help="Text to evaluate the model with",
    ),
    eval_voice: str = typer.Option("af_heart", help="Voice to evaluate the model with"),
) -> None:
    """
    Quantize an ONNX model using the Neural Compressor library.

    Args:
        model_path: Path to the input ONNX model
        output_path: Path to save the quantized model
    """
    print("Starting quantization process...")

    model = onnx.load(onnx_path)
    vocab = load_vocab()

    inputs = get_onnx_inputs(eval_voice, eval_text, vocab)

    init_session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    init_outputs = init_session.run(None, inputs)[0]

    def eval_func(model: onnx.ModelProto):
        inputs = get_onnx_inputs(eval_voice, eval_text, vocab)
        sess = ort.InferenceSession(
            model.SerializeToString(),
            providers=["CUDAExecutionProvider"]
            if torch.cuda.is_available()
            else ["CPUExecutionProvider"],
        )
        outputs = sess.run(None, inputs)[0]
        return 10 - mel_spectrogram_distance(init_outputs, outputs, distance_type="L2")

    calibration_dataloader = NeuralCalibrationDataloader(
        calibration_data, num_samples=samples, vocab=vocab
    )

    ops = {}

    specs = select_node_datatypes(
        quant_threshold=quant_threshold,
        fp16_threshold=fp16_threshold,
        trial_csv_path=trial_path,
    )

    spec_dict = {node.name: node for node in specs}

    def convert_dtype(dtype: DataType) -> str:
        if dtype == DataType.FLOAT16:
            return "fp16"
        elif dtype == DataType.INT8:
            return "int8"
        else:
            return "fp32"

    for node in model.graph.node:
        spec = spec_dict.get(node.name)
        if spec:
            ops[node.name] = {
                "weight": {"dtype": convert_dtype(spec.weights_type)},
                "activation": {"dtype": convert_dtype(spec.activations_type)},
            }
        else:
            ops[node.name] = {
                "weight": {"dtype": "fp32"},
                "activation": {"dtype": "fp32"},
            }

    quantization.fit(
        model,
        conf=PostTrainingQuantConfig(
            device="gpu" if torch.cuda.is_available() else "cpu",
            excluded_precisions=["bf16"],
            quant_format="QDQ",
            approach="static",
            tuning_criterion=TuningCriterion(
                strategy="basic",
                timeout=0,
                max_trials=1,
            ),
            accuracy_criterion=AccuracyCriterion(
                criterion="absolute",
                tolerable_loss=1.0,
            ),
            op_name_dict=ops,
            quant_level=1,
        ),
        calib_dataloader=calibration_dataloader,
        eval_func=eval_func,
    )


class NeuralCalibrationDataloader:
    def __init__(
        self,
        path: str,
        num_samples: Optional[int] = None,
        vocab: Optional[dict[str, int]] = None,
    ):
        self.vocab = vocab or load_vocab()
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.calibration_rows = list(reader)
        if num_samples is None:
            self.num_samples = len(self.calibration_rows)
        else:
            self.num_samples = num_samples
        self.batch_size = 1

    def __iter__(self):
        for row in self.calibration_rows[: self.num_samples]:
            inputs = get_onnx_inputs(row["Voice"], row["Text"], self.vocab)
            yield inputs, None


@app.command()
def quantize(
    onnx_path: str = typer.Option(
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

    data_reader = CsvCalibrationDataReader(calibration_data, samples)

    print("Starting calibration and quantization...")

    model = onnx.load(onnx_path)

    ops_to_include = ["Conv"]
    to_include = []
    ignored = []

    regex = re.compile(
        r"""(?x) # Enable verbose mode
        (?:   # Non-capturing group for multiple OR options
            ^/text_encoder/cnn.*
            | ^/decoder/generator/noise_res.* # adds 0.00001 loss
            # misc single convs in decoder
            | ^/decoder/F0_conv/.* # no loss
            | ^/decoder/N_conv/.* # 0.00002 increase in loss, not discernable to my ear
            | ^/decoder/asr_res/.* # no loss
            # lots of them like /decoder/decode.2/conv1/Conv
            | ^/decoder/decode\.\d/conv2.* # for all 0.00003 increase in loss, not discernable to my ear. 12 layers, this knocks down 4
            | ^/decoder/decode\..* # the rest
            | ^/N\..* # no loss
            # | ^/decoder/generator/resblocks\.[0].* # LOTS of params. From 0 to 5, each with 3 convs, so 15 layers. 0.00009 increase in loss if all are included. Certain hollowness to the sound. Limit to middle convolutions doesn't help much, but first layer doesn't seem to hurt
            | ^/decoder/generator/resblocks\..* # the rest
            | ^/decoder/generator/noise_convs\.\d/Conv
            | ^/decoder/generator/Conv.* # 0.00002 loss, for 2 layers
            # | ^/F0_proj/Conv # ginormous loss
            | ^/N_proj/Conv # 0.00001 loss, for 1 layer
            # | ^/F0\..* # terrible jump in loss. Lots of static
            # | ^/decoder/generator/conv_post/Conv # huge loss
            | ^/decoder/encode/conv2/Conv # 0.00002 loss, for 1 layer
            | ^/decoder/encode/.* # 0.00004 loss, for 3 layers
            | ^kmodel\.decoder.* # tons of things
        )
        """
    )
    # regex = re.compile(r"kmodel\.decoder.*")
    # Count each node type
    for node in model.graph.node:
        if node.op_type in ops_to_include:
            if regex.match(node.name):
                to_include.append(node.name)
            else:
                ignored.append(node.name)

    print("including nodes", to_include)
    print("ignored nodes", ignored)
    quantize_static(
        model_input=onnx_path,
        model_output=output_path,
        calibration_data_reader=data_reader,
        quant_format=QuantFormat.QOperator,
        op_types_to_quantize=ops_to_include,
        nodes_to_exclude=[
            # buggy blows up export
            "/text_encoder/cnn.0/cnn.0.2/LeakyRelu",
            "/text_encoder/cnn.1/cnn.0.2/LeakyRelu",
            "/text_encoder/cnn.2/cnn.0.2/LeakyRelu",
        ],
        nodes_to_quantize=to_include,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
    )

    model_size = Path(output_path).stat().st_size / (1024 * 1024)  # Convert to MB
    print(f"Quantization complete! Model size: {model_size:.2f} MB")


class CsvCalibrationDataReader(CalibrationDataReader):
    def __init__(
        self,
        path: str,
        samples: Optional[int] = None,
        vocab: Optional[dict[str, int]] = None,
    ):
        self.vocab = vocab or load_vocab()
        # Load calibration data using csv module
        calibration_rows = []
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            calibration_rows = list(reader)
        if samples is not None:
            calibration_rows = calibration_rows[:samples]
        self.rows = calibration_rows
        self.enum_data = iter(calibration_rows)
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            auto_refresh=False,
        )
        self.task = self.progress.add_task("Calibrating", total=len(calibration_rows))
        self.progress.start()

    def get_next(self):
        try:
            row = next(self.enum_data)
            # Extract language code from first character of voice name
            """Process a single row of calibration data."""
            text = row["Text"]
            voice = row["Voice"]
            processed = get_onnx_inputs(voice, text, self.vocab)
            self.progress.advance(self.task)
            self.progress.refresh()
            return processed
        except StopIteration:
            self.progress.stop()
            return None

    def rewind(self):
        self.enum_data = iter(self.rows)
        self.progress.reset(self.task)


@app.command()
def float16(
    onnx_path: str = typer.Option(
        "kokoro.onnx", help="Path to the ONNX model to convert"
    ),
    output_path: str = typer.Option(
        "kokoro_fp16.onnx", help="Path to save the FP16 model"
    ),
    sample_text: str = typer.Option(
        "Despite its lightweight architecture, it delivers comparable quality to larger models",
        help="Text to sample the model with for auto callibration",
    ),
    sample_voice: str = typer.Option(
        "af_heart", help="Voice to sample the model with for auto callibration"
    ),
) -> None:
    """Convert specific parts of model to float16."""

    print(f"Loading model from {onnx_path}")
    model = onnx.load(onnx_path)

    # Add debug output before conversion
    print("\nAnalyzing original model:")

    # Generate test inputs
    print("Generating test inputs...")
    vocab = load_vocab()
    inputs = get_onnx_inputs(sample_voice, sample_text, vocab)

    # Test original model
    print("\nTesting original model...")
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    original_outputs = sess.run(None, inputs)

    # Compare model with itself as sanity check
    print("\nSanity check - comparing original model with itself:")
    for i, out in enumerate(original_outputs):
        print(f"Output {i}: shape={out.shape}, dtype={out.dtype}")
        if np.isnan(out).any():
            print(f"WARNING: Output {i} contains NaN values!")

    # Define ops to convert to float16
    target_ops = {
        # "Conv",
        # "ConvTranspose",
        # "MatMul",
        # "Gemm",
        # "LayerNormalization",
        # "Add",  # Often used with residual connections
        # "Mul",  # Common in attention mechanisms
        "Div",
        # "LSTM",
    }
    node_block_list = [
        node.name for node in model.graph.node if node.op_type not in target_ops
    ]
    # Convert model to float16, targeting specific ops
    model_fp16 = convert_float_to_float16(
        model,
        keep_io_types=True,
        node_block_list=node_block_list,
    )
    print(f"\nSaving model to {output_path}...")
    onnx.save(model_fp16, output_path)

    # Test converted model
    print("\nTesting converted model...")
    try:
        sess_fp16 = ort.InferenceSession(
            output_path, providers=["CPUExecutionProvider"]
        )
        fp16_outputs = sess_fp16.run(None, inputs)
    except Exception as e:
        print(f"Error testing converted model: {e}")
        raise

    # Replace node type counting with model diffing
    print("\nComparing original and converted models:")
    diff_models(model, model_fp16)

    # Compare outputs
    print("\nComparing original and converted model outputs:")
    for i, (orig, conv) in enumerate(zip(original_outputs, fp16_outputs)):
        mse = mel_spectrogram_distance(orig, conv)
        print(f"  MSE: {mse}")

    print("Conversion complete!")

    # Add debug output after conversion
    print("\nAnalyzing converted model:")


def diff_models(model1, model2):
    """Compare two ONNX models and print their differences."""
    print("\nModel differences:")

    # Get nodes from both models
    nodes1 = {n.name: n for n in model1.graph.node}
    nodes2 = {n.name: n for n in model2.graph.node}

    # Find added and removed nodes
    added_nodes = set(nodes2.keys()) - set(nodes1.keys())
    removed_nodes = set(nodes1.keys()) - set(nodes2.keys())

    # Count operation types for added nodes
    added_ops = {}
    for name in added_nodes:
        op_type = nodes2[name].op_type
        added_ops[op_type] = added_ops.get(op_type, 0) + 1

    # Count operation types for removed nodes
    removed_ops = {}
    for name in removed_nodes:
        op_type = nodes1[name].op_type
        removed_ops[op_type] = removed_ops.get(op_type, 0) + 1

    # Print summary
    if added_ops:
        print("\nAdded operations:")
        for op_type, count in sorted(added_ops.items()):
            print(f"  {op_type}: {count} nodes")

    if removed_ops:
        print("\nRemoved operations:")
        for op_type, count in sorted(removed_ops.items()):
            print(f"  {op_type}: {count} nodes")

    if not (added_ops or removed_ops):
        print("  No structural changes detected")

    # Compare total node counts
    print(f"\nTotal nodes:")
    print(f"  Original: {len(nodes1)}")
    print(f"  Modified: {len(nodes2)}")
    print(f"  Difference: {len(nodes2) - len(nodes1):+d}")


def float16_validate_fn(
    mse_threshold: float = 0.0001,
) -> Callable[[np.ndarray, np.ndarray], bool]:
    def validate(res1, res2):
        for r1, r2 in zip(res1, res2):
            mse = mel_spectrogram_distance(r1, r2)
            print("mel_spectrogram_distance:", mse)
            if mse > mse_threshold:
                return False
        return True

    return validate


@app.command()
def run_trials(
    onnx_path: str = typer.Option(
        "kokoro.onnx", help="Path to the ONNX model to quantize"
    ),
    output_path: str = typer.Option(
        "quantization_trials.csv", help="Path to save trial results"
    ),
    calibration_data: str = typer.Option(
        "data/quant-calibration.csv", help="Path to calibration data CSV"
    ),
    samples: Optional[int] = typer.Option(
        None,
        help="Maximum calibration samples to use. Uses all if not provided.",
    ),
    eval_text: str = typer.Option(
        "Despite its lightweight architecture, it delivers comparable quality to larger models",
        help="Text to evaluate the model with",
    ),
    eval_voice: str = typer.Option("af_heart", help="Voice to evaluate the model with"),
) -> None:
    """
    Run quantization trials on individual nodes to measure their impact on model quality.
    Results are saved to a CSV file with columns: name, op_type, mel_distance
    """
    print("Loading model...")
    model = onnx.load(onnx_path)

    # Create list of nodes to try quantizing
    selection = QuantizationSelection(
        ops=[
            x.op
            for x in QUANT_RULES
            if x.min_activations == "int8" or x.min_weights == "int8"
        ],
        min_params=512,
    )
    nodes_to_reduce = [
        NodeToReduce(op_type=node.op_type, name=node.name)
        for node in model.graph.node
        if selection.matches(node, model.graph)
    ]

    print(f"Found {len(nodes_to_reduce)} nodes to evaluate")

    print("Running float16 trials...")
    run_float16_trials(
        model_path=onnx_path,
        selections=nodes_to_reduce,
        output_file=Path(output_path),
        test_text=eval_text,
        test_voice=eval_voice,
    )
    print("Running quantization trials...")
    data_reader = CsvCalibrationDataReader(calibration_data, samples)
    run_quantization_trials(
        model_path=onnx_path,
        calibration_data_reader=data_reader,
        selections=nodes_to_reduce,
        output_file=Path(output_path),
        test_text=eval_text,
        test_voice=eval_voice,
    )

    print(f"Trials complete! Results saved to {output_path}")


@app.command()
def estimate_size(
    onnx_path: str = typer.Option(
        "kokoro.onnx", help="Path to the ONNX model to analyze"
    ),
    trial_results: str = typer.Option(
        "quantization_trials.csv", help="Path to trial results CSV"
    ),
    quant_threshold: float = typer.Option(
        0.25, help="Maximum acceptable mel distance for quantization"
    ),
    fp16_threshold: float = typer.Option(
        0.25, help="Maximum acceptable mel distance for fp16 conversion"
    ),
) -> None:
    """
    Estimate model size after quantization/casting based on trial results and thresholds.
    """
    print("Loading model...")
    model = onnx.load(onnx_path)

    original_size = Path(onnx_path).stat().st_size / (1024 * 1024)  # Convert to MB
    print(f"Original model size: {original_size:.2f} MB")

    print(f"\nAnalyzing with thresholds:")
    print(f"  Quantization: {quant_threshold}")
    print(f"  FP16 casting: {fp16_threshold}")

    # Get quantization specifications based on thresholds
    specs = select_node_datatypes(
        quant_threshold=quant_threshold,
        fp16_threshold=fp16_threshold,
        trial_csv_path=trial_results,
    )

    # Count nodes by data type
    quant_counts = {
        "weights": {"float32": 0, "float16": 0, "int8": 0},
        "activations": {"float32": 0, "float16": 0, "int8": 0},
    }

    for spec in specs:
        quant_counts["weights"][spec.weights_type.value] += 1
        quant_counts["activations"][spec.activations_type.value] += 1

    print("\nNode data type distribution:")
    print("Weights:")
    for dtype, count in quant_counts["weights"].items():
        print(f"  {dtype}: {count}")
    print("Activations:")
    for dtype, count in quant_counts["activations"].items():
        print(f"  {dtype}: {count}")

    # Estimate size
    estimated_size = estimate_quantized_size(model, specs)
    estimated_mb = estimated_size / (1024 * 1024)  # Convert to MB

    print(f"\nEstimated model size: {estimated_mb:.2f} MB")
    print(f"Estimated reduction: {(1 - estimated_mb / original_size) * 100:.1f}%")


@app.command()
def export_optimized(
    onnx_path: str = typer.Option(
        "kokoro.onnx", help="Path to the ONNX model to optimize"
    ),
    output_path: str = typer.Option(
        "kokoro_optimized.onnx", help="Path to save the optimized model"
    ),
    trial_results: str = typer.Option(
        "quantization_trials.csv", help="Path to trial results CSV"
    ),
    quant_threshold: float = typer.Option(
        0.25, help="Maximum acceptable mel distance for quantization"
    ),
    fp16_threshold: float = typer.Option(
        0.25, help="Maximum acceptable mel distance for fp16 conversion"
    ),
    samples: Optional[int] = typer.Option(
        None, help="Number of samples to use for calibration"
    ),
    eval_text: str = typer.Option(
        "Despite its lightweight architecture, it delivers comparable quality to larger models",
        help="Text to evaluate the model with",
    ),
    eval_voice: str = typer.Option("af_heart", help="Voice to evaluate the model with"),
) -> None:
    """
    Export an optimized model using both FP16 and INT8 quantization based on trial results.
    """
    print("Loading model...")
    model = onnx.load(onnx_path)
    original_model = onnx.load(onnx_path)
    original_size = Path(onnx_path).stat().st_size / (1024 * 1024)
    print(f"Original model size: {original_size:.2f} MB")

    # Get test inputs and original outputs for comparison
    vocab = load_vocab()
    inputs = get_onnx_inputs(eval_voice, eval_text, vocab)
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    original_outputs = sess.run(None, inputs)

    # Get node specifications based on thresholds
    specs = select_node_datatypes(
        quant_threshold=quant_threshold,
        fp16_threshold=fp16_threshold,
        trial_csv_path=trial_results,
    )

    # Separate nodes based on quantization support
    qdq_nodes = []
    qop_nodes = []
    fp16_nodes = []

    for spec in specs:
        if spec.weights_type == DataType.INT8 or spec.activations_type == DataType.INT8:
            if spec.q_operator:  # Use the q_operator flag from the rules
                qop_nodes.append(spec.name)
            else:
                qdq_nodes.append(spec.name)
        elif (
            spec.weights_type == DataType.FLOAT16
            or spec.activations_type == DataType.FLOAT16
        ):
            fp16_nodes.append(spec.name)

    print(f"\nOptimizing model:")
    print(f"FP16 nodes: {len(fp16_nodes)}")
    print(f"QOperator nodes: {len(qop_nodes)}")
    print(f"QDQ nodes: {len(qdq_nodes)}")

    # First convert applicable nodes to FP16
    if fp16_nodes:
        print("\nConverting nodes to FP16...")
        node_block_list = [
            node.name for node in model.graph.node if node.name not in fp16_nodes
        ]
        model = convert_float_to_float16(
            model,
            keep_io_types=True,
            node_block_list=node_block_list,
        )

    # Then quantize nodes that support QOperator format
    if qop_nodes:
        print("\nQuantizing QOperator-compatible nodes...")
        temp_path = output_path + ".temp"
        onnx.save(model, temp_path)

        data_reader = CsvCalibrationDataReader("data/quant-calibration.csv", samples)
        quantize_static(
            model_input=temp_path,
            model_output=temp_path,  # Save to temp for next step
            calibration_data_reader=data_reader,
            quant_format=QuantFormat.QOperator,
            nodes_to_quantize=qop_nodes,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            calibrate_method=CalibrationMethod.MinMax,
        )
        model = onnx.load(temp_path)

    # Finally quantize nodes that only support QDQ
    if qdq_nodes:
        print("\nQuantizing QDQ-only nodes...")
        temp_path = output_path + ".temp"
        onnx.save(model, temp_path)

        data_reader = CsvCalibrationDataReader("data/quant-calibration.csv", samples)
        quantize_static(
            model_input=temp_path,
            model_output=output_path,
            calibration_data_reader=data_reader,
            quant_format=QuantFormat.QDQ,
            nodes_to_quantize=qdq_nodes,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            calibrate_method=CalibrationMethod.MinMax,
        )
        Path(temp_path).unlink()  # Clean up temp file
    else:
        # If no QDQ nodes, just save the current model
        onnx.save(model, output_path)

    diff_models(original_model, model)
    # Calculate final size and quality metrics
    final_size = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"\nFinal model size: {final_size:.2f} MB")
    print(f"Size reduction: {(1 - final_size / original_size) * 100:.1f}%")

    # Test optimized model
    print("\nTesting optimized model...")
    optimized_sess = ort.InferenceSession(
        output_path, providers=["CPUExecutionProvider"]
    )
    optimized_outputs = optimized_sess.run(None, inputs)

    # Compare outputs
    mse = mel_spectrogram_distance(original_outputs[0], optimized_outputs[0])
    sf.write("onnx_optimized.wav", optimized_outputs[0], 24000)
    print(f"Mel spectrogram distance from original: {mse}")
