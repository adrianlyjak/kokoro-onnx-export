import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generator, Literal, Optional

import numpy as np
import onnx
import onnxruntime as ort
import torch
import typer
from huggingface_hub import hf_hub_download
from kokoro.pipeline import KPipeline
from neural_compressor import PostTrainingQuantConfig, quantization
from neural_compressor.config import AccuracyCriterion, TuningCriterion
from onnxconverter_common.auto_mixed_precision import auto_convert_mixed_precision
from onnxruntime.quantization import (
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.calibrate import CalibrationDataReader
from onnxruntime.quantization.shape_inference import quant_pre_process

# from rich import print
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from kokoro_onnx.util import load_vocab

from .cli import app
from .convert_float_to_float16 import convert_float_to_float16
from .util import mel_spectrogram_distance


@app.command()
def quantize_neural_compressor(
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
        None, help="Number of samples to use for calibration"
    ),
    trials: int = typer.Option(500, help="Number of trials to run for quantization"),
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

    model = onnx.load(model_path)
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
        return mel_spectrogram_distance(init_outputs, outputs, distance_type="L2")

    calibration_dataloader = NeuralCalibrationDataloader(
        calibration_data, num_samples=samples, vocab=vocab
    )

    quantized_model = quantization.fit(
        model,
        conf=PostTrainingQuantConfig(
            device="gpu" if torch.cuda.is_available() else "cpu",
            excluded_precisions=["bf16"],
            quant_format="QOperator",
            approach="static",
            tuning_criterion=TuningCriterion(
                timeout=0,
                max_trials=trials,
            ),
            accuracy_criterion=AccuracyCriterion(
                higher_is_better=False,
                criterion="absolute",
                tolerable_loss=1.3,
            ),
        ),
        calib_dataloader=calibration_dataloader,
        eval_func=eval_func,
    )
    quantized_model.save(output_path)


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

    # Create calibration data reader
    data_reader = CsvCalibrationDataReader(calibration_rows, load_vocab())

    print("Starting calibration and quantization...")

    model = onnx.load(model_path)

    ops_to_include = [
        # "Add",
        # "Mul",
        # "Div",
        # "Slice",
        # "Reshape",
        # "Unsqueeze",
        # "Gather",
        # "ReduceMean",
        # "Pow",
        # "Transpose",
        # "MatMul",
        # "Shape",
        # "Div",
        "Conv",
        # "Sqrt",
        # "Concat",
        # "Gemm",
        # "Sub",
        # "Sin",
        # "Expand",
        # "LayerNormalization",
        # "LeakyRelu",
        # "Where",
        # "Cast",
        # "Tanh",
        # "Equal",
        # "Softmax",
        # "Range",
        # "ConstantOfShape",
        # "ScatterND",
        # "ConvTranspose",
        # "LSTM",
        # "Resize",
        # "Squeeze",
        # "Greater",
        # "SplitToSequence",
        # "Pad",
        # "SequenceEmpty",
        # "ReduceMax",
        # "TopK",
        # "ScatterElements",
        # "Not",
        # "Sigmoid",
        # "ReduceSum",
        # "Round",
        # "Clip",
        # "Loop",
        # "ConcatFromSequence",
        # "Floor",
        # "RandomUniformLike",
        # "CumSum",
        # "RandomNormalLike",
        # "Less",
        # "And",
        # "Atan",
        # "Exp",
        # "Cos",
    ]
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
        model_input=model_path,
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
        rows: list[dict[str, str]],
        vocab: Optional[dict[str, int]] = None,
    ):
        self.vocab = vocab or load_vocab()
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
    model_path: str = typer.Option(
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

    print(f"Loading model from {model_path}")
    model = onnx.load(model_path)

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


@dataclass
class QuantizationSelection:
    prefix: Optional[str] = None
    regex: Optional[re.Pattern] = None
    ops: Optional[list[str]] = None
    min_params: Optional[int] = None

    def matches(self, node: onnx.NodeProto) -> bool:
        if self.prefix is not None and not node.name.startswith(self.prefix):
            return False
        if self.regex is not None and not self.regex.match(node.name):
            return False
        if self.ops is not None and node.op_type not in self.ops:
            return False
            # Handle other nodes with tensor attributes
        params = 0
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.TENSOR:
                tensor = attr.t
                params += np.prod(tensor.dims)
            elif attr.type == onnx.AttributeProto.TENSORS:
                for tensor in attr.tensors:
                    params += np.prod(tensor.dims)
        if self.min_params is not None and params < self.min_params:
            return False
        return True


@dataclass
class OpQuantizationRule:
    op: str
    min_weights: Literal["int8", "float16", "float32"]
    min_activations: Literal["int8", "float16", "float32"]


QUANT_RULES = [
    OpQuantizationRule("Conv", "int8", "int8"),
    OpQuantizationRule("LSTM", "float16", "float16"),
    OpQuantizationRule("Gemm", "int8", "int8"),
    OpQuantizationRule("MatMul", "int8", "int8"),
    OpQuantizationRule("ConvTranspose", "int8", "float16"),
    OpQuantizationRule("Mul", "int8", "int8"),
    OpQuantizationRule("LayerNormalization", "float16", "float16"),
    OpQuantizationRule("Add", "int8", "int8"),
    OpQuantizationRule("InstanceNormalization", "float16", "float16"),
]
