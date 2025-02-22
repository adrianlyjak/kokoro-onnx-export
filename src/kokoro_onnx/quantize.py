import csv
import os
import re
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Literal, Optional

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationMethod,
    QuantType,
)
from onnxruntime.quantization.calibrate import (
    CalibrationDataReader,
    CalibrationMethod,
    TensorsData,
    create_calibrator,
)
from onnxruntime.quantization.onnx_quantizer import ONNXQuantizer
from onnxruntime.quantization.quant_utils import (
    QuantizationMode,
    QuantType,
)
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from kokoro_onnx.convert_float_to_float16 import convert_float_to_float16
from kokoro_onnx.util import (
    build_initializer_lookup,
    count_params_with_initializers,
    count_params_with_initializers_lookup,
    get_onnx_inputs,
    load_vocab,
    mel_spectrogram_distance,
)


def extract_tensors_range(
    model: onnx.ModelProto,
    calibration_data_reader: CalibrationDataReader,
    op_types_to_quantize=[],
    calibrate_method=CalibrationMethod.MinMax,
) -> TensorsData:
    with tempfile.TemporaryDirectory(prefix="ort.quant.") as quant_tmp_dir:
        output_path = str(Path(quant_tmp_dir) / "model_input.onnx")
        onnx.save_model(model, output_path)

        calibrator = create_calibrator(
            Path(output_path),
            op_types_to_quantize,
            augmented_model_path=Path(quant_tmp_dir)
            .joinpath("augmented_model.onnx")
            .as_posix(),
            calibrate_method=calibrate_method,
            use_external_data_format=False,
            extra_options={},
        )

        calibrator.collect_data(calibration_data_reader)
        return calibrator.compute_data()


def quantize_model_dynamic(
    model: onnx.ModelProto,
    nodes_to_quantize: list[str],
    weight_type: QuantType = QuantType.QUInt8,
) -> onnx.ModelProto:
    types = []
    for node in model.graph.node:
        if node.name in nodes_to_quantize:
            types.append(node.op_type)
    types = list(set(types))

    quantizer = ONNXQuantizer(
        model,
        per_channel=False,
        reduce_range=False,
        mode=QuantizationMode.QLinearOps,
        static=False,  # static
        weight_qType=weight_type,
        activation_qType=weight_type,
        tensors_range=None,
        nodes_to_quantize=nodes_to_quantize,
        nodes_to_exclude=[],
        op_types_to_quantize=types,
        extra_options=None,
    )

    quantized = quantizer.quantize_model()
    return quantized


def quantize_model_static(
    model: onnx.ModelProto,
    tensors_range: TensorsData,
    nodes_to_quantize: list[str],
    weight_type: QuantType = QuantType.QInt8,
    activation_type: QuantType = QuantType.QInt8,
) -> onnx.ModelProto:
    types = []
    for node in model.graph.node:
        if node.name in nodes_to_quantize:
            types.append(node.op_type)
    types = list(set(types))

    quantizer = ONNXQuantizer(
        model,
        per_channel=False,
        reduce_range=False,
        mode=QuantizationMode.QLinearOps,
        static=True,  # static
        weight_qType=weight_type,
        activation_qType=activation_type,
        tensors_range=tensors_range,
        nodes_to_quantize=nodes_to_quantize,
        nodes_to_exclude=[],
        op_types_to_quantize=types,
        extra_options=None,
    )

    quantized = quantizer.quantize_model()
    return quantized


def run_quantization_trials(
    model_path: str,
    calibration_data_reader: CalibrationDataReader,
    selections: list["NodeToReduce"],
    output_file: Path,
    test_text: str,
    test_voice: str,
    static: bool = False,
):
    model = onnx.load(model_path)

    initializers = build_initializer_lookup(model.graph)

    # Load existing results to skip completed trials
    completed_trials = set()
    if output_file.exists():
        with open(output_file, "r") as f:
            # Skip header
            next(f)
            for line in f:
                trial_name = line.split(",")[0]
                trial_type = line.split(",")[1]
                if trial_type == "quant":
                    completed_trials.add(trial_name)

    # Create file with header if it doesn't exist
    if not output_file.exists():
        with open(output_file, "w") as f:
            f.write("name,type,op_type,mel_distance\n")

    vocab = load_vocab()
    inputs = get_onnx_inputs(test_voice, test_text, vocab)

    init_session = ort.InferenceSession(model.SerializeToString())
    init_outputs = init_session.run(None, inputs)[0]
    del init_session

    print("Extracting tensor ranges...")
    op_types_to_quantize = [
        rule.op
        for rule in QUANT_RULES
        if rule.min_activations == "int8" or rule.min_weights == "int8"
    ]

    tensors_range = (
        extract_tensors_range(model, calibration_data_reader, op_types_to_quantize)
        if static
        else TensorsData(CalibrationMethod.MinMax, {})
    )

    # Count remaining trials
    remaining_trials = [
        node for node in selections if node.name not in completed_trials
    ]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task(
            "Running quantization trials...", total=len(remaining_trials)
        )

        for node in remaining_trials:
            progress.update(task, description=f"Quantizing {node.name}")

            should_quantize = False
            q_rule = None
            for rule in QUANT_RULES:
                if rule.op == node.op_type:
                    q_rule = rule
                    break
            should_quantize = (
                q_rule.min_activations == "int8" or q_rule.min_weights == "int8"
            )

            if not should_quantize:
                with open(output_file, "a") as f:
                    f.write(
                        f"{node.name},quant,{node.op_type},{node.params},{node.size},N/A\n"
                    )
                progress.advance(task)
                continue

            nodes_to_quantize = [node.name]

            model = onnx.load(model_path)
            if static:
                quantized = quantize_model_static(
                    model,
                    tensors_range,
                    nodes_to_quantize,
                    weight_type=QuantType.QInt8
                    if q_rule.min_weights == "int8"
                    else QuantType.QInt16,
                )
            else:
                quantized = quantize_model_dynamic(
                    model, nodes_to_quantize, weight_type=QuantType.QUInt8
                )

            quantized_session = ort.InferenceSession(quantized.SerializeToString())
            quantized_outputs = quantized_session.run(None, inputs)[0]
            del quantized_session

            mel_dist = mel_spectrogram_distance(
                init_outputs, quantized_outputs, distance_type="L2"
            )

            # Append result to CSV
            with open(output_file, "a") as f:
                f.write(
                    f"{node.name},quant,{node.op_type},{node.params},{node.size},{mel_dist}\n"
                )

            progress.advance(task)


def run_float16_trials(
    model_path: str,
    selections: list["NodeToReduce"],
    output_file: Path,
    test_text: str,
    test_voice: str,
):
    model = onnx.load(model_path)

    # Load existing results to skip completed trials
    completed_trials = set()
    if output_file.exists():
        with open(output_file, "r") as f:
            # Skip header
            next(f)
            for line in f:
                trial_name = line.split(",")[0]
                trial_type = line.split(",")[1]
                if trial_type == "cast":
                    completed_trials.add(trial_name)

    # Create file with header if it doesn't exist
    if not output_file.exists():
        with open(output_file, "w") as f:
            f.write("name,type,op_type,params,size,mel_distance\n")

    vocab = load_vocab()
    inputs = get_onnx_inputs(test_voice, test_text, vocab)

    init_session = ort.InferenceSession(model.SerializeToString())
    init_outputs = init_session.run(None, inputs)[0]
    del init_session

    # Count remaining trials
    remaining_trials = [
        node for node in selections if node.name not in completed_trials
    ]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task(
            "Running float16 trials...", total=len(remaining_trials)
        )

        for node in remaining_trials:
            progress.update(task, description=f"Converting {node.name} to float16")

            should_convert = False
            for rule in QUANT_RULES:
                if rule.op == node.op_type:
                    should_convert = (
                        rule.min_activations != "float32"
                        or rule.min_weights != "float32"
                    )
                    break

            if not should_convert:
                with open(output_file, "a") as f:
                    f.write(
                        f"{node.name},cast,{node.op_type},{node.params},{node.size},N/A\n"
                    )
                progress.advance(task)
                continue

            model = onnx.load(model_path)
            node_block_list = [n.name for n in model.graph.node if n.name != node.name]
            converted = convert_float_to_float16(
                model,
                keep_io_types=True,
                node_block_list=node_block_list,
                warn_truncate=False,
            )

            converted_session = ort.InferenceSession(converted.SerializeToString())
            converted_outputs = converted_session.run(None, inputs)[0]
            del converted_session

            import soundfile

            soundfile.write("fp16.wav", converted_outputs, samplerate=24000)

            mel_dist = mel_spectrogram_distance(
                init_outputs, converted_outputs, distance_type="L2"
            )

            # Append result to CSV
            with open(output_file, "a") as f:
                f.write(
                    f"{node.name},cast,{node.op_type},{node.params},{node.size},{mel_dist}\n"
                )

            progress.advance(task)


@dataclass
class NodeToReduce:
    op_type: str
    name: str
    params: int
    size: int


@dataclass
class QuantizationSelection:
    prefix: Optional[str] = None
    regex: Optional[re.Pattern] = None
    ops: Optional[list[str]] = None
    min_params: Optional[int] = None

    def matches(self, node: onnx.NodeProto, graph: onnx.GraphProto) -> bool:
        if self.prefix is not None and not node.name.startswith(self.prefix):
            return False
        if self.regex is not None and not self.regex.match(node.name):
            return False
        if self.ops is not None and node.op_type not in self.ops:
            return False

        if self.min_params is not None:
            params, _ = count_params_with_initializers(node, graph)
            if params < self.min_params:
                return False
        return True


@dataclass
class OpQuantizationRule:
    op: str
    q_operator: bool
    min_weights: Literal["int8", "float16", "float32"]
    min_activations: Literal["int8", "float16", "float32"]


QUANT_RULES = [
    OpQuantizationRule("Conv", True, "int8", "int8"),
    OpQuantizationRule("LSTM", False, "int8", "int8"),
    OpQuantizationRule("Gemm", True, "int8", "int8"),
    OpQuantizationRule("MatMul", True, "int8", "int8"),
    OpQuantizationRule("ConvTranspose", False, "float16", "float16"),
    OpQuantizationRule("Mul", False, "int8", "int8"),
    OpQuantizationRule("LayerNormalization", False, "int8", "int8"),
    OpQuantizationRule("Add", True, "int8", "int8"),
    OpQuantizationRule("InstanceNormalization", False, "int8", "int8"),
]


class DataType(Enum):
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    INT8 = "int8"


@dataclass
class NodeQuantSpec:
    name: str
    op_type: str
    weights_type: DataType
    activations_type: DataType
    q_operator: bool

    def supports_quantization(self) -> bool:
        return (
            self.weights_type == DataType.INT8 or self.activations_type == DataType.INT8
        )

    def supports_float16(self) -> bool:
        return (
            (
                self.weights_type == DataType.FLOAT16
                or self.activations_type == DataType.FLOAT16
            )
            or self.supports_quantization()
        )  # fallback to float16 if otherwise excluded from quantization


def select_node_datatypes(
    quant_threshold: float,
    fp16_threshold: float,
    trial_csv_path: str,
    quantizable_ops: Optional[list[str]] = None,
) -> List[NodeQuantSpec]:
    # Create mapping of rules by op type for easy lookup
    rules_by_op = {rule.op: rule for rule in QUANT_RULES}

    # Read trial results
    results: Dict[str, Dict[str, float]] = {}
    if not os.path.exists(trial_csv_path):
        return []
    with open(trial_csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            trial_type = row["type"]
            mel_dist = row["mel_distance"]
            if mel_dist == "N/A":
                continue

            if name not in results:
                results[name] = {}
            results[name][trial_type] = float(mel_dist)

    specs: List[NodeQuantSpec] = []

    for name, trials in results.items():
        # Get the op_type from any row with matching name
        with open(trial_csv_path, "r") as f:
            reader = csv.DictReader(f)
            op_type = next(row["op_type"] for row in reader if row["name"] == name)

        rule = rules_by_op.get(op_type)
        if not rule:
            continue

        # Default to FP32
        weights_type = DataType.FLOAT32
        activations_type = DataType.FLOAT32

        # Check quantization results if available
        if (
            "quant" in trials
            and trials["quant"] < quant_threshold
            and (quantizable_ops is None or op_type in quantizable_ops)
        ):
            # Respect minimum levels from rules
            if rule.min_weights == "int8":
                weights_type = DataType.INT8
            elif rule.min_weights == "float16":
                weights_type = DataType.FLOAT16

            if rule.min_activations == "int8":
                activations_type = DataType.INT8
            elif rule.min_activations == "float16":
                activations_type = DataType.FLOAT16

        # Check FP16 results if still at FP32
        elif "cast" in trials and trials["cast"] < fp16_threshold:
            if weights_type == DataType.FLOAT32 and rule.min_weights != "float32":
                weights_type = DataType.FLOAT16
            if (
                activations_type == DataType.FLOAT32
                and rule.min_activations != "float32"
            ):
                activations_type = DataType.FLOAT16

        specs.append(
            NodeQuantSpec(
                name=name,
                op_type=op_type,
                weights_type=weights_type,
                activations_type=activations_type,
                q_operator=rule.q_operator,
            )
        )

    return specs


def estimate_quantized_size(
    model: onnx.ModelProto, quant_specs: List[NodeQuantSpec]
) -> int:
    """
    Estimates the model size in bytes after quantization/casting according to specs
    """
    specs_by_name = {spec.name: spec for spec in quant_specs}
    total_size = 0

    # Helper to calculate size for a data type
    def get_bytes_per_element(dtype: DataType) -> int:
        if dtype == DataType.FLOAT16:
            return 2
        elif dtype == DataType.INT8:
            return 1
        return 4

    # Process initializers (weights)
    for initializer in model.graph.initializer:
        # Find which node this initializer belongs to
        connected_node = None
        for node in model.graph.node:
            if initializer.name in node.input:
                connected_node = node
                break

        if connected_node and connected_node.name in specs_by_name:
            spec = specs_by_name[connected_node.name]
            bytes_per_element = get_bytes_per_element(spec.weights_type)
        else:
            bytes_per_element = 4  # Default to float32

        size = np.prod(initializer.dims) * bytes_per_element
        total_size += size

    return total_size


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
        self.progress = None

    def get_next(self):
        if self.progress is None:
            self.progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                auto_refresh=False,
            )
            self.task = self.progress.add_task("Calibrating", total=len(self.rows))
            self.progress.start()
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
        if self.progress is not None:
            self.progress.reset(self.task)
