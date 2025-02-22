import re
from pathlib import Path
from typing import Optional

import onnx
import onnxruntime as ort
import soundfile as sf
import typer
from onnxruntime.quantization import (
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)

from kokoro_onnx.quantize import (
    QUANT_RULES,
    CsvCalibrationDataReader,
    NodeToReduce,
    QuantizationSelection,
    estimate_quantized_size,
    run_float16_trials,
    run_quantization_trials,
    select_node_datatypes,
)
from kokoro_onnx.util import execution_providers, load_vocab

from .cli import app
from .convert_float_to_float16 import convert_float_to_float16
from .util import (
    build_initializer_lookup,
    count_params_with_initializers_lookup,
    get_onnx_inputs,
    mel_spectrogram_distance,
)


@app.command()
def trial_quantization(
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
    static: bool = typer.Option(False, help="Whether to use static quantization"),
    min_params: int = typer.Option(
        256, help="Minimum number of parameters to quantize"
    ),
) -> None:
    """
    Run quantization trials on individual nodes to measure their impact on model quality.
    Results are saved to a CSV file with columns: name, op_type, mel_distance, params, size
    """
    print("Loading model...")
    model = onnx.load(onnx_path)

    # Create list of nodes to try quantizing
    selection = QuantizationSelection(
        ops=[x.op for x in QUANT_RULES],
        min_params=min_params,
    )
    nodes_to_reduce = []
    initializers = build_initializer_lookup(model.graph)
    for node in model.graph.node:
        if selection.matches(node, model.graph):
            params, size = count_params_with_initializers_lookup(node, initializers)
            nodes_to_reduce.append(
                NodeToReduce(
                    op_type=node.op_type, name=node.name, params=params, size=size
                )
            )

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
        static=static,
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
    quant_static: bool = typer.Option(False, help="Whether to use static quantization"),
    quant_exclude: Optional[str] = typer.Option(None, help="regex of nodes to exclude"),
    eval_voice: str = typer.Option("af_heart", help="Voice to evaluate the model with"),
    quant_type: str = typer.Option("QInt8", help="Quantization type to use"),
    quant_activation_type: Optional[str] = typer.Option(
        None, help="Quantization type to use for activations"
    ),
    quant_op_types: Optional[list[str]] = typer.Option(
        None, help="List of operation types to quantize"
    ),
) -> None:
    """
    Export an optimized model using both FP16 and INT8 quantization based on trial results.
    """
    print("Loading model...")
    quant_type_enum = QuantType.from_string(quant_type)
    quant_activation_type_enum = (
        QuantType.from_string(quant_activation_type)
        if quant_activation_type
        else quant_type_enum
    )
    model = onnx.load(onnx_path)
    original_model = onnx.load(onnx_path)
    original_size = Path(onnx_path).stat().st_size / (1024 * 1024)
    print(f"Original model size: {original_size:.2f} MB")

    # Get test inputs and original outputs for comparison
    vocab = load_vocab()
    inputs = get_onnx_inputs(eval_voice, eval_text, vocab)
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=execution_providers
    )
    original_outputs = sess.run(None, inputs)

    # Get node specifications based on thresholds
    specs = select_node_datatypes(
        quant_threshold=quant_threshold,
        fp16_threshold=fp16_threshold,
        trial_csv_path=trial_results,
        quantizable_ops=quant_op_types,
    )
    if not specs:
        print(
            "No nodes to quantize. Have you run trial-quantization yet to generate a quantization-trials.csv?"
        )
        return

    # Separate nodes based on quantization support
    q_nodes = []
    q_node_types: dict[str, str] = {}
    fp16_nodes = []

    quant_exclude_r = re.compile(quant_exclude) if quant_exclude else None
    for spec in specs:
        quantize = spec.supports_quantization()
        float16 = spec.supports_float16()
        if quantize and spec.op_type == "LSTM" and quant_static:
            # no LSTM in QOperator format :(
            quantize = False
        if quant_exclude_r and quant_exclude_r.search(spec.name):
            quantize = False
        if quantize:
            q_nodes.append(spec.name)
            q_node_types[spec.name] = spec.op_type
            if (
                spec.op_type == "Gemm"
            ):  # for some reason these nodes get munged up before they can get targeted
                q_nodes.append(spec.name + "_MatMul")
                q_node_types[spec.name + "_MatMul"] = "MatMul"
        elif float16 and fp16_threshold > 0:
            fp16_nodes.append(spec.name)

    print(f"\nOptimizing model:")
    print(f"FP16 nodes: {len(fp16_nodes)}")
    print(f"Q nodes: {len(q_nodes)}")

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
            warn_truncate=False,
        )

    # Then quantize nodes that support QOperator format
    if q_nodes:
        print("\nQuantizing nodes...")
        temp_path = output_path + ".temp"
        onnx.save(model, temp_path)

        if quant_static:
            data_reader = CsvCalibrationDataReader(
                "data/quant-calibration.csv", samples
            )
            quantize_static(
                model_input=temp_path,
                model_output=temp_path,
                calibration_data_reader=data_reader,
                quant_format=QuantFormat.QOperator,
                nodes_to_quantize=q_nodes,
                activation_type=quant_activation_type_enum,
                weight_type=quant_type_enum,
                calibrate_method=CalibrationMethod.MinMax,
            )
        else:
            if quant_type_enum == QuantType.QInt8:
                print(
                    "pre-Quantizing conv layers only to uint8, as they are not compatible with int8"
                )
                quantize_dynamic(
                    model_input=temp_path,
                    model_output=temp_path,
                    nodes_to_quantize=[q for q in q_nodes if q_node_types[q] == "Conv"],
                    weight_type=QuantType.QUInt8,
                )
            quantize_dynamic(
                model_input=temp_path,
                model_output=temp_path,
                nodes_to_quantize=[
                    q
                    for q in q_nodes
                    if q_node_types[q] != "Conv" or quant_type_enum != QuantType.QInt8
                ],
                weight_type=quant_type_enum,
            )
        model = onnx.load(temp_path)

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
