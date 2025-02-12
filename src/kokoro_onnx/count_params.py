import os
from collections import defaultdict

import numpy as np
import onnx
import typer
from rich import print

from .cli import app


@app.command()
def count_params(
    model_path: str = typer.Option("kokoro.onnx", help="Path to the ONNX model file"),
    depth: int = typer.Option(2, help="Depth of the prefix to count"),
):
    """
    Loads an ONNX model and summarizes its parameters by the second-level prefix.
    E.g.,
       - kmodel.predictor.F0.0.conv1.weight_v  ->  "kmodel.predictor"
       - kmodel.bert.encoder.albert_layer_groups.0...  ->  "kmodel.bert"
    """
    # Load model
    model = onnx.load(model_path)

    # Get file size in bytes
    file_size_bytes = os.path.getsize(model_path)
    file_size_kb = file_size_bytes / (1024)

    # Dictionary to accumulate parameter sizes per second-level prefix
    param_group_sizes = defaultdict(int)

    # Iterate over each initializer (tensor) in the graph
    for initializer in model.graph.initializer:
        # Count number of elements in tensor
        num_params = np.prod(initializer.dims)

        # Here we assume FP32 for size calculation (4 bytes per parameter).
        # Adjust if your model uses different datatypes (e.g., int8, FP16, BF16, etc.).
        param_size_bytes = num_params * 4

        # Parse out the prefix up to the second dot
        # e.g. "kmodel.predictor" from "kmodel.predictor.F0.0.conv1.weight_v"
        # Split only up to 2 dots to avoid splitting the entire string
        split_name = initializer.name.split(".", depth)

        if len(split_name) >= depth:
            prefix = ".".join(split_name[:depth])
        else:
            # If for some reason there isn't a second segment,
            # just use the entire name as the group
            prefix = initializer.name

        # Accumulate total bytes for this prefix
        param_group_sizes[prefix] += param_size_bytes

    # Print overall file size
    print(f"ONNX File Size on Disk: {file_size_kb:.2f} KB")
    print(f"Parameter Group Sizes (by {depth}-level prefix):")

    # Sort by descending total parameter size
    sorted_groups = sorted(param_group_sizes.items(), key=lambda x: x[1], reverse=True)

    for group, size_bytes in sorted_groups:
        print(f"  {group:40s} {size_bytes / (1024):.2f} KB")
    total_size_kb = sum(size_bytes / (1024) for group, size_bytes in sorted_groups)
    print(f"Total parameter size: {total_size_kb:.2f} KB")


@app.command()
def count_node_types(
    model_path: str = typer.Option("kokoro.onnx", help="Path to the ONNX model file"),
):
    """
    Loads an ONNX model and summarizes its nodes by type.
    """
    model = onnx.load(model_path)

    # Dictionary to count node types
    node_type_counts = defaultdict(int)

    # Count each node type
    for node in model.graph.node:
        node_type_counts[node.op_type] += 1

    # Print summary
    print("Node Types Summary:")

    # Sort by count in descending order
    sorted_types = sorted(node_type_counts.items(), key=lambda x: x[1], reverse=True)

    total_nodes = sum(count for _, count in sorted_types)

    # Print each type with count and percentage
    for node_type, count in sorted_types:
        percentage = (count / total_nodes) * 100
        print(f"  {node_type:20s} {count:4d} ({percentage:5.1f}%)")

    print(f"\nTotal number of nodes: {total_nodes}")
