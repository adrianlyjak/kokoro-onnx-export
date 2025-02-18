import os
from collections import defaultdict
from enum import Enum
from typing import Dict, Tuple

import numpy as np
import onnx
import typer
from onnx import AttributeProto, TensorProto, numpy_helper
from rich import print

from .cli import app


class CountBy(str, Enum):
    OP = "op"
    NAME = "name"


@app.command()
def count(
    model_path: str = typer.Option("kokoro.onnx", help="Path to the ONNX model file"),
    count_by: CountBy = typer.Option(
        CountBy.OP, help="Count by operation type or name prefix"
    ),
    name_depth: int = typer.Option(
        2, help="Depth of the name prefix to count by (when count_by=name)"
    ),
    size: bool = typer.Option(
        False, help="Count parameter sizes instead of number of nodes"
    ),
):
    """
    Analyzes an ONNX model, counting nodes or parameters by operation type or name prefix.
    """
    model = onnx.load(model_path)

    # Show file size when counting parameters
    if size:
        file_size_kb = os.path.getsize(model_path) / 1024
        print(f"ONNX File Size on Disk: {file_size_kb:.2f} KB")

    # Get grouping function based on count_by
    def get_group(name: str, op_type: str = None) -> str:
        if count_by == CountBy.OP:
            return op_type or "Unattached"
        else:
            split_name = name.split(".", name_depth)
            return (
                ".".join(split_name[:name_depth])
                if len(split_name) >= name_depth
                else name
            )

    # Count either nodes or parameters
    if size:
        # Map inputs to op types for parameter counting
        input_to_op = {}
        for node in model.graph.node:
            for input_name in node.input:
                input_to_op[input_name] = node.op_type

        # Count parameter sizes and counts
        sizes: Dict[str, int] = defaultdict(int)
        counts: Dict[str, int] = defaultdict(int)

        # Count initializers
        for initializer in model.graph.initializer:
            num_params = np.prod(initializer.dims)
            param_size_bytes = num_params * 4  # Assuming FP32

            group = get_group(initializer.name, input_to_op.get(initializer.name))
            sizes[group] += param_size_bytes
            counts[group] += num_params

        # Count embedded tensors in nodes
        for node in model.graph.node:
            group = get_group(node.name, node.op_type)

            # Handle Constant nodes which contain embedded tensor data
            if node.op_type == "Constant":
                for attr in node.attribute:
                    if attr.name == "value" and attr.t:
                        tensor = attr.t
                        num_params = np.prod(tensor.dims)
                        param_size_bytes = num_params * 4  # Assuming FP32
                        sizes[group] += param_size_bytes
                        counts[group] += num_params

            # Handle other nodes with tensor attributes
            for attr in node.attribute:
                if attr.type == AttributeProto.TENSOR:
                    tensor = attr.t
                    num_params = np.prod(tensor.dims)
                    param_size_bytes = num_params * 4
                    sizes[group] += param_size_bytes
                    counts[group] += num_params
                elif attr.type == AttributeProto.TENSORS:
                    for tensor in attr.tensors:
                        num_params = np.prod(tensor.dims)
                        param_size_bytes = num_params * 4
                        sizes[group] += param_size_bytes
                        counts[group] += num_params

        # Sort and prepare results
        sorted_items = sorted(sizes.items(), key=lambda x: x[1], reverse=True)
        total = sum(size for _, size in sorted_items)

        # Print results
        print(
            f"\nParameter Sizes by {'Operation Type' if count_by == CountBy.OP else f'Name Prefix (depth={name_depth})'}"
        )
        for group, size_bytes in sorted_items:
            size_kb = size_bytes / 1024
            count = counts[group]
            percentage = (size_bytes / total) * 100
            print(
                f"  {group:40s} {size_kb:8.2f} KB ({percentage:5.1f}%) - {count:,} params"
            )
        print(f"\nTotal parameter size: {total / 1024:.2f} KB")

    else:
        # Count nodes
        counts: Dict[str, int] = defaultdict(int)
        for node in model.graph.node:
            group = get_group(node.name, node.op_type)
            counts[group] += 1

        # Sort and prepare results
        sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        total = sum(count for _, count in sorted_items)

        # Print results
        print(
            f"Node Counts by {'Operation Type' if count_by == CountBy.OP else f'Name Prefix (depth={name_depth})'}"
        )
        for group, count in sorted_items:
            percentage = (count / total) * 100
            print(f"  {group:40s} {count:4d} ({percentage:5.1f}%)")
        print(f"\nTotal number of nodes: {total}")
