import os
from collections import defaultdict
from enum import Enum
from typing import Dict, Tuple

import numpy as np
import onnx
import typer
from onnx import AttributeProto, TensorProto, numpy_helper
from rich import print
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from .cli import app


class CountBy(str, Enum):
    OP = "op"
    NAME = "name"
    INDIVIDUAL = "individual"


@app.command()
def count(
    onnx_path: str = typer.Option("kokoro.onnx", help="Path to the ONNX model file"),
    count_by: CountBy = typer.Option(
        CountBy.OP, help="Count by operation type, name prefix, or individual nodes"
    ),
    name_depth: int = typer.Option(
        2, help="Depth of the name prefix to count by (when count_by=name)"
    ),
    size: bool = typer.Option(
        False, help="Count parameter sizes instead of number of nodes"
    ),
    filter_op: str = typer.Option(
        None, help="Only count nodes/params with this operation type"
    ),
    filter_name: str = typer.Option(
        None, help="Only count nodes/params with names starting with this prefix"
    ),
    max_rows: int = typer.Option(
        100, help="Maximum number of rows to display in the output"
    ),
):
    """
    Analyzes an ONNX model, counting nodes or parameters by operation type or name prefix.
    """
    model = onnx.load(onnx_path)

    # Show file size when counting parameters
    if size:
        file_size_kb = os.path.getsize(onnx_path) / 1024
        print(f"ONNX File Size on Disk: {file_size_kb:.2f} KB")

    # Get grouping function based on count_by
    def get_group(name: str, op_type: str = None) -> str:
        if count_by == CountBy.OP:
            return op_type or "Unattached"
        elif count_by == CountBy.INDIVIDUAL:
            return f"{name} ({op_type or 'Unattached'})"
        else:
            split_name = name.split(".", name_depth)
            return (
                ".".join(split_name[:name_depth])
                if len(split_name) >= name_depth
                else name
            )

    # Filter function to check if a node/parameter should be counted
    def should_count(name: str, op_type: str = None) -> bool:
        if filter_op and (op_type != filter_op):
            return False
        if filter_name and not name.startswith(filter_name):
            return False
        return True

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
            if not should_count(initializer.name, input_to_op.get(initializer.name)):
                continue

            num_params = np.prod(initializer.dims)
            param_size_bytes = num_params * 4  # Assuming FP32

            group = get_group(initializer.name, input_to_op.get(initializer.name))
            sizes[group] += param_size_bytes
            counts[group] += num_params

        # Count embedded tensors in nodes
        for node in model.graph.node:
            if not should_count(node.name, node.op_type):
                continue

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

        # Create and configure table
        table = Table(
            title=f"Parameter Sizes by {'Operation Type' if count_by == CountBy.OP else 'Individual Node' if count_by == CountBy.INDIVIDUAL else f'Name Prefix (depth={name_depth})'}"
        )
        table.add_column("Group", style="cyan")
        table.add_column("Size (KB)", justify="right", style="green")
        table.add_column("Percentage", justify="right")
        table.add_column("Parameters", justify="right")

        # Add rows
        for group, size_bytes in sorted_items[:max_rows]:
            size_kb = size_bytes / 1024
            count = counts[group]
            percentage = (size_bytes / total) * 100
            table.add_row(
                group,
                f"{size_kb:,.2f}",
                f"{percentage:,.1f}%",
                f"{count:,}",
            )

        if len(sorted_items) > max_rows:
            remaining_size = sum(size for _, size in sorted_items[max_rows:])
            remaining_count = sum(counts[group] for group, _ in sorted_items[max_rows:])
            remaining_percentage = (remaining_size / total) * 100
            table.add_row(
                f"... and {len(sorted_items) - max_rows} more rows",
                f"{remaining_size / 1024:,.2f}",
                f"{remaining_percentage:.1f}%",
                f"{remaining_count:,}",
                style="dim",
            )

        table.add_row(
            "Total",
            f"{total / 1024:,.2f}",
            "100%",
            f"{sum(counts.values()):,}",
            style="bold",
        )

        # Create console with both terminal and markdown output
        console = Console()
        console.print(table)

    else:
        # Count nodes
        counts: Dict[str, int] = defaultdict(int)
        for node in model.graph.node:
            if not should_count(node.name, node.op_type):
                continue
            group = get_group(node.name, node.op_type)
            counts[group] += 1

        # Sort and prepare results
        sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        total = sum(count for _, count in sorted_items)

        # Create table for node counts
        table = Table(
            title=f"Node Counts by {'Operation Type' if count_by == CountBy.OP else 'Individual Node' if count_by == CountBy.INDIVIDUAL else f'Name Prefix (depth={name_depth})'}"
        )
        table.add_column("Group", style="cyan")
        table.add_column("Count", justify="right", style="green")
        table.add_column("Percentage", justify="right")

        # Add rows
        for group, count in sorted_items[:max_rows]:
            percentage = (count / total) * 100
            table.add_row(group, str(count), f"{percentage:.1f}%")

        if len(sorted_items) > max_rows:
            remaining_count = sum(count for _, count in sorted_items[max_rows:])
            remaining_percentage = (remaining_count / total) * 100
            table.add_row(
                f"... and {len(sorted_items) - max_rows} more rows",
                str(remaining_count),
                f"{remaining_percentage:.1f}%",
                style="dim",
            )

        table.add_row("Total", str(total), "100%", style="bold")

        # Create console with both terminal and markdown output
        console = Console()
        console.print(table)
