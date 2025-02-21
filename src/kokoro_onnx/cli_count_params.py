import os
from collections import defaultdict
from enum import Enum
from typing import Dict

import numpy as np
import onnx
import typer
from onnx import AttributeProto
from rich import print
from rich.console import Console
from rich.table import Table

from kokoro_onnx.util import (
    TENSOR_NAME_TO_TYPE,
    TENSOR_TYPE_TO_NAME,
    TENSOR_TYPE_TO_SIZE,
    count_embedded_tensor_params,
)

from .cli import app


class CountBy(str, Enum):
    OP = "op"
    NAME = "name"
    INDIVIDUAL = "individual"
    DTYPE = "dtype"
    OP_DTYPE = "op+dtype"
    NAME_DTYPE = "name+dtype"


def get_table_title(count_by: CountBy, name_depth: int = None) -> str:
    """Generate table title based on counting method."""
    titles = {
        CountBy.OP: "Operation Type",
        CountBy.INDIVIDUAL: "Individual Node",
        CountBy.DTYPE: "Data Type",
        CountBy.OP_DTYPE: "Operation Type + Data Type",
        CountBy.NAME_DTYPE: "Name Prefix + Data Type",
        CountBy.NAME: f"Name Prefix (depth={name_depth})",
    }
    return titles[count_by]


def format_name_prefix(name: str, depth: int) -> str:
    """Format name prefix based on specified depth."""
    split_name = name.split(".", depth)
    return ".".join(split_name[:depth]) if len(split_name) >= depth else name


@app.command()
def count(
    onnx_path: str = typer.Option("kokoro.onnx", help="Path to the ONNX model file"),
    count_by: CountBy = typer.Option(
        CountBy.OP,
        help="Count by: operation type, name prefix, individual nodes, data type, or combinations (op+dtype, name+dtype)",
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
    filter_dtype: str = typer.Option(
        None, help="Only count nodes/params with this data type (e.g. FP32, INT8)"
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

    def get_type_name(data_type: int) -> str:
        """Get human readable name for a data type."""
        return TENSOR_TYPE_TO_NAME.get(data_type, "Unknown")

    def get_type_from_name(type_name: str) -> int:
        """Get ONNX tensor type from human readable name."""
        return TENSOR_NAME_TO_TYPE.get(type_name.upper())

    def get_group(name: str, op_type: str = None, data_type: int = None) -> str:
        if count_by == CountBy.OP:
            return op_type or "Unattached"
        elif count_by == CountBy.INDIVIDUAL:
            return f"{name} ({op_type or 'Unattached'})"
        elif count_by == CountBy.DTYPE:
            return get_type_name(data_type)
        elif count_by == CountBy.OP_DTYPE:
            type_str = get_type_name(data_type)
            return f"{op_type or 'Unattached'} ({type_str})"
        elif count_by == CountBy.NAME_DTYPE:
            type_str = get_type_name(data_type)
            name_prefix = format_name_prefix(name, name_depth)
            return f"{name_prefix} ({type_str})"
        else:  # CountBy.NAME
            return format_name_prefix(name, name_depth)

    def should_count(name: str, op_type: str = None, data_type: int = None) -> bool:
        if filter_op and (op_type != filter_op):
            return False
        if filter_name and not name.startswith(filter_name):
            return False
        if filter_dtype:
            filter_type = get_type_from_name(filter_dtype)
            if filter_type is None:
                print(f"Warning: Unknown data type filter '{filter_dtype}'")
                return False
            if data_type != filter_type:
                return False
        return True

    def get_tensor_element_size(tensor_type: int) -> int:
        """Get the size in bytes for a given ONNX tensor type."""
        return TENSOR_TYPE_TO_SIZE.get(
            tensor_type, 4
        )  # Default to 4 bytes if type unknown

    # Count either nodes or parameters
    if size:
        # Map inputs to op types and data types for parameter counting
        input_to_op = {}
        input_to_dtype = {}
        for node in model.graph.node:
            for input_name in node.input:
                input_to_op[input_name] = node.op_type

        # Track data types from initializers
        for initializer in model.graph.initializer:
            input_to_dtype[initializer.name] = initializer.data_type

        # Count parameter sizes and counts
        sizes: Dict[str, int] = defaultdict(int)
        counts: Dict[str, int] = defaultdict(int)

        # Count initializers
        for initializer in model.graph.initializer:
            if not should_count(
                initializer.name,
                input_to_op.get(initializer.name),
                initializer.data_type,
            ):
                continue

            num_params = np.prod(initializer.dims)
            element_size = get_tensor_element_size(initializer.data_type)
            param_size_bytes = num_params * element_size

            group = get_group(
                initializer.name,
                input_to_op.get(initializer.name),
                initializer.data_type,
            )
            sizes[group] += param_size_bytes
            counts[group] += num_params

        # Count embedded tensors in nodes
        for node in model.graph.node:
            # Find smallest data type for the node
            node_dtypes = []

            # Check Constant nodes
            if node.op_type == "Constant":
                for attr in node.attribute:
                    if attr.name == "value" and attr.t:
                        node_dtypes.append(attr.t.data_type)

            # Check tensor attributes
            for attr in node.attribute:
                if attr.type == AttributeProto.TENSOR:
                    node_dtypes.append(attr.t.data_type)
                elif attr.type == AttributeProto.TENSORS:
                    node_dtypes.extend(t.data_type for t in attr.tensors)

            # Choose smallest data type (if any found)
            smallest_dtype = None
            if node_dtypes:
                smallest_dtype = min(node_dtypes, key=get_tensor_element_size)

            if not should_count(node.name, node.op_type, smallest_dtype):
                continue

            group = get_group(node.name, node.op_type, smallest_dtype)

            params, size = count_embedded_tensor_params(node)
            sizes[group] += size
            counts[group] += params

        # Sort and prepare results
        sorted_items = sorted(sizes.items(), key=lambda x: x[1], reverse=True)
        total = sum(size for _, size in sorted_items)

        # Create and configure table
        table = Table(
            title=f"{'Parameter Sizes' if size else 'Node Counts'} by {get_table_title(count_by, name_depth)}"
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
            if not should_count(
                node.name, node.op_type, None
            ):  # Initially no data type for node
                continue
            group = get_group(node.name, node.op_type)
            counts[group] += 1

        # Sort and prepare results
        sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        total = sum(count for _, count in sorted_items)

        # Create table for node counts
        table = Table(
            title=f"Node Counts by {
                'Operation Type'
                if count_by == CountBy.OP
                else 'Individual Node'
                if count_by == CountBy.INDIVIDUAL
                else 'Data Type'
                if count_by == CountBy.DTYPE
                else 'Operation Type + Data Type'
                if count_by == CountBy.OP_DTYPE
                else 'Name Prefix + Data Type'
                if count_by == CountBy.NAME_DTYPE
                else f'Name Prefix (depth={name_depth})'
            }"
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
