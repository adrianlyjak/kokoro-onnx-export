import itertools
import os
import re

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnx import TensorProto, helper, numpy_helper

from kokoro_onnx.convert_float_to_float16 import convert_float_to_float16


def test_basic_fp16_conversion():
    """Test basic float32 to float16 conversion."""
    input_tensor = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    output_tensor = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])

    add_node = helper.make_node("Add", inputs=["X", "X"], outputs=["Y"], name="Add")

    graph = helper.make_graph(
        nodes=[add_node],
        name="BasicGraph",
        inputs=[input_tensor],
        outputs=[output_tensor],
    )

    model = make_model(graph)

    # Convert
    new_model = convert_float_to_float16(
        model,
        keep_io_types=False,
        check_fp16_ready=False,
    )

    # Validate with example inputs
    example_inputs = {
        "X": np.ones(
            (1, 4), dtype=np.float16
        )  # Note: float16 since keep_io_types=False
    }
    validate_model(new_model, "test_basic_fp16_conversion")

    # Additional checks...
    for i in new_model.graph.input:
        assert i.type.tensor_type.elem_type == TensorProto.FLOAT16


def test_partial_block():
    """
    Test: Two nodes in sequence, one blocked and one not blocked.
    Ensure the blocked node remains float32, the unblocked node becomes fp16,
    and that exactly one boundary Cast is inserted.
    """
    # (input float32) --[BlockedNode]--> (mid) --[UnblockedNode]--> (output)
    input_tensor = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])
    mid_tensor = helper.make_tensor_value_info("Mid", TensorProto.FLOAT, [1, 2])
    output_tensor = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])

    blocked_node = helper.make_node(
        "Relu",
        inputs=["X"],
        outputs=["Mid"],
        name="BlockedRelu",
    )
    unblocked_node = helper.make_node(
        "Sigmoid", inputs=["Mid"], outputs=["Y"], name="UnblockedSigmoid"
    )

    graph = helper.make_graph(
        [blocked_node, unblocked_node],
        "PartialBlockGraph",
        inputs=[input_tensor],
        outputs=[output_tensor],
    )
    model = make_model(graph)

    # Convert, with 'Relu' in the op_block_list => remain float32
    new_model = convert_float_to_float16(
        model,
        keep_io_types=False,
        op_block_list=["Relu"],
        node_block_list=None,
        check_fp16_ready=False,
    )

    validate_model(new_model, "test_partial_block")

    # Check for a boundary cast node from "BlockedRelu" -> "UnblockedSigmoid"
    cast_nodes = [n for n in new_model.graph.node if n.op_type == "Cast"]
    assert len(cast_nodes) == 1, "Expected one boundary Cast insertion."

    # Check outputs are float16 now, given keep_io_types=False
    for i in new_model.graph.input:  # still float 32, since the first node is blocked
        assert i.type.tensor_type.elem_type == TensorProto.FLOAT
    for o in new_model.graph.output:
        assert o.type.tensor_type.elem_type == TensorProto.FLOAT16
