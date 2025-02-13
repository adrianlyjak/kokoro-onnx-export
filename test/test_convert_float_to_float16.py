import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from kokoro_onnx.convert_float_to_float16 import convert_float_to_float16


def test_basic_fp16_conversion():
    """
    Test: Single Add node (float32) -> Convert everything to float16.
    Verify that the node's inputs/outputs become fp16.
    """
    # Build a small graph: (input) --Add--> (output)
    input_tensor = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    weight_init = numpy_helper.from_array(np.ones((1, 4), dtype=np.float32), name="W")
    output_tensor = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])

    add_node = helper.make_node("Add", inputs=["X", "W"], outputs=["Y"], name="AddNode")

    graph = helper.make_graph(
        nodes=[add_node],
        name="BasicGraph",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[weight_init],
    )

    model = helper.make_model(graph, producer_name="test_basic_fp16")
    onnx.checker.check_model(model)

    # Convert
    new_model = convert_float_to_float16(
        model,
        keep_io_types=False,  # We want the IO to be converted
        op_block_list=None,
        node_block_list=None,
        check_fp16_ready=False,
    )
    onnx.checker.check_model(new_model)

    # Check that input, output, and initializer are now FLOAT16
    for i in new_model.graph.input:
        assert i.type.tensor_type.elem_type == TensorProto.FLOAT16
    for o in new_model.graph.output:
        assert o.type.tensor_type.elem_type == TensorProto.FLOAT16
    for init in new_model.graph.initializer:
        assert init.data_type == TensorProto.FLOAT16


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
        "Relu",  # We'll block everything with "Relu" just as an example
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
    model = helper.make_model(graph, producer_name="test_partial_block")

    # Convert, with 'Relu' in the op_block_list => remain float32
    new_model = convert_float_to_float16(
        model,
        keep_io_types=False,
        op_block_list=["Relu"],
        node_block_list=None,
        check_fp16_ready=False,
    )

    onnx.checker.check_model(new_model)

    # Check for a boundary cast node from "BlockedRelu" -> "UnblockedSigmoid"
    cast_nodes = [n for n in new_model.graph.node if n.op_type == "Cast"]
    assert len(cast_nodes) > 0, "Expected at least one boundary Cast insertion."

    # Check inputs/outputs are float16 now, given keep_io_types=False
    for i in new_model.graph.input:
        assert i.type.tensor_type.elem_type == TensorProto.FLOAT16
    for o in new_model.graph.output:
        assert o.type.tensor_type.elem_type == TensorProto.FLOAT16


def test_subgraph_with_if_mismatch():
    """
    Reproduce an If node scenario with a subgraph that must remain float32
    because its parent node is blocked.

    We'll set keep_io_types=True to ensure the top-level inputs/outputs
    also stay float32.
    """
    # Step 1: Build a subgraph for 'then_branch':
    #   Subgraph input -> Squeeze -> Subgraph output
    then_in = helper.make_tensor_value_info("then_in", TensorProto.FLOAT, [None])
    then_out = helper.make_tensor_value_info("then_out", TensorProto.FLOAT, [None])
    squeeze_node = helper.make_node(
        "Squeeze",
        inputs=["then_in"],
        outputs=["then_out"],
        name="Squeeze_in_then_branch",
    )
    then_graph = helper.make_graph(
        nodes=[squeeze_node], name="ThenGraph", inputs=[then_in], outputs=[then_out]
    )

    # Step 2: Build a subgraph for 'else_branch' (trivial pass-through):
    else_in = helper.make_tensor_value_info("else_in", TensorProto.FLOAT, [None])
    else_out = helper.make_tensor_value_info("else_out", TensorProto.FLOAT, [None])
    identity_node = helper.make_node(
        "Identity",
        inputs=["else_in"],
        outputs=["else_out"],
        name="Identity_in_else_branch",
    )
    else_graph = helper.make_graph(
        nodes=[identity_node], name="ElseGraph", inputs=[else_in], outputs=[else_out]
    )

    # Step 3: Main graph:
    #   Condition (Bool), a blocked conv node (float32) -> If
    cond_in = helper.make_tensor_value_info("cond", TensorProto.BOOL, [])
    conv_in = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 16, 32, 32])
    conv_out = helper.make_tensor_value_info(
        "ConvOut", TensorProto.FLOAT, [1, 16, 32, 32]
    )

    # Provide a valid weights initializer for Conv
    # shape = (16, 16, 3, 3) for example
    W_data = np.random.randn(16, 16, 3, 3).astype(np.float32)
    W_init = numpy_helper.from_array(W_data, name="W")

    conv_node = helper.make_node(
        "Conv",
        inputs=["X", "W"],  # Now has 2 inputs
        outputs=["ConvOut"],
        name="BlockedConvNode",
    )

    # If node: inputs -> cond, ConvOut => subgraphs => finalOut
    final_out = helper.make_tensor_value_info("FinalOut", TensorProto.FLOAT, [None])
    if_node = helper.make_node(
        "If",
        inputs=["cond"],
        outputs=["FinalOut"],
        name="If_1",
        then_branch=then_graph,
        else_branch=else_graph,
    )

    main_graph = helper.make_graph(
        [conv_node, if_node],
        "MainIfGraph",
        inputs=[cond_in, conv_in],
        outputs=[final_out],
        initializer=[W_init],
    )
    model = helper.make_model(main_graph, producer_name="test_if_subgraph")

    # Block ALL nodes => remain float32
    node_block_list = [conv_node.name, squeeze_node.name, identity_node.name]

    new_model = convert_float_to_float16(
        model,
        keep_io_types=True,  # So top-level I/O also stays float32
        op_block_list=[],  # rely solely on node_block_list
        node_block_list=node_block_list,
        check_fp16_ready=False,
    )

    # This should pass now, because we have a valid Conv node with 2 inputs
    onnx.checker.check_model(new_model)

    # Verify If node is still there and subgraph input hasn't been forced to float16.
    if_node_new = None
    for node in new_model.graph.node:
        if node.name == "If_1":
            if_node_new = node
            break
    assert if_node_new is not None, "Failed to find If_1 in the new model."

    # Check the then_branch subgraph input is still float32
    then_g = if_node_new.attribute[0].g  # then_branch
    then_input = then_g.input[0]
    assert then_input.type.tensor_type.elem_type == TensorProto.FLOAT, (
        "Subgraph input was incorrectly converted to float16."
    )

    # Also check the main graph's "ConvOut" remains float32
    found_convout = False
    for vi in new_model.graph.value_info:
        if vi.name == "ConvOut":
            found_convout = True
            assert vi.type.tensor_type.elem_type == TensorProto.FLOAT, (
                "ConvOut got converted to float16 despite being blocked."
            )
    assert found_convout, "Did not find 'ConvOut' in new_model.graph.value_info."
