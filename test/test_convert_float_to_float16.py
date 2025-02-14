import os

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
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


def test_float16_multiple_outputs(tmp_path):
    """
    Create a small ONNX model that produces multiple outputs from
    the same intermediate tensor. This commonly triggers a name-collision
    bug if boundary-cast nodes are generated with duplicate names.
    """

    # 1) Build a simple graph with one Conv node
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 3, 3])
    W = helper.make_tensor_value_info("W", TensorProto.FLOAT, [1, 1, 3, 3])

    # The Conv output is reused by two subsequent Identity nodes
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 1, 1])
    Out1 = helper.make_tensor_value_info("Out1", TensorProto.FLOAT, [1, 1, 1, 1])
    Out2 = helper.make_tensor_value_info("Out2", TensorProto.FLOAT, [1, 1, 1, 1])

    conv_node = helper.make_node("Conv", inputs=["X", "W"], outputs=["Y"], name="conv1")

    # Two Identity nodes that each take "Y" as input
    id_node1 = helper.make_node("Identity", inputs=["Y"], outputs=["Out1"], name="id1")

    id_node2 = helper.make_node("Identity", inputs=["Y"], outputs=["Out2"], name="id2")

    # Construct the graph
    graph = helper.make_graph(
        nodes=[conv_node, id_node1, id_node2],
        name="multiple_outputs_graph",
        inputs=[X, W],
        outputs=[Out1, Out2],
    )

    # Create initializers for W
    W_init = np.ones((1, 1, 3, 3), dtype=np.float32)
    w_initializer = numpy_helper.from_array(W_init, name="W")

    graph.initializer.extend([w_initializer])

    onnx_model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 20)],
        producer_name="test_float16_multiple_outputs",
    )

    # 2) Load the model in onnxruntime (float32) to sanity-check
    sess_float32 = ort.InferenceSession(
        onnx_model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    inputs = {"X": np.ones((1, 1, 3, 3), dtype=np.float32)}
    outputs_float32 = sess_float32.run(None, inputs)
    assert len(outputs_float32) == 2, "Expected two outputs in float32 model."

    # 3) Call your float16 conversion (this is where you expect the name collision bug)
    # Replace this with whatever call you use in your `quantize.py:float16()` pipeline.

    fp16_model = convert_float_to_float16(
        onnx_model,
        keep_io_types=False,
    )  # or whatever your function signature is

    # 4) Try to load the new FP16 model. This step will fail if
    # there's a duplicate node name from boundary casting.
    sess_fp16 = ort.InferenceSession(
        fp16_model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    outputs_fp16 = sess_fp16.run(None, {"X": np.ones((1, 1, 3, 3), dtype=np.float16)})
    assert len(outputs_fp16) == 2, "Expected two outputs in float16 model."
    print("Test completed without duplicate naming issues.")


class ModelWithSqueeze(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 3)
        self.squeeze = nn.Flatten(start_dim=2)  # Similar effect to Squeeze

    def forward(self, x, condition):
        x = self.conv(x)
        # Simulate If node behavior
        if condition:
            return self.squeeze(x)
        return x


def test_subgraph_precision_mismatch(tmp_path):
    """
    Test to reproduce bug where Conv1d output gets converted to float16
    while the subsequent Squeeze remains float32.
    """
    model = ModelWithSqueeze()
    model.eval()

    # Create dummy inputs matching your shapes
    x = torch.randn(3, 256, 192)  # [batch, d_hid//2, length]

    # Export to ONNX in temp directory
    model_path = tmp_path / "model.onnx"
    torch.onnx.export(
        model,
        (x, torch.tensor(True)),
        str(model_path),
        input_names=["input", "condition"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch"},
            "output": {0: "batch"},
        },
        opset_version=20,
    )

    # Load the ONNX model
    onnx_model = onnx.load(model_path)

    only_ops = ["Conv"]
    # Convert to float16
    converted_model = convert_float_to_float16(
        onnx_model,
        keep_io_types=True,
        node_block_list=[
            x.name for x in onnx_model.graph.node if x.op_type not in only_ops
        ],
        check_fp16_ready=False,
    )

    # Add graph visualization before checking
    def print_graph(graph, indent=""):
        print(f"\n{indent}Graph: {graph.name}")
        print(f"{indent}----------------------")

        for node in graph.node:
            print(f"\n{indent}Node: {node.name} (Op: {node.op_type})")
            print(f"{indent}  Inputs:", node.input)
            print(f"{indent}  Outputs:", node.output)

            # Print data types for inputs
            for input_name in node.input:
                # Convert repeated fields to list and combine
                value_infos = (
                    list(graph.value_info) + list(graph.input) + list(graph.output)
                )
                for vi in value_infos:
                    if vi.name == input_name:
                        dtype = vi.type.tensor_type.elem_type
                        print(
                            f"{indent}  Input {input_name} type: "
                            f"{TensorProto.DataType.Name(dtype)}"
                        )

            # Print data types for outputs
            for output_name in node.output:
                value_infos = (
                    list(graph.value_info) + list(graph.input) + list(graph.output)
                )
                for vi in value_infos:
                    if vi.name == output_name:
                        dtype = vi.type.tensor_type.elem_type
                        print(
                            f"{indent}  Output {output_name} type: "
                            f"{TensorProto.DataType.Name(dtype)}"
                        )

            # Recursively print subgraphs if they exist
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    print(f"\n{indent}Subgraph from {node.name}.{attr.name}:")
                    print_graph(attr.g, indent + "  ")

    print("\nModel Graph Structure:")
    print_graph(converted_model.graph)

    # Verify the bug: Conv output should be float16 while Squeeze output remains float32
    conv_output_type = None
    squeeze_output_type = None

    # Find the Conv output type
    for vi in converted_model.graph.value_info:
        if "Conv" in vi.name:
            conv_output_type = vi.type.tensor_type.elem_type
            break

    # Find the Squeeze output type
    for node in converted_model.graph.node:
        if node.op_type == "Squeeze":
            for output in node.output:
                for vi in converted_model.graph.value_info:
                    if vi.name == output:
                        squeeze_output_type = vi.type.tensor_type.elem_type
                        break

    # The model should still pass the checker despite the mismatch
    onnx.checker.check_model(converted_model)
    ort.InferenceSession(
        converted_model.SerializeToString(), providers=["CPUExecutionProvider"]
    )

    # These assertions should be updated to match the expected behavior
    assert conv_output_type == TensorProto.FLOAT16, (
        "Conv output should be float16 since it's not blocked"
    )
    assert squeeze_output_type == TensorProto.FLOAT, (
        "Squeeze output should remain float32, since it's blocked"
    )


@torch.jit.script
def if_conv_flatten(
    x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, condition: bool
) -> torch.Tensor:
    """
    Applies a conv2d, then uses an If node:
      if condition: Flatten
      else: clone (i.e. identity)
    """
    x = F.conv2d(x, w, b, stride=1, padding=1)
    if condition:
        # 'then' branch
        return x.view(x.size(0), x.size(1), -1)  # flatten
    else:
        # 'else' branch
        return x.clone()


#
# Step B) Export it to ONNX with a dynamic boolean condition
#
def test_if_subgraph_blocking(tmp_path):
    # 1) Generate some dummy data
    x = torch.randn(2, 1, 5, 5, dtype=torch.float32)
    w = torch.randn(1, 1, 3, 3, dtype=torch.float32)
    b = torch.randn(1, dtype=torch.float32)

    # 2) Export to ONNX.  This definitely yields an If node
    model_path = tmp_path / "model_if.onnx"
    torch.onnx.export(
        if_conv_flatten,
        (x, w, b, True),  # Pass a bool that's not recognized as a constant by JIT
        str(model_path),
        opset_version=14,  # or 20, depending on your version
        input_names=["x", "w", "b", "condition"],
        output_names=["output"],
        dynamic_axes={"x": {0: "batch"}, "output": {0: "batch"}},
        do_constant_folding=False,
    )
    onnx_model = onnx.load(str(model_path))

    # 3) (Optional) Print the ops to confirm there's an If:
    print("OPS in exported model:", [n.op_type for n in onnx_model.graph.node])

    node_block_list = []
    for node in onnx_model.graph.node:
        # flatten in the "then_branch" is actually a Reshape op in ONNX,
        # so let's forcibly block any node of type "Reshape" or "View" or "Flatten":
        if node.op_type in ("Reshape", "Flatten", "View"):
            node_block_list.append(node.name)

    # keep_io_types=True means top-level x, w, b, condition stay float32
    # (which is typical if you want your user inputs in FP32).
    fp16_model = convert_float_to_float16(
        onnx_model,
        keep_io_types=True,
        node_block_list=node_block_list,
        check_fp16_ready=False,
    )

    # 5) Confirm the model is valid
    onnx.checker.check_model(fp16_model)

    # 6) Test in ONNXRuntime
    sess = ort.InferenceSession(fp16_model.SerializeToString())

    # condition=True path
    out_true = sess.run(
        None,
        {
            "x": x.numpy(),
            "w": w.numpy(),
            "b": b.numpy(),
            "condition": np.array(True, dtype=bool),
        },
    )
    print("Output shape for condition=True:", out_true[0].shape)

    # condition=False path
    out_false = sess.run(
        None,
        {
            "x": x.numpy(),
            "w": w.numpy(),
            "b": b.numpy(),
            "condition": np.array(False, dtype=bool),
        },
    )
    print("Output shape for condition=False:", out_false[0].shape)

    # If your sub-graph boundary logic is incomplete, you'll typically see
    # a TypeError or mismatch at the .run(...) call
