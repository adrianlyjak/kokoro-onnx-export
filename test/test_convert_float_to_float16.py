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


def create_if_graph_model():
    # Step 1: Build a subgraph for 'then_branch':
    #   Subgraph input -> Squeeze -> Subgraph output
    then_out = helper.make_tensor_value_info("then_out", TensorProto.FLOAT, [None])
    squeeze_node = helper.make_node(
        "Squeeze",
        inputs=["ConvOut"],
        outputs=["then_out"],
        name="Squeeze_in_then_branch",
    )
    then_graph = helper.make_graph(
        nodes=[squeeze_node], name="ThenGraph", inputs=[], outputs=[then_out]
    )

    # Step 2: Build a subgraph for 'else_branch' (trivial pass-through):
    else_out = helper.make_tensor_value_info("else_out", TensorProto.FLOAT, [None])
    identity_node = helper.make_node(
        "Identity",
        inputs=["X"],
        outputs=["else_out"],
        name="Identity_in_else_branch",
    )
    else_graph = helper.make_graph(
        nodes=[identity_node], name="ElseGraph", inputs=[], outputs=[else_out]
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
        inputs=["X", "W"],
        outputs=["ConvOut"],
        name="ConvNode",
    )

    # If node: inputs -> cond, ConvOut => subgraphs => finalOut
    final_out = helper.make_tensor_value_info("FinalOut", TensorProto.FLOAT, [None])
    if_node = helper.make_node(
        "If",
        inputs=["cond"],
        outputs=["IfOut"],
        name="If_1",
        then_branch=then_graph,
        else_branch=else_graph,
    )

    # Add a Relu node between If output and final output
    relu_node = helper.make_node(
        "Relu", inputs=["IfOut"], outputs=["FinalOut"], name="Relu_after_if"
    )

    main_graph = helper.make_graph(
        [conv_node, if_node, relu_node],
        "MainGraph",
        inputs=[cond_in, conv_in],
        outputs=[final_out],
        initializer=[W_init],
    )
    model = make_model(main_graph)
    return model


def test_subgraph_with_if_mismatch():
    """
    Reproduce an If node scenario with a subgraph that must remain float32
    because its parent node is blocked.

    We'll set keep_io_types=True to ensure the top-level inputs/outputs
    also stay float32.
    """
    model = create_if_graph_model()

    # Block ALL non-if nodes, and make sure the if tensor outputs match the if node's output type
    node_block_list = ["ConvNode", "Squeeze_in_then_branch", "Identity_in_else_branch"]

    new_model = convert_float_to_float16(
        model,
        keep_io_types=True,  # So top-level I/O also stays float32
        op_block_list=[],  # rely solely on node_block_list
        node_block_list=node_block_list,
        check_fp16_ready=False,
    )

    # This should pass now, because we have a valid Conv node with 2 inputs
    validate_model(new_model, "test_subgraph_with_if_mismatch")

    # Verify If node is still there and subgraph input hasn't been forced to float16.
    if_node_new = None
    for node in new_model.graph.node:
        if node.name == "If_1":
            if_node_new = node
            break
    assert if_node_new is not None, "Failed to find If_1 in the new model."

    # check the main graph's "ConvOut" remains float32
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

    onnx_model = make_model(graph)

    # 2) Load the model in onnxruntime (float32) to sanity-check
    validate_model(onnx_model, "test_float16_multiple_outputs (float32)")

    # 3) Call float16 conversion
    fp16_model = convert_float_to_float16(
        onnx_model,
        keep_io_types=False,
    )

    # 4) Validate the FP16 model
    validate_model(fp16_model, "test_float16_multiple_outputs (float16)")

    # Additional checks can stay as they are...


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
    model_path = "modif.onnx"  # tmp_path / "model_if.onnx"
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
    validate_model(onnx_model, "test_if_subgraph_blocking_before")

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
    validate_model(fp16_model, "test_if_subgraph_blocking")

    print(generate_mermaid_from_onnx(fp16_model))

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


def generate_mermaid_from_onnx(model: onnx.ModelProto) -> str:
    """Convert ONNX model to Mermaid diagram string."""
    graph = model.graph

    mermaid_lines = ["graph TD"]
    node_names: Set[str] = set()

    def sanitize_name(name: str) -> str:
        """Make node names Mermaid-compatible."""
        return name.replace(" ", "_").replace("/", "_").replace(":", "_")

    def process_graph(graph, prefix=""):
        """Process graph and subgraphs recursively."""
        # Add nodes
        for node in graph.node:
            node_id = (
                sanitize_name(node.name) if node.name else f"node_{len(node_names)}"
            )
            while node_id in node_names:
                node_id = f"{node_id}_{len(node_names)}"
            node_names.add(node_id)

            # Create node label with op_type
            label = f"{node_id}[{node.op_type}]"
            mermaid_lines.append(f"    {label}")

            # Add edges from inputs
            for input_name in node.input:
                if input_name:  # Skip empty inputs
                    input_id = sanitize_name(input_name)
                    mermaid_lines.append(f"    {input_id} --> {node_id}")

            # If node is an If node, process subgraphs
            if node.op_type == "If":
                for attr in node.attribute:
                    if attr.name in ["then_branch", "else_branch"]:
                        subgraph = attr.g
                        subgraph_prefix = f"{node_id}_{attr.name}"

                        # Create subgraph
                        mermaid_lines.append(f"    subgraph {subgraph_prefix}")
                        process_graph(subgraph, subgraph_prefix)
                        mermaid_lines.append("    end")

                        # Connect subgraph to parent node
                        mermaid_lines.append(f"    {node_id} --> {subgraph_prefix}")

    # Process initializers as input nodes
    for initializer in graph.initializer:
        init_id = sanitize_name(initializer.name)
        mermaid_lines.append(f"    {init_id}((Input))")
        node_names.add(init_id)

    # Process main graph
    process_graph(graph)

    return "\n".join(mermaid_lines)


def test_initializer_conversion():
    """
    Test that initializers are properly converted to float16 when used by unblocked nodes,
    and remain float32 when used by blocked nodes.
    """
    # Create a graph with two parallel paths:
    # Path 1: input -> Add(W1) -> output1  (blocked)
    # Path 2: input -> Add(W2) -> output2  (unblocked)

    input_tensor = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    output1 = helper.make_tensor_value_info("Y1", TensorProto.FLOAT, [1, 4])
    output2 = helper.make_tensor_value_info("Y2", TensorProto.FLOAT, [1, 4])

    # Create two weight initializers
    w1_data = np.ones((1, 4), dtype=np.float32)
    w2_data = np.ones((1, 4), dtype=np.float32) * 2
    w1_init = numpy_helper.from_array(w1_data, name="W1")
    w2_init = numpy_helper.from_array(w2_data, name="W2")

    add1 = helper.make_node(
        "Add", inputs=["X", "W1"], outputs=["Y1"], name="BlockedAdd"
    )
    add2 = helper.make_node(
        "Add", inputs=["X", "W2"], outputs=["Y2"], name="UnblockedAdd"
    )

    graph = helper.make_graph(
        nodes=[add1, add2],
        name="TestInitializers",
        inputs=[input_tensor],
        outputs=[output1, output2],
        initializer=[w1_init, w2_init],
    )

    model = make_model(graph)

    # Convert with BlockedAdd in the block list
    new_model = convert_float_to_float16(
        model,
        keep_io_types=True,
        node_block_list=["BlockedAdd"],
        check_fp16_ready=False,
    )

    validate_model(new_model, "test_initializer_conversion")

    # Verify W1 stayed float32 (used by blocked node) and W2 became float16
    for init in new_model.graph.initializer:
        if init.name == "W1":
            assert init.data_type == TensorProto.FLOAT
        elif init.name == "W2":
            assert init.data_type == TensorProto.FLOAT16


def test_boundary_cast_insertion():
    """
    Test that boundary casts are properly inserted between float32 and float16 nodes,
    and that unnecessary casts are not inserted between nodes of the same precision.
    """
    # Build graph: input -> Add -> Relu -> Sigmoid -> output
    # Where Add and Sigmoid will be float16, but Relu blocked as float32

    input_tensor = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    output_tensor = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])

    add_node = helper.make_node(
        "Add", inputs=["X", "X"], outputs=["add_out"], name="Add"
    )
    relu_node = helper.make_node(
        "Relu", inputs=["add_out"], outputs=["relu_out"], name="Relu"
    )
    sigmoid_node = helper.make_node(
        "Sigmoid", inputs=["relu_out"], outputs=["Y"], name="Sigmoid"
    )

    graph = helper.make_graph(
        nodes=[add_node, relu_node, sigmoid_node],
        name="TestBoundaryCasts",
        inputs=[input_tensor],
        outputs=[output_tensor],
    )

    model = make_model(graph)

    # Convert with Relu blocked
    new_model = convert_float_to_float16(
        model,
        keep_io_types=False,
        op_block_list=["Relu"],
        check_fp16_ready=False,
    )

    validate_model(new_model, "test_boundary_cast_insertion")

    # Count Cast nodes - should be exactly 2:
    # 1. Before Relu (float16->float32)
    # 2. After Relu (float32->float16)
    cast_nodes = [n for n in new_model.graph.node if n.op_type == "Cast"]
    assert len(cast_nodes) == 2, f"Expected 2 boundary casts, got {len(cast_nodes)}"

    # Verify cast directions
    casts_to_float32 = [n for n in cast_nodes if n.attribute[0].i == TensorProto.FLOAT]
    casts_to_float16 = [
        n for n in cast_nodes if n.attribute[0].i == TensorProto.FLOAT16
    ]
    assert len(casts_to_float32) == 1, "Expected 1 cast to float32"
    assert len(casts_to_float16) == 1, "Expected 1 cast to float16"


def test_keep_io_types():
    """Test keep_io_types=True preserves float32 I/O."""
    input_tensor = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    output_tensor = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])

    add_node = helper.make_node(
        "Add", inputs=["X", "X"], outputs=["add_out"], name="Add"
    )
    mul_node = helper.make_node(
        "Mul", inputs=["add_out", "add_out"], outputs=["Y"], name="Mul"
    )

    graph = helper.make_graph(
        nodes=[add_node, mul_node],
        name="TestKeepIOTypes",
        inputs=[input_tensor],
        outputs=[output_tensor],
    )

    model = make_model(graph)

    new_model = convert_float_to_float16(
        model,
        keep_io_types=True,
        op_block_list=[],
        node_block_list=None,
        check_fp16_ready=False,
    )

    # Validate with float32 inputs (since keep_io_types=True)
    example_inputs = {"X": np.ones((1, 4), dtype=np.float32)}
    validate_model(new_model, "test_keep_io_types")

    # Additional checks...
    assert new_model.graph.input[0].type.tensor_type.elem_type == TensorProto.FLOAT


def test_value_clamping():
    """
    Test that min_positive_val and max_finite_val properly clamp initializer values.
    """
    # Create initializer with values that will need clamping
    data = np.array([1e-8, 1e5, -1e-8, -1e5, 0.0, 1.0, -1.0], dtype=np.float32)
    init = numpy_helper.from_array(data, name="data")

    # Simple graph that just outputs the initializer
    output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [7])
    identity = helper.make_node(
        "Identity", inputs=["data"], outputs=["Y"], name="Identity"
    )

    graph = helper.make_graph(
        nodes=[identity],
        name="TestClamping",
        inputs=[],
        outputs=[output],
        initializer=[init],
    )

    model = make_model(graph)

    # Convert with specific clamping values
    min_positive = 1e-7
    max_finite = 1e4
    new_model = convert_float_to_float16(
        model,
        keep_io_types=False,
        min_positive_val=min_positive,
        max_finite_val=max_finite,
        check_fp16_ready=False,
    )

    validate_model(new_model, "test_value_clamping")

    # Extract the converted initializer data
    new_init = None
    for init in new_model.graph.initializer:
        if init.name == "data":
            print(f"init", init)
            new_init = numpy_helper.to_array(init)
            break

    assert new_init is not None, "Failed to find converted initializer"

    # Verify clamping behavior with much larger tolerance for float16
    np.testing.assert_allclose(
        new_init[0],
        min_positive,
        rtol=0.2,  # 20% tolerance for float16
        err_msg="Small positive should be clamped up",
    )
    np.testing.assert_allclose(
        new_init[1],
        max_finite,
        rtol=1e-3,
        err_msg="Large positive should be clamped down",
    )
    np.testing.assert_allclose(
        new_init[2],
        -min_positive,
        rtol=0.2,  # 20% tolerance for float16
        err_msg="Small negative should be clamped up",
    )
    np.testing.assert_allclose(
        new_init[3],
        -max_finite,
        rtol=1e-3,
        err_msg="Large negative should be clamped down",
    )
    np.testing.assert_allclose(
        new_init[4], 0.0, atol=1e-6, err_msg="Zero should remain unchanged"
    )
    np.testing.assert_allclose(
        new_init[5], 1.0, rtol=1e-3, err_msg="One should remain roughly unchanged"
    )
    np.testing.assert_allclose(
        new_init[6],
        -1.0,
        rtol=1e-3,
        err_msg="Negative one should remain roughly unchanged",
    )


def test_check_fp16_ready():
    """
    Test that check_fp16_ready properly detects and rejects models with float16.
    """
    # Create a model that's already partially float16
    input_tensor = helper.make_tensor_value_info("X", TensorProto.FLOAT16, [1, 4])
    output_tensor = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])

    node = helper.make_node("Identity", inputs=["X"], outputs=["Y"], name="Identity")

    graph = helper.make_graph(
        nodes=[node],
        name="TestFP16Ready",
        inputs=[input_tensor],
        outputs=[output_tensor],
    )

    model = make_model(graph)

    # Should raise when check_fp16_ready=True
    expected_msg = re.escape(
        "The model appears to be (partially) in float16 already. "
        "Set check_fp16_ready=False to override."
    )
    with pytest.raises(ValueError, match=expected_msg):
        convert_float_to_float16(model, check_fp16_ready=True)

    # Should not raise when check_fp16_ready=False
    converted = convert_float_to_float16(model, check_fp16_ready=False)
    assert converted is not None


def test_block_list_precedence():
    """
    Test that node_block_list takes precedence over op_block_list when they conflict.
    """
    input_tensor = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    output_tensor = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])

    # Create two Add nodes - one we'll block by name, one by op_type
    add1 = helper.make_node(
        "Add", inputs=["X", "X"], outputs=["add1_out"], name="BlockedByName"
    )
    add2 = helper.make_node(
        "Add", inputs=["add1_out", "add1_out"], outputs=["Y"], name="BlockedByType"
    )

    graph = helper.make_graph(
        nodes=[add1, add2],
        name="TestBlockLists",
        inputs=[input_tensor],
        outputs=[output_tensor],
    )

    model = make_model(graph)

    # Convert with conflicting block lists
    new_model = convert_float_to_float16(
        model,
        keep_io_types=True,
        op_block_list=["Add"],  # Block all Add ops
        node_block_list=["BlockedByName"],  # Explicitly block one Add
        check_fp16_ready=False,
    )

    # Both nodes should be blocked, but through different mechanisms
    # We can verify this by checking there are no Cast nodes inserted
    cast_nodes = [n for n in new_model.graph.node if n.op_type == "Cast"]
    assert len(cast_nodes) == 0, (
        "Expected no Cast nodes since both Adds should be blocked"
    )

    # Verify both nodes remained in float32 domain
    value_info = list(new_model.graph.value_info)
    found_float16 = False
    for vi in value_info:
        if vi.type.tensor_type.elem_type == TensorProto.FLOAT16:
            found_float16 = True
            break
    assert not found_float16, "Expected all internal tensors to remain float32"


def test_nonexistent_block_list():
    """
    Test that non-existent names in node_block_list are safely ignored.
    """
    input_tensor = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    output_tensor = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])

    node = helper.make_node("Add", inputs=["X", "X"], outputs=["Y"], name="add")

    graph = helper.make_graph(
        nodes=[node],
        name="TestNonexistentBlock",
        inputs=[input_tensor],
        outputs=[output_tensor],
    )

    model = make_model(graph)

    # Convert with non-existent node in block list
    new_model = convert_float_to_float16(
        model,
        keep_io_types=False,
        node_block_list=["nonexistent_node"],
        check_fp16_ready=False,
    )

    # Model should still be valid
    validate_model(new_model, "test_nonexistent_block_list")


def test_no_float32_tensors():
    """
    Test conversion of a model that has no float32 tensors (only int/bool).
    """
    input_tensor = helper.make_tensor_value_info("X", TensorProto.INT64, [1])
    output_tensor = helper.make_tensor_value_info("Y", TensorProto.BOOL, [1])

    node = helper.make_node("Greater", inputs=["X", "X"], outputs=["Y"], name="greater")

    graph = helper.make_graph(
        nodes=[node],
        name="TestNoFloat32",
        inputs=[input_tensor],
        outputs=[output_tensor],
    )

    model = make_model(graph)

    # Should handle this case gracefully
    new_model = convert_float_to_float16(model, check_fp16_ready=False)

    validate_model(new_model, "test_no_float32_tensors")

    # Model should be unchanged
    assert new_model.graph.node[0].op_type == "Greater"
    assert new_model.graph.input[0].type.tensor_type.elem_type == TensorProto.INT64
    assert new_model.graph.output[0].type.tensor_type.elem_type == TensorProto.BOOL


def test_node_name_handling():
    """
    Test handling of node names in block list:
    1. Names with special characters
    2. Empty names
    3. Very long names
    """
    input_tensor = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    output_tensor = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])

    # Create nodes with various name edge cases
    nodes = [
        helper.make_node(
            "Add",
            inputs=["X", "X"],
            outputs=["out1"],
            name="special/chars.here",  # Special characters
        ),
        helper.make_node(
            "Mul",
            inputs=["out1", "out1"],
            outputs=["out2"],
            name="",  # Empty name
        ),
        helper.make_node(
            "Relu",
            inputs=["out2"],
            outputs=["Y"],
            name="a" * 100,  # Very long name
        ),
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="TestNodeNames",
        inputs=[input_tensor],
        outputs=[output_tensor],
    )

    model = make_model(graph)

    # Block nodes with special characters and long name
    new_model = convert_float_to_float16(
        model,
        keep_io_types=True,
        node_block_list=["special/chars.here", "a" * 100],
        check_fp16_ready=False,
    )

    validate_model(new_model, "test_node_name_handling")

    # Verify the blocked nodes remained in float32 domain

    for node in new_model.graph.node:
        if node.name == "special/chars.here":  # Output of first blocked node
            output_tensor = node.output[0]
            print(f"output_tensor: {output_tensor}")
            vi_names = [vi.name for vi in new_model.graph.value_info]
            print(f"vi_names: {vi_names}")
            output_value = [
                vi for vi in new_model.graph.value_info if vi.name == output_tensor
            ][0]
            assert output_value.type.tensor_type.elem_type == TensorProto.FLOAT


def test_output_type_conversion():
    """
    Test that model outputs are converted to float16 when:
    1. keep_io_types=False
    2. The node producing the output is not blocked
    """
    input_tensor = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    output_tensor = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])

    # Create a simple Add node that feeds directly to output
    add_node = helper.make_node("Add", inputs=["X", "X"], outputs=["Y"], name="Add")

    graph = helper.make_graph(
        nodes=[add_node],
        name="TestOutputConversion",
        inputs=[input_tensor],
        outputs=[output_tensor],
    )

    model = make_model(graph)

    # Convert with keep_io_types=False and no blocking
    new_model = convert_float_to_float16(
        model,
        keep_io_types=False,
        op_block_list=[],
        node_block_list=None,
        check_fp16_ready=False,
    )

    validate_model(new_model, "test_output_type_conversion")

    # Verify output is float16
    assert (
        new_model.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT16
    ), (
        "Output should be float16 when keep_io_types=False and upstream node is not blocked"
    )

    # Also verify the Add node's output is float16
    add_output_info = [
        vi
        for vi in new_model.graph.value_info
        if vi.name == new_model.graph.node[0].output[0]
    ]
    if add_output_info:  # If the value_info exists
        assert add_output_info[0].type.tensor_type.elem_type == TensorProto.FLOAT16, (
            "Add node output should be float16"
        )


def test_existing_cast_input():
    """
    Test conversion when there's an existing Cast node from int64 to float32 as input.
    The existing Cast should be modified to output float16 instead of adding a new Cast.
    """
    # Create graph: input(int64) -> Cast(to float32) -> Add -> output
    input_tensor = helper.make_tensor_value_info("X", TensorProto.INT64, [1, 4])
    output_tensor = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])

    # Create Cast node from INT64 to FLOAT
    cast_node = helper.make_node(
        "Cast",
        inputs=["X"],
        outputs=["cast_out"],
        name="cast",
        to=TensorProto.FLOAT,  # Original cast to float32
    )

    add_node = helper.make_node(
        "Add",
        inputs=["cast_out", "cast_out"],
        outputs=["Y"],
        name="add",
    )

    graph = helper.make_graph(
        nodes=[cast_node, add_node],
        name="TestExistingCastInput",
        inputs=[input_tensor],
        outputs=[output_tensor],
    )

    model = make_model(graph)

    # Convert to float16
    new_model = convert_float_to_float16(
        model,
        keep_io_types=False,
        check_fp16_ready=False,
    )

    validate_model(new_model, "test_existing_cast_input")

    # Verify:
    # 1. The original Cast node now outputs float16
    # 2. No new Cast nodes were added
    cast_nodes = [n for n in new_model.graph.node if n.op_type == "Cast"]
    assert len(cast_nodes) == 1, "Expected exactly one Cast node"
    assert cast_nodes[0].attribute[0].i == TensorProto.FLOAT16, (
        "Original Cast node should now output float16"
    )


def test_existing_cast_output():
    """
    Test conversion when there's an existing Cast node from float32 to int64 as output.
    The float32 input to the existing Cast should become float16 without adding a new Cast.
    """
    # Create graph: input -> Add -> Cast(to int64) -> output
    input_tensor = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    output_tensor = helper.make_tensor_value_info("Y", TensorProto.INT64, [1, 4])

    add_node = helper.make_node(
        "Add",
        inputs=["X", "X"],
        outputs=["add_out"],
        name="add",
    )

    # Create Cast node from FLOAT to INT64
    cast_node = helper.make_node(
        "Cast",
        inputs=["add_out"],
        outputs=["Y"],
        name="cast",
        to=TensorProto.INT64,
    )

    graph = helper.make_graph(
        nodes=[add_node, cast_node],
        name="TestExistingCastOutput",
        inputs=[input_tensor],
        outputs=[output_tensor],
    )

    model = make_model(graph)

    # Convert to float16
    new_model = convert_float_to_float16(
        model,
        keep_io_types=False,
        check_fp16_ready=False,
    )

    validate_model(new_model, "test_existing_cast_output")

    # Verify:
    # 1. The Add node outputs float16
    # 2. The Cast node takes float16 input
    # 3. No new Cast nodes were added
    cast_nodes = [n for n in new_model.graph.node if n.op_type == "Cast"]
    assert len(cast_nodes) == 1, "Expected exactly one Cast node"

    # Find the value_info for add_out to verify it's float16
    add_output_info = [vi for vi in new_model.graph.value_info if vi.name == "add_out"]
    assert len(add_output_info) > 0, "Could not find add_out value_info"
    assert add_output_info[0].type.tensor_type.elem_type == TensorProto.FLOAT16, (
        "Add node should output float16"
    )


def validate_model(model: onnx.ModelProto, test_name: str):
    """
    Helper to validate an ONNX model by:
    1. Running ONNX model checker
    2. Loading into ONNX Runtime (which catches many issues the checker misses)

    Args:
        model: The ONNX model to validate
        test_name: Name of the test (for better error messages)
    """
    try:
        onnx.save(model, f"{test_name}.onnx")

        # Basic model check
        onnx.checker.check_model(model)

        # Try loading into ONNX Runtime
        sess = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )

    except Exception as e:
        raise AssertionError(f"{test_name} validation failed: {str(e)}") from e


def make_model(graph: onnx.GraphProto) -> onnx.ModelProto:
    """Helper to create an ONNX model with standard opset 20."""
    return helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 20)],  # Empty string means "ai.onnx"
    )
