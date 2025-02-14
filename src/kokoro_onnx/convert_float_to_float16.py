# Copied and heavily modified from https://github.com/microsoft/onnxconverter-common/blob/master/onnxconverter_common/float16.py
import itertools
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Union

import numpy as np
import onnx
import packaging.version as pv
from onnx import NodeProto, TensorProto, ValueInfoProto, helper, numpy_helper
from onnx import onnx_pb as onnx_proto

FLOAT32 = 1
FLOAT16 = 10

DEFAULT_OP_BLOCK_LIST = [
    "ArrayFeatureExtractor",
    "Binarizer",
    "CastMap",
    "CategoryMapper",
    "DictVectorizer",
    "FeatureVectorizer",
    "Imputer",
    "LabelEncoder",
    "LinearClassifier",
    "LinearRegressor",
    "Normalizer",
    "OneHotEncoder",
    "RandomUniformLike",
    "SVMClassifier",
    "SVMRegressor",
    "Scaler",
    "TreeEnsembleClassifier",
    "TreeEnsembleRegressor",
    "ZipMap",
    "NonMaxSuppression",
    "TopK",
    "RoiAlign",
    "Resize",
    "Range",
    "CumSum",
    "Min",
    "Max",
    "Upsample",
]


def convert_float_to_float16(
    model,
    min_positive_val=1e-7,
    max_finite_val=1e4,
    keep_io_types=False,
    disable_shape_infer=False,
    op_block_list=None,
    node_block_list=None,
    check_fp16_ready=True,
):
    """
    A reworked entry-point to convert an ONNX ModelProto from float32 to float16.
    This version does not insert up/down-casts around every blocked node input/output.
    Instead, it only inserts boundary casts between float32-blocked nodes and float16 nodes.

    :param model: ONNX ModelProto
    :param min_positive_val: clamp for small positive floats
    :param max_finite_val: clamp for large magnitude floats
    :param keep_io_types: if True, top-level model inputs/outputs remain float32
    :param disable_shape_infer: if True, skip the onnx.shape_inference pass
    :param op_block_list: list of op_types that must remain float32
    :param node_block_list: list of node names that must remain float32
    :param check_fp16_ready: if True, throw an error if model is already fp16
    :return: ModelProto in partial or full fp16
    """
    # Set up defaults
    if op_block_list is None:
        op_block_list = DEFAULT_OP_BLOCK_LIST
    if node_block_list is None:
        node_block_list = set()
    else:
        node_block_list = set(node_block_list)
    op_block_list = set(op_block_list)

    # Basic checking, optional shape inference
    model, func_infer_shape, is_fp16_ready_flag = initial_checking(
        model, disable_shape_infer
    )
    if is_fp16_ready_flag and check_fp16_ready:
        raise ValueError(
            "The model appears to be (partially) in float16 already. "
            "Set check_fp16_ready=False to override."
        )

    # accumulate all tensors
    # 1. input tensors
    # 2. output tensors
    # 3. initializer tensors
    # 4. value_info tensors??
    # 5. recurse from sub-graphs
    # Within all of these tensors, keep track of the
    #    (maybe 1) node that are outputs to it (not initializers nor inputs)
    #    the 0 to many nodes that they are inputs to
    #    whether they are IO tensors (first level graph only)
    #    what data type they are
    # Additionally, keep track of whether each is an IO tensor (on the root level)
    # then, for each, if its data type is fp32:
    #   if its IO and keep_io_types is True, skip
    #   if one of its inputs or outputs is not blocked, convert to fp16
    #   if converted, and any of its inputs or outputs are blocked, insert a cast, and reconnect the node to the cast node
    tensor_infos, all_graphs = _build_tensor_infos(model.graph)
    checker = _is_blocked_checker(op_block_list, node_block_list)
    used_names = set()
    for graph in all_graphs:
        for node in graph.node:
            used_names.add(node.name)
    for ti in tensor_infos:
        used_names.add(ti.name())
    print(f"ti names:")
    for ti in tensor_infos:
        print(f"  {ti.name()}")
    for ti in tensor_infos:
        _modify_graph_for_tensor_info(
            ti, checker, keep_io_types, used_names, min_positive_val, max_finite_val
        )
    for graph in all_graphs:
        for node in graph.node:
            if not checker(node):
                _modify_node_attribute_type(node, min_positive_val, max_finite_val)

    sort_topology(model.graph)
    remove_unnecessary_cast_node(model.graph)

    return model


def _is_blocked_checker(
    op_block_list: list[str], node_block_list: list[str]
) -> Callable[[NodeProto], bool]:
    op_block_list = set(op_block_list)
    node_block_list = set(node_block_list)

    def is_blocked(node: NodeProto) -> bool:
        return node.op_type in op_block_list or node.name in node_block_list

    return is_blocked


def _modify_graph_for_tensor_info(
    ti: "TensorInfo",
    is_node_blocked: Callable[[NodeProto], bool],
    keep_io_types: bool,
    used_node_names: set[str],
    min_positive_val: float,
    max_finite_val: float,
) -> None:
    if ti.data_type() != FLOAT32:
        return
    desired_type = FLOAT32
    print(f"{ti.name()}: not ti.is_io={not ti.is_io}, keep_io_types={keep_io_types}")
    if not ti.is_io or not keep_io_types:
        print(
            f"{ti.name()}: any(not is_node_blocked(n) for n in ti.io_nodes())={any(not is_node_blocked(n) for n in ti.io_nodes())} len(ti.io_nodes())={len(ti.io_nodes())}"
        )
        if any(not is_node_blocked(n) for n in ti.io_nodes()):
            desired_type = FLOAT16
    print(f"{ti.name()}: {ti.data_type()} -> {desired_type}")
    if ti.data_type() != desired_type and desired_type == FLOAT16:
        print(f"  setting {ti.name()} to {desired_type}")
        if isinstance(ti.proto, TensorProto):
            ti.proto.CopyFrom(
                convert_tensor_float_to_float16(
                    ti.proto, min_positive_val, max_finite_val
                )
            )
        else:
            ti.proto.type.tensor_type.elem_type = FLOAT16
    for output_node in ti.output_nodes:
        node_type = FLOAT32 if is_node_blocked(output_node) else FLOAT16
        print(
            f"  for input tensor {ti.name()}{{tensor_type={ti.data_type()}}} - checking output node {output_node.name}{{is_blocked={is_node_blocked(output_node)}, node_type={node_type}}} Matches={node_type == desired_type}"
        )
        if node_type != desired_type:
            cast_output_node = ti.cast_output_node
            cast_output = ti.cast_output
            if not cast_output_node:
                _create_cast_to(ti, node_type, used_node_names)
                cast_output_node = ti.cast_output_node
                cast_output = ti.cast_output
            for i, input_name in enumerate(output_node.input):
                if input_name == ti.name():
                    print(
                        f"  setting input {i} (input_name={input_name}) of {output_node.name} to {cast_output}"
                    )
                    output_node.input[i] = cast_output
    for input_node in ti.input_nodes:
        input_node_type = FLOAT32 if is_node_blocked(input_node) else FLOAT16
        print(
            f"  for output tensor {ti.name()}{{tensor_type={ti.data_type()}}} - checking input node {input_node.name}{{is_blocked={is_node_blocked(input_node)}, node_type={input_node_type}}}. Matches={input_node_type == desired_type}"
        )
        if input_node_type != desired_type:
            cast_input_node = ti.cast_input_node
            cast_input = ti.cast_input
            if not cast_input_node:
                _create_cast_from(ti, input_node_type, used_node_names)
                cast_input_node = ti.cast_input_node
                cast_input = ti.cast_input
            for i, output_name in enumerate(input_node.output):
                if output_name == ti.name():
                    print(
                        f"  setting output {i} (output_name={output_name}) of {input_node.name} to {cast_input}"
                    )
                    input_node.output[i] = cast_input


def _modify_node_attribute_type(
    node: NodeProto,
    min_positive_val: float,
    max_finite_val: float,
) -> None:
    for attr in node.attribute:
        # Single tensor
        if attr.HasField("t") and attr.t.data_type == onnx_proto.TensorProto.FLOAT:
            attr.t.CopyFrom(
                convert_tensor_float_to_float16(
                    attr.t, min_positive_val, max_finite_val
                )
            )
        # List of tensors
        for t in attr.tensors:
            if t.data_type == onnx_proto.TensorProto.FLOAT:
                t.CopyFrom(
                    convert_tensor_float_to_float16(t, min_positive_val, max_finite_val)
                )


def _create_unique_name(base_name: str, used_node_names: set[str]) -> str:
    idx = 0
    while f"{base_name}_{idx}" in used_node_names:
        idx += 1
    used_node_names.add(f"{base_name}_{idx}")
    return f"{base_name}_{idx}"


def _create_cast_to(ti: "TensorInfo", cast_to: int, used_node_names: set[str]) -> None:
    # Generate unique cast node name
    base_name = f"{ti.name()}_Cast_to_{_float_type_string(cast_to)}"
    cast_name = _create_unique_name(base_name, used_node_names)
    cast_output_tensor_name = _create_unique_name(
        f"{base_name}_output", used_node_names
    )

    # Create cast node
    cast_output = helper.make_node(
        "Cast",
        inputs=[ti.name()],
        outputs=[cast_output_tensor_name],
        name=cast_name,
        to=onnx_proto.TensorProto.FLOAT16
        if cast_to == FLOAT16
        else onnx_proto.TensorProto.FLOAT,
    )
    ti.graph.node.append(cast_output)
    ti.cast_output_node = cast_output
    ti.cast_output = cast_output_tensor_name


def _create_cast_from(
    ti: "TensorInfo", cast_from: int, used_node_names: set[str]
) -> None:
    # Generate unique cast node name
    cast_to = FLOAT32 if cast_from == FLOAT16 else FLOAT16
    base_name = f"{ti.name()}_Cast_to_{_float_type_string(cast_to)}"
    cast_name = _create_unique_name(base_name, used_node_names)
    cast_input_tensor_name = _create_unique_name(f"{base_name}_input", used_node_names)

    # Create cast node
    cast_input_node = helper.make_node(
        "Cast",
        inputs=[cast_input_tensor_name],
        outputs=[ti.name()],
        name=cast_name,
        to=onnx_proto.TensorProto.FLOAT16
        if cast_from == FLOAT32
        else onnx_proto.TensorProto.FLOAT,
    )
    ti.graph.node.append(cast_input_node)
    ti.cast_input_node = cast_input_node
    ti.cast_input = cast_input_tensor_name


@dataclass
class TensorInfo:
    """
    An accumulator for information about tensors in the model. On the "first" pass, these are instantiated, without
    casts, and then a subsequent pass iterates over all of the tensor infos, analyzing the connected nodes, creating and
    rewiring casts as necessary.
    """

    proto: Union[ValueInfoProto, TensorProto]
    graph: onnx.GraphProto
    is_io: bool = False
    is_initializer: bool = False
    input_nodes: List[NodeProto] = field(default_factory=list)
    output_nodes: List[NodeProto] = field(default_factory=list)
    cast_input_node: Optional[NodeProto] = None
    cast_input: Optional[str] = None  # fp32 if this is fp16, and vice versa
    cast_output_node: Optional[NodeProto] = None
    cast_output: Optional[str] = None  # fp32 if this is fp16, and vice versa

    def data_type(self) -> int:
        if isinstance(self.proto, ValueInfoProto):
            return self.proto.type.tensor_type.elem_type
        else:
            return self.proto.data_type

    def set_data_type(self, data_type: int):
        if isinstance(self.proto, ValueInfoProto):
            self.proto.type.tensor_type.elem_type = data_type
        else:
            self.proto.data_type = data_type

    def name(self) -> str:
        return self.proto.name

    def io_nodes(self) -> List[NodeProto]:
        return self.output_nodes + self.input_nodes


def _float_type_string(data_type: int) -> str:
    if data_type == FLOAT32:
        return "float32"
    elif data_type == FLOAT16:
        return "float16"
    else:
        return "other"


def _build_tensor_infos(
    root_graph: onnx.GraphProto,
) -> tuple[List[TensorInfo], List[onnx.GraphProto]]:
    tensor_infos = []
    tensor_to_outputs: dict[str, list[NodeProto]] = defaultdict(list)
    tensor_to_input: dict[str, list[NodeProto]] = defaultdict(list)
    graph_stack = [root_graph]
    all_graphs: list[onnx.GraphProto] = []
    while graph_stack:
        curr_graph = graph_stack.pop()
        all_graphs.append(curr_graph)
        for node in curr_graph.node:
            for input_name in node.input:
                tensor_to_outputs[input_name].append(node)
            for output_name in node.output:
                tensor_to_input[output_name].append(node)
            for attr in node.attribute:
                if attr.g.node:  # single sub-graph
                    graph_stack.append(attr.g)
                for g in attr.graphs:
                    if g.node:
                        graph_stack.append(g)

    for graph in all_graphs:
        is_root = graph == root_graph
        for vi in graph.input:
            tensor_infos.append(
                TensorInfo(
                    proto=vi,
                    graph=graph,
                    is_io=is_root,
                    output_nodes=tensor_to_outputs[vi.name],
                )
            )
        for vi in graph.output:
            print(f"output {vi.name} for graph {graph.name}")
            tensor_infos.append(
                TensorInfo(
                    proto=vi,
                    graph=graph,
                    is_io=is_root,
                    input_nodes=tensor_to_input[vi.name],
                )
            )
        for init in graph.initializer:
            tensor_infos.append(
                TensorInfo(
                    proto=init,
                    graph=graph,
                    is_initializer=True,
                    output_nodes=tensor_to_outputs[init.name],
                )
            )
        for tensor in graph.value_info:
            print(f"value_info {tensor.name} for graph {graph.name}")
            tensor_infos.append(
                TensorInfo(
                    proto=tensor,
                    graph=graph,
                    output_nodes=tensor_to_outputs[tensor.name],
                    input_nodes=tensor_to_input[tensor.name],
                )
            )
    return tensor_infos, all_graphs


def convert_graph_to_float16(
    graph: onnx_proto.GraphProto,
    op_block_list: set,
    node_block_list: set,
    is_top_level: bool,
    keep_io_types: bool,
    min_positive_val: float,
    max_finite_val: float,
):
    """
    Converts a single graph's nodes/initializers/values to float16, except for blocked ops/nodes.
    Inserts boundary casts only where float16 nodes connect to float32 nodes.
    """

    # 1) Mark each node as blocked or not
    blocked_node_names = mark_blocked_nodes(graph, op_block_list, node_block_list)

    # 2) Convert node attributes (tensors -> float16) for unblocked nodes
    process_tensor_in_node(graph, blocked_node_names, min_positive_val, max_finite_val)

    # 3) Convert initializers to float16 if exclusively used by unblocked nodes
    process_initializers(graph, blocked_node_names, min_positive_val, max_finite_val)

    # 4) Adjust graph inputs/outputs
    update_graph_io_dtypes(graph, blocked_node_names, is_top_level, keep_io_types)

    # 5) Insert boundary casts for node↔node edges
    insert_boundary_casts(graph, blocked_node_names)

    # 6) Convert leftover non-blocked ValueInfo to float16
    convert_value_infos(graph, blocked_node_names)

    # 7) Now handle sub-graphs for control-flow ops
    for node in graph.node:
        # If node is blocked => skip sub-graph conversion
        if node.name in blocked_node_names or node.op_type in op_block_list:
            continue

        # Otherwise, node is unblocked => any sub-graph belongs to the float16 domain
        # Insert boundary casts between node <-> sub-graph
        for attr in node.attribute:
            if attr.g.node:  # single sub-graph
                insert_if_subgraph_boundary_casts(
                    graph, node, attr.g, blocked_node_names
                )
                # Recurse
                convert_graph_to_float16(
                    attr.g,
                    op_block_list,
                    node_block_list,
                    is_top_level=False,  # sub-graph is not top-level
                    keep_io_types=False,  # sub-graph doesn't keep model IO types
                    min_positive_val=min_positive_val,
                    max_finite_val=max_finite_val,
                )
            for g in attr.graphs:
                if g.node:
                    insert_if_subgraph_boundary_casts(
                        graph, node, g, blocked_node_names
                    )
                    convert_graph_to_float16(
                        g,
                        op_block_list,
                        node_block_list,
                        is_top_level=False,
                        keep_io_types=False,
                        min_positive_val=min_positive_val,
                        max_finite_val=max_finite_val,
                    )

    # 8) Clean up casts + sort
    sort_topology(graph)
    # remove_unnecessary_cast_node(graph)

    # Print node info after conversion
    print("\nAfter conversion:")
    for node in graph.node:
        print(f"Node: {node.name} (op={node.op_type})")
        if node.op_type == "Cast":
            print(f"  Cast to: {node.attribute[0].i}")
        for i, input_name in enumerate(node.input):
            print(f"  Input {i}: {input_name}")
        for i, output_name in enumerate(node.output):
            print(f"  Output {i}: {output_name}")


def _npfloat16_to_int(np_list):
    """
    Convert numpy float16 to python int.

    :param np_list: numpy float16 list
    :return int_list: python int list
    """
    return [int(bin(_.view("H"))[2:].zfill(16), 2) for _ in np_list]


def convert_np_to_float16(np_array, min_positive_val=1e-7, max_finite_val=1e4):
    """
    Convert float32 numpy array to float16 without changing sign or finiteness.
    Positive values less than min_positive_val are mapped to min_positive_val.
    Positive finite values greater than max_finite_val are mapped to max_finite_val.
    Similar for negative values. NaN, 0, inf, and -inf are unchanged.
    """

    def between(a, b, c):
        return np.logical_and(a < b, b < c)

    if np_array[np.where(np_array > 0)].shape[0] > 0:
        pos_max = np_array[np.where(np_array > 0)].max()
        pos_min = np_array[np.where(np_array > 0)].min()

        if pos_max >= max_finite_val:
            warnings.warn(
                "the float32 number {} will be truncated to {}".format(
                    pos_max, max_finite_val
                )
            )

        if pos_min <= min_positive_val:
            warnings.warn(
                "the float32 number {} will be truncated to {}".format(
                    pos_min, min_positive_val
                )
            )

    if np_array[np.where(np_array < 0)].shape[0] > 0:
        neg_max = np_array[np.where(np_array < 0)].max()
        neg_min = np_array[np.where(np_array < 0)].min()

        if neg_min <= -max_finite_val:
            warnings.warn(
                "the float32 number {} will be truncated to {}".format(
                    neg_min, -max_finite_val
                )
            )

        if neg_max >= -min_positive_val:
            warnings.warn(
                "the float32 number {} will be truncated to {}".format(
                    neg_max, -min_positive_val
                )
            )

    np_array = np.where(
        between(0, np_array, min_positive_val), min_positive_val, np_array
    )
    np_array = np.where(
        between(-min_positive_val, np_array, 0), -min_positive_val, np_array
    )
    np_array = np.where(
        between(max_finite_val, np_array, float("inf")), max_finite_val, np_array
    )
    np_array = np.where(
        between(float("-inf"), np_array, -max_finite_val), -max_finite_val, np_array
    )
    return np.float16(np_array)


def convert_tensor_float_to_float16(
    tensor: TensorProto, min_positive_val=1e-7, max_finite_val=1e4
):
    """
    Convert tensor float to float16.

    :param tensor: TensorProto object
    :return tensor_float16: converted TensorProto object

    Example:

    ::

        from onnxmltools.utils.float16_converter import convert_tensor_float_to_float16
        new_tensor = convert_tensor_float_to_float16(tensor)

    """
    if not isinstance(tensor, onnx_proto.TensorProto):
        raise ValueError(
            "Expected input type is an ONNX TensorProto but got %s" % type(tensor)
        )

    if tensor.data_type == onnx_proto.TensorProto.FLOAT:
        tensor.data_type = onnx_proto.TensorProto.FLOAT16
        # convert float_data (float type) to float16 and write to int32_data
        if tensor.float_data:
            float16_data = convert_np_to_float16(
                np.array(tensor.float_data), min_positive_val, max_finite_val
            )
            int_list = _npfloat16_to_int(float16_data)
            tensor.int32_data[:] = int_list
            tensor.float_data[:] = []
        # convert raw_data (bytes type)
        if tensor.raw_data:
            # convert n.raw_data to float
            float32_list = np.fromstring(tensor.raw_data, dtype="float32")
            # convert float to float16
            float16_list = convert_np_to_float16(
                float32_list, min_positive_val, max_finite_val
            )
            # convert float16 to bytes and write back to raw_data
            tensor.raw_data = float16_list.tostring()
    return tensor


def make_value_info_from_tensor(tensor):
    shape = numpy_helper.to_array(tensor).shape
    return helper.make_tensor_value_info(tensor.name, tensor.data_type, shape)


def initial_checking(model, disable_shape_infer):
    func_infer_shape = None
    if not disable_shape_infer and pv.Version(onnx.__version__) >= pv.Version("1.2"):
        try:
            from onnx.shape_inference import infer_shapes

            func_infer_shape = infer_shapes
        finally:
            pass

    if not isinstance(model, onnx_proto.ModelProto):
        raise ValueError(
            "Expected model type is an ONNX ModelProto but got %s" % type(model)
        )

    if func_infer_shape is not None:
        model = func_infer_shape(model)

    is_fp16_ready_flag = check_if_fp16_ready(model.graph)

    return model, func_infer_shape, is_fp16_ready_flag


def convert_value_infos(graph: onnx_proto.GraphProto, blocked_node_names: set):
    """
    Convert ValueInfo dtypes to float16 for unblocked edges.
    """
    print("\nConverting ValueInfos:")

    # Build map: output_name -> node_name
    output_to_node = {}
    for node in graph.node:
        for o_name in node.output:
            output_to_node[o_name] = node.name

    # Helper function to convert a value info if its producer is not blocked
    def convert_if_unblocked(value_info):
        producer_name = output_to_node.get(value_info.name, None)
        print(f"ValueInfo: {value_info.name}")
        print(f"  Producer: {producer_name}")
        print(f"  Current type: {value_info.type.tensor_type.elem_type}")

        # If there's no known producer, it might be a graph input, skip it here
        if producer_name is None:
            print("  No producer found - skipping")
            return

        # If producer is blocked => remain float32
        if producer_name not in blocked_node_names:
            # Convert to float16
            if value_info.type.tensor_type.elem_type == onnx_proto.TensorProto.FLOAT:
                print("  Converting to float16")
                value_info.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT16
        else:
            print("  Producer is blocked - keeping float32")

    # Convert both value_info and output types
    for value_info in graph.value_info:
        convert_if_unblocked(value_info)

    # Also convert output types if their producer is not blocked
    for output in graph.output:
        convert_if_unblocked(output)


def update_graph_io_dtypes(
    graph: onnx_proto.GraphProto,
    blocked_node_names: set,
    is_top_level: bool,
    keep_io_types: bool,
):
    """
    Adjust the graph I/O dtypes if needed. If keep_io_types=True and we are at top-level,
    we do NOT forcibly convert them to float16. Otherwise, if !keep_io_types at top-level,
    we do convert them to float16. For sub-graphs, we typically want them to match the parent's domain,
    so we do the usual logic.
    """

    if is_top_level and keep_io_types:
        # 1) Do NOT forcibly change top-level inputs/outputs to float16
        #    We leave them as float32 (or whatever they already are).
        # 2) That’s it. We'll insert boundary casts as needed elsewhere.
        return

    if is_top_level and not keep_io_types:
        # Force all top-level inputs to float16
        for vi in graph.input:
            if vi.type.tensor_type.elem_type == onnx_proto.TensorProto.FLOAT:
                vi.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT16

        # Force all top-level outputs to float16
        for vi in graph.output:
            if vi.type.tensor_type.elem_type == onnx_proto.TensorProto.FLOAT:
                vi.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT16

    # For both top-level (if not keep_io_types) or sub-graphs, we still check
    # if an input or output is definitely consumed/produced by a blocked node.
    # If so, keep it float32. Otherwise, we can convert to float16.

    # Build consumer map
    input_to_nodes = {}
    for node in graph.node:
        for inp_name in node.input:
            input_to_nodes.setdefault(inp_name, []).append(node.name)

    # For each graph input that is currently float32, see if all consumers are unblocked
    for vi in graph.input:
        if vi.type.tensor_type.elem_type == onnx_proto.TensorProto.FLOAT:
            consumers = input_to_nodes.get(vi.name, [])
            # If any consumer is a blocked node => remain float32
            # Otherwise, convert to float16
            if consumers and all(c not in blocked_node_names for c in consumers):
                vi.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT16

    # Build producer map
    output_to_producer = {}
    for node in graph.node:
        for o_name in node.output:
            output_to_producer[o_name] = node.name

    # For each graph output that is currently float32, see if its producer is unblocked
    for vi in graph.output:
        if vi.type.tensor_type.elem_type == onnx_proto.TensorProto.FLOAT:
            producer_name = output_to_producer.get(vi.name, None)
            if producer_name is not None and (producer_name not in blocked_node_names):
                vi.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT16


def insert_if_subgraph_boundary_casts(
    parent_graph: onnx_proto.GraphProto,
    parent_node: onnx_proto.NodeProto,
    sub_graph: onnx_proto.GraphProto,
    blocked_node_names: set,
):
    """
    Inserts cast ops in the *parent_graph* so that the sub-graph's inputs/outputs
    have the correct float16 vs. float32 type, matching the parent's domain.

    Because we've already decided `parent_node` is "unblocked" => we want sub-graph in float16.
    If the parent node is blocked => sub-graph remains float32 and we skip conversion.

    Note:
      - 'parent_graph' is the ONNX GraphProto that contains 'parent_node'.
      - 'sub_graph' is the If/Loop sub-graph we are about to recurse into.
      - We do boundary casts here if there's a mismatch in data type.
    """

    # First, explicitly handle references in the sub-graph that come from the parent
    _resolve_subgraph_external_inputs(
        parent_graph, parent_node, sub_graph, blocked_node_names
    )

    # 1) Check if parent_node is blocked or not
    parent_is_blocked = parent_node.name in blocked_node_names
    desired_elem = (
        onnx_proto.TensorProto.FLOAT
        if parent_is_blocked
        else onnx_proto.TensorProto.FLOAT16
    )

    # 2) We'll gather all used names in the parent graph, to avoid collisions
    used_names = set()
    for n in parent_graph.node:
        used_names.update(n.input)
        used_names.update(n.output)
    for vi in parent_graph.value_info:
        used_names.add(vi.name)
    for vi in parent_graph.input:
        used_names.add(vi.name)
    for vi in parent_graph.output:
        used_names.add(vi.name)
    for init in parent_graph.initializer:
        used_names.add(init.name)

    # 3) For each sub_graph input:
    #    - If it's float32 or float16 but does NOT match 'desired_elem',
    #      we rename the sub_graph input and insert a cast node in the parent graph.
    for sg_input in sub_graph.input:
        dt = sg_input.type.tensor_type.elem_type
        if dt not in (onnx_proto.TensorProto.FLOAT, onnx_proto.TensorProto.FLOAT16):
            # e.g. bool, int, etc. No cast needed
            continue

        if dt == desired_elem:
            continue  # already matches

        # We have a mismatch => rename the sub_graph input and create a cast in the parent graph
        old_name = sg_input.name
        base_new_name = old_name + "_subgraph_casted"
        new_name = base_new_name
        idx = 0
        while new_name in used_names:
            new_name = base_new_name + f"_{idx}"
            idx += 1
        used_names.add(new_name)

        # (A) rename sub_graph input
        sg_input.name = new_name

        # (B) create a cast node in parent graph from old_name -> new_name
        cast_node_name = old_name + "_parent_cast_subgraph_in"
        cast_node = helper.make_node(
            "Cast",
            inputs=[old_name],
            outputs=[new_name],
            name=cast_node_name,
            to=desired_elem,
        )
        parent_graph.node.append(cast_node)

        # (C) optionally add ValueInfo for the new_name in the parent graph
        new_vi = parent_graph.value_info.add()
        new_vi.name = new_name
        new_vi.type.tensor_type.elem_type = desired_elem
        # shape is unknown unless you do something more elaborate

    # 4) For sub_graph outputs:
    #    In ONNX “If”, each sub-graph must produce a tensor that exactly matches
    #    the If node’s output type. So if the parent is unblocked => float16,
    #    the sub-graph must produce float16. If there's a mismatch,
    #    ONNXRuntime will fail.
    #
    #    We'll simply set sub_graph.output to 'desired_elem' so that
    #    the sub-graph is consistent with the parent's domain.
    #    If you do partial blocking *inside* the sub-graph, it should handle
    #    internal boundary casts. But the final output must match parent's domain.
    #
    #    If the parent is blocked => float32, we typically skip converting sub-graph,
    #    so this is also consistent in that scenario.
    for sg_output in sub_graph.output:
        dt = sg_output.type.tensor_type.elem_type
        if dt in (onnx_proto.TensorProto.FLOAT, onnx_proto.TensorProto.FLOAT16):
            if dt != desired_elem:
                sg_output.type.tensor_type.elem_type = desired_elem

    # Done. The sub_graph boundary is now consistent with the parent's domain.


def mark_blocked_nodes(
    graph: onnx_proto.GraphProto, op_block_list: set, node_block_list: set
) -> set[str]:
    """
    Return a set of node.names that must remain float32.
    A node is blocked if:
    - Its op_type is in op_block_list (ops to keep in float32)
    - OR its name is in node_block_list
    """
    blocked = set(
        [
            node.name
            for node in graph.node
            if node.name in node_block_list or node.op_type in op_block_list
        ]
    )
    return blocked


def insert_boundary_casts(graph: onnx_proto.GraphProto, blocked_node_names: set):
    """
    Insert cast ops only on edges where a float16 (unblocked) node output
    feeds a float32 (blocked) node input, or vice versa.

    We generate a unique cast output name each time, to avoid collisions
    in large or repeated networks.
    """
    # Collect info about which outputs come from which node
    name_to_node_dict = {}
    for node in graph.node:
        for out_name in node.output:
            name_to_node_dict[out_name] = node.name

    # Keep track of existing tensor names to avoid collisions
    used_tensor_names = set()
    for node in graph.node:
        used_tensor_names.update(node.input)
        used_tensor_names.update(node.output)
    # Also consider graph inputs/outputs
    for vi in graph.input:
        used_tensor_names.add(vi.name)
    for vi in graph.output:
        used_tensor_names.add(vi.name)
    for vi in graph.value_info:
        used_tensor_names.add(vi.name)
    for init in graph.initializer:
        used_tensor_names.add(init.name)

    new_nodes = []
    # We'll increment a counter each time we need to disambiguate
    boundary_cast_counter = 0

    for node in graph.node:
        cur_node_blocked = node.name in blocked_node_names
        for i, inp_name in enumerate(node.input):
            if inp_name not in name_to_node_dict:
                # Possibly a graph input or initializer
                continue

            upstream_node_name = name_to_node_dict[inp_name]
            upstream_blocked = upstream_node_name in blocked_node_names
            if upstream_blocked != cur_node_blocked:
                # We have a boundary crossing: float32 -> float16 or float16 -> float32

                # We'll build a unique cast node name
                cast_node_name = (
                    f"Cast_boundary_{upstream_node_name}_to_{node.name}_{i}"
                )
                # Build a unique output name to avoid collisions
                cast_output_name_base = inp_name + "_boundary_cast"
                cast_output_name = cast_output_name_base
                while cast_output_name in used_tensor_names:
                    cast_output_name = (
                        f"{cast_output_name_base}_{boundary_cast_counter}"
                    )
                    boundary_cast_counter += 1

                # Mark it used
                used_tensor_names.add(cast_output_name)

                # Choose the 'to' type based on the current node's status
                to_type = (
                    onnx_proto.TensorProto.FLOAT16
                    if not cur_node_blocked
                    else onnx_proto.TensorProto.FLOAT
                )

                # Create the new cast node
                cast_node = helper.make_node(
                    "Cast",
                    inputs=[inp_name],
                    outputs=[cast_output_name],
                    name=cast_node_name,
                    to=to_type,
                )
                new_nodes.append(cast_node)

                # Redirect the node input to the cast's output
                node.input[i] = cast_output_name

    # Append new cast nodes at the end
    graph.node.extend(new_nodes)


def _resolve_subgraph_external_inputs(
    parent_graph: onnx_proto.GraphProto,
    parent_node: onnx_proto.NodeProto,
    sub_graph: onnx_proto.GraphProto,
    blocked_node_names: set,
):
    """
    For each node in sub_graph, find any 'input' that isn't local to sub_graph.
    That means it must come from the parent graph. Insert a sub_graph input for it,
    and insert a boundary cast in the parent graph so that sub_graph sees the right float type.
    """

    # 1) Build a set of local names in the sub_graph
    local_names = set()
    for vi in itertools.chain(sub_graph.input, sub_graph.initializer):
        local_names.add(vi.name)
    for node in sub_graph.node:
        for out_name in node.output:
            local_names.add(out_name)

    # 2) Identify external references
    external_refs = []
    for node in sub_graph.node:
        for in_name in node.input:
            if in_name and (in_name not in local_names):
                external_refs.append(in_name)
    if not external_refs:
        return  # nothing to do

    # 3) The sub_graph is either "unblocked" float16 or "blocked" float32
    #    If parent_node is blocked => sub_graph must remain float32
    parent_is_blocked = parent_node.name in blocked_node_names
    desired_elem = (
        onnx_proto.TensorProto.FLOAT
        if parent_is_blocked
        else onnx_proto.TensorProto.FLOAT16
    )

    # 4) Prepare a set of used names in the parent graph to avoid collisions
    used_parent_names = set()
    for n in parent_graph.node:
        used_parent_names.update(n.input)
        used_parent_names.update(n.output)
    for vi in parent_graph.input:
        used_parent_names.add(vi.name)
    for vi in parent_graph.output:
        used_parent_names.add(vi.name)
    for vi in parent_graph.value_info:
        used_parent_names.add(vi.name)
    for init in parent_graph.initializer:
        used_parent_names.add(init.name)

    # 5) For each external reference, add sub_graph input + Cast in parent
    for external_name in set(external_refs):
        # If it's already declared as input, skip
        # (Possible if multiple sub-graphs share that input.)
        if any(vi.name == external_name for vi in sub_graph.input):
            continue

        # (A) rename usage inside sub_graph (to something local)
        new_name = external_name + "_subgraph_in"
        suffix_idx = 0
        while new_name in local_names:
            new_name = f"{external_name}_subgraph_in_{suffix_idx}"
            suffix_idx += 1
        local_names.add(new_name)

        # We must rewrite the sub-graph node input references
        for node in sub_graph.node:
            for i, inp in enumerate(node.input):
                if inp == external_name:
                    node.input[i] = new_name

        # (B) Add a new Graph input to sub_graph for `new_name`
        new_input = sub_graph.input.add()
        new_input.name = new_name
        # The shape might be unknown here, but at least set the type
        new_input.type.tensor_type.elem_type = desired_elem

        # (C) Insert a boundary Cast in the parent graph from external_name -> new_name
        cast_node_name = f"{external_name}_to_subgraph_cast"
        cast_output_name = new_name
        # For safety, we can also rename cast_output_name if it collides in the parent
        if cast_output_name in used_parent_names:
            idx = 0
            while cast_output_name in used_parent_names:
                cast_output_name = f"{new_name}_{idx}"
                idx += 1
            new_input.name = cast_output_name
        used_parent_names.add(cast_output_name)

        cast_node = helper.make_node(
            "Cast",
            inputs=[external_name],
            outputs=[cast_output_name],
            name=cast_node_name,
            to=desired_elem,
        )
        parent_graph.node.append(cast_node)


def insert_io_casts_for_keep_io_types(
    graph: onnx_proto.GraphProto,
    blocked_node_names: set,
):
    """
    If keep_io_types=True, the top-level graph inputs/outputs remain float32.
    But unblocked (float16) nodes that consume/produce those tensors need boundary casts.
    """

    # 1) Handle top-level inputs that remain float32 but feed unblocked float16 nodes
    #    - If an input goes to at least one unblocked consumer, insert one float32->float16 Cast
    #    - Blocked consumers stay on float32

    # Mapping input_name -> list of nodes that consume it
    input_to_nodes = {}
    for node in graph.node:
        for inp_name in node.input:
            input_to_nodes.setdefault(inp_name, []).append(node)

    # Keep track of used tensor names to avoid collisions
    used_names = set()
    for node in graph.node:
        used_names.update(node.input)
        used_names.update(node.output)
    for vi in graph.input:
        used_names.add(vi.name)
    for vi in graph.output:
        used_names.add(vi.name)
    for vi in graph.value_info:
        used_names.add(vi.name)
    for init in graph.initializer:
        used_names.add(init.name)

    new_nodes = []
    for g_input in graph.input:
        if g_input.type.tensor_type.elem_type != onnx_proto.TensorProto.FLOAT:
            continue  # Only worry about float32 inputs
        consumers = input_to_nodes.get(g_input.name, [])
        # Partition consumers: some may be blocked (want float32), some unblocked (now float16)
        unblocked_consumers = [c for c in consumers if c.name not in blocked_node_names]
        # If no unblocked consumers, no cast needed
        if not unblocked_consumers:
            continue

        # Create a single cast node from input float32 → float16
        cast_name = f"Cast_input_{g_input.name}_to_fp16"
        cast_output_name = f"{g_input.name}_fp16"
        # Ensure uniqueness
        base_out = cast_output_name
        counter = 0
        while cast_output_name in used_names:
            cast_output_name = f"{base_out}_{counter}"
            counter += 1
        used_names.add(cast_output_name)

        cast_node = helper.make_node(
            "Cast",
            inputs=[g_input.name],
            outputs=[cast_output_name],
            name=cast_name,
            to=onnx_proto.TensorProto.FLOAT16,
        )
        new_nodes.append(cast_node)

        # Redirect all unblocked consumers to use the cast node output
        for node_ in unblocked_consumers:
            for i, inp_name in enumerate(node_.input):
                if inp_name == g_input.name:
                    node_.input[i] = cast_output_name

    # 2) Handle top-level outputs that remain float32 but might be produced by unblocked float16 nodes
    #    - If an output is float32, but is produced by an unblocked node, insert float16->float32 Cast.

    # Build map: output_tensor_name -> producing node
    output_to_node = {}
    for node in graph.node:
        for o_name in node.output:
            output_to_node[o_name] = node.name

    for g_output in graph.output:
        if g_output.type.tensor_type.elem_type != onnx_proto.TensorProto.FLOAT:
            continue  # Only worry about float32 outputs
        producer = output_to_node.get(g_output.name, None)
        if producer is None:
            continue  # Not produced by any node, might be an unused graph input or something
        if producer in blocked_node_names:
            continue  # Produced by a blocked (float32) node => no cast needed

        # So the node is unblocked => node outputs float16, but we want top-level float32.
        # Insert a cast node float16 -> float32
        cast_name = f"Cast_output_{g_output.name}_to_fp32"
        cast_input_name = g_output.name
        cast_output_name = f"{g_output.name}_fp32"

        base_out = cast_output_name
        counter = 0
        while cast_output_name in used_names:
            cast_output_name = f"{base_out}_{counter}"
            counter += 1
        used_names.add(cast_output_name)

        cast_node = helper.make_node(
            "Cast",
            inputs=[cast_input_name],
            outputs=[cast_output_name],
            name=cast_name,
            to=onnx_proto.TensorProto.FLOAT,
        )
        new_nodes.append(cast_node)

        # Now rename the top-level output to the cast's output
        g_output.name = cast_output_name

    # Finally, insert these new cast nodes at the end of graph.node
    if new_nodes:
        graph.node.extend(new_nodes)


def process_tensor_in_node(
    graph: onnx_proto.GraphProto,
    blocked_node_names: set,
    min_positive_val: float,
    max_finite_val: float,
):
    """
    For each node not blocked, convert all float32 attributes/tensors to float16.
    """
    for node in graph.node:
        if node.name in blocked_node_names:
            # Skip converting any attributes; remain float32
            continue
        # Convert float32 --> float16 in attributes
        for attr in node.attribute:
            # Single tensor
            if attr.HasField("t") and attr.t.data_type == onnx_proto.TensorProto.FLOAT:
                attr.t.CopyFrom(
                    convert_tensor_float_to_float16(
                        attr.t, min_positive_val, max_finite_val
                    )
                )
            # List of tensors
            for t in attr.tensors:
                if t.data_type == onnx_proto.TensorProto.FLOAT:
                    t.CopyFrom(
                        convert_tensor_float_to_float16(
                            t, min_positive_val, max_finite_val
                        )
                    )


def process_initializers(
    graph: onnx_proto.GraphProto,
    blocked_node_names: set,
    min_positive_val: float,
    max_finite_val: float,
):
    """
    Convert float32 initializers to float16 if all consumer nodes are unblocked.
    """
    # Figure out which nodes consume each initializer
    init_name_to_consumers = {init.name: [] for init in graph.initializer}
    for node in graph.node:
        for input_name in node.input:
            if input_name in init_name_to_consumers:
                init_name_to_consumers[input_name].append(node.name)

    # Decide which inits can be converted
    for init in graph.initializer:
        consumer_list = init_name_to_consumers[init.name]
        if not consumer_list:
            # Not consumed by any node => safe to convert
            can_convert = True
        else:
            # If any consumer is blocked => keep it float32
            can_convert = all(c not in blocked_node_names for c in consumer_list)

        if can_convert and init.data_type == onnx_proto.TensorProto.FLOAT:
            convert_tensor_float_to_float16(init, min_positive_val, max_finite_val)


def sort_graph_node(graph_proto):
    # find the "first" node in Nodes that its input is not any node's output
    def find_first_node(output2node_dict):
        for node in org_nodes:
            is_not_first_node = any(item in output2node_dict for item in node.input)
            if not is_not_first_node:
                return node
        return None

    # remove the node from output2node_dict using output as key
    def remove_first_node_from_dict2(first_node):
        for output in first_node.output:
            if output in output2node_dict:
                del output2node_dict[output]

    org_nodes = graph_proto.node
    # create a dict to store output as key and node as value
    output2node_dict = {}
    for node in org_nodes:
        for output in node.output:
            output2node_dict[output] = node

    # save the final node after sorted
    sorted_node = []
    # traverse the Nodes to find the first node
    while len(output2node_dict) > 0:
        first_node = find_first_node(output2node_dict)
        sorted_node.append(first_node)
        remove_first_node_from_dict2(first_node)
        # del node from original nodes list to avoid duplicate traverse
        org_nodes.remove(first_node)

    for new_node in sorted_node:
        graph_proto.node.extend([new_node])


# The input graph should be mode.graph
# Recursevly sort the topology for each sub-graph
def sort_topology(graph_proto):
    assert isinstance(graph_proto, onnx_proto.GraphProto)
    sort_graph_node(graph_proto)  # sort global graph
    for node in graph_proto.node:
        for attr in node.attribute:
            if isinstance(attr.g, onnx_proto.GraphProto) and len(attr.g.node) > 0:
                sort_topology(attr.g)  # sort sub-graph
            for g in attr.graphs:
                if isinstance(g, onnx_proto.GraphProto):
                    sort_topology(g)  # sort sub-graph


def remove_unnecessary_cast_node(graph_proto: onnx_proto.GraphProto):
    # 1. find all cast nodes in the graph
    cast_node_list = []
    input_name_to_cast_node_dict = {}
    output_name_to_cast_node_dict = {}
    # using name as key to point to a node. because node object cannot be key
    name_to_node_dict = {}
    for node in graph_proto.node:
        if node.op_type == "Cast":
            # if node.name not in ["graph_input_cast0", "graph_output_cast0"]:
            cast_node_list.append(node)

            name_to_node_dict[node.name] = node
            for input_name in node.input:
                input_name_to_cast_node_dict[input_name] = node
            for output_name in node.output:
                output_name_to_cast_node_dict[output_name] = node

    # 2. find upstream and downstream node of the cast node
    cast_node_upstream_dict = {}  # mapping cast node(name) to its upstream node
    cast_node_downstream_dict = {}  # mapping cast node(name) to its downstream node
    for current_node in graph_proto.node:
        # find the downstream node(s)
        for input_name in current_node.input:
            if input_name in output_name_to_cast_node_dict:
                # found the downstream node of the cast node, might be multiple
                cast_node = output_name_to_cast_node_dict[input_name]
                if cast_node.name not in cast_node_downstream_dict:
                    cast_node_downstream_dict[cast_node.name] = current_node
                else:  # already exists one downstream node, make it a list
                    existing_downstream_nodes = cast_node_downstream_dict[
                        cast_node.name
                    ]
                    if isinstance(existing_downstream_nodes, list):
                        existing_downstream_nodes.append(current_node)
                    else:  # make a list
                        existing_downstream_nodes = [
                            existing_downstream_nodes,
                            current_node,
                        ]
                        cast_node_downstream_dict[cast_node.name] = (
                            existing_downstream_nodes
                        )
        # find the upstream node
        for output_name in current_node.output:
            if output_name in input_name_to_cast_node_dict:
                # found the upstream node of the cast node, should be unique
                cast_node = input_name_to_cast_node_dict[output_name]
                cast_node_upstream_dict[cast_node.name] = current_node

    # 3. remove the cast node which upstream is 'Constant'
    for cast_node_name, upstream_node in cast_node_upstream_dict.items():
        cast_node = name_to_node_dict[cast_node_name]
        if upstream_node.op_type == "Constant":
            cast_node_list.remove(cast_node)

    # 4. find the cast(to16) node which downstream is Cast(to32)
    remove_candidate = []
    for cast_node_name, downstream_node in cast_node_downstream_dict.items():
        cast_node = name_to_node_dict[cast_node_name]
        if isinstance(downstream_node, list):
            for dn in downstream_node:
                if (
                    dn.op_type == "Cast"
                    and dn.attribute[0].i == 32
                    and cast_node.attribute[0].i == 16
                    and dn in cast_node_list
                    and cast_node in cast_node_list
                ):
                    remove_candidate.append((cast_node, dn))
        else:
            if (
                downstream_node.op_type == "Cast"
                and cast_node.attribute[0].i == 10
                and downstream_node.attribute[0].i == 1
                and downstream_node in cast_node_list
                and cast_node in cast_node_list
            ):
                remove_candidate.append((cast_node, downstream_node))

    # 5. change the connection of "upstream->cast16->cast32->downstream" to "upstream->downstream"
    for cast_node_pair in remove_candidate:
        first_cast_node = cast_node_pair[0]
        second_cast_node = cast_node_pair[1]
        upstream_node = cast_node_upstream_dict.get(first_cast_node.name)
        downstream_node = cast_node_downstream_dict.get(second_cast_node.name)
        if upstream_node is None and downstream_node is not None:
            # The upstream_node should be graph input
            out = first_cast_node.input[0]
            for i, input_name in enumerate(downstream_node.input):
                for output_name in second_cast_node.output:
                    if input_name == output_name:
                        # change the input as the upstream node's output
                        downstream_node.input[i] = out
        elif upstream_node is not None and downstream_node is None:
            raise ValueError(
                "The downstream node of the second cast node should be graph output"
            )
        else:
            # find the upstream node's output to first_cast_node
            out = None
            for output_name in upstream_node.output:
                if output_name == first_cast_node.input[0]:
                    out = output_name
                    break
            # find the downstream node's input as second_cast_node's output
            for i, input_name in enumerate(downstream_node.input):
                for output_name in second_cast_node.output:
                    if input_name == output_name:
                        # change the input as the upstream node's output
                        downstream_node.input[i] = out

    # 6. remove the cast node pair
    for cast_node_pair in remove_candidate:
        graph_proto.node.remove(cast_node_pair[0])
        graph_proto.node.remove(cast_node_pair[1])


# Check if the model is already converted to float16
def check_if_fp16_ready(graph_proto):
    # Check graph input and ouput
    is_value_info_fp16 = False
    for value_info in itertools.chain(
        graph_proto.output, graph_proto.input, graph_proto.value_info
    ):
        if value_info.type.tensor_type.elem_type == onnx_proto.TensorProto.FLOAT16:
            is_value_info_fp16 = True
            break

    # Check initializer
    is_initializer_fp16 = False
    for initializer in graph_proto.initializer:
        if initializer.data_type == onnx_proto.TensorProto.FLOAT16:
            is_initializer_fp16 = True
            break

    # Check cast node
    has_cast_node_fp16 = False
    for node in graph_proto.node:
        if node.op_type == "Cast" and node.attribute[0].i == FLOAT16:
            has_cast_node_fp16 = True
            break

    # Any of above flags is True, return True
    if is_value_info_fp16 or is_initializer_fp16 or has_cast_node_fp16:
        return True  # already converted to float16
    else:
        return False  # not converted to float16 yet
