# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###########################################################################

import itertools
import warnings

import numpy as np
import onnx
import packaging.version as pv
from onnx import helper, numpy_helper
from onnx import onnx_pb as onnx_proto

FLOAT32 = 1
FLOAT16 = 10


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


def convert_tensor_float_to_float16(tensor, min_positive_val=1e-7, max_finite_val=1e4):
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

    # Recursively convert all graphs (the main graph and any sub-graphs, e.g. in If/Loop/Scan)
    graph_stack = [(model.graph, True)]  # (graph, is_top_level)
    while graph_stack:
        curr_graph, is_top_level = graph_stack.pop()
        convert_graph_to_float16(
            curr_graph,
            op_block_list,
            node_block_list,
            is_top_level,
            keep_io_types,
            min_positive_val,
            max_finite_val,
        )
        # Gather sub-graphs to process
        for node in curr_graph.node:
            for attr in node.attribute:
                if attr.g.node:  # single GraphProto
                    graph_stack.append((attr.g, False))
                for g in attr.graphs:  # list of GraphProto
                    if g.node:
                        graph_stack.append((g, False))

    # Sort and remove unneeded casts
    sort_topology(model.graph)
    remove_unnecessary_cast_node(model.graph)

    return model


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

    :param graph: the GraphProto to modify in-place
    :param op_block_list: set of op_types to keep in float32
    :param node_block_list: set of node names to keep in float32
    :param is_top_level: True if this is the model's top-level graph
    :param keep_io_types: if True, do not alter input/output dtypes of the top-level graph
    :param min_positive_val: clamp for small positive floats
    :param max_finite_val: clamp for large magnitude floats
    """

    # 1) Mark each node as blocked or not
    blocked_node_names = mark_blocked_nodes(graph, op_block_list, node_block_list)

    # 2) Convert attributes for unblocked nodes to fp16
    #    (We won't force a cast around blocked nodes; we simply keep them in fp32.)
    process_tensor_in_node(graph, blocked_node_names, min_positive_val, max_finite_val)

    # 3) Convert initializers to fp16 if they are exclusively used by unblocked nodes
    process_initializers(graph, blocked_node_names, min_positive_val, max_finite_val)

    # 4) Adjust graph inputs/outputs
    #    - If top-level and keep_io_types=True, do NOT alter input/output dtypes.
    #    - Otherwise, we switch them to fp16 unless they feed or come from a blocked node.
    update_graph_io_dtypes(graph, blocked_node_names, is_top_level, keep_io_types)

    # 5) Insert casts only on boundaries between float16 vs float32 nodes
    insert_boundary_casts(graph, blocked_node_names)

    # 6) Convert leftover non-blocked ValueInfo to float16
    #    (Excluding the ones specifically used by blocked ops)
    convert_value_infos(graph, blocked_node_names)


def convert_value_infos(graph: onnx_proto.GraphProto, blocked_node_names: set):
    """
    Convert ValueInfo dtypes to float16 for unblocked edges.
    """
    # Build map: output_name -> node_name
    output_to_node = {}
    for node in graph.node:
        for o_name in node.output:
            output_to_node[o_name] = node.name

    for value_info in graph.value_info:
        producer_name = output_to_node.get(value_info.name, None)
        # If there's no known producer, it might be a graph input, skip it here
        if producer_name is None:
            continue
        # If producer is blocked => remain float32
        if producer_name not in blocked_node_names:
            # Convert to float16
            if value_info.type.tensor_type.elem_type == onnx_proto.TensorProto.FLOAT:
                value_info.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT16


def update_graph_io_dtypes(
    graph: onnx_proto.GraphProto,
    blocked_node_names: set,
    is_top_level: bool,
    keep_io_types: bool,
):
    """
    - If is_top_level and keep_io_types is False, forcibly set top-level inputs/outputs to FLOAT16.
    - Otherwise (sub-graphs or keep_io_types=True), leave them as-is.
    - We rely on boundary casts if a blocked node requires float32 internally.
    """
    if is_top_level and not keep_io_types:
        # Force all top-level inputs to float16
        for vi in graph.input:
            if vi.type.tensor_type.elem_type == onnx_proto.TensorProto.FLOAT:
                vi.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT16

        # Force all top-level outputs to float16
        for vi in graph.output:
            if vi.type.tensor_type.elem_type == onnx_proto.TensorProto.FLOAT:
                vi.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT16
    # If keep_io_types=True, or this is a sub-graph, we do not modify the I/O dtypes here.
    if is_top_level and keep_io_types:
        return  # don't change top-level IO

    # If a top-level input is used by blocked nodes, keep it float32; else convert to float16
    input_to_nodes = {}
    for node in graph.node:
        for inp_name in node.input:
            if inp_name not in input_to_nodes:
                input_to_nodes[inp_name] = []
            input_to_nodes[inp_name].append(node.name)

    for vi in graph.input:
        if vi.type.tensor_type.elem_type == onnx_proto.TensorProto.FLOAT:
            # If any consumer is blocked => keep it float32
            consumers = input_to_nodes.get(vi.name, [])
            if any(c in blocked_node_names for c in consumers):
                continue
            # else convert to float16
            vi.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT16

    # For outputs, check the node that produces the output
    output_to_producer = {}
    for node in graph.node:
        for o_name in node.output:
            output_to_producer[o_name] = node.name

    for vi in graph.output:
        if vi.type.tensor_type.elem_type == onnx_proto.TensorProto.FLOAT:
            producer_name = output_to_producer.get(vi.name, None)
            if producer_name is None:
                continue
            if producer_name in blocked_node_names:
                # produced by a blocked node => keep float32
                continue
            vi.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT16


# Change the input/output of the node to the new output name after Cast node for sub-graph
# Because there have NO value_info start from
def process_node_input_output(
    graph: onnx_proto.GraphProto, global_input_name_dict: dict
):
    for node in graph.node:
        for i, input_name in enumerate(node.input):
            if input_name in global_input_name_dict:
                node.input[i] = global_input_name_dict[input_name]
        for i, output_name in enumerate(node.output):
            if output_name in global_input_name_dict:
                node.output[i] = global_input_name_dict[output_name]


def process_graph_input(
    graph: onnx_proto.GraphProto,
    is_top_level: bool,
    is_io_fp32: bool,
    global_input_name_dict: dict,
):
    # The input dtype is float32, need to cast to fp16
    if is_top_level and is_io_fp32:
        for graph_input in graph.input:  # n_input is ValueInfoProto
            if graph_input.type.tensor_type.elem_type == onnx_proto.TensorProto.FLOAT:
                downstream_nodes = find_donwstream_node_by_input_name(
                    graph, graph_input.name
                )
                for d_node in downstream_nodes:
                    cast_node_name = graph_input.name + "_cast_to_" + d_node.name
                    cast_node_output_name = graph_input.name + "_cast_to_" + d_node.name
                    add_cast_node(
                        graph,
                        [graph_input.name],
                        [cast_node_output_name],
                        cast_node_name,
                        FLOAT16,
                    )
                    add_new_value_info(
                        graph,
                        graph_input,
                        cast_node_output_name,
                        onnx_proto.TensorProto.FLOAT16,
                    )
                    for i, input_name in enumerate(d_node.input):
                        if input_name == graph_input.name:
                            d_node.input[i] = (
                                cast_node_output_name  # Change the input of the second node
                            )
                            global_input_name_dict[graph_input.name] = (
                                cast_node_output_name
                            )

    # For the sub-graph, don't do cast
    else:  # Change the input dtype to fp16 without any cast
        for graph_input in graph.input:
            if graph_input.type.tensor_type.elem_type == onnx_proto.TensorProto.FLOAT:
                graph_input.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT16


def process_graph_output(
    graph: onnx_proto.GraphProto, is_top_level: bool, is_io_fp32: bool
):
    if is_top_level and is_io_fp32:  # the output dtype is float32, need to cast to fp16
        for i, graph_output in enumerate(graph.output):
            if graph_output.type.tensor_type.elem_type == onnx_proto.TensorProto.FLOAT:
                upstream_nodes = find_upstream_node_by_output_name(
                    graph, graph_output.name
                )
                for u_node in upstream_nodes:
                    cast_node_name = u_node.name + "_cast_to_" + graph_output.name
                    cast_node_output_name = (
                        u_node.name + "_cast_to_" + graph_output.name
                    )
                    add_cast_node(
                        graph,
                        [cast_node_output_name],
                        [graph_output.name],
                        cast_node_name,
                        FLOAT32,
                    )
                    add_new_value_info(
                        graph,
                        graph_output,
                        cast_node_output_name,
                        onnx_proto.TensorProto.FLOAT16,
                    )
                    for i, output_name in enumerate(u_node.output):
                        if output_name == graph_output.name:
                            u_node.output[i] = cast_node_output_name
    else:  # change the output dtype to fp16 in tensor
        for graph_output in graph.output:
            if graph_output.type.tensor_type.elem_type == onnx_proto.TensorProto.FLOAT:
                graph_output.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT16


def mark_blocked_nodes(
    graph: onnx_proto.GraphProto, op_block_list: set, node_block_list: set
) -> set:
    """
    Return a set of node.names that must remain float32.
    This includes:
    - If the node's op_type is in op_block_list
    - If the node's name is in node_block_list
    """
    blocked = set()
    for node in graph.node:
        if node.op_type in op_block_list:
            blocked.add(node.name)
        if node.name in node_block_list:
            blocked.add(node.name)
    return blocked


def insert_boundary_casts(graph: onnx_proto.GraphProto, blocked_node_names: set):
    """
    Insert cast ops only on edges where a float16 (unblocked) node output
    feeds a float32 (blocked) node input, or vice versa.

    We'll do a single pass:
      For each node input, check the upstream node that produces it.
      If upstream's blocked status differs from current node's blocked status,
      we insert exactly one Cast on that edge.
    """
    # Collect info about which outputs come from which node
    name_to_node_dict = {}
    for node in graph.node:
        for out_name in node.output:
            name_to_node_dict[out_name] = node.name

    # We'll create new cast nodes as needed
    new_nodes = []
    for node in graph.node:
        cur_node_blocked = node.name in blocked_node_names
        for i, inp_name in enumerate(node.input):
            if inp_name not in name_to_node_dict:
                # Possibly a graph input or initializer; handle that separately if needed
                continue
            upstream_node_name = name_to_node_dict[inp_name]
            upstream_blocked = upstream_node_name in blocked_node_names
            if upstream_blocked != cur_node_blocked:
                # We have a boundary crossing: float32 -> float16 or float16 -> float32
                # Insert a cast in between
                cast_node_name = (
                    f"Cast_boundary_{upstream_node_name}_to_{node.name}_{i}"
                )
                cast_output_name = inp_name + "_boundary_cast"
                # Choose the 'to' type based on the current node's status
                to_type = (
                    onnx_proto.TensorProto.FLOAT16
                    if not cur_node_blocked
                    else onnx_proto.TensorProto.FLOAT
                )
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


# Change all the value info type from float32 to float16 if not in block list
def process_value_info(graph: onnx_proto.GraphProto, value_info_block_list: list):
    for value_info in graph.value_info:
        if value_info.name in value_info_block_list:
            continue
        else:
            if value_info.type.tensor_type.elem_type == onnx_proto.TensorProto.FLOAT:
                value_info.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT16


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


def get_next_level_graph(
    graph: onnx_proto.GraphProto, op_block_list: list, node_block_list: list
):
    sub_graph_list = []
    for node in graph.node:
        if node.op_type in op_block_list or node.name in node_block_list:
            continue
        for attr in node.attribute:
            # Check if sub-graph exist
            if len(attr.g.node) > 0:  # single sub-graph
                sub_graph_list.append(attr.g)
            for g in attr.graphs:
                if len(g.node) > 0:  # multiple sub-graphs
                    sub_graph_list.append(g)
    return sub_graph_list


def add_cast_node(
    graph: onnx_proto.GraphProto,
    inputs: list,
    outputs: list,
    node_name: str,
    to_type: int,
):
    new_node = [helper.make_node("Cast", inputs, outputs, to=to_type, name=node_name)]
    graph.node.extend(new_node)


def add_new_value_info(
    graph: onnx_proto.GraphProto,
    exist_value_info: onnx_proto.ValueInfoProto,
    name: str,
    dtype: int,
):
    new_value_info = graph.value_info.add()
    new_value_info.CopyFrom(exist_value_info)
    new_value_info.name = name
    new_value_info.type.tensor_type.elem_type = dtype


# Find the node that has the specified output name
def find_upstream_node_by_output_name(graph: onnx_proto.GraphProto, output_name: str):
    nodes = []
    for node in graph.node:
        if output_name in node.output:
            nodes.append(node)
    assert len(nodes) <= 1  # Suppose there is less than one node found
    return nodes


# Find the node that has the specified input name
def find_donwstream_node_by_input_name(graph: onnx_proto.GraphProto, input_name: str):
    nodes = []
    for node in graph.node:
        if input_name in node.input:
            nodes.append(node)
    return nodes


# Remove identity node
def remove_identity_node_from_model(model: onnx_proto.ModelProto):
    remove_identity_node_from_graph(model.graph)
    try:
        from onnx.shape_inference import infer_shapes

        func_infer_shape = infer_shapes
        model = func_infer_shape(model)
        return model
    finally:
        pass


# Remove identity node
def remove_identity_node_from_graph(graph: onnx_proto.GraphProto):
    for curr_node in graph.node:
        if curr_node.op_type == "Identity":
            for input_name in curr_node.input:
                upstream_nodes = find_upstream_node_by_output_name(graph, input_name)
                for u_node in upstream_nodes:
                    if u_node is not None:
                        u_node.output[0] = curr_node.output[0]
                        graph.node.remove(curr_node)


def convert_float_to_float16_model_path(
    model_path, min_positive_val=1e-7, max_finite_val=1e4, keep_io_types=False
):
    """
    Convert tensor float type in the ONNX Model to tensor float16.
    *It is to fix an issue that infer_shapes func cannot be used to infer >2GB models.
    *But this function can be applied to all model sizes.
    :param model_path: ONNX Model path
    :return: converted ONNX ModelProto object
    Examples
    ::
        #Convert to ONNX ModelProto object and save model binary file:
        from onnxmltools.utils.float16_converter import convert_float_to_float16_model_path
        new_onnx_model = convert_float_to_float16_model_path('model.onnx')
        onnx.save(new_onnx_model, 'new_model.onnx')
    """

    disable_shape_infer = False
    if pv.Version(onnx.__version__) >= pv.Version("1.8"):
        try:
            # infer_shapes_path can be applied to all model sizes
            import os
            import tempfile

            from onnx.shape_inference import infer_shapes_path

            # shape_infer_model_path should be in the same folder of model_path
            with tempfile.NamedTemporaryFile(
                dir=os.path.dirname(model_path)
            ) as tmpfile:
                shape_infer_model_path = tmpfile.name
                infer_shapes_path(model_path, shape_infer_model_path)
                model = onnx.load(shape_infer_model_path)
                disable_shape_infer = True
        finally:
            pass
    if not disable_shape_infer:
        model = onnx.load(model_path)
    return convert_float_to_float16(
        model, min_positive_val, max_finite_val, keep_io_types, disable_shape_infer
    )


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
