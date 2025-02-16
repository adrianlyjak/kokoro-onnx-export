# Copied and heavily modified from https://github.com/microsoft/onnxconverter-common/blob/master/onnxconverter_common/float16.py
import itertools
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Generator, List, Optional, Union

import numpy as np
import onnx
import packaging.version as pv
from onnx import (
    GraphProto,
    NodeProto,
    TensorProto,
    ValueInfoProto,
    helper,
    numpy_helper,
)
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
    checker = _is_blocked_checker(op_block_list.union({"Cast"}), node_block_list)
    used_names = set()
    for graph in all_graphs:
        for node in graph.node:
            used_names.add(node.name)
    for ti in tensor_infos:
        used_names.add(ti.name())
    for ti in tensor_infos:
        _modify_graph_for_tensor_info(
            ti,
            checker,
            keep_io_types,
            used_names,
            min_positive_val,
            max_finite_val,
            tensor_infos,
        )
    for graph in all_graphs:
        for node in graph.node:
            if not checker(node):
                _modify_node_attribute_type(node, min_positive_val, max_finite_val)

    sort_topology(model.graph)

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
    hack: list["TensorInfo"] = [],
) -> None:
    if ti.data_type() != FLOAT32:
        return
    desired_type = FLOAT32
    parent_node_if_output = (
        [ti.parent_node]
        if ti.parent_node is not None and ti.output_index is not None
        else []
    )
    if not ti.is_root_io or not keep_io_types:
        all_blocked = all(
            is_node_blocked(n) for n in ti.io_nodes() + (parent_node_if_output)
        )
        if not all_blocked:
            desired_type = FLOAT16
    if ti.data_type() != desired_type and desired_type == FLOAT16:
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
        if node_type != desired_type:
            cast_output_node = ti.cast_output_node
            cast_output = ti.cast_output
            if not cast_output_node:
                _create_cast_to(ti, node_type, used_node_names)
                cast_output_node = ti.cast_output_node
                cast_output = ti.cast_output
            for i, input_name in enumerate(output_node.input):
                if input_name == ti.name():
                    output_node.input[i] = cast_output
    for parent_node in parent_node_if_output:
        parent_node_desired_type = FLOAT32 if is_node_blocked(parent_node) else FLOAT16
        if parent_node_desired_type != desired_type:
            cast_output_node = ti.cast_output_node
            cast_output = ti.cast_output
            if not cast_output_node:
                _create_cast_to(
                    ti,
                    parent_node_desired_type,
                    used_node_names,
                    create_value_info=False,
                )
                cast_output_node = ti.cast_output_node
                cast_output = ti.cast_output
            old_output: ValueInfoProto = ti.graph.output[ti.output_index]
            old_as_vi = ti.graph.value_info.add()
            old_as_vi.name = old_output.name
            old_as_vi.type.tensor_type.elem_type = old_output.type.tensor_type.elem_type
            ti.graph.output.remove(old_output)
            new_output_vi = ti.graph.output.add()
            new_output_vi.name = cast_output
            new_output_vi.type.tensor_type.elem_type = parent_node_desired_type
            new_output_vi.type.tensor_type.shape.CopyFrom(
                old_output.type.tensor_type.shape
            )
    for input_node in ti.input_nodes:
        input_node_type = FLOAT32 if is_node_blocked(input_node) else FLOAT16
        if input_node_type != desired_type:
            cast_input_node = ti.cast_input_node
            cast_input = ti.cast_input
            if not cast_input_node:
                _create_cast_from(ti, input_node_type, used_node_names)
                cast_input_node = ti.cast_input_node
                cast_input = ti.cast_input
            for i, output_name in enumerate(input_node.output):
                if output_name == ti.name():
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


def _create_cast_to(
    ti: "TensorInfo",
    cast_to: int,
    used_node_names: set[str],
    create_value_info: bool = True,
) -> None:
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

    if create_value_info:
        new_vi = ti.graph.value_info.add()
        new_vi.name = cast_output_tensor_name
        new_vi.type.tensor_type.elem_type = cast_to
        if isinstance(ti.proto, ValueInfoProto):
            new_vi.type.tensor_type.shape.CopyFrom(ti.proto.type.tensor_type.shape)


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

    new_vi = ti.graph.value_info.add()
    new_vi.name = cast_input_tensor_name
    new_vi.type.tensor_type.elem_type = cast_from
    if isinstance(ti.proto, ValueInfoProto):
        new_vi.type.tensor_type.shape.CopyFrom(ti.proto.type.tensor_type.shape)


@dataclass
class TensorInfo:
    """
    An accumulator for information about tensors in the model. On the "first" pass, these are instantiated, without
    casts, and then a subsequent pass iterates over all of the tensor infos, analyzing the connected nodes, creating and
    rewiring casts as necessary.
    """

    proto: Union[ValueInfoProto, TensorProto]
    graph: onnx.GraphProto
    parent_node: Optional[NodeProto] = None
    is_root_io: bool = False
    output_index: Optional[int] = None
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
    graph_stack: list[tuple[onnx.GraphProto, Optional[NodeProto]]] = [
        (root_graph, None)
    ]
    all_graphs: list[tuple[onnx.GraphProto, Optional[NodeProto]]] = []
    while graph_stack:
        curr_graph, parent_node = graph_stack.pop()
        all_graphs.append((curr_graph, parent_node))
        for node in curr_graph.node:
            for input_name in node.input:
                tensor_to_outputs[input_name].append(node)
            for output_name in node.output:
                tensor_to_input[output_name].append(node)
            for attr in node.attribute:
                if attr.g.node:  # single sub-graph
                    graph_stack.append((attr.g, node))
                for g in attr.graphs:
                    if g.node:
                        graph_stack.append((g, node))

    for graph, parent_node in all_graphs:
        is_root = graph == root_graph
        for vi in graph.input:
            tensor_infos.append(
                TensorInfo(
                    proto=vi,
                    graph=graph,
                    parent_node=parent_node,
                    is_root_io=is_root,
                    output_nodes=tensor_to_outputs[vi.name],
                )
            )
        for i, vi in enumerate(graph.output):
            tensor_infos.append(
                TensorInfo(
                    proto=vi,
                    graph=graph,
                    parent_node=parent_node,
                    output_index=i,
                    is_root_io=is_root,
                    input_nodes=tensor_to_input[vi.name],
                )
            )
        for init in graph.initializer:
            tensor_infos.append(
                TensorInfo(
                    proto=init,
                    graph=graph,
                    parent_node=parent_node,
                    is_initializer=True,
                    output_nodes=tensor_to_outputs[init.name],
                )
            )
        for tensor in graph.value_info:
            tensor_infos.append(
                TensorInfo(
                    proto=tensor,
                    graph=graph,
                    parent_node=parent_node,
                    output_nodes=tensor_to_outputs[tensor.name],
                    input_nodes=tensor_to_input[tensor.name],
                )
            )
    return tensor_infos, [g for (g, _) in all_graphs]


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


def _get_child_graphs(node) -> Generator[GraphProto, None, None]:
    """Return a list of child graphs for the given node"""
    for attr in node.attribute:
        if attr.g:
            yield attr.g
            for n in attr.g.node:
                yield from _get_child_graphs(n)
        for g in attr.graphs:
            yield g
            for n in g.node:
                yield from _get_child_graphs(n)


def _get_node_inputs_including_child_graphs(node) -> list[str]:
    """Also determine inputs from the child graph if any"""
    inputs = [i for i in node.input]
    for child_graph in _get_child_graphs(node):
        for node in child_graph.node:
            inputs.extend([i for i in node.input])
    return inputs


def sort_graph_node(graph_proto):
    node_inputs: dict[str, list[str]] = {
        n.name: _get_node_inputs_including_child_graphs(n) for n in graph_proto.node
    }

    # find the "first" node in Nodes that its input is not any node's output
    def find_first_node(output2node_dict):
        for node in org_nodes:
            is_not_first_node = any(
                item in output2node_dict for item in node_inputs[node.name]
            )
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
