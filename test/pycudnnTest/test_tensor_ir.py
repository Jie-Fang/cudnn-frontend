import test_graph as tg
import torch
import utils
from dataclasses import dataclass
from abc import ABC, abstractmethod
from collections import namedtuple

from nv_tensor_ir import ir
from nv_tensor_ir.dialects import nv_tensor_ir

import nv_tensor_ir.extras.types as T
from data_types import DataType, convert_datatype


def cal_shapeK(input_data_type):
    if input_data_type in [DataType.FLOAT, DataType.INT32]:
        return 8
    elif input_data_type in [DataType.FP8_E4M3]:
        return 32
    elif input_data_type in [DataType.HALF, DataType.BFLOAT16]:
        return 16
    else:
        raise ValueError(f"Unsupported data type: {input_data_type}")


def generate_tensorir_compilation_configs(kmmaShapeK=16, cta_count=1):
    stream_k = False

    kphase = [1, 1, 4]

    kcta_count = [cta_count, 1, 1]

    mma_shapes = [[64, 128, kmmaShapeK], [128, 128, kmmaShapeK], [128, 256, kmmaShapeK]]

    cluster_shapes = [
        [1, 1, 1],
        [1, 2, 1],
        [1, 4, 1],
        [2, 1, 1],
        [2, 2, 1],
        [2, 4, 1],
        [4, 1, 1],
        [4, 2, 1],
        [4, 4, 1],
    ]
    configs = []

    for mma_shape in mma_shapes:
        for cluster_shape in cluster_shapes:
            tile_size = [
                int(m * c * k / cta)
                for m, c, k, cta in zip(mma_shape, cluster_shape, kphase, kcta_count)
            ]
            configs.append([tile_size, mma_shape, cluster_shape, cta_count, stream_k])

    return configs


def get_tensorir_compilation_config(tensorir_args, concrete_test_dict):
    tile_size = [128, 128, 64]
    mma_shape = [128, 128, 16]
    cluster_shape = [1, 1, 1]
    cta_count = 1
    stream_k = False

    if hasattr(tensorir_args, "tile_size") and tensorir_args.tile_size is not None:
        tile_size = list(map(int, tensorir_args.tile_size.split(",")))

    if (
        hasattr(tensorir_args, "cluster_shape")
        and tensorir_args.cluster_shape is not None
    ):
        cluster_shape = list(map(int, tensorir_args.cluster_shape.split(",")))

    if hasattr(tensorir_args, "mma_shape") and tensorir_args.mma_shape is not None:
        mma_shape = list(map(int, tensorir_args.mma_shape.split(",")))

    if hasattr(tensorir_args, "cta_count") and tensorir_args.cta_count is not None:
        cta_count = int(tensorir_args.cta_count)

    if hasattr(tensorir_args, "stream_k") and tensorir_args.stream_k is not None:
        stream_k = bool(tensorir_args.stream_k)

    concrete_test_dict["tile_size"] = tile_size
    concrete_test_dict["cluster_shape"] = cluster_shape
    concrete_test_dict["mma_shape"] = mma_shape
    concrete_test_dict["cta_count"] = cta_count
    concrete_test_dict["stream_k"] = stream_k

    return [tile_size, mma_shape, cluster_shape, cta_count, stream_k]


def find_mismatches(tensor_a, tensor_b, atol, rtol):
    # Use isclose with equal_nan=True: this will consider NaNs as equal.
    close_mask = torch.isclose(tensor_a, tensor_b, atol=atol, rtol=rtol, equal_nan=True)
    # Get indices where the tensors do not match
    diff_indices = torch.nonzero(~close_mask)

    mismatches = []
    for idx in diff_indices:
        index_tuple = tuple(idx.tolist())
        a_val = tensor_a[index_tuple]
        b_val = tensor_b[index_tuple]

        # Compute the absolute difference. Note: if one value is NaN (and the other is not),
        # the result will be NaN, so we handle that by assigning it an infinite diff.
        diff_val = torch.abs(a_val - b_val)
        if torch.isnan(diff_val):
            diff_val = float("inf")
        else:
            diff_val = diff_val.item()  # Convert tensor to a Python number for sorting

        # Also convert a_val and b_val to Python scalars if possible
        a_num = a_val.item() if hasattr(a_val, "item") else a_val
        b_num = b_val.item() if hasattr(b_val, "item") else b_val
        mismatches.append((index_tuple, a_num, b_num, diff_val))

    # Sort mismatches by the absolute difference in descending order.
    mismatches.sort(key=lambda x: x[3], reverse=True)

    # Print the largest 10 mismatches to reduce output length.
    for idx, a_num, b_num, diff_val in mismatches[:10]:
        print(
            f"Index: {idx}, Tensor A Value: {a_num}, Tensor B Value: {b_num}, diff = {diff_val}"
        )


@dataclass
class TensorInfo:
    tensor_type: nv_tensor_ir.TensorType
    dtype: ir.Type
    shape: list[int]
    shape_div: list[int]
    stride: list[int]
    stride_div: list[int]


def calculate_divisibility(out_stride):
    """Calculate shape and stride divisibility for tensor operations.

    Args:
        out_stride: List of stride values

    Returns:
        Tuple of (shape_div, stride_div) lists containing divisibility values
    """
    shape_div = []
    stride_div = []
    divisibility = 8
    stride_div_flag = True
    has_stride_zero = False

    for s in out_stride:
        if s == 0:
            shape_div.append(1)
            stride_div.append(1)
            has_stride_zero = True
        else:
            if s != 1:
                shape_div.append(1)
                if stride_div_flag and has_stride_zero:
                    stride_div.append(divisibility)
                    stride_div_flag = False
                else:
                    stride_div.append(1)
            else:
                shape_div.append(divisibility)
                stride_div.append(1)

    if has_stride_zero:
        stride_div[0] = divisibility

    return shape_div, stride_div


class TensorIRNode(ABC):
    """Base class for all Tensor IR operation nodes."""

    def __init__(self, node, node_map, ip, tensor_ir_test):
        self.node = node
        self.node_map = node_map
        self.ip = ip
        self.tensor_ir_test = tensor_ir_test
        self.children = [node_map[child] for child in node.producer_nodes]

        # Get output tensor info with compute data type from node kwargs or output data type
        self.output_tensor_info = (
            self.tensor_ir_test.determine_tensor_ir_inout_tensor_type(
                node,
                (
                    node.kwargs["compute_data_type"]
                    if "compute_data_type" in node.kwargs
                    else node.output[0].data_type
                ),
            )
        )

    @abstractmethod
    def run(self):
        """Run the operation and store the result in node_map"""
        pass

    def convert_and_splat_for_binary_pointwise(self, lsh, rsh, out_type):
        """Helper method for binary operations to handle input type conversions"""
        return self.tensor_ir_test.convert_and_splat_for_binary_pointwise(
            lsh, rsh, out_type
        )

    def convert_scalar_tensor(self, scalar_tensor, target_type):
        """Helper method to convert scalar tensors"""
        return self.tensor_ir_test.convert_scalar_tensor(scalar_tensor, target_type)

    def is_float(self, value):
        if isinstance(nv_tensor_ir.get_value_datatype(value), ir.FloatType):
            return True
        else:
            return False

    def is_integer(self, value):
        if isinstance(nv_tensor_ir.get_value_datatype(value), ir.IntegerType):
            return True
        else:
            return False


class ReductionNode(TensorIRNode):
    """Node for reduction operations"""

    def run(self):
        with self.ip:
            accumulator_type = ir.TypeAttr.get(
                eval(
                    convert_datatype(
                        self.tensor_ir_test.test_graph.compute_data_type, "tensorir"
                    )
                )
            )

            reduction_mode = None
            if "reduction_mode.ADD" in self.node.kwargs["mode"]:
                reduction_mode = nv_tensor_ir.ReductionMode.add
            elif "reduction_mode.AMAX" in self.node.kwargs["mode"]:
                reduction_mode = nv_tensor_ir.ReductionMode.amax
            elif "reduction_mode.MIN" in self.node.kwargs["mode"]:
                reduction_mode = nv_tensor_ir.ReductionMode.min
            elif "reduction_mode.MAX" in self.node.kwargs["mode"]:
                reduction_mode = nv_tensor_ir.ReductionMode.max

            input_datatype = nv_tensor_ir.get_tensor_datatype(self.children[0].type)
            output_datatype = nv_tensor_ir.get_tensor_datatype(
                self.output_tensor_info.tensor_type
            )

            shape_div, _ = calculate_divisibility(self.output_tensor_info.stride)

            if input_datatype != output_datatype:
                convert_value = nv_tensor_ir.convert(
                    nv_tensor_ir.TensorType.get(
                        shape=self.output_tensor_info.shape,
                        datatype=output_datatype,
                        shape_divisibility=shape_div,
                    ),
                    self.children[0],
                )
            else:
                convert_value = self.children[0]

            reduction_dimensions = []
            reduction_dim = 0
            for s in self.output_tensor_info.stride:
                if s == 0:
                    reduction_dimensions.append(reduction_dim)
                reduction_dim += 1

            mlir_value = nv_tensor_ir.reduce(
                nv_tensor_ir.TensorType.get(
                    shape=self.output_tensor_info.shape,
                    datatype=output_datatype,
                    shape_divisibility=shape_div,
                ),
                convert_value,
                reduction_dimensions,
                reduction_mode,
            )

            self.node_map[self.node] = mlir_value


class MatmulNode(TensorIRNode):
    """Node for matmul operations"""

    def run(self):
        with self.ip:
            compute_data_type = self.node.kwargs["compute_data_type"]
            compute_tensor_info = (
                self.tensor_ir_test.determine_tensor_ir_inout_tensor_type(
                    self.node, compute_data_type
                )
            )

            lsh = self.children[0]
            rsh = self.children[1]

            if nv_tensor_ir.get_tensor_datatype(
                lsh.type
            ) != nv_tensor_ir.get_tensor_datatype(rsh.type):
                datatype = nv_tensor_ir.get_tensor_datatype(
                    compute_tensor_info.tensor_type
                )
                if datatype != nv_tensor_ir.get_tensor_datatype(lsh.type):
                    lsh = nv_tensor_ir.convert(
                        nv_tensor_ir.TensorType.get_from_tensor_type(
                            lsh.type, datatype
                        ),
                        lsh,
                    )
                if datatype != nv_tensor_ir.get_tensor_datatype(rsh.type):
                    rsh = nv_tensor_ir.convert(
                        nv_tensor_ir.TensorType.get_from_tensor_type(
                            rsh.type, datatype
                        ),
                        rsh,
                    )

            matmul_value = nv_tensor_ir.matmul(
                compute_tensor_info.tensor_type, lsh, rsh
            )
            self.node_map[self.node] = matmul_value


class BinaryOperationNode(TensorIRNode):
    """Node for binary operations"""

    # Map operation names to unified tensor operation functions
    BINARY_OP_MAP = {
        "add": nv_tensor_ir.add,
        "bias": nv_tensor_ir.add,
        "sub": nv_tensor_ir.sub,
        "mul": nv_tensor_ir.mul,
        "max": nv_tensor_ir.max,
        "min": nv_tensor_ir.min,
        "div": nv_tensor_ir.div,
        "mod": nv_tensor_ir.mod,
        "pow": nv_tensor_ir.pow,
        "add_square": nv_tensor_ir.add_square,
        "atan2": nv_tensor_ir.atan2,
        "relu_backward": nv_tensor_ir.relu_bwd,
        "tanh_backward": nv_tensor_ir.tanh_bwd,
        "sigmoid_backward": nv_tensor_ir.sigmoid_bwd,
        "gelu_backward": nv_tensor_ir.gelu_bwd,
        "gelu_approx_tanh_backward": nv_tensor_ir.gelu_approx_tanh_bwd,
        "logical_or": nv_tensor_ir.or_,
        "logical_and": nv_tensor_ir.and_,
    }

    def run(self):
        with self.ip:
            lsh, rsh = self.convert_and_splat_for_binary_pointwise(
                self.children[0], self.children[1], self.output_tensor_info.tensor_type
            )

            op_name = self.node.op_name
            if op_name not in self.BINARY_OP_MAP:
                print(
                    f"Unimplemented binary Operation in Lowering to Tensor IR: {op_name}"
                )
                return

            # Get the unified operation function
            op_func = self.BINARY_OP_MAP[op_name]

            # Handle operations with special calling conventions
            if op_name in ["logical_or", "logical_and", "pow"]:
                self.node_map[self.node] = op_func(lsh, rsh)
            else:
                self.node_map[self.node] = op_func(
                    self.output_tensor_info.tensor_type, lsh, rsh
                )


class UnaryOperationNode(TensorIRNode):
    """Node for unary operations"""

    # Map operation names to unified tensor operation functions
    UNARY_OP_MAP = {
        "abs": nv_tensor_ir.abs,
        "tanh": nv_tensor_ir.tanh_fwd,
        "ceil": nv_tensor_ir.ceil,
        "floor": nv_tensor_ir.floor,
        "cos": nv_tensor_ir.cos,
        "sin": nv_tensor_ir.sin,
        "tan": nv_tensor_ir.tan,
        "exp": nv_tensor_ir.exp,
        "log": nv_tensor_ir.log,
        "neg": nv_tensor_ir.neg,
        "rsqrt": nv_tensor_ir.rsqrt,
        "sqrt": nv_tensor_ir.sqrt,
        "erf": nv_tensor_ir.erf,
        "reciprocal": nv_tensor_ir.reciprocal,
        "relu": nv_tensor_ir.relu_fwd,
        "sigmoid": nv_tensor_ir.sigmoid_fwd,
        "gelu": nv_tensor_ir.gelu_fwd,
        "gelu_approx_tanh": nv_tensor_ir.gelu_approx_tanh_fwd,
        "logical_not": nv_tensor_ir.not_,
    }

    def run(self):
        with self.ip:
            if self.output_tensor_info.tensor_type != self.children[0].type:
                convert_value = nv_tensor_ir.convert(
                    self.output_tensor_info.tensor_type, self.children[0]
                )
            else:
                convert_value = self.children[0]

            op_name = self.node.op_name
            if op_name not in self.UNARY_OP_MAP:
                print(
                    f"Unimplemented unary Operation in Lowering to Tensor IR: {op_name}"
                )
                return

            # Get the unified operation function
            op_func = self.UNARY_OP_MAP[op_name]

            # Execute the operation - all unary ops use the same calling convention
            self.node_map[self.node] = op_func(convert_value)


class ComparatorNode(TensorIRNode):
    """Node for comparison operations"""

    # Map operation names to comparator types
    COMPARATOR_MAP = {
        "cmp_lt": nv_tensor_ir.Comparator.olt,
        "cmp_ge": nv_tensor_ir.Comparator.oge,
        "cmp_gt": nv_tensor_ir.Comparator.ogt,
        "cmp_le": nv_tensor_ir.Comparator.ole,
        "cmp_eq": nv_tensor_ir.Comparator.oeq,
        "cmp_ne": nv_tensor_ir.Comparator.one,
    }

    def run(self):
        with self.ip:
            op_name = self.node.op_name
            if op_name not in self.COMPARATOR_MAP:
                print(f"Unimplemented comparator operation: {op_name}")
                return

            # Ensure we have float tensors for comparisons
            if not self.is_float(self.children[0]) or not self.is_float(
                self.children[1]
            ):
                raise ValueError(f"Comparator {op_name} requires float tensors")

            self.node_map[self.node] = nv_tensor_ir.cmpf(
                self.COMPARATOR_MAP[op_name], self.children[0], self.children[1]
            )


class IdentityNode(TensorIRNode):
    """Node for identity operations"""

    def run(self):
        with self.ip:
            self.node_map[self.node] = nv_tensor_ir.convert(
                self.output_tensor_info.tensor_type,
                self.children[0],
            )


class ConvolutionNode(TensorIRNode):
    """Base node for convolution operations"""

    def _prepare_common_params(self):
        accumulator_type_attr = ir.TypeAttr.get(
            eval(
                convert_datatype(
                    self.tensor_ir_test.test_graph.compute_data_type, "tensorir"
                )
            )
        )
        pre_padding = self.node.kwargs["padding"]
        post_padding = self.node.kwargs["padding"]  # FIXME: what should this be?
        stride = self.node.kwargs["stride"]
        dilation = self.node.kwargs["dilation"]

        return accumulator_type_attr, pre_padding, post_padding, stride, dilation


class ConvFpropNode(ConvolutionNode):
    """Node for forward convolution"""

    def run(self):
        with self.ip:
            x = self.children[0]
            w = self.children[1]

            accumulator_type_attr, pre_padding, post_padding, stride, dilation = (
                self._prepare_common_params()
            )

            self.node_map[self.node] = nv_tensor_ir.conv_fprop(
                self.output_tensor_info.tensor_type,
                x,
                w,
                accumulator_type_attr,
                pre_padding,
                post_padding,
                stride,
                dilation,
            )


class ConvDgradNode(ConvolutionNode):
    """Node for gradient w.r.t. data convolution"""

    def run(self):
        with self.ip:
            dy = self.children[0]
            w = self.children[1]

            accumulator_type_attr, pre_padding, post_padding, stride, dilation = (
                self._prepare_common_params()
            )

            self.node_map[self.node] = nv_tensor_ir.conv_dgrad(
                self.output_tensor_info.tensor_type,
                dy,
                w,
                accumulator_type_attr,
                pre_padding,
                post_padding,
                stride,
                dilation,
            )


class ConvWgradNode(ConvolutionNode):
    """Node for gradient w.r.t. weight convolution"""

    def run(self):
        with self.ip:
            dy = self.children[0]
            w = self.children[1]

            accumulator_type_attr, pre_padding, post_padding, stride, dilation = (
                self._prepare_common_params()
            )

            self.node_map[self.node] = nv_tensor_ir.conv_dgrad(
                self.output_tensor_info.tensor_type,
                dy,
                w,
                accumulator_type_attr,
                pre_padding,
                post_padding,
                stride,
                dilation,
            )


class ActivationForwardNode(TensorIRNode):
    """Node for activation forward operations that take beta parameter"""

    # Map operation names to activation functions
    ACTIVATION_MAP = {
        "swish": nv_tensor_ir.swish_fwd,
        "softplus": nv_tensor_ir.softplus_fwd,
        "elu": nv_tensor_ir.elu_fwd,
    }

    def run(self):
        with self.ip:
            beta = self.tensor_ir_test.get_beta_attr()
            if beta is None:
                raise ValueError(f"Beta attribute not found for {self.node.op_name}")

            converted_x, _ = self.convert_and_splat_for_binary_pointwise(
                self.children[0], None, self.output_tensor_info.tensor_type
            )

            # Ensure we have float tensors for activations
            if not self.is_float(converted_x):
                raise ValueError(
                    f"Activation operation {self.node.op_name} requires float tensors"
                )

            op_name = self.node.op_name
            if op_name in self.ACTIVATION_MAP:
                self.node_map[self.node] = self.ACTIVATION_MAP[op_name](
                    converted_x, beta
                )
            else:
                print(f"Unimplemented activation forward operation: {op_name}")


class ActivationBackwardNode(TensorIRNode):
    """Node for activation backward operations that take beta parameter"""

    # Map operation names to activation backward functions
    ACTIVATION_BWD_MAP = {
        "swish_backward": nv_tensor_ir.swish_bwd,
        "softplus_backward": nv_tensor_ir.softplus_bwd,
        "elu_backward": nv_tensor_ir.elu_bwd,
    }

    def run(self):
        with self.ip:
            beta = self.tensor_ir_test.get_beta_attr()
            if beta is None:
                raise ValueError(f"Beta attribute not found for {self.node.op_name}")

            converted_x, converted_grad = self.convert_and_splat_for_binary_pointwise(
                self.children[0], self.children[1], self.output_tensor_info.tensor_type
            )

            # Ensure we have float tensors for activation gradients
            if not self.is_float(converted_x) or not self.is_float(converted_grad):
                raise ValueError(
                    f"Activation gradient operation {self.node.op_name} requires float tensors"
                )

            op_name = self.node.op_name
            if op_name in self.ACTIVATION_BWD_MAP:
                self.node_map[self.node] = self.ACTIVATION_BWD_MAP[op_name](
                    converted_x, converted_grad, beta
                )
            else:
                print(f"Unimplemented activation backward operation: {op_name}")


class BinarySelectNode(TensorIRNode):
    """Node for binary select operations"""

    def run(self):
        with self.ip:
            condition = self.children[0]
            true_value = self.children[1]
            false_value = self.children[2]

            # Type checking: condition must be integer/boolean, true/false values must be float
            if not self.is_integer(condition):
                raise ValueError(
                    f"binary_select condition (first operand) must be an integer tensor, got {condition.type}"
                )

            if not self.is_float(true_value) or not self.is_float(false_value):
                raise ValueError(
                    f"binary_select true and false values (second and third operands) must be float tensors"
                )

            self.node_map[self.node] = nv_tensor_ir.binary_select(
                condition, true_value, false_value
            )


class test_tensor_ir:
    # Group operations by their handler class for cleaner lookup
    OPERATION_GROUPS = {
        ReductionNode: ["reduction"],
        MatmulNode: ["matmul"],
        BinaryOperationNode: [
            "add",
            "bias",
            "sub",
            "mul",
            "max",
            "min",
            "div",
            "mod",
            "atan2",
            "pow",
            "add_square",
            "logical_or",
            "logical_and",
            "relu_backward",
            "tanh_backward",
            "sigmoid_backward",
            "gelu_backward",
            "gelu_approx_tanh_backward",
        ],
        UnaryOperationNode: [
            "tanh",
            "abs",
            "ceil",
            "floor",
            "cos",
            "sin",
            "tan",
            "exp",
            "log",
            "neg",
            "rsqrt",
            "sqrt",
            "erf",
            "logical_not",
            "reciprocal",
            "relu",
            "sigmoid",
            "gelu",
            "gelu_approx_tanh",
        ],
        ComparatorNode: ["cmp_lt", "cmp_ge", "cmp_gt", "cmp_le", "cmp_eq", "cmp_ne"],
        BinarySelectNode: ["binary_select"],
        ActivationForwardNode: ["swish", "softplus", "elu"],
        ActivationBackwardNode: ["swish_backward", "softplus_backward", "elu_backward"],
        IdentityNode: ["identity"],
        ConvWgradNode: ["conv_wgrad"],
        ConvDgradNode: ["conv_dgrad"],
        ConvFpropNode: ["conv_fprop"],
    }

    # Map of operation names to node classes - built from the operation groups
    NODE_CLASS_MAP = {}

    # Build the NODE_CLASS_MAP from the operation groups
    @classmethod
    def _init_node_class_map(cls):
        for node_class, op_names in cls.OPERATION_GROUPS.items():
            for op_name in op_names:
                cls.NODE_CLASS_MAP[op_name] = node_class

    def __init__(self, test_graph):
        self.test_graph = test_graph
        self.outputs = []
        self.ref_outputs = None

        # Initialize the node class map if it's empty
        if not self.NODE_CLASS_MAP:
            self._init_node_class_map()

    def determine_tensor_ir_inout_tensor_type(self, node, dtype=None):
        assert len(node.output) == 1
        test_tensor = node.output[0]
        if dtype is None:
            dtype = eval(convert_datatype(test_tensor.data_type, "tensorir"))
        else:
            dtype = eval(convert_datatype(dtype, "tensorir"))

        if hasattr(node, "is_by_value") and node.is_by_value:
            return TensorInfo(
                tensor_type=dtype,
                dtype=dtype,
                shape=[1],
                shape_div=[],
                stride=[1],
                stride_div=[],
            )

        shape = []
        stride = []
        idx = 0

        ori_stride = test_tensor.stride
        ori_shape = test_tensor.dim

        # Check if tensor is a scalar tensor (all strides 1 and all dims 1 in json definition)
        isScalarTensor = all(s == 1 for s in ori_stride) and all(
            d == 1 for d in ori_shape
        )

        if isScalarTensor:
            # If tensor is scalar in json definition, convert to dense memref with shape to [-1] and stride to [0]
            shape = [-1] * len(ori_shape)
            shape_div = [1] * len(ori_shape)
            stride = [0] * len(ori_stride)
            stride_div = [1] * len(ori_stride)
            return TensorInfo(
                tensor_type=nv_tensor_ir.TensorType.get(shape=shape, datatype=dtype),
                dtype=dtype,
                shape=shape,
                shape_div=shape_div,
                stride=stride,
                stride_div=stride_div,
            )
        for s, d in zip(ori_stride, ori_shape):
            if idx > 0 and d == 1:
                shape.append(-1)  # row broadcast need to set broadcast dim to `?`
                stride.append(0)
            else:
                if s != 1:
                    stride.append(-1)
                else:
                    stride.append(1)
                shape.append(-1)
            idx += 1

        shape_div, stride_div = calculate_divisibility(stride)
        ty = nv_tensor_ir.TensorType.get(
            shape=shape,
            datatype=dtype,
            shape_divisibility=shape_div,
        )
        return TensorInfo(
            tensor_type=ty,
            dtype=dtype,
            shape=shape,
            shape_div=shape_div,
            stride=stride,
            stride_div=stride_div,
        )

    def calc_ref(self):
        self.ref_outputs = self.test_graph.calc_reference()

    def tensorir_compare_to_reference(self, atol=1e-2, rtol=1e-2):
        assert len(self.ref_outputs) == len(self.outputs)

        # Use the base class method for comparison
        return tg.test_graph.compare_to_reference(
            self.ref_outputs, self.outputs, atol=atol, rtol=rtol
        )

    def get_beta_attr(self):
        for node in self.test_graph.entrance_nodes:
            if node.name == "beta":
                beta_tensor = node.get_value()
                # Handle beta tensor of any dimension by accessing the first element
                return (
                    beta_tensor.item()
                    if beta_tensor.numel() == 1
                    else beta_tensor.flatten()[0].item()
                )
        return None

    def run_tensor_ir_module(
        self,
        module,
        kernel_configs,
        dump_ir_path,
        load_ir_path,
        mlir_timing,
        timing_loop=1,
        atol=1e-2,
        rtol=1e-2,
    ):
        graph_analysis = nv_tensor_ir.GraphAnalysis(module)
        m_idx = graph_analysis.get_matmul_m_index()
        n_idx = graph_analysis.get_matmul_n_index()
        k_idx = graph_analysis.get_matmul_k_index()
        B = self.test_graph.entrance_nodes[m_idx[0]].output[0].dim[0]
        M = self.test_graph.entrance_nodes[m_idx[0]].output[0].dim[m_idx[1]]
        N = self.test_graph.entrance_nodes[n_idx[0]].output[0].dim[n_idx[1]]
        K = self.test_graph.entrance_nodes[k_idx[0]].output[0].dim[k_idx[1]]
        problem_size = nv_tensor_ir.GemmProblemSize(B, M, N, K)
        print("problem_size:", B, M, N, K)
        device = torch.device("cuda")

        # iterate all kernel inputs automatically
        tensor_desc = nv_tensor_ir.VectorTensorOperandDescriptor()
        inputs_gpu = []
        for node in self.test_graph.entrance_nodes:
            torch_mem = node.get_value()
            if not node.output[0].is_by_value:
                gpu_mem = torch_mem.to(device=device)
                inputs_gpu.append(gpu_mem)  # need to save gpu_mem in case of releasing
                strides = []
                for i, (shape, stride) in enumerate(
                    zip(torch_mem.shape, torch_mem.stride())
                ):
                    strides.append(0) if shape == 1 else strides.append(stride)
                tensor_operand = nv_tensor_ir.TensorOperandDescriptor(
                    nv_tensor_ir.TensorDescriptor(
                        gpu_mem.data_ptr(), nv_tensor_ir.LayoutDescriptor(strides)
                    )
                )
                tensor_desc.append(tensor_operand)
            else:
                cpu_mem = torch_mem.to("cpu")
                tensor_operand = nv_tensor_ir.TensorOperandDescriptor(
                    nv_tensor_ir.ScalarDescriptor(
                        eval(convert_datatype(node.output[0].data_type, "cask")),
                        cpu_mem[0].data_ptr(),
                    )
                )
                tensor_desc.append(tensor_operand)
        if not self.ref_outputs:
            self.calc_ref()
        outputs_gpu = [
            torch.as_strided(
                torch.empty(
                    tuple(node.output[0].dim),
                    dtype=eval(convert_datatype(node.output[0].data_type, "torch")),
                    device=device,
                ),
                size=tuple(node.output[0].dim),
                stride=tuple(node.output[0].stride),
            )
            for node in self.test_graph.nodes
            if node.is_output_node()
        ]
        for torch_gpu in outputs_gpu:
            tensor_desc.append(
                nv_tensor_ir.TensorOperandDescriptor(
                    nv_tensor_ir.TensorDescriptor(
                        torch_gpu.data_ptr(),
                        nv_tensor_ir.LayoutDescriptor(torch_gpu.stride()),
                    )
                )
            )

        args = nv_tensor_ir.ArgumentsView(problem_size, tensor_desc)

        with ir.Context() as ctx, ir.Location.unknown():
            nv_tensor_ir.register_dialect()

            best_perf = float("inf")
            best_config = dict(
                tile_size=[], mma_shape=[], cluster_shape=[], cta_count=[]
            )
            for (
                tile_size,
                mma_shape,
                cluster_shape,
                cta_count,
                stream_k,
            ) in kernel_configs:
                cloned_module = ir.Module.parse(str(module))
                cask_context = nv_tensor_ir.create_cask_context()
                compiler = nv_tensor_ir.Compiler(cask_context)

                compile_option = nv_tensor_ir.TensorIRCompilationOption(
                    10,  # Hardcoded for blackwell
                    nv_tensor_ir.GraphCategory.kGemm,  # Hardcode for Gemm
                    nv_tensor_ir.TensorConversionOptions(
                        tile_size, mma_shape, cluster_shape, cta_count, stream_k
                    ),
                    nv_tensor_ir.DebugOptions(dump_ir_path, load_ir_path, mlir_timing),
                )

                print(
                    f"#### Running tile_size={tile_size}, mma_shape={mma_shape}, cluster_shape={cluster_shape}, cta_count={cta_count}, stream_k={stream_k}"
                )
                shader = compiler.compile(cloned_module, compile_option)
                execution_plan = nv_tensor_ir.ExecutionPlan(shader, args)
                execution_plan.initialize()
                device_workspace_size = execution_plan.query_max_device_workspace_size()
                device_workspace_mem_cpu = torch.zeros(
                    device_workspace_size, dtype=torch.int8, device="cpu"
                )
                device_workspace_mem_gpu = (
                    device_workspace_mem_cpu.clone().detach().to("cuda")
                )
                device_workspace = nv_tensor_ir.DeviceWorkspace(
                    device_workspace_mem_gpu.data_ptr(), device_workspace_size
                )
                # TODO: Do we need to dump the launch config for debugging?
                # execution_plan.dump_launch_config()
                if timing_loop == 0:
                    execution_plan.launch(device_workspace)
                    self.outputs = outputs_gpu
                    if not self.ref_outputs:
                        self.calc_ref()
                    passed = self.tensorir_compare_to_reference(atol, rtol)
                    assert passed, "Mismatch between TensorIR and reference"
                elif timing_loop == 1:
                    execution_plan.launch(device_workspace)
                else:
                    # warm the caches
                    execution_plan.launch(device_workspace)
                    import utils

                    # TODO: Shall we use median instead of average?
                    (_, avg_rt, _) = utils.measure_gpu_runtime(
                        lambda: execution_plan.launch(device_workspace), timing_loop
                    )
                    if avg_rt < best_perf:
                        best_perf = avg_rt
                        best_config = {
                            "tile_size": tile_size,
                            "mma_shape": mma_shape,
                            "cluster_shape": cluster_shape,
                            "cta_count": cta_count,
                        }

            print(
                f"@@@@ Best perf achieved is {best_perf / 1000} msec with kernel config: {best_config}"
            )

    def build_tensor_ir_module(self, json_test_name="graph"):
        input_tensors = []
        for node in self.test_graph.entrance_nodes:
            input_tensors.append(node)
        output_tensors = [
            node for node in self.test_graph.nodes if node.is_output_node()
        ]

        align_name = "nv_tensor_ir.alignment"
        stride_name = "nv_tensor_ir.stride"
        with ir.Context() as ctx, ir.Location.unknown() as loc:
            ctx.enable_multithreading(False)
            nv_tensor_ir.register_dialect()

            module = ir.Module.create(loc)

            # Convert a stride list to string format, replacing -1 with '?'.
            # E.g. stride =[1, -1, 1] and stride_div = [1, 8, 1] -> "(1,?{div=8},1)"
            def stride_list_to_string(stride, stride_div):
                stride_elements = [
                    "?{div=" + str(div) + "}" if s == -1 else str(s)
                    for s, div in zip(stride, stride_div)
                ]
                return f"({','.join(stride_elements)})"

            # Get alignment in bytes based on the data type
            def get_alignment(dtype):
                # TODO: check if this is correct
                # Usually GMEM address must be 16B aligned for tma
                if dtype.width >= 8:
                    return 16
                # But 4-bit and 6-bit must 32B aligned
                else:
                    return 32

            input_tensor_infos = list(
                map(
                    lambda x: self.determine_tensor_ir_inout_tensor_type(x),
                    input_tensors,
                )
            )

            arg_attrs = []
            input_types = []
            for tensor_info in input_tensor_infos:
                input_types.append(tensor_info.tensor_type)
                # Add stride and alignment attributes if it's a tensor not a scalar passed by value
                if tensor_info.tensor_type != tensor_info.dtype:
                    stride_str = ir.StringAttr.get(
                        stride_list_to_string(
                            tensor_info.stride, tensor_info.stride_div
                        )
                    )
                    align_attr = ir.IntegerAttr.get(
                        ir.IntegerType.get_signless(64),
                        get_alignment(tensor_info.dtype),
                    )
                    arg_attrs.append(
                        ir.DictAttr.get(
                            {stride_name: stride_str, align_name: align_attr}
                        )
                    )
                else:
                    arg_attrs.append(ir.DictAttr.get({}))

            output_tensor_infos = list(
                map(
                    lambda x: self.determine_tensor_ir_inout_tensor_type(x),
                    output_tensors,
                )
            )

            res_attrs = []
            output_types = []
            for tensor_info in output_tensor_infos:
                output_types.append(tensor_info.tensor_type)
                # Add stride and alignment attributes if it's a tensor not a scalar passed by value
                if tensor_info.tensor_type != tensor_info.dtype:
                    stride_str = ir.StringAttr.get(
                        stride_list_to_string(
                            tensor_info.stride, tensor_info.stride_div
                        )
                    )
                    align_attr = ir.IntegerAttr.get(
                        ir.IntegerType.get_signless(64),
                        get_alignment(tensor_info.dtype),
                    )
                    res_attrs.append(
                        ir.DictAttr.get(
                            {stride_name: stride_str, align_name: align_attr}
                        )
                    )
                else:
                    res_attrs.append(ir.DictAttr.get({}))

            # Create a mlir function signature for the kernel.
            function_type = ir.TypeAttr.get(
                T.function(inputs=input_types, results=output_types)
            )
            function_name = json_test_name

            # Create tensor ir graph
            graph = nv_tensor_ir.graph(
                function_name,
                function_type=function_type,
                arg_attrs=arg_attrs,
                res_attrs=res_attrs,
                ip=ir.InsertionPoint(module.body),
            )

            node_map = (
                {}
            )  # nodes -> tensor_ir values (Do python semantics mess with this?)

            # Generate arguments
            graph.regions[0].blocks.append(*input_types)

            # Assign inputs to arguments
            for i, node in enumerate(input_tensors):
                node_map[node] = graph.regions[0].blocks[0].arguments[i]

            with ir.InsertionPoint(graph.regions[0].blocks[0]) as ip:
                for (
                    node
                ) in output_tensors:  # This should be built from output nodes no?
                    # Recursively lower ops
                    self.build_tensor_ir_recursive(
                        node, node_map, ip
                    )  # This is passed the insertion point, which provides a mutable reference to the module. # Consider passing other things, as order in graph module is said to not matter.

                # Generate return op
                result = []
                for node in output_tensors:
                    if (
                        "compute_data_type" in node.kwargs
                        and node.output[0]._data_type
                        != node.kwargs["compute_data_type"]
                    ):
                        tensor_info = self.determine_tensor_ir_inout_tensor_type(
                            node, node.output[0]._data_type
                        )
                        node_map[node] = nv_tensor_ir.convert(
                            tensor_info.tensor_type,
                            node_map[node],
                        )
                    result.append(node_map[node])

                nv_tensor_ir.results_(result)
        module.operation.verify()

        return module

    def convert_and_splat_for_binary_pointwise(self, lsh, rsh, out_type):
        if lsh is not None and rsh is not None:
            if isinstance(lsh.type, (ir.IntegerType, ir.FloatType)) and isinstance(
                rsh.type, nv_tensor_ir.TensorType
            ):
                lsh = nv_tensor_ir.splat(
                    nv_tensor_ir.TensorType.get_from_tensor_type(rsh.type, lsh.type),
                    lsh,
                )
            elif isinstance(rsh.type, (ir.IntegerType, ir.FloatType)) and isinstance(
                lsh.type, nv_tensor_ir.TensorType
            ):
                rsh = nv_tensor_ir.splat(
                    nv_tensor_ir.TensorType.get_from_tensor_type(lsh.type, rsh.type),
                    rsh,
                )
        out_type_datatype = nv_tensor_ir.get_tensor_datatype(out_type)
        if lsh is not None and out_type_datatype != nv_tensor_ir.get_tensor_datatype(
            lsh.type
        ):
            convert_value0 = nv_tensor_ir.convert(
                nv_tensor_ir.TensorType.get_from_tensor_type(
                    lsh.type, out_type_datatype
                ),
                lsh,
            )
        else:
            convert_value0 = lsh
        if rsh is not None and out_type_datatype != nv_tensor_ir.get_tensor_datatype(
            rsh.type
        ):
            convert_value1 = nv_tensor_ir.convert(
                nv_tensor_ir.TensorType.get_from_tensor_type(
                    rsh.type, out_type_datatype
                ),
                rsh,
            )
        else:
            convert_value1 = rsh
        return convert_value0, convert_value1

    def convert_scalar_tensor(self, scalar_tensor, target_type):
        if nv_tensor_ir.get_tensor_datatype(
            target_type
        ) != nv_tensor_ir.get_tensor_datatype(scalar_tensor.type):
            return nv_tensor_ir.convert(
                nv_tensor_ir.TensorType.get_from_tensor_type(
                    scalar_tensor.type, nv_tensor_ir.get_tensor_datatype(target_type)
                ),
                scalar_tensor,
            )
        else:
            return scalar_tensor

    def build_tensor_ir_recursive(self, node, node_map, ip):
        """Build tensor IR representation recursively for the given node."""
        # Skip if node is already processed
        if node in node_map.keys():
            return

        # Process all children first
        for child in node.producer_nodes:
            self.build_tensor_ir_recursive(child, node_map, ip)

        # Only process operation nodes
        if not isinstance(node, tg.operation):
            print("not an op")
            return

        op_name = node.op_name

        # Get the node class for this operation
        node_class = self.NODE_CLASS_MAP.get(op_name)

        if node_class is None:
            print(f"Unimplemented Operation in Lowering to Tensor IR: {op_name}")
            return

        # Create and run the node
        ir_node = node_class(node, node_map, ip, self)
        ir_node.run()
