import test_graph as tg
import torch
import cudnn
import utils

import nv_tensor_ir
from nv_tensor_ir import ir
from nv_tensor_ir.dialects import nv_tensor_ir, func, arith, scf

import nv_tensor_ir.extras.types as T


class test_tensor_ir:
    def __init__(self, test_graph):
        self.test_graph = test_graph

    # Determine graph input and output types
    def determine_tensor_ir_inout_tensor_type(self, node, dtype={}):
        assert len(node.output) == 1
        test_tensor = node.output[0]
        if dtype is self.determine_tensor_ir_inout_tensor_type.__defaults__[0]:
            dtype = self.determine_tensor_ir_dtype(test_tensor.data_type)
        else:
            dtype = self.determine_tensor_ir_dtype(dtype)
        if hasattr(node, "is_by_value"):
            return dtype

        shape = []
        stride = []
        idx = 0
        if test_tensor.ref_data == None:
            ori_stride = test_tensor.cudnn_tensor.get_stride()
            ori_shape = test_tensor.cudnn_tensor.get_dim()
        else:
            ori_stride = test_tensor.ref_data.stride()
            ori_shape = test_tensor.ref_data.shape
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
        # ty = nv_tensor_ir.TensorType.get(
        #     shape=ori_shape,
        #     stride=ori_stride,
        #     datatype=dtype,
        # )
        # print("test_tensor.cudnn_tensor stride:", stride, " shape:", shape)
        # print("test_tensor.cudnn_tensor shape:", test_tensor.cudnn_tensor.get_dim(), " test_tensor.cudnn_tensor.get_stride():", test_tensor.cudnn_tensor.get_stride())
        ty = nv_tensor_ir.TensorType.get(shape=shape, stride=stride, datatype=dtype)
        return ty, shape, stride, dtype

    def determine_tensor_ir_dtype(self, cudnn_datatype):
        match cudnn_datatype:
            case cudnn.data_type.INT32:
                return T.si32()
            case cudnn.data_type.FLOAT:
                return T.f32()
            case cudnn.data_type.HALF:
                return T.f16()
            case cudnn.data_type.BFLOAT16:
                return T.bf16()
        assert False

    def determine_cask_dtype(self, cudnn_datatype):
        match cudnn_datatype:
            case cudnn.data_type.INT32:
                return nv_tensor_ir.NumericTypeID.kS32
            case cudnn.data_type.FLOAT:
                return nv_tensor_ir.NumericTypeID.kF32
            case cudnn.data_type.HALF:
                return nv_tensor_ir.NumericTypeID.kF16
            case cudnn.data_type.BFLOAT16:
                return nv_tensor_ir.NumericTypeID.kBF16
        assert False

    def tensorir_compare_to_reference(self, atol=1e-2, rtol=1e-2):
        # Run the reference
        print("Computing reference")
        ref_outputs = self.test_graph.calc_reference()
        assert len(ref_outputs) == len(self.test_graph.getOutputs())
        number_outputs_tested = 0
        output_idx = 0
        # Compare with reference
        for Y_expected, Y_actual in zip(ref_outputs, self.test_graph.getOutputs()):
            # TODO (@mbreughe): Clean up this assumption:
            # If there are None's in the output, it's because the reference didn't provide actual output (eg batchnorm)
            # For now, we can assume that we don't care about this output and just let the reference pass
            # To be on the safe side, we will make sure at least one output was checked
            if Y_expected is None:
                continue
            if Y_expected.dtype != Y_actual.dtype:
                print(
                    "WARNING: reference and actual output types differ ({} resp., {})".format(
                        Y_expected.dtype, Y_actual.dtype
                    )
                )

            if Y_expected.shape != Y_actual.shape:
                print(
                    "WARNING: reference and actual output shapes differ ({} resp., {})".format(
                        Y_expected.shape, Y_actual.shape
                    )
                )
            torch.testing.assert_close(Y_expected, Y_actual, atol=atol, rtol=rtol)

            number_outputs_tested += 1
            output_idx += 1
        assert number_outputs_tested >= 1

        utils.reportCurrentTime("assert_close")

    def run_tensor_ir_module(
        self,
        module,
        compile_option=nv_tensor_ir.TensorConversionOptions(
            [128, 128, 64], [128, 128, 16], [1, 1, 1], 1
        ),
    ):
        kernel_name = "graph"
        graph_analysis = nv_tensor_ir.GraphAnalysis(module)
        a_partition_idx = 0
        b_partition_idx = 0
        for i in range(len(self.test_graph.entrance_nodes)):
            if (
                nv_tensor_ir.GraphPartitionType.GRAPH_PARTITION_MAINLOOP_A
                == graph_analysis.get_graph_operand_partition(module, i)
            ):
                a_partition_idx = i
            if (
                nv_tensor_ir.GraphPartitionType.GRAPH_PARTITION_MAINLOOP_B
                == graph_analysis.get_graph_operand_partition(module, i)
            ):
                b_partition_idx = i
        a_tensor_dim = (
            self.test_graph.entrance_nodes[a_partition_idx]
            .output[0]
            .cudnn_tensor.get_dim()
        )
        b_tensor_dim = (
            self.test_graph.entrance_nodes[b_partition_idx]
            .output[0]
            .cudnn_tensor.get_dim()
        )
        B = a_tensor_dim[0]
        M = a_tensor_dim[1]
        N = b_tensor_dim[2]
        K = a_tensor_dim[2]
        problem_size = nv_tensor_ir.GemmProblemSize(B, M, N, K)
        print("problem_size:", B, M, N, K)
        device = torch.device("cuda:0")

        # iterate all kernel inputs automatically
        tensor_desc = nv_tensor_ir.VectorTensorOperandDescriptor()
        inputs_gpu = []
        for node in self.test_graph.entrance_nodes:
            torch_mem = node.output[0].ref_data
            if not node.output[0].is_by_value:
                gpu_mem = torch.tensor(torch_mem, device=device)
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
                tensor_operand = nv_tensor_ir.TensorOperandDescriptor(
                    nv_tensor_ir.ScalarDescriptor(
                        self.determine_cask_dtype(node.output[0].data_type),
                        torch_mem[0].data_ptr(),
                    )
                )
                tensor_desc.append(tensor_operand)

        workspace, variant_pack = self.test_graph.create_workspace_and_variantpack()
        outputs = self.test_graph.getOutputs()
        outputs_gpu = []
        for output in outputs:
            torch_gpu = torch.tensor(output, device=device)
            outputs_gpu.append(torch_gpu)  # need to save torch_gpu in case of releasing
            tensor_operand = nv_tensor_ir.TensorOperandDescriptor(
                nv_tensor_ir.TensorDescriptor(
                    torch_gpu.data_ptr(), nv_tensor_ir.LayoutDescriptor(output.stride())
                )
            )
            tensor_desc.append(tensor_operand)

        args = nv_tensor_ir.ArgumentsView(problem_size, tensor_desc)
        with ir.Context() as ctx, ir.Location.unknown():
            nv_tensor_ir.register_dialect()

            cask_context = nv_tensor_ir.create_cask_context()

            compiler = nv_tensor_ir.Compiler(cask_context)
            shader = compiler.compile(module, compile_option)

            print("BEFORE EXECUTE")
            nv_tensor_ir.cask_execute_shader_complete(shader, args)

            print("AFTER EXECUTE")

        outputs = self.test_graph.getOutputs()
        for i in range(len(outputs)):
            outputs[i].copy_(outputs_gpu[i])
        self.tensorir_compare_to_reference(atol=1e-2, rtol=1e-2)
        # try:
        #     self.tensorir_compare_to_reference(atol=1e-2, rtol=1e-2)
        # except:
        #     print("MISMATCH!")
        # else:
        #     print("PASSED!")

    def build_tensor_ir_module(self):
        input_tensors = []
        for node in self.test_graph.entrance_nodes:
            node.output[0].ref_data = torch.as_strided(
                node.output[0].ref_data,
                node.output[0].ref_data.size(),
                stride=node.kwargs["stride"],
            )
            input_tensors.append(node)
        output_tensors = [
            node for node in self.test_graph.nodes if node.is_output_node()
        ]

        with ir.Context() as ctx, ir.Location.unknown() as loc:
            nv_tensor_ir.register_dialect()

            module = ir.Module.create()
            input_types = list(
                map(
                    lambda x: next(iter(self.determine_tensor_ir_inout_tensor_type(x))),
                    input_tensors,
                )
            )
            output_types = list(
                map(
                    lambda x: next(iter(self.determine_tensor_ir_inout_tensor_type(x))),
                    output_tensors,
                )
            )

            # Create a mlir function signature for the kernel.
            function_type = ir.TypeAttr.get(
                T.function(inputs=input_types, results=output_types)
            )
            function_name = "graph"  # FIXME: Get this better

            # Create tensor ir graph
            graph = nv_tensor_ir.graph(
                function_name,
                function_type=function_type,
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
                        node_map[node] = nv_tensor_ir.convert(
                            next(
                                iter(
                                    self.determine_tensor_ir_inout_tensor_type(
                                        node, node.output[0]._data_type
                                    )
                                )
                            ),
                            node_map[node],
                        )
                    result.append(node_map[node])
                nv_tensor_ir.results_(result)
        print(module)
        module.operation.verify()

        return module

    def convert_and_splat_for_binary_pointwise(self, lsh, rsh, out_type):
        if isinstance(lsh.type, (ir.IntegerType, ir.FloatType)) and isinstance(
            rsh.type, nv_tensor_ir.TensorType
        ):
            lsh = nv_tensor_ir.splat(rsh.type, lsh)
        elif isinstance(rsh.type, (ir.IntegerType, ir.FloatType)) and isinstance(
            lsh.type, nv_tensor_ir.TensorType
        ):
            rsh = nv_tensor_ir.splat(lsh.type, rsh)
        if out_type != lsh.type:
            convert_value0 = nv_tensor_ir.convert(out_type, lsh)
        else:
            convert_value0 = lsh
        if out_type != rsh.type:
            convert_value1 = nv_tensor_ir.convert(out_type, rsh)
        else:
            convert_value1 = rsh
        return convert_value0, convert_value1

    def build_binary_operation(self, op_name, node, children, out_type, node_map):
        lsh, rsh = self.convert_and_splat_for_binary_pointwise(
            children[0], children[1], out_type
        )
        match op_name:
            case "add":
                node_map[node] = nv_tensor_ir.add(out_type, lsh, rsh)
            case "sub":
                node_map[node] = nv_tensor_ir.sub(out_type, lsh, rsh)
            case "mul":
                node_map[node] = nv_tensor_ir.mul(out_type, lsh, rsh)
            case "bias":
                node_map[node] = nv_tensor_ir.add(out_type, lsh, rsh)
            case "max":
                node_map[node] = nv_tensor_ir.max(out_type, lsh, rsh)
            case "min":
                node_map[node] = nv_tensor_ir.min(out_type, lsh, rsh)
            case "div":
                node_map[node] = nv_tensor_ir.div(out_type, lsh, rsh)
            case "mod":
                node_map[node] = nv_tensor_ir.mod(out_type, lsh, rsh)
            case "add_square":
                node_map[node] = nv_tensor_ir.add_square(out_type, lsh, rsh)
            case "logical_or":
                node_map[node] = nv_tensor_ir.or_(out_type, lsh, rsh)
            case "logical_and":
                node_map[node] = nv_tensor_ir.and_(out_type, lsh, rsh)
            case "logical_not":
                node_map[node] = nv_tensor_ir.not_(out_type, lsh, rsh)
            case "relu_backward":
                node_map[node] = nv_tensor_ir.relu_bwd(out_type, lsh, rsh)
            case _:
                print(
                    "Unimplemented binary Operation in Lowering to Tensor IR: ", op_name
                )

    def build_unary_operation(self, op_name, node, children, out_type, node_map):
        if out_type != children[0].type:
            convert_value = nv_tensor_ir.convert(out_type, children[0])
        else:
            convert_value = children[0]
        match op_name:
            case "tanh":
                node_map[node] = nv_tensor_ir.tanh_fwd(convert_value)
            case "abs":
                node_map[node] = nv_tensor_ir.abs(convert_value)
            case "ceil":
                node_map[node] = nv_tensor_ir.ceil(convert_value)
            case "floor":
                node_map[node] = nv_tensor_ir.floor(convert_value)
            case "cos":
                node_map[node] = nv_tensor_ir.cos(convert_value)
            case "sin":
                node_map[node] = nv_tensor_ir.sin(convert_value)
            case "tan":
                node_map[node] = nv_tensor_ir.tan(convert_value)
            case "exp":
                node_map[node] = nv_tensor_ir.exp(convert_value)
            case "log":
                node_map[node] = nv_tensor_ir.log(convert_value)
            case "neg":
                node_map[node] = nv_tensor_ir.neg(convert_value)
            case "rsqrt":
                node_map[node] = nv_tensor_ir.rsqrt(convert_value)
            case "sqrt":
                node_map[node] = nv_tensor_ir.sqrt(convert_value)
            case "erf":
                node_map[node] = nv_tensor_ir.erf(convert_value)
            case "reciprocal":
                node_map[node] = nv_tensor_ir.reciprocal(convert_value)
            case "relu":
                node_map[node] = nv_tensor_ir.relu_fwd(convert_value)
            case "sigmoid":
                node_map[node] = nv_tensor_ir.sigmoid_fwd(convert_value)
            case "elu":
                node_map[node] = nv_tensor_ir.elu_fwd(convert_value)
            case "gelu":
                node_map[node] = nv_tensor_ir.gelu_fwd(convert_value)
            case "gelu_approx_tanh":
                node_map[node] = nv_tensor_ir.gelu_approx_tanh_fwd(convert_value)
            case _:
                print(
                    "Unimplemented unary Operation in Lowering to Tensor IR: ", op_name
                )

    def build_tensor_ir_recursive(self, node, node_map, ip):
        if node in node_map.keys():
            return
        # map nodes -> tensor_ir values
        children = node.producer_nodes

        # Get children's values.
        for child in children:
            self.build_tensor_ir_recursive(child, node_map, ip)
        with ip as InsertionPoint:
            # No inheritance, just switch.
            if isinstance(node, tg.operation):
                # Cast children to math precision

                new_children = []
                # : Also if output data type is different from compute data type, cast output.
                for child in children:
                    new_children.append(node_map[child])

                children = new_children
                out_type, out_shape, out_stride, _ = (
                    self.determine_tensor_ir_inout_tensor_type(
                        node,
                        (
                            node.kwargs["compute_data_type"]
                            if "compute_data_type" in node.kwargs
                            else node.output[0].data_type
                        ),
                    )
                )
                match node.cudnn_op.__name__:  # FIXME(@xrouth): Try to match on something less fragile than "__name__"
                    case "reduction":
                        accumulator_type = ir.TypeAttr.get(
                            self.determine_tensor_ir_dtype(
                                self.test_graph.compute_data_type
                            )
                        )

                        reduction_mode = None

                        match node.kwargs["mode"]:
                            # FIXME (@xrouth): Support more reduction modes
                            case cudnn.reduction_mode.ADD:
                                reduction_mode = nv_tensor_ir.ReductionMode.add
                            case cudnn.reduction_mode.AMAX:
                                reduction_mode = nv_tensor_ir.ReductionMode.amax
                            case cudnn.reduction_mode.MIN:
                                reduction_mode = nv_tensor_ir.ReductionMode.min
                            case cudnn.reduction_mode.MAX:
                                reduction_mode = nv_tensor_ir.ReductionMode.max

                        out_dims = node.output[0].cudnn_tensor.get_dim()
                        print(
                            "out_type:",
                            out_type,
                            "out_shape:",
                            out_shape,
                            " out_stride:",
                            out_stride,
                        )
                        reduction_dim = 0
                        for s in out_stride:
                            if s == 0:
                                break
                            reduction_dim += 1
                        reduction_dimensions = [reduction_dim]
                        mlir_value = nv_tensor_ir.reduce(
                            out_type, children[0], reduction_dimensions, reduction_mode
                        )
                        node_map[node] = mlir_value
                    case "matmul":
                        compute_data_type = node.kwargs["compute_data_type"]
                        node_output = node.output[0]
                        compute_type, _, _, _ = (
                            self.determine_tensor_ir_inout_tensor_type(
                                node, compute_data_type
                            )
                        )
                        matmul_value = nv_tensor_ir.matmul(
                            compute_type, children[0], children[1]
                        )
                        node_map[node] = matmul_value
                    # POINTWISE:
                    case (
                        "bias"
                        | "add"
                        | "sub"
                        | "mul"
                        | "max"
                        | "min"
                        | "div"
                        | "mod"
                        | "add_square"
                        | "logical_or"
                        | "logical_and"
                        | "logical_not"
                        | "relu_backward"
                    ):
                        self.build_binary_operation(
                            node.cudnn_op.__name__, node, children, out_type, node_map
                        )
                    case (
                        "tanh"
                        | "abs"
                        | "ceil"
                        | "floor"
                        | "cos"
                        | "sin"
                        | "tan"
                        | "exp"
                        | "log"
                        | "neg"
                        | "rsqrt"
                        | "ert"
                        | "reciprocal"
                        | "relu"
                        | "sigmoid"
                        | "elu"
                        | "gelu"
                        | "gelu_approx_tanh"
                    ):
                        self.build_unary_operation(
                            node.cudnn_op.__name__, node, children, out_type, node_map
                        )
                    case "cmp_lt":
                        comparator = None
                        node_map[node] = nv_tensor_ir.cmp(
                            out_type, comparator, children[0], children[1]
                        )
                    case "cmp_ge":
                        comparator = None
                        node_map[node] = nv_tensor_ir.cmp(
                            out_type, comparator, children[0], children[1]
                        )
                    case "cmp_gt":
                        comparator = None
                        node_map[node] = nv_tensor_ir.cmp(
                            out_type, comparator, children[0], children[1]
                        )
                    case "pow":
                        # Is pow really vec-vec?
                        node_map[node] = nv_tensor_ir.pow(
                            out_type, children[0], children[1]
                        )
                    case "identity":
                        node_map[node] = node
                    case "conv_wgrad":
                        dy = children[0]
                        w = children[1]
                        accumulator_type_attr = ir.TypeAttr.get(
                            self.determine_tensor_ir_dtype(
                                self.test_graph.compute_data_type
                            )
                        )

                        out_dims = node.output[0].cudnn_tensor.get_dim()

                        pre_padding = node.kwargs["padding"]
                        post_padding = node.kwargs[
                            "padding"
                        ]  #  [] # FIXME(@xrouth): what should this be?
                        stride = node.kwargs["stride"]
                        dilation = node.kwargs["dilation"]

                        node_map[node] = nv_tensor_ir.conv_dgrad(
                            out_type,
                            dy,
                            w,
                            accumulator_type_attr,
                            pre_padding,
                            post_padding,
                            stride,
                            dilation,
                        )
                    case "conv_dgrad":
                        dy = children[0]
                        w = children[1]
                        accumulator_type_attr = ir.TypeAttr.get(
                            self.determine_tensor_ir_dtype(
                                self.test_graph.compute_data_type
                            )
                        )

                        out_dims = node.output[0].cudnn_tensor.get_dim()

                        pre_padding = node.kwargs["padding"]
                        post_padding = node.kwargs[
                            "padding"
                        ]  # [] # FIXME(@xrouth): what should this be?
                        stride = node.kwargs["stride"]
                        dilation = node.kwargs["dilation"]

                        node_map[node] = nv_tensor_ir.conv_dgrad(
                            out_type,
                            dy,
                            w,
                            accumulator_type_attr,
                            pre_padding,
                            post_padding,
                            stride,
                            dilation,
                        )
                    case "conv_fprop":
                        x = children[0]
                        w = children[1]
                        accumulator_type_attr = ir.TypeAttr.get(
                            self.determine_tensor_ir_dtype(
                                self.test_graph.compute_data_type
                            )
                        )

                        pre_padding = node.kwargs["padding"]
                        post_padding = node.kwargs[
                            "padding"
                        ]  # [] # FIXME(@xrouth): what should this be?
                        stride = node.kwargs["stride"]
                        dilation = node.kwargs["dilation"]

                        node_map[node] = nv_tensor_ir.conv_fprop(
                            out_type,
                            x,
                            w,
                            accumulator_type_attr,
                            pre_padding,
                            post_padding,
                            stride,
                            dilation,
                        )
                    case _:
                        print(
                            "Unimplemented Operation in Lowering to Tensor IR: ",
                            node.cudnn_op.__name__,
                        )
            else:
                print("not an op")
                # Nothing
