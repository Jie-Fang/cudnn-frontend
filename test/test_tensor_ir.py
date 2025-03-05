import test_graph as tg
import torch
import utils

from nv_tensor_ir import ir
from nv_tensor_ir.dialects import nv_tensor_ir

import nv_tensor_ir.extras.types as T
from data_types import DataType, convert_datatype


def cal_shapeK(input_data_type):
    if input_data_type in [DataType.FLOAT, DataType.INT32]:
        return 8
    else:
        return 16


def generate_tensorir_compilation_configs(kmmaShapeK=16, cta_count=1):
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
            configs.append([tile_size, mma_shape, cluster_shape, cta_count])

    return configs


def get_tensorir_compilation_config(tensorir_args, concrete_test_dict):
    tile_size = [128, 128, 64]
    mma_shape = [128, 128, 16]
    cluster_shape = [1, 1, 1]
    cta_count = 1

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

    concrete_test_dict["tile_size"] = tile_size
    concrete_test_dict["cluster_shape"] = cluster_shape
    concrete_test_dict["mma_shape"] = mma_shape
    concrete_test_dict["cta_count"] = cta_count

    return [tile_size, mma_shape, cluster_shape, cta_count]


def find_mismatches(tensor_a, tensor_b, atol, rtol):
    close_mask = torch.isclose(tensor_a, tensor_b, atol=atol, rtol=rtol)
    diff_indices = torch.nonzero(~close_mask)
    for idx in diff_indices:
        diff = tensor_a[tuple(idx.tolist())] - tensor_b[tuple(idx.tolist())]
        print(
            f"Index: {tuple(idx.tolist())}, Tensor A Value: {tensor_a[tuple(idx.tolist())]}, Tensor B Value: {tensor_b[tuple(idx.tolist())]}, diff = {diff}"
        )


class test_tensor_ir:
    def __init__(self, test_graph):
        self.test_graph = test_graph
        self.outputs = []
        self.ref_outputs = None

    def determine_tensor_ir_inout_tensor_type(self, node, dtype=None):
        assert len(node.output) == 1
        test_tensor = node.output[0]
        if dtype is None:
            dtype = eval(convert_datatype(test_tensor.data_type, "tensorir"))
        else:
            dtype = eval(convert_datatype(dtype, "tensorir"))

        if hasattr(node, "is_by_value"):
            return dtype, [1], [1], dtype

        shape = []
        stride = []
        idx = 0

        if test_tensor.ref_data is None:
            ori_stride = test_tensor.stride
            ori_shape = test_tensor.dim
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
        ty = nv_tensor_ir.TensorType.get(shape=shape, stride=stride, datatype=dtype)
        return ty, shape, stride, dtype

    def calc_ref(self):
        self.ref_outputs = self.test_graph.calc_reference()

    def tensorir_compare_to_reference(self, atol=1e-2, rtol=1e-2):
        passed = True
        assert len(self.ref_outputs) == len(self.outputs)
        number_outputs_tested = 0
        output_idx = 0
        # Compare with reference
        for Y_expected, Y_actual in zip(self.ref_outputs, self.outputs):
            if Y_expected.device.type != Y_actual.device.type:
                if Y_expected.device.type == "cuda":
                    Y_expected = Y_expected.to("cpu")
                else:
                    Y_actual = Y_actual.to("cpu")
            # TODO (@mbreughe): Clean up this assumption:
            # If there are None's in the output, it's because the reference didn't provide actual output (eg batchnorm)
            # For now, we can assume that we don't care about this output and just let the reference pass
            # To be on the safe side, we will make sure at least one output was checked
            if Y_expected is None:
                continue

            if Y_expected.shape != Y_actual.shape:
                print(
                    "WARNING: reference and actual output shapes differ ({} resp., {})".format(
                        Y_expected.shape, Y_actual.shape
                    )
                )

            # find_mismatches(Y_expected, Y_actual, atol, rtol)#
            try:
                torch.testing.assert_close(Y_expected, Y_actual, atol=atol, rtol=rtol)
            except Exception as e:
                passed = False
                print("Assertion Error:", str(e))
                print("Stack trace:")
                import traceback

                traceback.print_exc()

            number_outputs_tested += 1
            output_idx += 1
        assert number_outputs_tested >= 1

        utils.reportCurrentTime("assert_close")
        return passed

    def run_tensor_ir_module(
        self,
        module,
        kernel_configs,
        dump_ir_path,
        mlir_timing,
        host_jitting,
        timing_loop=1,
        atol=1e-2,
        rtol=1e-2,
    ):
        graph_analysis = nv_tensor_ir.GraphAnalysis(module)
        a_partition_idx, b_partition_idx = -1, -1
        # Determine partition indices for A and B
        for i, node in enumerate(self.test_graph.entrance_nodes):
            partition_type = graph_analysis.get_graph_operand_partition(module, i)
            if (
                partition_type
                == nv_tensor_ir.GraphPartitionType.GRAPH_PARTITION_MAINLOOP_A
            ):
                a_partition_idx = i
            elif (
                partition_type
                == nv_tensor_ir.GraphPartitionType.GRAPH_PARTITION_MAINLOOP_B
            ):
                b_partition_idx = i
        # Ensure valid partition indices were found
        assert (
            a_partition_idx != -1 and b_partition_idx != -1
        ), "Invalid partition indices."

        a_tensor_dim = self.test_graph.entrance_nodes[a_partition_idx].output[0].dim
        b_tensor_dim = self.test_graph.entrance_nodes[b_partition_idx].output[0].dim
        B, M, N, K = a_tensor_dim[0], a_tensor_dim[1], b_tensor_dim[2], a_tensor_dim[2]
        problem_size = nv_tensor_ir.GemmProblemSize(B, M, N, K)
        print("problem_size:", B, M, N, K)
        device = torch.device("cuda:0")

        # iterate all kernel inputs automatically
        tensor_desc = nv_tensor_ir.VectorTensorOperandDescriptor()
        inputs_gpu = []
        for node in self.test_graph.entrance_nodes:
            torch_mem = node.get_value()
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
                        eval(convert_datatype(node.output[0].data_type, "cask")),
                        torch_mem[0].data_ptr(),
                    )
                )
                tensor_desc.append(tensor_operand)

        outputs_gpu = []
        for node in self.test_graph.nodes:
            if node.is_output_node():
                output_tensor = node.output[0]

                gpu_mem = torch.empty(
                    output_tensor.dim,
                    dtype=eval(convert_datatype(output_tensor.data_type, "torch")),
                    device=device,
                )
                outputs_gpu.append(gpu_mem)
                tensor_desc.append(
                    nv_tensor_ir.TensorOperandDescriptor(
                        nv_tensor_ir.TensorDescriptor(
                            gpu_mem.data_ptr(),
                            nv_tensor_ir.LayoutDescriptor(gpu_mem.stride()),
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
            for tile_size, mma_shape, cluster_shape, cta_count in kernel_configs:
                cloned_module = ir.Module.parse(str(module))
                cask_context = nv_tensor_ir.create_cask_context()
                compiler = nv_tensor_ir.Compiler(cask_context)

                compile_option = nv_tensor_ir.TensorIRCompilationOption(
                    10,  # Hardcoded for blackwell
                    nv_tensor_ir.GraphCategory.kGemm,  # Hardcode for Gemm
                    nv_tensor_ir.TensorConversionOptions(
                        tile_size, mma_shape, cluster_shape, cta_count
                    ),
                    nv_tensor_ir.DebugOptions(dump_ir_path, mlir_timing),
                    host_jitting,
                )

                print(
                    f"#### Running tile_size={tile_size}, mma_shape={mma_shape}, cluster_shape={cluster_shape}, cta_count={cta_count}"
                )
                shader = compiler.compile(cloned_module, compile_option)
                execution_plan = nv_tensor_ir.ExecutionPlan(shader, args)
                execution_plan.initialize()
                # TODO: Do we need to dump the launch config for debugging?
                # execution_plan.dump_launch_config()
                if timing_loop == 0:
                    execution_plan.launch()
                    self.outputs = outputs_gpu
                    if not self.ref_outputs:
                        self.calc_ref()
                    passed = self.tensorir_compare_to_reference(atol, rtol)
                    assert passed, "Mismatch between TensorIR and reference"
                elif timing_loop == 1:
                    execution_plan.launch()
                else:
                    # warm the caches
                    execution_plan.launch()
                    import utils

                    # TODO: Shall we use median instead of average?
                    (_, avg_rt, _) = utils.measure_gpu_runtime(
                        lambda: execution_plan.launch(), timing_loop
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

        with ir.Context() as ctx, ir.Location.unknown() as loc:
            ctx.enable_multithreading(False)
            nv_tensor_ir.register_dialect()

            module = ir.Module.create(loc)
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
            function_name = json_test_name

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
        module.operation.verify()

        return module

    def convert_and_splat_for_binary_pointwise(self, lsh, rsh, out_type):
        if isinstance(lsh.type, (ir.IntegerType, ir.FloatType)) and isinstance(
            rsh.type, nv_tensor_ir.TensorType
        ):
            lsh = nv_tensor_ir.splat(
                nv_tensor_ir.TensorType.get_from_tensor_type(rsh.type, lsh.type), lsh
            )
        elif isinstance(rsh.type, (ir.IntegerType, ir.FloatType)) and isinstance(
            lsh.type, nv_tensor_ir.TensorType
        ):
            rsh = nv_tensor_ir.splat(
                nv_tensor_ir.TensorType.get_from_tensor_type(lsh.type, rsh.type), rsh
            )
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
                match node.op_name:  # FIXME(@xrouth): Try to match on something less fragile than "__name__"
                    case "reduction":
                        accumulator_type = ir.TypeAttr.get(
                            eval(
                                convert_datatype(
                                    self.test_graph.compute_data_type, "tensorir"
                                )
                            )
                        )

                        reduction_mode = None
                        if "reduction_mode.ADD" in node.kwargs["mode"]:
                            reduction_mode = nv_tensor_ir.ReductionMode.add
                        elif "reduction_mode.AMAX" in node.kwargs["mode"]:
                            reduction_mode = nv_tensor_ir.ReductionMode.amax
                        elif "reduction_mode.MIN" in node.kwargs["mode"]:
                            reduction_mode = nv_tensor_ir.ReductionMode.min
                        elif "reduction_mode.MAX" in node.kwargs["mode"]:
                            reduction_mode = nv_tensor_ir.ReductionMode.max
                        input_datatype = nv_tensor_ir.get_tensor_datatype(
                            children[0].type
                        )
                        output_datatype = nv_tensor_ir.get_tensor_datatype(out_type)
                        if input_datatype != output_datatype:
                            convert_value = nv_tensor_ir.convert(
                                nv_tensor_ir.TensorType.get(
                                    shape=out_shape,
                                    stride=out_stride,
                                    datatype=output_datatype,
                                ),
                                children[0],
                            )
                        else:
                            convert_value = children[0]
                        reduction_dimensions = []
                        reduction_dim = 0
                        for s in out_stride:
                            if s == 0:
                                reduction_dimensions.append(reduction_dim)
                            reduction_dim += 1
                        mlir_value = nv_tensor_ir.reduce(
                            nv_tensor_ir.TensorType.get(
                                shape=out_shape,
                                stride=out_stride,
                                datatype=output_datatype,
                            ),
                            convert_value,
                            reduction_dimensions,
                            reduction_mode,
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
                            node.op_name, node, children, out_type, node_map
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
                        | "erf"
                        | "reciprocal"
                        | "relu"
                        | "sigmoid"
                        | "elu"
                        | "gelu"
                        | "gelu_approx_tanh"
                    ):
                        self.build_unary_operation(
                            node.op_name, node, children, out_type, node_map
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
                            eval(
                                convert_datatype(
                                    self.test_graph.compute_data_type, "tensorir"
                                )
                            )
                        )

                        out_dims = node.output[0].dim

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
                            eval(
                                convert_datatype(
                                    self.test_graph.compute_data_type, "tensorir"
                                )
                            )
                        )

                        out_dims = node.output[0].dim

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
                            eval(
                                convert_datatype(
                                    self.test_graph.compute_data_type, "tensorir"
                                )
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
                            node.op_name,
                        )
            else:
                print("not an op")
                # Nothing
