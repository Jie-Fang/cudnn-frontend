from data_types import convert_to_torch_type
import torch
import cudnn
from test_graph import (
    test_graph,
    ConstantTensor,
)
import utils


def is_column_major(cudnn_tensor):
    strides = cudnn_tensor.get_stride()
    assert len(strides) == 3
    return strides[2] > strides[1]


# @brief convert the strides of a torch_tensor to the ones of cudnn_tensor
# @param torch_tensor: tensor created by torch
# @param cudnn_tensor: tensor created by cudnn
def convert_strides(torch_tensor, cudnn_tensor):
    cudnn_stride = tuple(cudnn_tensor.get_stride())

    # Ensure we setup the correct strides
    torch_tensor = torch.as_strided(
        torch_tensor, torch_tensor.size(), tuple(cudnn_stride)
    )

    return torch_tensor


class cudnn_test_graph(test_graph):
    __test__ = False

    # Add data types, custom test name ,etc.
    def __init__(
        self,
        io_data_type=cudnn.data_type.HALF,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    ):
        super().__init__(io_data_type, intermediate_data_type, compute_data_type)
        self.set_backend_engine(-1)
        self.handle = cudnn.create_handle()

    def set_backend_engine(self, backendEngine):
        self.heuristics = []
        # Some backendEngines are ints, some are strings. Make them all string.
        engine = str(backendEngine)
        if engine == "-1":
            self.set_heuristics([cudnn.heur_mode.A])
        elif engine == "-2":
            self.set_heuristics([cudnn.heur_mode.B])
        elif engine == "-3":
            self.set_heuristics([cudnn.heur_mode.FALLBACK])
        else:
            print(
                "MB Unkown heuristic for backendEngine {} (type {}), trying A and FALLBACK".format(
                    engine, type(engine)
                )
            )
            self.set_heuristics([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])

    def set_heuristics(self, heuristics):
        self.heuristics = heuristics

    def conv_fprop(self, **kwargs):
        return self.create_and_add_operation(kwargs, cudnn.pygraph.conv_fprop)

    def conv_dgrad(self, **kwargs):
        return self.create_and_add_operation(kwargs, cudnn.pygraph.conv_dgrad)

    def conv_wgrad(self, **kwargs):
        return self.create_and_add_operation(kwargs, cudnn.pygraph.conv_wgrad)

    # @brief: Add a relu to the graph
    def relu(self, **kwargs):
        return self.create_and_add_operation(kwargs, cudnn.pygraph.relu)

    def batchnorm(self, **kwargs):
        return self.create_and_add_operation(kwargs, cudnn.pygraph.batchnorm)

    def matmul(self, **kwargs):
        return self.create_and_add_operation(kwargs, cudnn.pygraph.matmul)

    def add(self, **kwargs):
        return self.create_and_add_operation(kwargs, cudnn.pygraph.add)

    def bias(self, **kwargs):
        return self.create_and_add_operation(kwargs, cudnn.pygraph.bias)

    def reduction(self, **kwargs):
        return self.create_and_add_operation(kwargs, cudnn.pygraph.reduction)

    def tensor_cpu_constant(self, value, **kwargs):
        # Create a name if none provided
        if "name" in kwargs:
            name = kwargs["name"]
        else:
            name = self.create_unique_name("Tensor")

        node = ConstantTensor(kwargs, name, self.io_data_type, value)
        self.nodes.append(node)
        self.entrance_nodes.append(node)
        return node.output[0]

    # @brief: Create a pycudnn node from the legacy_op
    # @brief: Create an operation, pass through the kwargs and set up dependencies
    # @param kwargs: the named arguments passed to a cudnn function
    # @param cudnn_op: the cudnn operation (e.g., cudnn.pygraph.conv)
    # @return: test_tensor (the output from the added operation)
    def create_and_add_operation(self, kwargs, cudnn_op):
        return super().create_and_add_operation(kwargs, op=cudnn_op)

    def create_test_graph_node(self, legacy_op):
        name = legacy_op.get_name()
        op_name = legacy_op.get_pycudnn_operation_name()
        op_ptr = getattr(cudnn.pygraph, op_name)
        return test_graph.create_operation(op_name, name, op_ptr)

    def mark_implicit_output_nodes(self):
        for node in self.nodes:
            if node.is_output_node():
                print("Setting {} as output".format(node.name))
                for output in node.output:
                    if output.cudnn_tensor is not None:
                        output.cudnn_tensor.set_output(True)

    # @brief: check whether correct shape inferencing took place
    # @pre: build_cudnn_graph needs to have been invoked first
    def frontend_check(self, expected_dims):
        # Create a mapping of output names and their dimensions
        node_dim_mapping = {}
        for node in self.nodes:
            for output_tensor in node.output:
                node_dim_mapping[output_tensor.name] = (
                    output_tensor.cudnn_tensor.get_dim()
                )

        # For every output we wish to check, check it
        for name in expected_dims:
            assert name in node_dim_mapping
            assert expected_dims[name] == node_dim_mapping[name]

    # @brief: Build the cudnn graph
    # @note: this temporarily modifies the is_visited status of the nodes
    # @return the cudnn graph
    # @note we are relying on the user not the alter the graph. We can instead return them a copy, but this would be at a cost
    def build_cudnn_graph(self, use_heuristic_list=True):

        # Setting up graph
        graph = cudnn.pygraph(
            self.graph_name,
            io_data_type=self.io_data_type,
            intermediate_data_type=self.intermediate_data_type,
            compute_data_type=self.compute_data_type,
            handle=self.handle,
        )
        utils.reportCurrentTime("cudnn.pygraph")

        # TODO(@mbreughe): Change this. We don't want to invoke cudnn calls this way since we change the order
        # a developer may have intended. It is useful when building from json graphs, but not when
        # manually setting up graphs. This is like building a house of cards.
        self.clear_node_meta_data()
        for node in self.entrance_nodes:
            node.build_cudnntree_recursive(graph)

        # Once we constructed the cudnn graph, it's time to apply any explicit modifiers to each node's output tensors
        # test_graph creates dummy output tensors to allow constructing of a graph.
        # However, the actual output tensors are created once we call build_cudnntree_recursive. This means any modifications
        # such as Y.set_data_type(FLOAT) actually happened on the dummy tensors. Here we propogate this to the real tensors
        self.apply_modifiers_to_node_output_tensors()

        # Setting implicit output nodes"
        self.mark_implicit_output_nodes()

        utils.reportCurrentTime("recursive_tree_build")

        graph.build(self.heuristics) if use_heuristic_list else graph.build()
        utils.reportCurrentTime("graph.build")

        # Clear the "is_visited" status of the nodes
        self.clear_node_meta_data()

        self.cudnn_graph = graph

        utils.reportCurrentTime("post_build")
        return graph

    def create_workspace_and_variantpack(self):
        # Creating workspace
        try:
            workspace = torch.empty(
                self.cudnn_graph.get_workspace_size(), device="cuda", dtype=torch.uint8
            )
        except Exception as error:
            workspace = torch.empty(
                self.cudnn_graph.get_workspace_size(), device="cpu", dtype=torch.uint8
            )

        variant_pack = {}
        for node in self.entrance_nodes:
            for output in node.output:
                variant_pack[output.cudnn_tensor] = node.get_value()

        for node in self.nodes:
            if node.is_output_node():
                for output in node.output:
                    try:
                        output_tensor = torch.zeros(
                            *output.cudnn_tensor.get_dim(),
                            dtype=eval(
                                convert_to_torch_type(
                                    output.cudnn_tensor.get_data_type()
                                )
                            ),
                            device="cuda",
                        )
                    except Exception as error:
                        output_tensor = torch.zeros(
                            *output.cudnn_tensor.get_dim(),
                            dtype=eval(
                                convert_to_torch_type(
                                    output.cudnn_tensor.get_data_type()
                                )
                            ),
                            device="cpu",
                        )

                    output_tensor = convert_strides(output_tensor, output.cudnn_tensor)

                    self.output_tensors.append(output_tensor)
                    variant_pack[output.cudnn_tensor] = self.output_tensors[-1]

        utils.reportCurrentTime("create_workspace_and_variantpack")

        return (workspace, variant_pack)

    def cudnn_execute(self, timingLoop=1):
        # Set the random seed here: all random tensors that are entry nodes
        # are created by create_workspace_and_variantpack().
        # TODO(@mbreughe): verify the above statement and move seed initialization
        # to variantpack creation routine
        # TODO(@mbreughe): allow dialing in any seed
        torch.manual_seed(0)
        workspace, variant_pack = self.create_workspace_and_variantpack()

        # Run the cudnn graph
        print("Executing graph through cudnn")

        # TODO(@mbreughe): modify so that every run has cold caches
        # warm the caches
        self.cudnn_graph.execute(variant_pack, workspace)

        # TODO(@mbreughe:) Handle the case for -T1. Right now, -T1 and -T0 both run the graph only once, without timing
        if timingLoop > 1:
            # TODO(@mbreughe): Support cold caches by using multiple variant_packs
            (min_rt, avg_rt, max_rt) = utils.measure_gpu_runtime(
                lambda: self.cudnn_graph.execute(variant_pack, workspace), timingLoop
            )

        utils.reportCurrentTime("graph.execute")

        torch.cuda.synchronize()

        cudnn.destroy_handle(self.handle)

    # @brief: Run the cudnn implementation and the reference, and compare
    # @note: This assumes build_cudnn_graph has already been run
    # @pre: build_cudnn_graph needs to be called first
    def cudnn_execute_and_compare_to_reference(self, atol=1e-2, rtol=1e-2):
        # Set up workspace and variant pack, then execute.
        self.cudnn_execute(timingLoop=1)

        # Run the reference
        print("Computing reference")
        ref_outputs = self.calc_reference()

        assert len(ref_outputs) == len(self.getOutputs())

        number_outputs_tested = 0

        # Compare with reference
        for Y_expected, Y_actual in zip(ref_outputs, self.getOutputs()):
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

        assert number_outputs_tested >= 1
        print("PASSED: cudnn and reference match")

        utils.reportCurrentTime("assert_close")
