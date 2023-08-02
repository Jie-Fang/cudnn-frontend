import cudnn
import torch
from typing import Any
from dataclasses import dataclass, asdict, field
import copy

# @brief: Reference code
# @details: the methods mirror cudnn.pygraph methods and class constructors(__init__)
# @note: we can easily replace PytorchReference by CustomReference to use a different reference framework (one LoC change in test_graph below)
class PytorchReference:
    # @brief: run convolution without bias
    # @param kwargs: these are the named parameters used in the associated cudnn.pygraph.conv function
    #   The only difference is that the input tensors are replaced by pytorch tensors
    # @details: all this function needs to do is unpack the cudnn.pygraph function arguments and pass them to the pytorch equivalent
    @staticmethod
    def conv(kwargs):
        return [torch.nn.functional.conv2d(kwargs['image'], kwargs['weight'], bias = None, padding=kwargs["padding"], stride=kwargs["stride"], dilation=kwargs["dilation"])]

    # @brief: run relu
    # @details: unpack the cudnn.pygraph.relu parameters and pass them to the pytorch equivalent
    @staticmethod
    def relu(kwargs):
        return [torch.nn.functional.relu(kwargs["input"])]
    
    @staticmethod
    def batchnorm(kwargs):
        is_training = kwargs["norm_forward_phase"] == cudnn.norm_forward_phase.TRAINING
        momentum = kwargs["momentum"].item()
        epsilon=kwargs["epsilon"].item()
        output = torch.nn.functional.batch_norm(kwargs["input"], kwargs["in_running_mean"], kwargs["in_running_var"], weight= kwargs["scale"], bias=kwargs["bias"], training=is_training, momentum=momentum, eps=epsilon)
        
        output = [output]

        # torch's implementation only returns 1 output. 
        # Filling out the others with an amount of None's and have the reference check deal with it
        output.extend([None]*4)
        return output

    @staticmethod
    def matmul(kwargs):
        output = torch.bmm(kwargs['A'], kwargs['B'])
        return [output]

# Base class for Tensor and operation nodes
class test_node:
    __test__ = False
    def __init__(self, name):
        self.name = name
        self.visited = False
        self.is_explicit_output = False
        self.consumer_nodes = []
        self.producer_nodes = []

    def set_output_node(self, is_output):
        self.is_explicit_output = is_output

    def is_output_node(self):
        return self.is_explicit_output or len(self.consumer_nodes) == 0

    def clear_meta_data(self):
        self.visited = False

    def add_producer_node(self, node):
        self.producer_nodes.append(node)
        node.consumer_nodes.append(self)

    def set_visited(self):
        self.visited = True
    
    def is_visited(self):
        return self.visited

    def is_prereq_satisfied(self):
        prereq_satisfied = True
        # Since an input tensor doesnt have any producers, this will result in true
        for node in self.producer_nodes:
            prereq_satisfied = prereq_satisfied and node.is_visited()
        return prereq_satisfied

    # Function that needs to be overriden by its child classes
    def run_cudnn_code(self, cudnn_graph):
        print("NOT IMPLEMENTED")

    def build_cudnntree_recursive(self, cudnn_graph):
        if not self.is_visited() and self.is_prereq_satisfied():
            #print ("Checking {}".format(self.name))
            self.run_cudnn_code(cudnn_graph)
            self.set_visited()
            for node in self.consumer_nodes:
                node.build_cudnntree_recursive(cudnn_graph)

    def run_reftree_recursive(self):
        if not self.is_visited() and self.is_prereq_satisfied():
            #print ("Checking {}".format(self.name))
            self.run_ref()
            self.set_visited()
            for node in self.consumer_nodes:
                node.run_reftree_recursive()

class operation(test_node):
    
    # @param pyCuddnOp: cudnn.pygraph operation (e.g., cudnn.pygraph.conv)
    # @param ref_func: reference function for the associated cudnn_op
    # @param name: name for this operation (could be passed by kwargs as well)
    def __init__(self, cudnn_op, ref_func, name, num_outputs=1):
        super().__init__(name)
        
        self.cudnn_op = cudnn_op
        self.ref_func = ref_func

        self.output = []

        if num_outputs > 1:
            for i in range(num_outputs):
                self.output.append(test_tensor("{}_out_{}".format(name, i), self))
        else:
            self.output.append(test_tensor("{}_out".format(name), self))


    # @param kwargs: parameters for the associated cudnn_op
    # @details: All this function needs to do is: 
    #   * add the correct producers
    #   * store the kwargs (named parameters from the associated cudnn_op)
    def set_kwargs(self, kwargs):
        self.kwargs = kwargs
        # TODO(@mbreughe): Does .values() make an unnecessary copy?
        for v in self.kwargs.values():
            if isinstance(v, test_tensor):
                self.add_producer_node(v.parent_op)

    # @brief: Run the cudnn node
    # TODO(@mbreughe): extended to multiple output tensors
    def run_cudnn_code(self, cudnn_graph):
        # Besides input tensors, all kwargs can just be passed through to the cudnn method.
        # For the input tensor we need to extract the test_tensor's cudnn tensor.
        # Therefore: copy all kwargs, except for test_tensors
        # TODO(@mbreughe) Avoid copying these kwargs twice (ref and cudnn). Let's do this in the init function once
        new_kwargs = {x: self.kwargs[x] for x in self.kwargs if not isinstance(self.kwargs[x], test_tensor)}
        for x in self.kwargs:
            if isinstance(self.kwargs[x], test_tensor):
                new_kwargs[x] = self.kwargs[x].cudnn_tensor

        cudnn_res = self.cudnn_op(cudnn_graph, **new_kwargs)

        # in case we have multiple outputs
        if isinstance(cudnn_res, list):
            assert len(cudnn_res) == len(self.output)
            for output, cudnn_out in zip(self.output, cudnn_res):
                output.cudnn_tensor = cudnn_out
        else:
            self.output[0].cudnn_tensor = cudnn_res

    def run_ref(self):
        new_kwargs = {x: self.kwargs[x] for x in self.kwargs if not isinstance(self.kwargs[x], test_tensor)}
        for x in self.kwargs:
            if isinstance(self.kwargs[x], test_tensor):
                new_kwargs[x] = self.kwargs[x].ref_data
        ref_output = self.ref_func(new_kwargs)

        for output, ref_out in zip(self.output, ref_output):
            output.ref_data = ref_out

class random_tensor_generator(test_node):
    __test__ = False

    def __init__(self, kwargs, name, io_data_type):
        super().__init__(name)
        self.kwargs = kwargs

        self.output = [test_tensor(name+"_out", self)]

        data_type = io_data_type if not "data_type" in self.kwargs else self.kwargs["data_type"]
        self.output[0].set_data_type(data_type)

    def initialize_random_tensor(self):
        if self.output[0].ref_data is None:
            self.output[0].ref_data = torch.randn(self.kwargs["dim"], requires_grad=False, device="cuda", dtype=convert_to_torch_type(self.output[0].data_type))
            
            if self.get_layout() == "NHWC":
                self.output[0].ref_data = self.output[0].ref_data.to(memory_format=torch.channels_last)
    
    def get_value(self):
        self.initialize_random_tensor()
        return self.output[0].ref_data
    
    def get_layout(self):
        # TODO(mbreughe): Assume NCHW layout by default for now
        return "NCHW" if not "layout" in self.kwargs else self.kwargs["layout"]

    def run_cudnn_code(self, cudnn_graph):
        self.initialize_random_tensor()
        self.output[0].cudnn_tensor = cudnn_graph.tensor(name = self.name, dim = self.output[0].ref_data.size(), stride = self.output[0].ref_data.stride(), data_type = self.output[0].data_type)

    def run_ref(self):
        return self.get_value()
    
# TODO(@mbreughe): maybe subclass this from random_tensor_generator
# TODO(@mbreughe): consider putting the layout-kwargs as a separate helper function instead of setting it in the kwargs
class ConstantTensor(test_node):
    def __init__(self, kwargs, name, io_data_type, value):
        super().__init__(name)
        self.kwargs = kwargs

        self.output = [test_tensor(name+"_out", self)]

        data_type = io_data_type if not "data_type" in self.kwargs else self.kwargs["data_type"]
        self.output[0].set_data_type(data_type)

        self.value = value

    def instantiate(self):
        if self.output[0].ref_data is None:
            self.output[0].ref_data = torch.full(self.kwargs["dim"], self.value, requires_grad=False, device="cpu", dtype=convert_to_torch_type(self.output[0].data_type))
            
            if self.get_layout == "NHWC":
                self.output[0].ref_data = self.output.ref_data.to(memory_format=torch.channels_last)
    
    def get_value(self):
        self.instantiate()
        return self.output[0].ref_data
    
    def get_layout(self):
        # TODO(mbreughe): Assume NCHW layout by default for now
        return "NCHW" if not "layout" in self.kwargs else self.kwargs["layout"]

    def run_cudnn_code(self, cudnn_graph):
        self.instantiate()
        self.output[0].cudnn_tensor = cudnn_graph.tensor(name = self.name, dim = self.output[0].ref_data.size(), stride = self.output[0].ref_data.stride(), data_type = self.output[0].data_type)

    def run_ref(self):
        return self.get_value()

class test_tensor:
    __test__ = False

    def __init__(self, name, parent_op):
        self.name = name
        # The cudnn.pygraph.tensor instance associated with test_tensor
        self.cudnn_tensor = None
        # The reference data for this tensor
        self.ref_data = None

        self.parent_op = parent_op

    def set_data_type(self, data_type):
        self.data_type = data_type

        # Apply it immediately if a cudnn tensor was already created
        if self.cudnn_tensor is not None:
            self.cudnn_tensor.set_data_type(data_type)

    def set_stride(self, stride):
        self.stride = stride

        if self.cudnn_tensor is not None:
            self.cudnn_tensor.set_stride(stride)

    # TODO(@mbreughe): refactor this to avoid looking up strings
    def apply_modifiers(self):
        # If we ever specified a data type, apply it
        if "data_type" in dir(self):
            self.cudnn_tensor.set_data_type(self.data_type)

        if "stride" in dir(self):
            self.cudnn_tensor.set_stride(self.stride)
    

def convert_to_cudnn_type(torch_type):
    if torch_type == torch.float16:
        return cudnn.data_type.HALF
    elif torch_type == torch.float32:
        return cudnn.data_type.FLOAT
    else:
        raise ValueError("Unsupported tensor data type.")

    return

def convert_to_torch_type(cudnn_type):
    if cudnn_type == cudnn.data_type.HALF:
        return torch.float16
    elif cudnn_type == cudnn.data_type.FLOAT:
        return torch.float32
    else:
        raise ValueError("Unsupported tensor data type.")

    return

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
    torch_tensor = torch.as_strided(torch_tensor, torch_tensor.size(), tuple(cudnn_stride))

    return torch_tensor

# @brief: test_graph that mirrors cudnn.pygraph
# @details: this contains functionality to run both cudnn code as well as a reference
class test_graph:
    __test__ = False
    # Add data types, custom test name ,etc.
    def __init__(self, io_data_type = cudnn.data_type.HALF, intermediate_data_type = cudnn.data_type.FLOAT, compute_data_type = cudnn.data_type.FLOAT):
        self.uid_counter = 0
        self.nodes = []
        self.entrance_nodes = []
        self.graph_name = "test_graph"
        self.output_tensors = []
        self.io_data_type = io_data_type
        self.intermediate_data_type = intermediate_data_type
        self.compute_data_type = compute_data_type

    def getOutputs(self):
        return self.output_tensors

    # @brief: Add a convolution node to the graph
    def conv(self, **kwargs):
        return self.create_and_add_operation(kwargs, cudnn.pygraph.conv)

    # @brief: Add a relu to the graph
    def relu(self, **kwargs):
        return self.create_and_add_operation(kwargs, cudnn.pygraph.relu)
    
    def batchnorm(self, **kwargs):
        return self.create_and_add_operation(kwargs, cudnn.pygraph.batchnorm)
    
    def matmul(self, **kwargs):
        return self.create_and_add_operation(kwargs, cudnn.pygraph.matmul)

    # @brief: Add an input tensor to the graph
    def tensor(self, **kwargs):
        # Create a name if none provided
        if "name" in kwargs:
            name = kwargs["name"]
        else:
            name = self.create_unique_name("Tensor")

        test_tensor = random_tensor_generator(kwargs, name, self.io_data_type)

        self.nodes.append(test_tensor)
        # we are assuming only input tensors are explicitly created
        self.entrance_nodes.append(test_tensor)
        return test_tensor.output[0]

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
    
    # @brief: utility function to create unique names for the graph
    def create_unique_name(self, prefix):
        name = prefix + "_{}".format(self.uid_counter)
        self.uid_counter += 1
        return name

    # @brief: Create an operation, pass through the kwargs and set up dependencies
    # @param kwargs: the named arguments passed to a cudnn function
    # @param cudnn_op: the cudnn operation (e.g., cudnn.pygraph.conv)
    # @return: test_tensor (the output from the added operation)
    def create_and_add_operation(self, kwargs, cudnn_op):
        if "name" in kwargs:
            name = kwargs["name"]
        else:
            cudnn_opName = cudnn_op.__name__
            name = self.create_unique_name(cudnn_opName)

        node = test_graph.create_operation(cudnn_op, name)
        node.set_kwargs(kwargs)
        self.nodes.append(node)

        # cudnn returns either a single tensor, or a list of tensors
        # internally, we always store a list to allow for generalization
        if len(node.output) == 1:
            return node.output[0]
        else:
            return node.output

    # @brief: In esssence, this function is a factory function:
    # @param cudnn_op: the cudnn operation (e.g., cudnn.pygraph.conv)
    # @return: operation
    # it builds an operation by:
    #   * discovering the reference function based on the name of the cudnn op
    @staticmethod
    def create_operation(cudnn_op, name):
        cudnn_opName = cudnn_op.__name__
        # Fetch the reference function from the reference framework
        # Note that we use PytorchReference here, but we can make this arbitrary
        # TODO(@mbreughe): Add error handling code in case the function is not found
        ref_func = getattr(PytorchReference, cudnn_opName)

        # TODO(@mbreughe): automate this
        num_outputs = 1
        if cudnn_opName == "batchnorm":
            num_outputs = 5
        
        node = operation(cudnn_op, ref_func, name, num_outputs)
        return node

    # @brief: Utility function
    def clear_node_meta_data(self):
        for node in self.nodes:
            node.clear_meta_data()

    # @brief: Utility function to discover output nodes
    def mark_implicit_output_nodes(self):
        for node in self.nodes:
            if node.is_output_node():
                print ("Setting {} as output".format(node.name))
                for output in node.output:
                    output.cudnn_tensor.set_output(True)

    def apply_modifiers_to_node_output_tensors(self):
        for node in self.nodes:
            for tensor in node.output:
                tensor.apply_modifiers()

    # @brief: Build the cudnn graph
    # @note: this temporarily modifies the is_visited status of the nodes
    # @return the cudnn graph
    # @note we are relying on the user not the alter the graph. We can instead return them a copy, but this would be at a cost
    def build_cudnn_graph(self):
        # Setting up graph
        graph = cudnn.pygraph(self.graph_name, io_data_type = self.io_data_type, intermediate_data_type = self.intermediate_data_type, compute_data_type = self.compute_data_type)
        
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

        # Building graph
        graph.build()

        # Clear the "is_visited" status of the nodes
        self.clear_node_meta_data()

        self.cudnn_graph = graph
        return graph
    
    # @brief: check whether correct shape inferencing took place
    # @pre: build_cudnn_graph needs to have been invoked first
    def frontend_check(self, expected_dims):
        # Create a mapping of output names and their dimensions
        node_dim_mapping = {}
        for node in self.nodes:
            for output_tensor in node.output:
                node_dim_mapping[output_tensor.name] = output_tensor.cudnn_tensor.get_dim()
        
        # For every output we wish to check, check it
        for name in expected_dims:
            assert name in node_dim_mapping
            assert expected_dims[name] == node_dim_mapping[name]

    def set_io_data_type(self, data_type):
        self.io_data_type = data_type

    def set_intermediate_data_type(self, data_type):
        self.set_intermediate_data_type = data_type

    def set_compute_data_type(self, data_type):
        self.set_compute_data_type(data_type)

    # @brief: Run the reference for the associated graph
    # @note: this temporarily modifies the is_visited status of the nodes
    # TODO(@mbreughe): to preserve resources, we could consider clearing intermediate results when they are no longer needed
    # @pre: build_cudnn_graph may need to be invoked first because of its call to apply_modifiers_to_node_output_tensors, which may modify tensor layouts,
    # but also because it sets output nodes
    def calc_reference(self):
        # Clear the "is_visited" status of the nodes
        self.clear_node_meta_data()
        for node in self.entrance_nodes:
            node.run_reftree_recursive()
        
        output = []
        for node in self.nodes:
            if node.is_output_node():
                for out in node.output:
                    output.append(out.ref_data)

        # Clear the "is_visited" status of the nodes
        self.clear_node_meta_data()
        return output
    
    def create_workspace_and_variantpack(self):
        # Creating workspace
        workspace = torch.empty(self.cudnn_graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

        variant_pack = {}
        for node in self.entrance_nodes:
            for output in node.output:
                variant_pack[output.cudnn_tensor] = node.get_value()

        for node in self.nodes:
            if node.is_output_node():
                for output in node.output:
                    output_tensor = torch.zeros(*output.cudnn_tensor.get_dim(), dtype=convert_to_torch_type(output.cudnn_tensor.get_data_type()), device='cuda')

                    output_tensor = convert_strides(output_tensor, output.cudnn_tensor)

                    self.output_tensors.append(output_tensor)
                    variant_pack[output.cudnn_tensor] = self.output_tensors[-1]

        return (workspace, variant_pack)
    
    # @brief: Run the cudnn implementation and the reference, and compare
    # @note: This assumes build_cudnn_graph has already been run
    # @pre: build_cudnn_graph needs to be called first
    def cudnn_execute_and_compare_to_reference(self, atol=1e-2, rtol=1e-2):
        workspace, variant_pack = self.create_workspace_and_variantpack()

        # Run the cudnn graph
        print("Executing graph through cudnn")
        self.cudnn_graph.execute(variant_pack, workspace)

        # Run the reference
        print("Computing reference")
        ref_outputs = self.calc_reference()

        assert len(ref_outputs ) == len(self.getOutputs())

        number_outputs_tested = 0
        # Compare with reference
        for Y_expected, Y_actual in zip(ref_outputs, self.getOutputs()):
            # TODO (@mbreughe): Clean up this assumption:
            # If there are None's in the output, it's because the reference didn't provide actual output (eg batchnorm)
            # For now, we can assume that we don't care about this output and just let the reference pass
            # To be on the safe side, we will make sure at least one output was checked
            if Y_expected is None:
                continue
            
            torch.testing.assert_close(Y_expected, Y_actual, atol=atol, rtol=rtol)
            number_outputs_tested += 1
        
        assert number_outputs_tested >= 1