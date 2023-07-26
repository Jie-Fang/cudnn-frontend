import pycudnn
import torch
from typing import Any
from dataclasses import dataclass, asdict, field
import copy

# @brief: Reference code
# @details: the methods mirror pycudnn.pygraph methods and class constructors(__init__)
# @note: we can easily replace PytorchReference by CustomReference to use a different reference framework (one LoC change in TestGraph below)
class PytorchReference:
    # @brief: run convolution without bias
    # @param kwargs: these are the named parameters used in the associated pycudnn.pygraph.conv function
    #   The only difference is that the input tensors are replaced by pytorch tensors
    # @details: all this function needs to do is unpack the pycudnn.pygraph function arguments and pass them to the pytorch equivalent
    @staticmethod
    def conv(kwargs):
        return [torch.nn.functional.conv2d(kwargs['image'], kwargs['weight'], bias = None, padding=kwargs["padding"], stride=kwargs["stride"], dilation=kwargs["dilation"])]

    # @brief: run relu
    # @details: unpack the pycudnn.pygraph.relu parameters and pass them to the pytorch equivalent
    @staticmethod
    def relu(kwargs):
        return [torch.nn.functional.relu(kwargs["input"])]
    
    @staticmethod
    def batchnorm(kwargs):
        is_training = kwargs["norm_forward_phase"] == pycudnn.norm_forward_phase.TRAINING
        momentum = kwargs["momentum"].item()
        epsilon=kwargs["epsilon"].item()
        output = torch.nn.functional.batch_norm(kwargs["input"], kwargs["in_running_mean"], kwargs["in_running_var"], weight= kwargs["scale"], bias=kwargs["bias"], training=is_training, momentum=momentum, eps=epsilon)
        
        output = [output]

        # torch's implementation only returns 1 output. 
        # Filling out the others with an amount of None's and have the reference check deal with it
        output.extend([None]*4)
        return output

# Base class for Tensor and Operation nodes
class TestNode:
    __test__ = False
    def __init__(self, name):
        self.name = name
        self.is_visited = False
        self.is_explicit_output = False
        self.consumerNodes = []
        self.producerNodes = []

    def setOutputNode(self, is_output):
        self.is_explicit_output = is_output

    def isOutputNode(self):
        return self.is_explicit_output or len(self.consumerNodes) == 0

    def clearMetaData(self):
        self.is_visited = False

    def addProducerNode(self, node):
        self.producerNodes.append(node)
        node.consumerNodes.append(self)

    def setVisited(self):
        self.is_visited = True
    
    def isVisited(self):
        return self.is_visited

    def isPrereqSatisfied(self):
        preReqSatisfied = True
        # Since an input tensor doesnt have any producers, this will result in true
        for node in self.producerNodes:
            preReqSatisfied = preReqSatisfied and node.isVisited()
        return preReqSatisfied

    # Function that needs to be overriden by its child classes
    def runPyCudnnCode(self, pyCudnnGraph):
        print("NOT IMPLEMENTED")

    def buildPycudnnTreeRecursive(self, pyCudnnGraph):
        if not self.isVisited() and self.isPrereqSatisfied():
            #print ("Checking {}".format(self.name))
            self.runPyCudnnCode(pyCudnnGraph)
            self.setVisited()
            for node in self.consumerNodes:
                node.buildPycudnnTreeRecursive(pyCudnnGraph)

    def runRefTreeRecursive(self):
        if not self.isVisited() and self.isPrereqSatisfied():
            #print ("Checking {}".format(self.name))
            self.runRef()
            self.setVisited()
            for node in self.consumerNodes:
                node.runRefTreeRecursive()

class Operation(TestNode):
    
    # @param pyCuddnOp: pycudnn.pygraph operation (e.g., pycudnn.pygraph.conv)
    # @param refFunc: reference function for the associated pyCudnnOp
    # @param name: name for this Operation (could be passed by kwargs as well)
    def __init__(self, pyCudnnOp, refFunc, name, num_outputs=1):
        super().__init__(name)
        
        self.pyCudnnOp = pyCudnnOp
        self.refFunc = refFunc

        self.output = []
        for i in range(num_outputs):
            self.output.append(TestTensor("{}_out_{}".format(name, i), self))


    # @param kwargs: parameters for the associated pyCudnnOp
    # @details: All this function needs to do is: 
    #   * add the correct producers
    #   * store the kwargs (named parameters from the associated pyCudnnOp)
    def setKwargs(self, kwargs):
        self.kwargs = kwargs
        # TODO(@mbreughe): Does .values() make an unnecessary copy?
        for v in self.kwargs.values():
            if isinstance(v, TestTensor):
                self.addProducerNode(v.parent_op)

    # @brief: Run the pycudnn node
    # TODO(@mbreughe): extended to multiple output tensors
    def runPyCudnnCode(self, pyCudnnGraph):
        # Besides input tensors, all kwargs can just be passed through to the pycudnn method.
        # For the input tensor we need to extract the TestTensor's pycudnn tensor.
        # Therefore: copy all kwargs, except for TestTensors
        # TODO(@mbreughe) Avoid copying these kwargs twice (ref and pycudnn). Let's do this in the init function once
        new_kwargs = {x: self.kwargs[x] for x in self.kwargs if not isinstance(self.kwargs[x], TestTensor)}
        for x in self.kwargs:
            if isinstance(self.kwargs[x], TestTensor):
                new_kwargs[x] = self.kwargs[x].pyCudnnTensor

        pycudnn_res = self.pyCudnnOp(pyCudnnGraph, **new_kwargs)

        # in case we have multiple outputs
        if isinstance(pycudnn_res, list):
            assert len(pycudnn_res) == len(self.output)
            for output, pycudnn_out in zip(self.output, pycudnn_res):
                output.pyCudnnTensor = pycudnn_out
        else:
            self.output[0].pyCudnnTensor = pycudnn_res

    def runRef(self):
        new_kwargs = {x: self.kwargs[x] for x in self.kwargs if not isinstance(self.kwargs[x], TestTensor)}
        for x in self.kwargs:
            if isinstance(self.kwargs[x], TestTensor):
                new_kwargs[x] = self.kwargs[x].ref_data
        ref_output = self.refFunc(new_kwargs)

        for output, ref_out in zip(self.output, ref_output):
            output.ref_data = ref_out

class RandomTensorGenerator(TestNode):
    __test__ = False

    def __init__(self, kwargs, name, io_data_type):
        super().__init__(name)
        self.kwargs = kwargs

        self.output = [TestTensor(name+"_out", self)]

        data_type = io_data_type if not "data_type" in self.kwargs else self.kwargs["data_type"]
        self.output[0].set_data_type(data_type)

    def instantiateRandomTensor(self):
        if self.output[0].ref_data is None:
            self.output[0].ref_data = torch.randn(self.kwargs["dim"], requires_grad=False, device="cuda", dtype=convert_to_torch_type(self.output[0].data_type))
            
            if self.getLayout() == "NHWC":
                self.output[0].ref_data = self.output[0].ref_data.to(memory_format=torch.channels_last)
    
    def getValue(self):
        self.instantiateRandomTensor()
        return self.output[0].ref_data
    
    def getLayout(self):
        # TODO(mbreughe): Assume NCHW layout by default for now
        return "NCHW" if not "layout" in self.kwargs else self.kwargs["layout"]

    def runPyCudnnCode(self, pyCudnnGraph):
        self.instantiateRandomTensor()
        self.output[0].pyCudnnTensor = pyCudnnGraph.tensor(name = self.name, dim = self.output[0].ref_data.size(), stride = self.output[0].ref_data.stride(), data_type = self.output[0].data_type)

    def runRef(self):
        return self.getValue()
    
# TODO(@mbreughe): maybe subclass this from RandomTensorGenerator
# TODO(@mbreughe): consider putting the layout-kwargs as a separate helper function instead of setting it in the kwargs
class ConstantTensor(TestNode):
    def __init__(self, kwargs, name, io_data_type, value):
        super().__init__(name)
        self.kwargs = kwargs

        self.output = [TestTensor(name+"_out", self)]

        data_type = io_data_type if not "data_type" in self.kwargs else self.kwargs["data_type"]
        self.output[0].set_data_type(data_type)

        self.value = value

    def instantiate(self):
        if self.output[0].ref_data is None:
            self.output[0].ref_data = torch.full(self.kwargs["dim"], self.value, requires_grad=False, device="cpu", dtype=convert_to_torch_type(self.output[0].data_type))
            
            if self.getLayout == "NHWC":
                self.output[0].ref_data = self.output.ref_data.to(memory_format=torch.channels_last)
    
    def getValue(self):
        self.instantiate()
        return self.output[0].ref_data
    
    def getLayout(self):
        # TODO(mbreughe): Assume NCHW layout by default for now
        return "NCHW" if not "layout" in self.kwargs else self.kwargs["layout"]

    def runPyCudnnCode(self, pyCudnnGraph):
        self.instantiate()
        self.output[0].pyCudnnTensor = pyCudnnGraph.tensor(name = self.name, dim = self.output[0].ref_data.size(), stride = self.output[0].ref_data.stride(), data_type = self.output[0].data_type)

    def runRef(self):
        return self.getValue()

class TestTensor:
    __test__ = False

    def __init__(self, name, parent_op):
        self.name = name
        # The pycudnn.pygraph.tensor instance associated with TestTensor
        self.pyCudnnTensor = None
        # The reference data for this tensor
        self.ref_data = None

        self.parent_op = parent_op

    def set_data_type(self, data_type):
        self.data_type = data_type

        # Apply it immediately if a pycudnn tensor was already created
        if self.pyCudnnTensor is not None:
            self.pyCudnnTensor.set_data_type(data_type)

    def apply_modifiers(self):
        # If we ever specified a data type, apply it
        if "data_type" in dir(self):
            self.pyCudnnTensor.set_data_type(self.data_type)
    

def convert_to_cudnn_type(torch_type):
    if torch_type == torch.float16:
        return pycudnn.data_type.HALF
    elif torch_type == torch.float32:
        return pycudnn.data_type.FLOAT
    else:
        raise ValueError("Unsupported tensor data type.")

    return

def convert_to_torch_type(cudnn_type):
    if cudnn_type == pycudnn.data_type.HALF:
        return torch.float16
    elif cudnn_type == pycudnn.data_type.FLOAT:
        return torch.float32
    else:
        raise ValueError("Unsupported tensor data type.")

    return

# @brief: TestGraph that mirrors pycudnn.pygraph
# @details: this contains functionality to run both pycudnn code as well as a reference
class TestGraph:
    __test__ = False
    # Add data types, custom test name ,etc.
    def __init__(self, io_data_type = pycudnn.data_type.HALF, intermediate_data_type = pycudnn.data_type.FLOAT, compute_data_type = pycudnn.data_type.FLOAT):
        self.uid_counter = 0
        self.nodes = []
        self.entrance_nodes = []
        self.graph_name = "TestGraph"
        self.output_tensors = []
        self.io_data_type = io_data_type
        self.intermediate_data_type = intermediate_data_type
        self.compute_data_type = compute_data_type

    def getOutputs(self):
        return self.output_tensors

    # @brief: Add a convolution node to the graph
    def conv(self, **kwargs):
        return self.createAndAddOperation(kwargs, pycudnn.pygraph.conv)

    # @brief: Add a relu to the graph
    def relu(self, **kwargs):
        return self.createAndAddOperation(kwargs, pycudnn.pygraph.relu)
    
    def batchnorm(self, **kwargs):
        return self.createAndAddOperation(kwargs, pycudnn.pygraph.batchnorm)

    # @brief: Add an input tensor to the graph
    def tensor(self, **kwargs):
        # Create a name if none provided
        if "name" in kwargs:
            name = kwargs["name"]
        else:
            name = self.createUniqueName("Tensor")

        testTensor = RandomTensorGenerator(kwargs, name, self.io_data_type)

        self.nodes.append(testTensor)
        # we are assuming only input tensors are explicitly created
        self.entrance_nodes.append(testTensor)
        return testTensor.output[0]

    def tensor_cpu_constant(self, value, **kwargs):
        # Create a name if none provided
        if "name" in kwargs:
            name = kwargs["name"]
        else:
            name = self.createUniqueName("Tensor")

        node = ConstantTensor(kwargs, name, self.io_data_type, value)
        self.nodes.append(node)
        self.entrance_nodes.append(node)
        return node.output[0]
    
    # @brief: utility function to create unique names for the graph
    def createUniqueName(self, prefix):
        name = prefix + "_{}".format(self.uid_counter)
        self.uid_counter += 1
        return name

    # @brief: Create an operation, pass through the kwargs and set up dependencies
    # @param kwargs: the named arguments passed to a pycudnn function
    # @param pyCudnnOp: the pycudnn operation (e.g., pycudnn.pygraph.conv)
    # @return: TestTensor (the output from the added operation)
    def createAndAddOperation(self, kwargs, pyCudnnOp):
        if "name" in kwargs:
            name = kwargs["name"]
        else:
            pyCudnnOpName = pyCudnnOp.__name__
            name = self.createUniqueName(pyCudnnOpName)

        node = TestGraph.createOperation(pyCudnnOp, name)
        node.setKwargs(kwargs)
        self.nodes.append(node)

        # pycudnn returns either a single tensor, or a list of tensors
        # internally, we always store a list to allow for generalization
        if len(node.output) == 1:
            return node.output[0]
        else:
            return node.output

    # @brief: In esssence, this function is a factory function:
    # @param pyCudnnOp: the pycudnn operation (e.g., pycudnn.pygraph.conv)
    # @return: Operation
    # it builds an operation by:
    #   * discovering the reference function based on the name of the pycudnn op
    @staticmethod
    def createOperation(pyCudnnOp, name):
        pyCudnnOpName = pyCudnnOp.__name__
        # Fetch the reference function from the reference framework
        # Note that we use PytorchReference here, but we can make this arbitrary
        # TODO(@mbreughe): Add error handling code in case the function is not found
        refFunc = getattr(PytorchReference, pyCudnnOpName)

        num_outputs = 1
        if pyCudnnOpName == "batchnorm":
            num_outputs = 5
        
        node = Operation(pyCudnnOp, refFunc, name, num_outputs)
        return node

    # @brief: Utility function
    def clearNodeMetaData(self):
        for node in self.nodes:
            node.clearMetaData()

    # @brief: Utility function to discover output nodes
    def markImplicitOutputNodes(self):
        for node in self.nodes:
            if node.isOutputNode():
                print ("Setting {} as output".format(node.name))
                for output in node.output:
                    output.pyCudnnTensor.set_output(True)

    def apply_modifiers_to_node_output_tensors(self):
        for node in self.nodes:
            for tensor in node.output:
                tensor.apply_modifiers()

    # @brief: Build the pycudnn graph
    # @note: this temporarily modifies the isVisited status of the nodes
    # @return the pycudnn graph
    # @note we are relying on the user not the alter the graph. We can instead return them a copy, but this would be at a cost
    def buildPyCudnnGraph(self):
        # Setting up graph
        graph = pycudnn.pygraph(self.graph_name, io_data_type = self.io_data_type, intermediate_data_type = self.intermediate_data_type, compute_data_type = self.compute_data_type)
        
        # TODO(@mbreughe): Change this. We don't want to invoke pycudnn calls this way since we change the order
        # a developer may have intended. It is useful when building from json graphs, but not when 
        # manually setting up graphs. This is like building a house of cards.
        self.clearNodeMetaData()
        for node in self.entrance_nodes:
            node.buildPycudnnTreeRecursive(graph)
        
        # Once we constructed the pycudnn graph, it's time to apply any explicit modifiers to each node's output tensors
        # TestGraph creates dummy output tensors to allow constructing of a graph.
        # However, the actual output tensors are created once we call buildPycudnnTreeRecursive. This means any modifications
        # such as Y.set_data_type(FLOAT) actually happened on the dummy tensors. Here we propogate this to the real tensors
        self.apply_modifiers_to_node_output_tensors()

        # Setting implicit output nodes"
        self.markImplicitOutputNodes()

        # Building graph
        graph.build()

        # Clear the "isVisited" status of the nodes
        self.clearNodeMetaData()

        self.cudnn_graph = graph
        return graph

    def set_io_data_type(self, data_type):
        self.io_data_type = data_type

    def set_intermediate_data_type(self, data_type):
        self.set_intermediate_data_type = data_type

    def set_compute_data_type(self, data_type):
        self.set_compute_data_type(data_type)

    # @brief: Run the reference for the associated graph
    # @note: this temporarily modifies the isVisited status of the nodes
    # TODO(@mbreughe): to preserve resources, we could consider clearing intermediate results when they are no longer needed
    def calcReference(self):
        # Clear the "isVisited" status of the nodes
        self.clearNodeMetaData()
        for node in self.entrance_nodes:
            node.runRefTreeRecursive()
        
        output = []
        for node in self.nodes:
            if node.isOutputNode():
                for out in node.output:
                    output.append(out.ref_data)

        # Clear the "isVisited" status of the nodes
        self.clearNodeMetaData()
        return output
    
    def createWorkspaceAndVariantPack(self):
        # Creating workspace
        workspace = torch.empty(self.cudnn_graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

        variant_pack = {}
        for node in self.entrance_nodes:
            for output in node.output:
                variant_pack[output.pyCudnnTensor] = node.getValue()

        for node in self.nodes:
            if node.isOutputNode():
                for output in node.output:
                    # TODO(@mbreughe): infer layout
                    output_tensor = torch.zeros(*output.pyCudnnTensor.get_dim(), dtype=convert_to_torch_type(output.pyCudnnTensor.get_data_type()), device='cuda', layout=torch.strided).to(memory_format=torch.channels_last)
                    self.output_tensors.append(output_tensor)
                    variant_pack[output.pyCudnnTensor] = self.output_tensors[-1]

        return (workspace, variant_pack)
    
    # @brief: Run the pycudnn implementation and the reference, and compare
    # @note: This assumes buildPyCudnnGraph has already been run
    def cudnnExecuteAndCompareToReference(self, atol=1e-2, rtol=1e-2):
        workspace, variant_pack = self.createWorkspaceAndVariantPack()

        # Run the pycudnn graph
        print("Executing graph through pycudnn")
        self.cudnn_graph.execute(variant_pack, workspace)

        # Run the reference
        print("Computing reference")
        ref_outputs = self.calcReference()

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