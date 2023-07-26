import cudnn
import torch
from typing import Any
from dataclasses import dataclass, asdict, field
import copy

# @brief: Reference code
# @details: the methods mirror cudnn.pygraph methods and class constructors(__init__)
# @note: we can easily replace PytorchReference by CustomReference to use a different reference framework (one LoC change in TestGraph below)
class PytorchReference:
    # @brief: run convolution without bias
    # @param kwargs: these are the named parameters used in the associated cudnn.pygraph.conv function
    #   The only difference is that the input tensors are replaced by pytorch tensors
    # @details: all this function needs to do is unpack the cudnn.pygraph function arguments and pass them to the pytorch equivalent
    @staticmethod
    def conv(kwargs):
        return torch.nn.functional.conv2d(kwargs['image'], kwargs['weight'], bias = None, padding=kwargs["padding"], stride=kwargs["stride"], dilation=kwargs["dilation"])

    # @brief: run relu
    # @details: unpack the cudnn.pygraph.relu parameters and pass them to the pytorch equivalent
    @staticmethod
    def relu(kwargs):
        return torch.nn.functional.relu(kwargs["input"])

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
    # @param kwargs: parameters for the associated pyCudnnOp
    # @param pyCuddnOp: cudnn.pygraph operation (e.g., cudnn.pygraph.conv)
    # @param refFunc: reference function for the associated pyCudnnOp
    # @param name: name for this Operation (could be passed by kwargs as well)
    # @ details: All this function needs to do is: 
    #   * add the correct producers
    #   * store the kwargs (named parameters from the associated pyCudnnOp)
    #   * store the cudnn function to call
    def __init__(self, kwargs, pyCudnnOp, refFunc, name):
        super().__init__(name)
        self.kwargs = kwargs
        # TODO(@mbreughe): Does .values() make an unnecessary copy?
        for v in self.kwargs.values():
            if isinstance(v, TestNode):
                self.addProducerNode(v)
        
        self.pyCudnnOp = pyCudnnOp
        self.refFunc = refFunc

    # @brief: Run the cudnn node
    # TODO(@mbreughe): extended to multiple output tensors
    def runPyCudnnCode(self, pyCudnnGraph):
        # Besides input tensors, all kwargs can just be passed through to the cudnn method.
        # For the input tensor we need to extract the TestTensor's cudnn tensor.
        # Therefore: copy all kwargs, except for TestTensors
        # TODO(@mbreughe) Avoid copying these kwargs twice (ref and cudnn). Let's do this in the init function once
        new_kwargs = {x: self.kwargs[x] for x in self.kwargs if not isinstance(self.kwargs[x], TestNode)}
        for x in self.kwargs:
            if isinstance(self.kwargs[x], TestNode):
                new_kwargs[x] = self.kwargs[x].pyCudnnTensor

        self.pyCudnnTensor = self.pyCudnnOp(pyCudnnGraph, **new_kwargs)

    def runRef(self):
        new_kwargs = {x: self.kwargs[x] for x in self.kwargs if not isinstance(self.kwargs[x], TestNode)}
        for x in self.kwargs:
            if isinstance(self.kwargs[x], TestNode):
                new_kwargs[x] = self.kwargs[x].ref_data
        self.ref_data = self.refFunc(new_kwargs)

class TestTensor(TestNode):
    __test__ = False

    def __init__(self, kwargs, name):
        super().__init__(name)
        self.kwargs = kwargs
        # The cudnn.pygraph.tensor instance associated with TestTensor
        self.pyCudnnTensor = None
        # The reference data for this tensor
        self.ref_data = None
        # TODO(mbreughe): Assume NHWC layout for now
        self.layout = "NHWC"
        # Use fp16 by default
        # TODO(@mbreughe): use cudnn convention instead (like extracting it from the cudnn graph)
        if not "data_type" in self.kwargs:
            self.data_type(cudnn.data_type.HALF)
    
    def data_type(self, data_type):
        self.kwargs["data_type"] = data_type

    def instantiateRandomTensor(self):
        if self.ref_data is None:
            self.ref_data = torch.randn(self.kwargs["dim"], requires_grad=False, device="cuda", dtype=convert_to_torch_type(self.kwargs["data_type"]))
            
            if self.layout == "NHWC":
                self.ref_data = self.ref_data.to(memory_format=torch.channels_last)
    
    def getValue(self):
        self.instantiateRandomTensor()
        return self.ref_data

    def runPyCudnnCode(self, pyCudnnGraph):
        self.instantiateRandomTensor()
        self.pyCudnnTensor = pyCudnnGraph.tensor(name = self.name, dim = self.ref_data.size(), stride = self.ref_data.stride(), data_type = convert_to_cudnn_type(self.ref_data.dtype))

    def runRef(self):
        return self.getValue()
    

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

# @brief: TestGraph that mirrors cudnn.pygraph
# @details: this contains functionality to run both cudnn code as well as a reference
class TestGraph:
    __test__ = False
    uid_counter = 0
    # Add data types, custom test name ,etc.
    def __init__(self):
        self.nodes = []
        self.entrance_nodes = []
        self.graph_name = "TestGraph"
        self.output_tensors = []

    # @brief: Add a convolution node to the graph
    def conv(self, **kwargs):
        return self.createAndAddOperation(kwargs, cudnn.pygraph.conv)

    # @brief: Add a relu to the graph
    def relu(self, **kwargs):
        return self.createAndAddOperation(kwargs, cudnn.pygraph.relu)

    # @brief: Add an input tensor to the graph
    def tensor(self, **kwargs):
        # Create a name if none provided
        if "name" in kwargs:
            name = kwargs["name"]
        else:
            name = TestGraph.createUniqueName("Tensor")

        testTensor = TestTensor(kwargs, name)

        self.nodes.append(testTensor)
        # TODO(@mbreughe): we are assuming only input tensors are explicitly created
        self.entrance_nodes.append(testTensor)
        return testTensor
    
    # @brief: utility function to create unique names
    @staticmethod
    def createUniqueName(prefix):
        name = prefix + "_{}".format(TestGraph.uid_counter)
        TestGraph.uid_counter += 1
        return name

    # @brief: In esssence, this function is a factory function:
    # @param kwargs: the named arguments passed to a cudnn function
    # @param pyCudnnOp: the cudnn operation (e.g., cudnn.pygraph.conv)
    # @return: Operation
    # it builds an operation by:
    #   * discovering the reference function based on the name of the cudnn op
    #   * creating a name
    #   * passing through the kwargs
    def createAndAddOperation(self, kwargs, pyCudnnOp):
        pyCudnnOpName = pyCudnnOp.__name__
        
        if "name" in kwargs:
            name = kwargs["name"]
        else:
            name = TestGraph.createUniqueName(pyCudnnOpName)

        # Fetch the reference function from the reference framework
        # Note that we use PytorchReference here, but we can make this arbitrary
        # TODO(@mbreughe): Add error handling code in case the function is not found
        refFunc = getattr(PytorchReference, pyCudnnOpName)
        
        node = Operation(kwargs, pyCudnnOp, refFunc, name)
        self.nodes.append(node)

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
                node.pyCudnnTensor.set_output(True)

    # @brief: Build the cudnn graph
    # @note: this temporarily modifies the isVisited status of the nodes
    # @return the cudnn graph
    # @note we are relying on the user not the alter the graph. We can instead return them a copy, but this would be at a cost
    def buildPyCudnnGraph(self):
        # Setting up graph
        graph = cudnn.pygraph(self.graph_name, io_data_type = cudnn.data_type.HALF, intermediate_data_type = cudnn.data_type.FLOAT, compute_data_type = cudnn.data_type.FLOAT)
        self.clearNodeMetaData()
        for node in self.entrance_nodes:
            node.buildPycudnnTreeRecursive(graph)

        # Setting implicit output nodes"
        self.markImplicitOutputNodes()

        # Building graph
        graph.build()

        # Clear the "isVisited" status of the nodes
        self.clearNodeMetaData()

        self.cudnn_graph = graph
        return graph

    # @brief: Run the reference for the associated graph
    # @note: this temporarily modifies the isVisited status of the nodes
    # TODO(@mbreughe): to preserve resources, we could consider clearing intermediate results when they are no longer needed
    def getReference(self):
        # Clear the "isVisited" status of the nodes
        self.clearNodeMetaData()
        for node in self.entrance_nodes:
            node.runRefTreeRecursive()
        
        output = []
        for node in self.nodes:
            if node.isOutputNode():
                output.append(node.ref_data)

        # Clear the "isVisited" status of the nodes
        self.clearNodeMetaData()
        return output
    
    # @brief: Run the cudnn implementation and the reference, and compare
    # @note: This assumes buildPyCudnnGraph has already been run
    def referenceCheck(self, atol=1e-2, rtol=1e-2):
        # Creating workspace
        workspace = torch.empty(self.cudnn_graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

        variant_pack = {}
        for node in self.entrance_nodes:
            variant_pack[node.pyCudnnTensor] = node.getValue()

        for node in self.nodes:
            if node.isOutputNode():
                # TODO(@mbreughe): infer layout
                output_tensor = torch.zeros(*node.pyCudnnTensor.get_dim(), dtype=convert_to_torch_type(node.pyCudnnTensor.get_data_type()), device='cuda', layout=torch.strided).to(memory_format=torch.channels_last)
                self.output_tensors.append(output_tensor)
                variant_pack[node.pyCudnnTensor] = self.output_tensors[-1]

        # Run the cudnn graph
        self.cudnn_graph.execute(variant_pack, workspace)

        # TODO(@mbreughe): adjust this for multiple outputs.
        Y_actual = self.output_tensors[-1]

        # Run the reference
        Y_expected = self.getReference()[-1]

        # Compare with reference
        torch.testing.assert_close(Y_expected, Y_actual, atol=1e-2, rtol=1e-2)