import pycudnn
import torch
from typing import Any
from dataclasses import dataclass, asdict, field
import copy

class PytorchReference:
    @staticmethod
    def conv(kwargs):
        return torch.nn.functional.conv2d(kwargs['image'].data, kwargs['weight'].data, bias = None, padding=kwargs["padding"], stride=kwargs["stride"], dilation=kwargs["dilation"])

    @staticmethod
    def relu(kwargs):
        return torch.nn.functional.relu(kwargs["input"])

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

    def genPyCudnnNode(self, pyCudnnGraph):
        print("NOT IMPLEMENTED")

    def buildPycudnnTreeRecursive(self, pyCudnnGraph):

        if not self.isVisited() and self.isPrereqSatisfied():
            #print ("Checking {}".format(self.name))
            self.genPyCudnnNode(pyCudnnGraph)
            self.setVisited()
            for node in self.consumerNodes:
                node.buildPycudnnTreeRecursive(pyCudnnGraph)

    def runRefTreeRecursive(self):
        if not self.isVisited() and self.isPrereqSatisfied():
            #print ("Checking {}".format(self.name))
            self.genRef()
            self.setVisited()
            for node in self.consumerNodes:
                node.runRefTreeRecursive()

class Operation(TestNode):
    # All this function needs to do is: 
    #   * add the correct producers
    #   * store the rest of the kwargs
    #   * store the pycudnn function to call
    def __init__(self, kwargs, pyCudnnOp, refFunc, name):
        # TBD Fix default name
        # Take into account that kwargs has a name already
        super().__init__(name)
        self.kwargs = kwargs
        # DOes .values() make an unnecessary copy?
        for v in self.kwargs.values():
            if isinstance(v, TestNode):
                print("Node {} is adding producer {}".format(name, v.name))
                self.addProducerNode(v)
        
        self.pyCudnnOp = pyCudnnOp
        self.refFunc = refFunc

    def genPyCudnnNode(self, pyCudnnGraph):
        # Besides input tensors, all kwargs can just be passed through to the pycudnn method.
        # For the input tensor we need to extract the TestTensor's pycudnn tensor.
        # Therefore: copy all kwargs, except for TestTensors
        # TODO(@mbreughe) Avoid copying these kwargs twice (ref and pycudnn). Let's do this in the init function once
        new_kwargs = {x: self.kwargs[x] for x in self.kwargs if not isinstance(self.kwargs[x], TestNode)}
        for x in self.kwargs:
            if isinstance(self.kwargs[x], TestNode):
                new_kwargs[x] = self.kwargs[x].pyCudnnTensor

        self.pyCudnnTensor = self.pyCudnnOp(pyCudnnGraph, **new_kwargs)

    def genRef(self):
        new_kwargs = {x: self.kwargs[x] for x in self.kwargs if not isinstance(self.kwargs[x], TestNode)}
        for x in self.kwargs:
            if isinstance(self.kwargs[x], TestNode):
                new_kwargs[x] = self.kwargs[x].data
        self.data = self.refFunc(new_kwargs)

@dataclass
class TestTensor(TestNode):
    __test__ = False

    def __init__(self, kwargs, name):
        super().__init__(name)
        self.kwargs = kwargs
        self.pyCudnnTensor = None
        self.data = None
        # TODO(mbreughe): Assume NHWC layout for now
        self.layout = "NHWC"
        # Use fp16 by default
        # TODO(@mbreughe): use pycudnn convention instead (like extracting it from the pycudnn graph)
        if not "data_type" in self.kwargs:
            self.data_type(pycudnn.data_type.HALF)
    
    def data_type(self, data_type):
        self.kwargs["data_type"] = data_type

    def instantiateRandomTensor(self):
        if self.data is None:
            self.data = torch.randn(self.kwargs["dim"], requires_grad=False, device="cuda", dtype=convert_to_torch_type(self.kwargs["data_type"]))
            
            if self.layout == "NHWC":
                self.data = self.data.to(memory_format=torch.channels_last)
    
    def getValue(self):
        self.instantiateRandomTensor()
        return self.data

    def genPyCudnnNode(self, pyCudnnGraph):
        self.instantiateRandomTensor()
        new_kwargs = {x: self.kwargs[x] for x in self.kwargs}
        print(type(pyCudnnGraph))
        self.pyCudnnTensor = pyCudnnGraph.tensor(name = self.name, dim = self.data.size(), stride = self.data.stride(), data_type = convert_to_cudnn_type(self.data.dtype))

    def genRef(self):
        return self.getValue()
    

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

class TestGraph:
    __test__ = False
    uid_counter = 0
    # Add data types, custom test name ,etc.
    def __init__(self):
        self.nodes = []
        self.entrance_nodes = []
        self.graph_name = "TestGraph"
        self.output_tensors = []

    def conv(self, **kwargs):
        return self.createAndAddOperation(kwargs, pycudnn.pygraph.conv)

    def relu(self, **kwargs):
        return self.createAndAddOperation(kwargs, pycudnn.pygraph.relu)

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
    
    @staticmethod
    def createUniqueName(prefix):
        name = prefix + "_{}".format(TestGraph.uid_counter)
        TestGraph.uid_counter += 1
        return name



    # @brief: In esssence, this function is a factory function:
    # @param kwargs: the named arguments passed to a pycudnn function
    # @param pyCudnnOp: the pycudnn operation (e.g., pycudnn.pygraph.conv)
    # @return: Operation
    # it builds an operation by:
    #   * discovering the reference function based on the name of the pycudnn op
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
        refFunc = getattr(PytorchReference, pyCudnnOpName)
        
        node = Operation(kwargs, pyCudnnOp, refFunc, name)
        self.nodes.append(node)

        return node

    def clearNodeMetaData(self):
        for node in self.nodes:
            node.clearMetaData()

    def markImplicitOutputNodes(self):
        for node in self.nodes:
            if node.isOutputNode():
                print ("Setting {} as output".format(node.name))
                node.pyCudnnTensor.set_output(True)


    # Note: this temporarily modifies the isVisited status of the nodes
    def buildPyCudnnGraph(self):
        print("Setting up graph")
        graph = pycudnn.pygraph(self.graph_name, io_data_type = pycudnn.data_type.HALF, intermediate_data_type = pycudnn.data_type.FLOAT, compute_data_type = pycudnn.data_type.FLOAT)
        self.clearNodeMetaData()
        for node in self.entrance_nodes:
            node.buildPycudnnTreeRecursive(graph)

        print ("Setting implicit output nodes")
        self.markImplicitOutputNodes()

        print("Building graph")
        graph.build()

        print("Creating workspace")
        workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

        variant_pack = {}
        for node in self.entrance_nodes:
            variant_pack[node.pyCudnnTensor] = node.getValue()

        for node in self.nodes:
            if node.isOutputNode():
                output_tensor = torch.zeros(*node.pyCudnnTensor.get_dim(), dtype=torch.float16, device='cuda', layout=torch.strided).to(memory_format=torch.channels_last)
                self.output_tensors.append(output_tensor)
                variant_pack[node.pyCudnnTensor] = self.output_tensors[-1]

        # Clear the "isVisited" status of the nodes
        self.clearNodeMetaData()
        return graph, variant_pack, workspace

    # Note: this temporarily modifies the isVisited status of the nodes
    # TODO(@mbreughe): to preserve resources, we could consider clearing intermediate results when they are no longer needed
    def getReference(self):
        # Clear the "isVisited" status of the nodes
        self.clearNodeMetaData()
        for node in self.entrance_nodes:
            node.runRefTreeRecursive()
        
        output = []
        for node in self.nodes:
            if node.isOutputNode():
                output.append(node.data)

        # Clear the "isVisited" status of the nodes
        self.clearNodeMetaData()
        return output