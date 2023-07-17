import pycudnn
import torch
from typing import Any
from dataclasses import dataclass, asdict, field
import copy

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

class Convolution(TestNode):
    # All this function needs to do is: 
    #   * add the correct producers
    #   * store the rest of the kwargs
    #   * store the pycudnn function to call
    def __init__(self, kwargs):
        # TBD change default name
        # Take into account that kwargs has a name already
        super().__init__("Conv_Node")
        self.kwargs = kwargs
        # DOes .values() make an unnecessary copy?
        for v in self.kwargs.values():
            if type(v) is TestTensor:
                self.addProducerNode(v)

    def genPyCudnnNode(self, pyCudnnGraph):
        # Besides input tensors, all kwargs can just be passed through to the pycudnn method.
        # For the input tensor we need to extract the TestTensor's pycudnn tensor.
        # Therefore: copy all kwargs, except for TestTensors
        new_kwargs = {x: self.kwargs[x] for x in self.kwargs if not type(self.kwargs[x]) is TestTensor}
        for x in self.kwargs:
            if type(self.kwargs[x]) is TestTensor:
                new_kwargs[x] = self.kwargs[x].pyCudnnTensor

        # Once we make this a function pointer, we no longer need the convolution class
        self.pyCudnnTensor = pyCudnnGraph.conv(**new_kwargs)

    # While the arguments in pytorch look very similar, we should probably explicitly build up the function call
    # We should pass in a Reference object/function pointer to the TestNode, instead of hard coding the call here.
    # This will also allow us to dial in arbirtary ref test code
    def genRef(self):
        self.data = torch.nn.functional.conv2d(self.kwargs['image'].data, self.kwargs['weight'].data, bias = None, padding=self.kwargs["padding"], stride=self.kwargs["stride"], dilation=self.kwargs["dilation"])
        

class ReLU(TestNode):
    def __init__(self, input):
        super().__init__("RELU_NODE")
        self.input = input
        self.addProducerNode(input)
        self.pyCudnnTensor = None

    def genPyCudnnNode(self, pyCudnnGraph):
        self.pyCudnnTensor = pyCudnnGraph.relu(name = self.name, input = self.input.pyCudnnTensor)

    def genRef(self):
        self.data = torch.nn.functional.relu(self.input.data)

@dataclass
class TestTensor(TestNode):
    __test__ = False
    UID: str
    #dataType: str
    # TODO(mbreughe): Assume NHWC layout
    layout: str
    dim: list = field(default=None)
    isVirtual: bool = field(default=False)
    data: Any = field(default = None)

    def __init__(self, UID, dim):
        super().__init__("Tensor_{}".format(UID))
        self.UID = UID
        self.dim = dim
        self.pyCudnnTensor = None
        self.layout = "NHWC"

    def instantiateRandomTensor(self):
        if self.data is None:
            # TBD specify data type
            self.data = torch.randn(self.dim, requires_grad=False, device="cuda", dtype=torch.float16)
            if self.layout == "NHWC":
                self.data = self.data.to(memory_format=torch.channels_last)
    
    def getValue(self):
        self.instantiateRandomTensor()
        return self.data

    def genPyCudnnNode(self, pyCudnnGraph):
        self.instantiateRandomTensor()
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
        node = Convolution(kwargs)
        self.nodes.append(node)
        return node

    # Add name field and create name -> uid mapping. May help with debugging
    # Should we limit this to input tensors only?
    def addTensor(self, dim):
        testTensor = TestTensor(UID=TestGraph.uid_counter, dim = dim)
        TestGraph.uid_counter += 1

        self.nodes.append(testTensor)
        self.entrance_nodes.append(testTensor)
        return testTensor

    # This will need to be modified if we have more than 1 output
    def addOperation(self, node):
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