from typing import Any
from dataclasses import dataclass, asdict, field
import pytest
import pycudnn
import torch


def runReference(testGraph):
    for node in testGraph:
        node.runReference()



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

class Convolution(TestNode):
    def __init__(self, image, weight, padding = [0,0], stride = [1,1], dilation = [1,1]):
        # TBD change default name
        super().__init__("Conv_Node")
        self.image = image
        self.weight = weight
        self.addProducerNode(image)
        self.addProducerNode(weight)
        self.padding = padding
        self.stride = stride
        self.dilation = dilation

    def genPyCudnnNode(self, pyCudnnGraph):
        self.pyCudnnTensor = pyCudnnGraph.conv(name = self.name, image = self.image.pyCudnnTensor, weight = self.weight.pyCudnnTensor, padding = self.padding, stride = self.stride, dilation = self.dilation)
        

class ReLU(TestNode):
    def __init__(self, input):
        super().__init__("RELU_NODE")
        self.input = input
        self.addProducerNode(input)

    def genPyCudnnNode(self, pyCudnnGraph):
        self.pyCudnnTensor = pyCudnnGraph.relu(name = self.name, input = self.input.pyCudnnTensor)

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

def convert_to_cudnn_type(torch_type):
    if torch_type == torch.float16:
        return pycudnn.data_type.HALF
    elif torch_type == torch.float32:
        return pycudnn.data_type.FLOAT
    else:
        raise ValueError("Unsupported tensor data type.")

    return

class CSBR(torch.nn.Module):
    def forward(self, x, w, b = None, padding = [1,1], stride = [1,1], dilation = [1,1]):
        if b is not None:
            b = b.reshape(-1) # Conv2d needs a 1D tensor
        conv_output = torch.nn.functional.conv2d(x, w, bias = b, padding=padding, stride=stride, dilation=dilation)
        return torch.nn.functional.relu(conv_output)

class TestGraph:
    __test__ = False
    uid_counter = 0
    # Add data types, custom test name ,etc.
    def __init__(self):
        self.nodes = []
        self.entrance_nodes = []
        self.graph_name = "TestGraph"
        self.output_tensors = []

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

    def buildPyCudnnGraph(self):
        print("Setting up graph")
        graph = pycudnn.pygraph(self.graph_name, io_data_type = pycudnn.data_type.HALF, intermediate_data_type = pycudnn.data_type.FLOAT, compute_data_type = pycudnn.data_type.FLOAT)
        self.clearNodeMetaData()
        for node in self.entrance_nodes:
            node.buildPycudnnTreeRecursive(graph)

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

        return graph, variant_pack, workspace

    def getReference(self):
        # TODO(@mbreughe): Replace this with recursive reference
        padding = [0,1]
        stride = [2,3]
        dilation = [1,1]
        model = CSBR().eval().to("cuda").to(torch.float16)
        Y_expected = model(self.entrance_nodes[0].data, self.entrance_nodes[1].data, padding = padding, stride = stride, dilation = dilation)
        return Y_expected



def test_conv_relu():
    padding = [0,1]
    stride = [2,3]
    dilation = [1,1]

    testGraph = TestGraph()
    X = testGraph.addTensor([20, 40, 30, 40])
    W = testGraph.addTensor([54, 40, 3, 4])
    
    conv_out = testGraph.addOperation(Convolution(image = X, weight = W, padding = padding, stride = stride, dilation = dilation))
    Y = testGraph.addOperation(ReLU(input = conv_out))

    graph, variant_pack, workspace = testGraph.buildPyCudnnGraph()

    print(graph)
    graph.execute(variant_pack, workspace)
    Y_actual = testGraph.output_tensors[-1]

    print (Y_actual)

    # TODO(@mbreughe): hard coded for now
    Y_expected = testGraph.getReference()

    print(Y_expected)
   
    # Compare
    torch.testing.assert_close(Y_expected, Y_actual, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("in_dim, expected_gemm_out_dim", [([16, 32, 64, 128], [16,32,64,])])
def gemm_relu_with_pyGraphRunner(in_dim, expected_gemm_out_dim):
    B, M, N, K = in_dim

    # Build the common test graph
    testGraph = GRGraph()
    X_Tensor = GRTensor(name="X", dim = [B,M,K], dataType=torch.float16)
    W_Tensor = GRTensor(name="W", dim = [B,K,N], dataType=torch.float16)
    testGraph.addTensor(X_Tensor)
    testGraph.addTensor(W_Tensor)

    gemmNode = GRGemmOp(X_Tensor, W_Tensor, out_name= "gemm_out")
    testGraph.addOperation(gemmNode)
    reluNode = GRPointWiseOp(gemmNode.out)
    testGraph.addOperation(reluNode)

    # Convert the test graph into a Pycudnn graph
    mb_graph, tensors = setupPyCudnnGraph(testGraph)

    output = torch.zeros(mb_graph.problem_size, dtype=torch.float16, device='cuda')

    mb_graph.variant_pack[mb_graph.last_output] = output.data_ptr()

    mb_graph.cudnn_graph.build()

    # Check expected shapes
    assert expected_gemm_out_dim == tensors["gemm_out"].get_dim()

    # Run each node of the test graph through Pytorch
    pytorch_out = runRefTreeRecursive(testGraph)

    # Compare pycudnn output with the reference implementation
    mb_graph.cudnn_graph.execute(mb_graph.variant_pack)
    torch.testing.assert_close(output, pyt_out)


if __name__ == "__main__":
    test_conv_relu()