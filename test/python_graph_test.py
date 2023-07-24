
from test_graph import TestGraph
# @pytest.fixture.parametrize(test_name)
#def python_graph_test(test_name, jparams):

def test_python_graph(test_name):
    testGraph = TestGraph()
    test_name({"in_dim":[32,32,7,7], "filter_dim":[256,32,1,1], "padding":[0,0], "stride":[1,1], "dilation":[1,1], "expected_conv_out_dim": [32,256,7,7]}, testGraph)
    graph = testGraph.buildPyCudnnGraph()

    testGraph.cudnnExecuteAndCompareToReference(atol=1e-2,rtol=1e-2)