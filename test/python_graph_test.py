
from test_graph import TestGraph
# @pytest.fixture.parametrize(test_name)
#def python_graph_test(test_name, jparams):

def test_python_graph(test_name, jparams):
    testGraph = TestGraph()
    test_name(jparams, testGraph)
    graph = testGraph.buildPyCudnnGraph()
    
    testGraph.cudnnExecuteAndCompareToReference(atol=1e-2,rtol=1e-2)