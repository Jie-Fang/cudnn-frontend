
from test_graph import TestGraph
# @pytest.fixture.parametrize(test_name)
#def python_graph_test(test_name, jparams):

def test_python_graph(test_name, jparams):
    EXPECTED_KEY = "expected_dim"
    testGraph = TestGraph()
    test_name(jparams, testGraph)
    graph = testGraph.buildPyCudnnGraph()

    # Perform a front-end check if expected dimensions were specified
    if EXPECTED_KEY in jparams:
        testGraph.frontend_check(jparams[EXPECTED_KEY])
    
    testGraph.cudnnExecuteAndCompareToReference(atol=1e-2,rtol=1e-2)