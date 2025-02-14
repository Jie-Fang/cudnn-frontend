from cudnn_test_graph import cudnn_test_graph


# @param graph_builder_fptr is set through conftest.py
# @param jparams is set through conftest.py
def test_python_graph(graph_builder_fptr, jparams):
    EXPECTED_KEY = "expected_dim"
    testgraph = cudnn_test_graph()
    graph_builder_fptr(jparams, testgraph)
    graph = testgraph.build_cudnn_graph()

    # Perform a front-end check if expected dimensions were specified
    if EXPECTED_KEY in jparams:
        testgraph.frontend_check(jparams[EXPECTED_KEY])

    testgraph.cudnn_execute_and_compare_to_reference(atol=1e-2, rtol=1e-2)
