from test_graph import test_graph
import test_tensor_ir as tti

# @param graph_builder_fptr is set through conftest.py
# @param jparams is set through conftest.py
def test_tensor_ir_pygraph(graph_builder_fptr, jparams):
    EXPECTED_KEY = "expected_dim"
    testgraph = test_graph()
    graph_builder_fptr(jparams, testgraph)

    testgraph.build_cudnn_graph_only()
    tensor_ir_tester = tti.test_tensor_ir(testgraph)
    tensor_ir_module = tensor_ir_tester.build_tensor_ir_module()
    tensor_ir_tester.run_tensor_ir_module(tensor_ir_module)

