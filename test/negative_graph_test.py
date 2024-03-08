
from test_graph import test_graph
import pytest
import re
import cudnn
import threading
from pytest_reraise import Reraise

# @param graph_builder_fptr is set through conftest.py
# @param jparams is set through conftest.py
def test_negative_graph(graph_builder_fptr, jparams):   
    def tester():
        testgraph = test_graph()
        graph_builder_fptr(problem_size, testgraph)  

        with pytest.raises(Exception) as e:
            # For negative tests, we expect some sort of exception should be thrown. Otherwise, we should fail the test.
            graph_builder_fptr(problem_size, testgraph)   
            graph = testgraph.build_cudnn_graph()

        # If expected_err_msg or expected_err_code is set, we'll do furthur verification.
        err_detail = str(e.value)
        err_code = re.search(r'cudnn_status: (CUDNN_STATUS_\S+)', err_detail)    
        if "expected_err_msg" in verification:
            # get_last_err_str() is not available in the public python bindings yet. 
            # The following implementation is based on a modified version of the python bindings. 
            # Note, if the error is thrown by the frontend, cudnnGetLastErrorString() will return nothing.
            # To distinguish this, for error thrown by the FE, there'll just be an error code.
            # For error thrown by the backend, there will be a 'reason' field which contains the return of cudnnGetLastErrorString().
            # For error msg checking, we only check error thrown by the backend, that means, there should be a "reason" in the exception str.
            # For now we'll skip the err msg checking. 
            if hasattr(cudnn, 'get_last_err_str') and "reason" in str(e.value):
                # If the error is thrown by the backend, we'll retrieve the error msg by calling cudnnGetLastErrorString() in the modified cudnn python module.
                assert re.match(r".*%s.*" % verification["expected_err_msg"], str(e.value))            
                # The second call to cudnnGetLastErrorString() should return an empty string
                assert cudnn.get_last_err_str() == ""

        if "expected_err_code" in verification:
            assert err_code is not None
            actual_err_code = err_code.group(1)
            assert actual_err_code == verification["expected_err_code"]
            
    problem_size = jparams["problem_size"]
    verification = jparams["verification"]
    thread_num = 1 if "thread" not in jparams else jparams["thread"]    

    if not verification["is_negative"]:
        pytest.skip("Not a negative test")

    if thread_num == 1:
        tester()
    else:
        reraise = Reraise()
        threads = [threading.Thread(target=reraise.wrap(tester)) for i in range(thread_num)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()            
        reraise()


