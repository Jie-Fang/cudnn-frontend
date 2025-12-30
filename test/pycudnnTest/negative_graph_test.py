from cudnn_test_graph import cudnn_test_graph
import pytest
import re
import cudnn
import threading
from pytest_reraise import Reraise
from utils import OutputGrabber
import sys

# @param graph_builder_fptr is set through conftest.py
# @param jparams is set through conftest.py


def check_log(log_str):
    ret = []
    matches = re.finditer(r"function \S+ called:\s+e!\s+Error: (CUDNN_STATUS_\S+); Reason:(.*)", log_str)
    for m in matches:
        ret.append([m.group(1).strip(), m.group(2).strip()])
    return ret


def test_negative_graph(graph_builder_fptr, jparams):
    def tester():
        testgraph = cudnn_test_graph()
        with pytest.raises(Exception) as e:
            # For negative tests, we expect some sort of exception should be thrown. Otherwise, we should fail the test.
            graph_builder_fptr(problem_size, testgraph)
            graph = testgraph.build_cudnn_graph()
        print("Problem size: %s" % str(problem_size))
        print("Exception caught: %s" % str(e.value))

    problem_size = jparams["problem_size"]
    verification = jparams["verification"]
    thread_num = 1 if "thread" not in jparams else jparams["thread"]

    if not verification["is_negative"]:
        pytest.skip("Not a negative test")

    with OutputGrabber(sys.stderr) as api_log:
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

    last_err_code = None
    last_err_msg = None
    extracted_err = check_log(api_log.capturedtext)
    print(api_log.capturedtext)
    for err_code, err_msg in extracted_err:
        assert err_code is not None and err_msg is not None
        assert last_err_code is None or last_err_code == err_code
        assert last_err_msg is None or last_err_msg == err_msg
        last_err_code = err_code
        last_err_msg = err_msg

    if last_err_code is not None and last_err_msg is not None:
        # this means the error was thrown by the backend
        assert thread_num == len(extracted_err)
        print("Error code: %s, Error msg: %s" % (last_err_code, last_err_msg))
        if "expected_err_msg" in verification:
            assert re.match(r".*%s.*" % verification["expected_err_msg"], last_err_msg)
        if hasattr(cudnn, "cudnnGetLastErrorString"):
            # If the error is thrown by the backend, the second call to cudnnGetLastErrorString() should return an empty string
            # cudnnGetLastErrorString() is not available in the public python bindings yet.
            # For now we'll skip the err msg checking.
            # TODO(@adshen): bug 4600431 is to track adding cudnnGetLastErrorString() into the frontend python bindings. Will need to revisit these lines after it is implemented.
            # There's a plan to add cudnnGetLastErrorString() to the frontend exception in the near future.
            # If we can ensure cudnnGetLastErrorString() is called in the frontend whenever a backend error occurs, we'll only need to call cudnnGetLastErrorString() once in the test and ensure it's empty.
            assert cudnn.cudnnGetLastErrorString() == ""
        if "expected_err_code" in verification:
            assert last_err_code == verification["expected_err_code"]
    else:
        print("Skip error code and error message checking, as the error was not thrown by the backend API.")
