import cudnn
import pytest
from looseversion import LooseVersion

def test_gemm(jparams, testgraph):
    B, M, N, K = jparams["in_dim"]

    tensor_a = testgraph.tensor(
        name="image", dim=[B, M, K], data_type=cudnn.data_type.HALF
    )
    tensor_b = testgraph.tensor(
        name="weight", dim=[B, K, N], data_type=cudnn.data_type.HALF
    )

    gemm_output = testgraph.matmul(
        name="mb_matmul", A=tensor_a, B=tensor_b, compute_data_type=cudnn.data_type.HALF
    )
    gemm_output.set_stride([M*N, N, 1])


def test_gemm_relu(jparams, testgraph):
    B, M, N, K = jparams["in_dim"]

    image = testgraph.tensor(
        name="image", dim=[B, M, K], data_type=cudnn.data_type.HALF
    )
    weight = testgraph.tensor(
        name="weight", dim=[B, K, N], data_type=cudnn.data_type.HALF
    )

    gemm_output = testgraph.matmul(
        name="mb_matmul", A=image, B=weight, compute_data_type=cudnn.data_type.FLOAT
    )
    # Make intermediate tensor output row-major:
    gemm_output.set_stride([M * N, N, 1])
    relu_output = testgraph.relu(input=gemm_output)


def test_gemm_bias_relu(jparams, testgraph):
    B, M, N, K = jparams["in_dim"]

    image = testgraph.tensor(
        name="image", dim=[B, M, K], data_type=cudnn.data_type.HALF
    )
    weight = testgraph.tensor(
        name="weight", dim=[B, K, N], data_type=cudnn.data_type.HALF
    )
    bias = testgraph.tensor(
        name="weight", dim=[B, M, N], data_type=cudnn.data_type.HALF
    )

    gemm_output = testgraph.matmul(
        name="mb_matmul", A=image, B=weight, compute_data_type=cudnn.data_type.FLOAT
    )
    # Make intermediate tensor output row-major:
    gemm_output.set_stride([M * N, N, 1])

    bias_out = testgraph.bias(name="bias", input=gemm_output, bias=bias)
    relu_output = testgraph.relu(input=bias_out)
