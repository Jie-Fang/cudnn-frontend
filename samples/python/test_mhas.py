import pycudnn
import pytest
import torch

def convert_to_cudnn_type(torch_type):
    if torch_type == torch.float16:
        return pycudnn.data_type.HALF
    elif torch_type == torch.float32:
        return pycudnn.data_type.FLOAT
    elif torch_type == torch.int32:
        return pycudnn.data_type.INT32
    elif torch_type == torch.int64:
        return pycudnn.data_type.INT64
    else:
        raise ValueError("Unsupported tensor data type.")

@pytest.mark.skipif(pycudnn.get_cudnn_version() < 8900, reason="requires cudnn 8.9 or higher")
def test_scale_dot_product_attention_with_dropout_rng():
    b = 32
    h = 16
    s_q = 512
    s_kv = 512
    d = 64

    shape_Q = (b, h, s_q, d)
    stride_Q = (s_q * 3 * h * d, d, 3 * h * d, 1)

    shape_K = (b, h, d, s_kv)
    stride_K = (s_kv * 3 * h * d, d, 1, 3 * h * d)

    shape_V = (b, h, s_kv, d)
    stride_V = (s_kv * 3 * h * d, d, 3 * h * d, 1)

    offset_Q = 0
    offset_K = h * d
    offset_V = 2 * h * d

    qkv_gpu = torch.empty(b * s_q * 3 * h * d, dtype=torch.float16, device="cuda")
    Q_gpu = torch.as_strided(qkv_gpu, shape_Q, stride_Q, storage_offset=offset_Q)
    K_gpu = torch.as_strided(qkv_gpu, shape_K, stride_K, storage_offset=offset_K)
    V_gpu = torch.as_strided(qkv_gpu, shape_V, stride_V, storage_offset=offset_V)

    SEQ_LEN_Q_gpu = torch.full((b,1,1,1), 32, dtype=torch.int32, device="cuda")
    SEQ_LEN_K_gpu = torch.full((b,1,1,1), 32, dtype=torch.int32, device="cuda")

    Bias_gpu = torch.empty((1, h, s_q, s_kv), dtype=torch.float16, device="cuda")
    
    # Cudnn graph
    graph = pycudnn.pygraph("mha", io_data_type = pycudnn.data_type.HALF, intermediate_data_type = pycudnn.data_type.FLOAT, compute_data_type = pycudnn.data_type.FLOAT)
    Q = graph.tensor(name = "Q", dim = Q_gpu.size(), stride = Q_gpu.stride(), data_type = convert_to_cudnn_type(Q_gpu.dtype))
    K = graph.tensor(name = "K", dim = K_gpu.size(), stride = K_gpu.stride(), data_type = convert_to_cudnn_type(K_gpu.dtype))
    V = graph.tensor(name = "V", dim = V_gpu.size(), stride = V_gpu.stride(), data_type = convert_to_cudnn_type(V_gpu.dtype))
    SEQ_LEN_Q = graph.tensor(name = "SEQ_LEN_Q", dim = SEQ_LEN_Q_gpu.size(), stride = SEQ_LEN_Q_gpu.stride(), data_type = convert_to_cudnn_type(SEQ_LEN_Q_gpu.dtype))
    SEQ_LEN_K = graph.tensor(name = "SEQ_LEN_K", dim = SEQ_LEN_K_gpu.size(), stride = SEQ_LEN_K_gpu.stride(), data_type = convert_to_cudnn_type(SEQ_LEN_K_gpu.dtype))
    Bias = graph.tensor(name = "Bias", dim = Bias_gpu.size(), stride = Bias_gpu.stride(), data_type = convert_to_cudnn_type(Bias_gpu.dtype))
    S, O = graph.scaled_dot_product_attention(name = "scaled_dot_product_attention"
                                              , q = Q, k = K, v = V, seq_len_q = SEQ_LEN_Q, seq_len_k = SEQ_LEN_K
                                              , is_inference = False
                                              , scale_k = 0.5
                                              , bias = Bias
                                              , use_padding_mask = True
                                              , use_causal_mask = True
                                              , dropout = (0.5, 123456)
                                              )
    O.set_output(True)
    S.set_output(True)
    graph.build()
    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

    O_actual = torch.zeros(b * s_q * h * d, dtype=torch.float16, device="cuda")
    S_actual = torch.zeros(b * h * s_q * s_kv, dtype=torch.float16, device="cuda")

    graph.execute({Q: Q_gpu, K: K_gpu, V: V_gpu, SEQ_LEN_Q: SEQ_LEN_Q_gpu, SEQ_LEN_K: SEQ_LEN_K_gpu
                   , Bias: Bias_gpu
                   , O: O_actual, S: S_actual}
                   , workspace)
         
@pytest.mark.skipif(pycudnn.get_cudnn_version() < 8900, reason="requires cudnn 8.9 or higher")
def test_scale_dot_product_flash_attention():
    b = 1
    h = 2
    s_q = 2048
    s_kv = 2048
    d = 128

    shape_Q = (b, h, s_q, d)
    stride_Q = (3 * h * d, 3 * d, 3 * b * h * d, 1)

    shape_K = (b, h, d, s_kv)
    stride_K = (3 * h * d, 3 * d, 1, 3 * b * h * d)

    shape_V = (b, h, s_kv, d)
    stride_V = (3 * h * d, 3 * d, 3 * b * h * d, 1)

    offset_Q = 0
    offset_K = d
    offset_V = 2 * d

    qkv_gpu = torch.empty(b * s_q * 3 * h * d, dtype=torch.float16, device="cuda")
    Q_gpu = torch.as_strided(qkv_gpu, shape_Q, stride_Q, storage_offset=offset_Q)
    K_gpu = torch.as_strided(qkv_gpu, shape_K, stride_K, storage_offset=offset_K)
    V_gpu = torch.as_strided(qkv_gpu, shape_V, stride_V, storage_offset=offset_V)

    Seed_gpu = torch.full((1,1,1,1), 123456, dtype=torch.int64, device="cuda")
    Offset_gpu = torch.full((1,1,1,1), 1, dtype=torch.int64, device="cuda")
    
    # Cudnn graph
    graph = pycudnn.pygraph("mha", io_data_type = pycudnn.data_type.HALF, intermediate_data_type = pycudnn.data_type.FLOAT, compute_data_type = pycudnn.data_type.FLOAT)
    Q = graph.tensor(name = "Q", dim = Q_gpu.size(), stride = Q_gpu.stride(), data_type = convert_to_cudnn_type(Q_gpu.dtype))
    K = graph.tensor(name = "K", dim = K_gpu.size(), stride = K_gpu.stride(), data_type = convert_to_cudnn_type(K_gpu.dtype))
    V = graph.tensor(name = "V", dim = V_gpu.size(), stride = V_gpu.stride(), data_type = convert_to_cudnn_type(V_gpu.dtype))
    Seed = graph.tensor(name = "Seed", dim = Seed_gpu.size(), stride = Seed_gpu.stride(), data_type = convert_to_cudnn_type(Seed_gpu.dtype))
    Offset = graph.tensor(name = "Offset", dim = Offset_gpu.size(), stride = Offset_gpu.stride(), data_type = convert_to_cudnn_type(Offset_gpu.dtype))
    O, Stats = graph.scaled_dot_product_flash_attention(name = "scaled_dot_product_flash_attention"
                                              , q = Q, k = K, v = V
                                              , is_inference = False
                                              , scale_k = 0.5
                                              , use_padding_mask = True
                                              , use_causal_mask = True
                                              , dropout = (0.2, Seed, Offset)
                                              )
    O.set_output(True)
    Stats.set_output(True)
    graph.build()
    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

    O_actual = torch.zeros(b * s_q * h * d, dtype=torch.float16, device="cuda")
    Stats_actual = torch.zeros(b * h * s_q * 1, dtype=torch.float16, device="cuda")

    graph.execute({Q: Q_gpu, K: K_gpu, V: V_gpu, Seed: Seed_gpu, Offset: Offset_gpu
                   , O: O_actual, Stats: Stats_actual}
                   , workspace)

if __name__ == "__main__":
    test_scale_dot_product_attention_with_dropout_rng()