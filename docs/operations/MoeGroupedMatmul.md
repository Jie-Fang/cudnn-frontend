# Matmul

The MoE Grouped Matmul operation computes a grouped matmul operation based on given first token offset, token index, and token ks in three modes (None, Gather, and Scatter):

In None and Scatter modes:
$$ Output[1, S * topK, N] = Token[1, S * topK, K] * Weight[E, K, N] $$

In Gather mode:
$$ Output[1, S * topK, N] = Token[1, S, K] * Weight[E, K, N] $$

FirstTokenOffset has shape [B * E, 1, 1] and is used in all three modes.

TokenIndex has shape [1, S * topK, 1] and is used in the Gather and Scatter modes.

TokenKs has shape [1, S * topK, 1] and is used in the Scatter mode.

TopK as an int32_t element needs to be explicitly provided in the Scatter mode.

The operation also has broadcasting capabilities which are described in {ref}`cudnn backend's moe grouped matmul operation <CUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_DESCRIPTOR>`.

## C++ API

```
std::shared_ptr<Tensor_attributes>
moe_grouped_matmul(std::shared_ptr<Tensor_attributes> token, std::shared_ptr<Tensor_attributes> weight, std::shared_ptr<Tensor_attributes> first_token_offset, std::shared_ptr<Tensor_attributes> token_index, std::shared_ptr<Tensor_attributes> token_ks, moe_grouped_matmul_attribute);
```

Moe_grouped_matmul attributes is a lightweight structure with setters:  
```
Moe_grouped_matmul&
set_name(std::string const&)

Moe_grouped_matmul&
set_mode(MoeGroupedMatmulMode_t mode)

Moe_grouped_matmul&
set_compute_data_type(DataType_t value)

Moe_grouped_matmul&
set_top_k(int32_t top_k_value)
```
