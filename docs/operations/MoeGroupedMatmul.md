# MoE Grouped Matmul

The MoE Grouped Matmul operation computes a grouped matmul operation based on given first token offset, token index, and token ks in three modes (None, Gather, and Scatter):

In None and Scatter modes:
$$ Output[1, S * topK, N] = Token[1, S * topK, K] * Weight[E, K, N] $$

In Gather mode:
$$ Output[1, S * topK, N] = Token[1, S, K] * Weight[E, K, N] $$

FirstTokenOffset has shape [B * E, 1, 1] and is used in all three modes.

TokenIndex has shape [1, S * topK, 1] and is used in the Gather and Scatter modes.

TokenKs has shape [1, S * topK, 1] and is used in the Scatter mode.

TopK as an int32_t element needs to be explicitly provided in the Scatter mode.

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

# MoE Grouped Matmul Bwd

The MoE Grouped Matmul Bwd operation computes the backward propagation for weight of a grouped matmul operation

transposed(dOutput) ([1, N, S]) @ token ([1, S, K]) -> transposed(dWeight) ([E, N, K])

## C++ API

```
std::shared_ptr<Tensor_attributes>
moe_grouped_matmul_bwd(std::shared_ptr<Tensor_attributes> doutput, std::shared_ptr<Tensor_attributes> token, std::shared_ptr<Tensor_attributes> first_token_offset, moe_grouped_matmul_bwd_attribute);
```

Moe_grouped_matmul_bwd attributes is a lightweight structure with setters:  
```
Moe_grouped_matmul_bwd&
set_name(std::string const&)

Moe_grouped_matmul_bwd&
set_compute_data_type(DataType_t value)
```
