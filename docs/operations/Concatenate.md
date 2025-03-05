# Concatenate

The Concatenate operation merges two or more tensors into one, along the specified axis.  The user may also specify an in-place merge.  The operation provides the capabilities of the {ref}`cudnn backend's concatenate operation                          <CUDNN_BACKEND_OPERATION_CONCAT_DESCRIPTOR>`.

## C++ API

```
std::shared_ptr<Tensor_attributes>
concatenate(std::vector<std::shared_ptr<Tensor_attributes>>, Concatenate_attributes);
```

Concatenate attributes is a lightweight structure with inputs, outputs, and setters:  
```
std::vector<std::shared_ptr<Tensor_attributes>> inputs;

std::unordered_map<output_names, std::shared_ptr<Tensor_attributes>> outputs;

Concatenate_attributes&
set_axis(int64_t const value)

Concatenate_attributes&
set_in_place_index(int64_t const value)
```
