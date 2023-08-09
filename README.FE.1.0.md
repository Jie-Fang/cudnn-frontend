# cuDNN FrontEnd(FE) v1.0 API

## Introduction
FE v1.0 API is aimed to extend functionality and usage exposed by the [cuDNN C backend API](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnn-backend-api). Both C++ and python APIs are provided with both having functional parity.  
For a general introduction to FE, please first refer README.md.

## Workflow
The steps involved in building and running a cudnn graph are as follows:
- Create a cudnn graph and specify the global properties. The global properties like compute precision and input/output data type help infer properties that are not explicitly mentioned.
- Create and add the input tensors 
- Create and add the operation noes. The outputs of these operation are of tensor type and can be sequentially used as inputs to the next node.
- Validate the operation graph. This step makes sure the graph is well built and does not have hanging tensors or node.
- Build the cudnn operation graph. This step lowers the graph into cudnn dialect.
- Get the execution plan, based on the heuristics type of your choice.
- Filter out the plans by your custom criteria (Optional)
- Run autotuning on the filter plan (Optional)
- Execute the plan with the relevant data pointers.
    
## APIs
FE v1.0 API follows a functional style of building a graph. Operations take in input tensors and return output tensors. This also allows composition of operations. 

| Purpose                 | C++ API                                                   | Python API   |
| ---                     | ---                                                       | ---          |
| Create tensor           | tensor(...)                                               | tesnor(...)  |
| Convolution Fprop       | conv_fprop <br>Conv_fprop_attributes                      | conv_fprop   |
| Convolution Dgrad       | conv_dgrad <br>Conv_dgrad_attributes                      | conv_dgrad   |
| Convolution Wgrad       | conv_wgrad <br>Conv_wgrad_attributes                      | conv_wgrad   |
| Matrix Multiplication   | matmul <br> Matmul_attributes                             | matmul       |
| Pointwise Operations    | pointwise <br> Pointwise_attributes                       | <br>- add<br>- bias<br>- rqsrt<br>- sub<br>- mul<br>- scale<br>- relu<br>- elu<br>- gelu<br>- cmp_gt       |
| Batch Normalization     | batchnorm <br>Batchnorm_attributes                        | batchnorm    |
| Batch Norm bprop        | batchnorm_backward <br>batchnorm_backward_attributes      | batchnorm    |
| Generate stats of output| genstats <br>Genstats_attributes                          | genstats     |
| BN Finalize of stats    | bn_finalize <br>BN_finalize_attributes                    | bn_finalize  |
| Dbn weight              | dbn_weight <br>DBN_weight_attributes                      | dbn_weight   |
| Scale dot product flash attention | scaled_dot_product_flash_attention<br> Scaled_dot_product_flash_attention_attributes | scaled_dot_product_flash_attention|

### Create Graph
Instantiate an object of class `cudnn_frontend::graph::Graph` which will house tensors and operations.  

Optional graph level attributes can be set on the object:
- `cudnn_frontend::graph::Graph& set_io_data_type(cudnn_frontend::DataType_t)`
- `cudnn_frontend::graph::Graph& set_intermediate_data_type(cudnn_frontend::DataType_t)`
- `cudnn_frontend::graph::Graph& set_compute_data_type(cudnn_frontend::DataType_t)`
These attributes are meant to used as default in case they are not provided for constituent tensors and operations.

### Define Tensors
Users create input tensors to provide to operations within a graph. To add tensors in a graph, use:  
`std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> cudnn_frontend::graph::tensor(cudnn_frontend::graph::Tensor_attributes)`.  
As the API returns a shared pointer, both the user and FE graph are owners of the tensor.  

Tensor attributes is a lighweight structure with setters for each attribute.  
- `cudnn_frontend::graph::Tensor_attributes& set_data_type(cudnn_frontend::DataType_t)`
- `cudnn_frontend::graph::Tensor_attributes& set_dim(std::vector<int64_t>&)`
- `cudnn_frontend::graph::Tensor_attributes& set_stride(std::vector<int64_t>&)`
- `cudnn_frontend::graph::Tensor_attributes& set_is_virtual(bool)`
- `cudnn_frontend::graph::Tensor_attributes& set_is_pass_by_value(bool)`
- `cudnn_frontend::graph::Tensor_attributes& set_reordering_type(cudnn_frontend::TensorReordering_t)`
- `cudnn_frontend::graph::Tensor_attributes& set_name(std::string&)`

### Define Operations
Operations take in mandatory input tensor via positional arguments. Optional input tensors are provided using corresponding setters in operation attributes. 

Operations return an ordered array of output tensors. Any optional outputs if not present will have their shared pointers pointing to `std::nullptr`.

#### Batchnorm Forward
Batchnorm operation computes:
$$ output = scale*{input + mean \over \sqrt{variance + epsilon}} + bias $$

Optionally the operation also computes:
$$ next\_running\_mean = (1 - momentum)*previous\_running\_mean + momentum*current\_running\_mean $$
$$ next\_running\_variance = (1 - momentum)*previous\_running\_variance + momentum*current\_running\_variance $$

The API to achieve above equations is:  
```
std::array<std::shared_ptr<Tensor_attributes>, 5> batchnorm(std::shared_ptr<Tensor_attributes>& input,
                                                            std::shared_ptr<Tensor_attributes>& scale,
                                                            std::shared_ptr<Tensor_attributes>& bias,
                                                            Batchnorm_attributes attribues); 
```
where the output array has tensors in order of: `[output, saved_mean, saved_invariance, next_running_mean, next_running_variance]`

Batchnorm attributes is a lighweight structure with setters for providing optoinal input tensors and other operation attributes:  
```
Batchnorm_attributes&
set_previous_running_stats(std::shared_ptr<Tensor_attributes>& previous_running_mean,
                            std::shared_ptr<Tensor_attributes>& previous_running_variance,
                            std::shared_ptr<Tensor_attributes>& momentum)

Batchnorm_attributes&
set_name(std::string const&)

Batchnorm_attributes&
set_compute_data_type(DataType_t value)
```

#### Batchnorm Finalize

#### Convolution Fprop
Convolution fprop computes:
$$ response = image * filter $$

The API to achieve above is:  
```
std::shared_ptr<Tensor_attributes> conv_fprop(std::shared_ptr<Tensor_attributes> image,
                                                  std::shared_ptr<Tensor_attributes> filter,
                                                  Conv_fprop_attributes);
```

Conv_fprop attributes is a lighweight structure with setters:  
```
Conv_fprop_attributes&
set_padding(std::vector<int64_t>)

Conv_fprop_attributes&
set_stride(std::vector<int64_t>)

Conv_fprop_attributes&
set_dilation(std::vector<int64_t>)

Conv_fprop_attributes&
set_name(std::string const&)

Conv_fprop_attributes&
set_compute_data_type(DataType_t value)
```

#### Convolution Dgrad
Convolution dgrad computes data gradient during backpropagation.

The API to achieve above is:  
```
std::shared_ptr<Tensor_attributes> conv_dgrad(std::shared_ptr<Tensor_attributes> image,
                                                  std::shared_ptr<Tensor_attributes> filter,
                                                  Conv_dgrad_attributes);
```

Conv_dgrad attributes is a lighweight structure with setters:  
```
Conv_dgrad_attributes&
set_padding(std::vector<int64_t>)

Conv_dgrad_attributes&
set_stride(std::vector<int64_t>)

Conv_dgrad_attributes&
set_dilation(std::vector<int64_t>)

Conv_dgrad_attributes&
set_name(std::string const&)

Conv_dgrad_attributes&
set_compute_data_type(DataType_t value)
```

#### Convolution Wgrad
Convolution wgrad computes weight gradient during backpropagation.

The API to achieve above is:  
```
std::shared_ptr<Tensor_attributes> conv_wgrad(std::shared_ptr<Tensor_attributes> image,
                                                  std::shared_ptr<Tensor_attributes> filter,
                                                  Conv_wgrad_attributes);
```

Conv_wgrad attributes is a lighweight structure with setters:  
```
Conv_wgrad_attributes&
set_padding(std::vector<int64_t>)

Conv_wgrad_attributes&
set_stride(std::vector<int64_t>)

Conv_wgrad_attributes&
set_dilation(std::vector<int64_t>)

Conv_wgrad_attributes&
set_name(std::string const&)

Conv_wgrad_attributes&
set_compute_data_type(DataType_t value)
```

#### Batchnorm Backward(DBN)
DBN operation computes data graident, scale gradient, bias gradient during backpropagation of batchnorm forward operation.

The API to achieve above is:  
```
std::array<std::shared_ptr<Tensor_attributes>, 3> batchnorm_backward(std::shared_ptr<Tensor_attributes> loss,
                                                                         std::shared_ptr<Tensor_attributes> input,
                                                                         std::shared_ptr<Tensor_attributes> scale,
                                                                         batchnorm_backward_attributes);
```
where the output array has tensors in order of: `[input gradient, scale gradient, bias gradient]`.

DBN attributes is a lighweight structure with setters:  
```
batchnorm_backward_attributes&
set_saved_mean_and_inv_variance(std::shared_ptr<Tensor_attributes> saved_mean,
                                std::shared_ptr<Tensor_attributes> saved_inverse_variance)
                                
batchnorm_backward_attributes&
set_epsilon(std::shared_ptr<Tensor_attributes> epsilon)

batchnorm_backward_attributes&
set_name(std::string const&)

batchnorm_backward_attributes&
set_compute_data_type(DataType_t value)
```
Only setting either (saved mean and inverse_variance) or (epsilon) is necessary.

#### Batchnorm Backward weights

#### Generate Stats
Genstats operation computes sum and sum of squares per-channel dimension.

The API to achieve above is:  
```
std::array<std::shared_ptr<Tensor_attributes>, 2>
cudnn_frontend::graph::genstats(std::shared_ptr<Tensor_attributes>, Genstats_attributes);
```
where the output array has tensors in order of: `[sum, square_sum]`

Genstats attributes is a lighweight structure with setters:  
```
Genstats_attributes&
set_name(std::string const&)

Genstats_attributes&
set_compute_data_type(DataType_t value)
```

#### Matmul
Matmul operation computes:
$$ C[M, N] = A[M, K] * B[K, N] $$
Last two dimensions of input dimensions are interpretted as M, N, K. All other preceding dimensions are interpretted as batch dimensions.  
The operation also has broadcasting capabilites which is described in [cudnn Backend's matmul operation](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR).

The API to achieve above is:  
```
std::shared_ptr<Tensor_attributes>
Matmul(std::shared_ptr<Tensor_attributes> a, std::shared_ptr<Tensor_attributes> b, Matmul_attributes);
```

Matmul attributes is a lighweight structure with setters:  
```
Matmul_attributes&
set_name(std::string const&)

Matmul_attributes&
set_compute_data_type(DataType_t value)
```

#### Pointwise
Pointwise performs an elementwise operation between two tensors. The operation used is controlled by pointwise mode `cudnn_frontend::PointwiseMode_t`.   

The API to achieve above is:  
```
std::shared_ptr<Tensor_attributes>
pointwise(std::shared_ptr<Tensor_attributes>,
          Pointwise_attributes);

std::shared_ptr<Tensor_attributes>
pointwise(std::shared_ptr<Tensor_attributes>,
          std::shared_ptr<Tensor_attributes>,
          Pointwise_attributes);

std::shared_ptr<Tensor_attributes>
pointwise(std::shared_ptr<Tensor_attributes>,
          std::shared_ptr<Tensor_attributes>,
          std::shared_ptr<Tensor_attributes>,
          Pointwise_attributes);
```
where the pointwise mode dictates the API among the choices above.
Please refer to documentation of `cudnn_frontend::PointwiseMode_t` for details.

Pointwise attributes is a lighweight structure with setters:  
```
Pointwise_attributes&
set_mode(PointwiseMode_t)

Pointwise_attributes&
set_axis(int64_t)

Pointwise_attributes&
set_name(std::string const&)

Pointwise_attributes&
set_compute_data_type(DataType_t value)
```

#### Reduction
Reduction operation reduces an input tensor using an operation controlled by `cudnn_frontend::ReductionMode_t`.
The dimensions in input tensors to reduce are deduced using output tensor dimensions.

The API to achieve above is:  
```
std::shared_ptr<Tensor_attributes>
reduction(std::shared_ptr<Tensor_attributes> input, Reduction_attributes);
```

Reduction attributes is a lighweight structure with setters:  
```
Reduction_attributes&
set_mode(ReductionMode_t)

Reduction_attributes&
set_name(std::string const&)

Reduction_attributes&
set_compute_data_type(DataType_t value)
```

#### Rng

#### Scaled Dot Product Attention

#### Scaled Dot Product Flash Attention

### Validate graph
Validate API ensures API usage is sound, checks against dangling tensors, etc.
Internally, any unspecified properties like dimensions, strides, etc are inferred.

```
cudnn_frontend::error_t cudnn_frontend::graph::Graph::validate()
```

### Build cudnn backend graph
This method creates cudnn backend descriptors for all constituents of the graph.

```
cudnn_frontend::error_t cudnn_frontend::graph::Graph::build_operation_graph(cudnnHandle_t handle)
```

### Get Execution plans
This method returns a list of execution plans that can potentially run the FE graph.

```
cudnn_frontend::graph::Plans cudnn_frontend::graph::Graph::get_execution_plans(heur_mode_t)
```

### Filter plans
Users can filter out plans against numerical, behavioral notes, or plans that do not provide desired functional correctness.

```
cudnn_frontend::graph::Plans& cudnn_frontend::graph::Plans::filter_out_numeric_notes(std::vector<cudnnBackendNumericalNote_t> const&);
cudnn_frontend::graph::Plans& cudnn_frontend::graph::Plans::filter_out_behavior_notes(std::vector<cudnnBackendBehaviorNote_t> const&);
cudnn_frontend::graph::Plans& cudnn_frontend::graph::Plans::filter_out_workspace_greater_than(int64_t max_allowed_workspace);
```

### Check graph support
This method guarantees that executing the graph using plans queried will succeed.

```
cudnn_frontend::error_t Plans::check_support();
```

### Autotune


### Set Execution plans
After checking support, filtering and/or autotuning, execution plans can be set in descending order of preference.

```
cudnn_frontend::error_t
cudnn_frontend::graph::Graph::set_execution_plans(cudnn_frontend::::graph::Plans const&)
```

### Execute
Executing graph requires device pointers to all input output tensors and a user alloaction device workspace pointer.

```
cudnn_frontend::error_t
cudnn_frontend::graph::Graph::execute(cudnnHandle_t handle,
                                        std::unordered_map<std::shared_ptr<Tensor>, void *> var_pack,
                                        void* workspace);
```

## Samples
Samples are meant to illustrate FE v1.0 API usage to users.  
- `samples/cpp` contains samples that use C++ API.  
- `samples/python` contains samples that use python API.

C++ samples are written using [Catch2](https://github.com/catchorg/Catch2) test framework.  
Python samples are written using [pytest](https://github.com/pytest-dev/pytest) and [pytorch](https://pytorch.org), with both requiring external installation. 
