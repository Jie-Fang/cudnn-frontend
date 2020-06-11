# CUDNN API WRAP

CUDNN API provides a wraper to use the CUDNN C API. The cudnn "C" API is documented in the [cudnn developer page](https://cudnn.developer.page). 

The header only library requires:
cudnn_backend_wrap.h

There are multiple header files which correspond to each cudnnBackendDescriptorType_t documented in the enum.
    - Tensor.h -> CUDNN_BACKEND_TENSOR_DESCRIPTOR
    - ConvDesc.h -> CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR
    - Operation.h -> CUDNN_BACKEND_OPERATION_*_DESCRIPTOR
    - OperationGraph.h -> CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR
    - Engine.h -> CUDNN_BACKEND_ENGINE_DESCRIPTOR
    - EngineConfig.h -> CUDNN_BACKEND_ENGINECFG_DESCRIPTOR
    - ExecutionPlan.h -> CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR
    - VariantPack.h -> CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR


Outstanding task(s)
    - [] Adding get attributes
    - [] Use an engine with workspace
    - [] Write sample example for
        - [] wgrad
        - [] conv
        - [] dgrad
    - [] Confirm naming of the cudnn_backend_wrap.h



How to build:
     - CUDA_PATH has the cuda installation. 
        - Include files are in CUDA_PATH/include
        - Link files are in CUDA_PATH/lib64
     - CUDNN_WRAP_PATH has the wrapper header files.

     make CUDA_PATH=/usr/local/cuda CUDNN_WRAP_PATH=/usr/local/include/
