# CUDNN FRONTEND API

## Introduction
CUDNN Frontend API is a c++ header-only library to use the CUDNN C backend API. The cudnn "C" API is documented in the [cudnn developer page](https://cudnn.developer.page). 

## Usage
In order to include the entire library include the cudnn_frontend into your compilation unit.
cudnn_frontend.h

## Organization
Each cudnnBackendDescriptorType_t documented in the enum is organized into its header file.
- cudnn_frontend_Tensor.h         -> CUDNN_BACKEND_TENSOR_DESCRIPTOR
- cudnn_frontend_ConvDesc.h       -> CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR
- cudnn_frontend_PointWiseDesc.h  -> CUDNN_BACKEND_POINTWISE_DESCRIPTOR
- cudnn_frontend_Operation.h      -> CUDNN_BACKEND_OPERATION_*_DESCRIPTOR
- cudnn_frontend_OperationGraph.h -> CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR
- cudnn_frontend_Heuristics.h     -> CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR
- cudnn_frontend_Engine.h         -> CUDNN_BACKEND_ENGINE_DESCRIPTOR
- cudnn_frontend_EngineConfig.h   -> CUDNN_BACKEND_ENGINECFG_DESCRIPTOR
- cudnn_frontend_ExecutionPlan.h  -> CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR
- cudnn_frontend_VariantPack.h    -> CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR

Utility Functions
- cudnn_frontend_find_plan.h -> Implements the cudnnFindPlan function
- cudnn_frontend_get_plan.h  -> Implements the cudnnGetPlan function
- cudnn_frontend_Filters.h   -> List of helpful utility function to filter out execution plans

Error Handling 
- cudnn_frontend_utils.h

Fallback Lists
- cudnn_frontend_EngineFallbackList.h -> Provides fallback engine id if heuristics do not provide an executable engine.

## Samples
Multiple samples of convoultion, dgrad, wgrad and convBiasAct are addeded in the samples/test_list.cpp and samples/conv_sample.cpp.  

Sample tests are written using the https://github.com/catchorg/Catch2 c++ test framework.

##### How to build samples:
     - CUDA_PATH has the cuda installation. 
        - Include files are in CUDA_PATH/include
        - Link files are in CUDA_PATH/lib64
     - CUDNN_WRAP_PATH has the wrapper header files.

     make CUDA_PATH=/usr/local/cuda CUDNN_WRAP_PATH=/usr/local/include/
    
## cudnnFindPlan and cudnnGetPlan:
Prior to cudnn_v8 we had cudnnFindConvoultion* and cudnnGetConvoultion* which provided a way to sample all the algorithms for a given problem and study the run times. This can be further used to cache the best algorithms for a given problem.  This has been replaced with cudnnFindPlan and cudnnGetPlan.

In order to use the cudnnFindPlan an user needs to provide:
- Source of pruned list of engineConfig for the given problem statement.
- Filter function to Filter out the execution plan based on the prerequisite conditions.

The cudnnFindPlan in turn
- Creates a set of execution plans that are supported.
- Execute each filtered plan and ranks them in order of execution plan.

The most common engineConfig generation is the built-in heuristics of cudnn_v8. Generally this is appended with the fallback list. An example of usage can be seen in run_from_cudnn_find(...) function in conv_sample.cpp
