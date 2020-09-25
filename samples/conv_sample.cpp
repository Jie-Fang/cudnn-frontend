/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */ 


#include "conv_sample.h"
using common_descriptors = std::tuple<
           cudnn_frontend::Tensor,
           cudnn_frontend::Tensor,
           cudnn_frontend::Tensor,
           cudnn_frontend::ConvDesc>;

common_descriptors
create_common_descriptors(int64_t* x_dim_padded,
    int64_t* padA,
    int64_t* convstrideA,
    int64_t* dilationA,
    int64_t* w_dim_padded,
    int64_t* y_dim_padded,
    cudnnDataType_t dataType,
    cudnnConvolutionMode_t mode) {
    
    const int convDim = 2;

    int64_t strideA_padded[4];
    int64_t outstrideA_padded[4];
    int64_t filterstrideA_padded[4];

    generateStrides(w_dim_padded, filterstrideA_padded, 4, CUDNN_TENSOR_NCHW);
    generateStrides(x_dim_padded, strideA_padded, 4, CUDNN_TENSOR_NCHW);
    generateStrides(y_dim_padded, outstrideA_padded, 4, CUDNN_TENSOR_NCHW);

    return 
        common_descriptors( 
            cudnn_frontend::TensorBuilder()
                .setDim(4, x_dim_padded)
                .setStrides(4,strideA_padded)
                .setId('x')
                .setAlignment(4)
                .setDataType(dataType)
                .build(),
            cudnn_frontend::TensorBuilder()
                .setDim(4, y_dim_padded)
                .setStrides(4,outstrideA_padded)
                .setId('y')
                .setAlignment(4)
                .setDataType(dataType)
                .build(),
            cudnn_frontend::TensorBuilder()
                .setDim(4, w_dim_padded)
                .setStrides(4,filterstrideA_padded)
                .setId('w')
                .setAlignment(4)
                .setDataType(dataType)
                .build(),
            cudnn_frontend::ConvDescBuilder()
                .setDataType(dataType)
                .setMathMode(mode)
                .setNDims(convDim)
                .setStrides(convDim, convstrideA)
                .setPrePadding(convDim, padA)
                .setPostPadding(convDim, padA)
                .setDilation(convDim, dilationA)
                .build()
        );
}

cudnn_frontend::OperationGraph 
create_operation_graph(common_descriptors &descriptors, cudnnBackendDescriptorType_t mode, cudnnHandle_t handle_) {
    float alpha     = 1.0f;
    float beta      = 0.0;

    auto op = cudnn_frontend::OperationBuilder()
        .setxDesc(std::get<0>(descriptors))
        .setyDesc(std::get<1>(descriptors))
        .setwDesc(std::get<2>(descriptors))
        .setcDesc(std::get<3>(descriptors))
        .setAlpha(alpha)
        .setBeta(beta)
        .setOpMode(mode)
        .build();
    
    std::cout << op.describe() << std::endl;

    std::array<cudnn_frontend::Operation const *, 1> ops = {&op};

    return cudnn_frontend::OperationGraphBuilder()
        .setHandle(handle_)
        .setOperationGraph(ops.size(), ops.data())
        .build();
}

void run_from_heuristics (
    int64_t* x_dim_padded,
    int64_t* padA,
    int64_t* convstrideA,
    int64_t* dilationA,
    int64_t* w_dim_padded,
    int64_t* y_dim_padded,
    cudnnDataType_t dataType,
    cudnnConvolutionMode_t mode,
    float * devPtrX,
    float * devPtrW,
    float * devPtrY)
{
    cudnnHandle_t handle_;

    try {
        checkCudnnErr(cudnnCreate(&handle_));
        common_descriptors descriptors = create_common_descriptors(x_dim_padded, padA, convstrideA, dilationA, w_dim_padded, y_dim_padded, dataType, mode);

        std::cout << std::get<0>(descriptors).describe() << std::endl;
        std::cout << std::get<1>(descriptors).describe() << std::endl;
        std::cout << std::get<2>(descriptors).describe() << std::endl;
        std::cout << std::get<3>(descriptors).describe() << std::endl;

        auto opGraph = create_operation_graph(descriptors, CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR, handle_);
        std::cout << opGraph.describe() << std::endl;

        auto heuristics = cudnn_frontend::EngineHeuristicsBuilder()
            .setOperationGraph(opGraph)
            .setHeurMode(CUDNN_HEUR_MODE_INSTANT)
            .build();

        std::cout << "Heuristic has " << heuristics.getEngineConfigCount() << " configurations " << std::endl;
        auto &engine_config = heuristics.getEngineConfig();

        auto plan = cudnn_frontend::ExecutionPlanBuilder()
            .setHandle(handle_)
            .setEngineConfig(engine_config[0])
            .build();

        auto workspace_size = plan.getWorkspaceSize(); 
        std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;
        void *workspace_ptr = nullptr;
        if (workspace_size > 0) {
            checkCudaErr(cudaMalloc(&workspace_ptr, workspace_size));
        }
        void * data_ptrs[] = {devPtrX, devPtrY, devPtrW};
        int64_t uids[] = {'x', 'y', 'w'};
        auto variantPack = cudnn_frontend::VariantPackBuilder()
            .setWorkspacePointer(workspace_ptr)
            .setDataPointers(3, data_ptrs)
            .setUids(3, uids)
            .build();
        std::cout << "variantPack " << variantPack.describe() << std::endl;
        cudnnStatus_t status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
        if (workspace_size > 0) {
            checkCudaErr(cudaFree(workspace_ptr));
        }
        cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error");

    } catch (cudnn_frontend::cudnnException e) {
        std::cout << "[ERROR] Exception " << e.what() << std::endl;
    }

    if (handle_) cudnnDestroy(handle_);
    return;

}

void run_from_global_index (
    int64_t* x_dim_padded,
    int64_t* padA,
    int64_t* convstrideA,
    int64_t* dilationA,
    int64_t* w_dim_padded,
    int64_t* y_dim_padded,
    cudnnDataType_t dataType,
    cudnnConvolutionMode_t mode,
    float * devPtrX,
    float * devPtrW,
    float * devPtrY)
{
    cudnnHandle_t handle_;

    try {
        checkCudnnErr(cudnnCreate(&handle_));
        common_descriptors descriptors = create_common_descriptors(x_dim_padded, padA, convstrideA, dilationA, w_dim_padded, y_dim_padded, dataType, mode);

        std::cout << std::get<0>(descriptors).describe() << std::endl;
        std::cout << std::get<1>(descriptors).describe() << std::endl;
        std::cout << std::get<2>(descriptors).describe() << std::endl;
        std::cout << std::get<3>(descriptors).describe() << std::endl;

        auto opGraph = create_operation_graph(descriptors, CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR, handle_);
        std::cout << opGraph.describe() << std::endl;

        auto total_engines = opGraph.getEngineCount();
        // We have to randomly pick one engine from [0, total_engines)
        // Selecting "0" by default
        auto engine = cudnn_frontend::EngineBuilder()
            .setGlobalEngineIdx(0)
            .setOperationGraph(opGraph)
            .build();
        std::cout << engine.describe() << std::endl;
        auto knobs = engine.getKnobs();
        for (auto it = std::begin(knobs); it != std::end(knobs); ++it) {
            std::cout << it->describe() << std::endl;
        }
        if (knobs.begin() != knobs.end()) {
            std::cout << "Updated knob choice" << std::endl;
            knobs.begin()->setChoice(knobs.begin()->getMinValue() + 1);
            std::cout << knobs.begin()->describe() << std::endl;
        }
        auto engine_config = cudnn_frontend::EngineConfigBuilder()
            .setEngine(engine)
            .build();
        std::cout << engine_config.describe() << std::endl;
        auto plan = cudnn_frontend::ExecutionPlanBuilder()
            .setHandle(handle_)
            .setEngineConfig(engine_config)
            .build();

        auto workspace_size = plan.getWorkspaceSize(); 
        std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;

        void * data_ptrs[] = {devPtrX, devPtrY, devPtrW};
        int64_t uids[] = {'x', 'y', 'w'};
        auto variantPack = cudnn_frontend::VariantPackBuilder()
            .setWorkspacePointer(nullptr)
            .setDataPointers(3, data_ptrs)
            .setUids(3, uids)
            .build();
        std::cout << "variantPack " << variantPack.describe() << std::endl;
        cudnnStatus_t status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
        cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error");

    } catch (cudnn_frontend::cudnnException e) {
        std::cout << "[ERROR] Exception " << e.what() << std::endl;
    }

    if (handle_) cudnnDestroy(handle_);
}

void run_with_external_config (
    int64_t* x_dim_padded,
    int64_t* padA,
    int64_t* convstrideA,
    int64_t* dilationA,
    int64_t* w_dim_padded,
    int64_t* y_dim_padded,
    cudnnDataType_t dataType,
    cudnnConvolutionMode_t mode,
    float * devPtrX,
    float * devPtrW,
    float * devPtrY)
{
    cudnnHandle_t handle_;

    try {
        checkCudnnErr(cudnnCreate(&handle_));
        common_descriptors descriptors = create_common_descriptors(x_dim_padded, padA, convstrideA, dilationA, w_dim_padded, y_dim_padded, dataType, mode);

        std::cout << std::get<0>(descriptors).describe() << std::endl;
        std::cout << std::get<1>(descriptors).describe() << std::endl;
        std::cout << std::get<2>(descriptors).describe() << std::endl;
        std::cout << std::get<3>(descriptors).describe() << std::endl;

        auto opGraph = create_operation_graph(descriptors, CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR, handle_);
        std::cout << opGraph.describe() << std::endl;

        auto heuristics = cudnn_frontend::EngineHeuristicsBuilder()
            .setOperationGraph(opGraph)
            .setHeurMode(CUDNN_HEUR_MODE_INSTANT)
            .build();

        std::cout << "Heuristic has " << heuristics.getEngineConfigCount() << " configurations " << std::endl;
        auto &engine_config = heuristics.getEngineConfig();

        auto fallback = cudnn_frontend::EngineFallbackListBuilder()
            .setOperationGraph(opGraph)
            .setOperation(CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR)
            .build();
        auto &fallback_list = fallback.getFallbackList();
        std::cout << "Fallback List has " << fallback_list.size() << " configurations " << std::endl;

        cudnn_frontend::EngineConfigList filtered_configs;
        cudnn_frontend::filter(engine_config, filtered_configs, cudnn_frontend::isNonDeterministic);
        cudnn_frontend::filter(fallback_list, filtered_configs, cudnn_frontend::isNonDeterministic);

        std::cout << "Heuristic has " << heuristics.getEngineConfigCount() << " configurations " << std::endl;
        std::cout << "Fallback List has " << fallback_list.size() << " configurations " << std::endl;
        std::cout << "Filter config list has " << filtered_configs.size() << " configurations " << std::endl;

        auto plan = cudnn_frontend::ExecutionPlanBuilder()
            .setHandle(handle_)
            .setEngineConfig(filtered_configs[0])
            .build();

        std::cout << plan.describe() << std::endl;
        auto workspace_size = plan.getWorkspaceSize(); 
        void * data_ptrs[] = {devPtrX, devPtrY, devPtrW};
        int64_t uids[] = {'x', 'y', 'w'};
        auto variantPack = cudnn_frontend::VariantPackBuilder()
            .setWorkspacePointer(nullptr)
            .setDataPointers(3, data_ptrs)
            .setUids(3, uids)
            .build();
        std::cout << "variantPack " << variantPack.describe() << std::endl;
        cudnnStatus_t status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
        cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error");

    } catch (cudnn_frontend::cudnnException e) {
        std::cout << "[ERROR] Exception " << e.what() << std::endl;
    }

    if (handle_) cudnnDestroy(handle_);

    return;
}

