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

#include "norm_sample.h"
#include <cudnn_frontend.h>
#include "cudnn_backend.h"
#include "error_util.h"

bool
AllowAll(cudnnBackendDescriptor_t engine_config) {
    (void)engine_config;
    return false;
}

void
run_batch_norm_forward(
    int64_t *perChannelSum, 
    int64_t *epsilon,
    int64_t *tensorDims,
    int64_t *peerDims,

    void *xDevPtr, 
    void *yDevPtr, 
    void *scaledevPtr, 
    void *biasdevPtr, 
    void *in_meandevPtr, 
    void *in_vardevPtr, 
    void *out_meandevPtr, 
    void *out_vardevPtr,
    void *saved_meandevPtr, 
    void *saved_inv_vardevPtr, 
    void *peer_devPtr, 

    double epsilon_val,
    double exponential_decay_factor)

{
    cudnnHandle_t handle_;
    std::cout << "================ Running Batch Norm Forward ======================= " << std::endl;
    try {
        // Create the cudnn handle
        checkCudnnErr(cudnnCreate(&handle_));

        // Creates the necessary tensor descriptors
        int64_t tensor_stride[4];
        int64_t stride[4];
        int64_t peer_stride[4];

        generateStrides(tensorDims, tensor_stride, 4, CUDNN_TENSOR_NHWC);
        generateStrides(peerDims, peer_stride, 4, CUDNN_TENSOR_NHWC);

        auto tensor_create = [&tensor_stride, &tensorDims](cudnnDataType_t type,
                                int64_t id) {
            return cudnn_frontend::TensorBuilder()
                   .setDim(4, tensorDims)
                   .setStrides(4, tensor_stride)
                   .setId(id)
                   .setAlignment(16)
                   .setDataType(type)
                   .build();
        };

        auto peer_tensor_create = [&peer_stride, &tensorDims](cudnnDataType_t type,
                                int64_t id) {
            return cudnn_frontend::TensorBuilder()
                   .setDim(4, tensorDims)
                   .setStrides(4, peer_stride)
                   .setId(id)
                   .setAlignment(16)
                   .setDataType(type)
                   .build();
        };


        generateStrides(perChannelSum, stride, 4, CUDNN_TENSOR_NHWC);

        auto per_channel_tensor_create = [&stride, &perChannelSum](cudnnDataType_t type,
                                int64_t id) {
            return cudnn_frontend::TensorBuilder()
                   .setDim(4, perChannelSum)
                   .setStrides(4, stride)
                   .setId(id)
                   .setAlignment(16)
                   .setDataType(type)
                   .build();
        };

        auto xTensor             = tensor_create(CUDNN_DATA_HALF, 100);
        auto yTensor             = tensor_create(CUDNN_DATA_HALF, 101);
        auto scaleTensor         = per_channel_tensor_create(CUDNN_DATA_FLOAT, 102);
        auto biasTensor          = per_channel_tensor_create(CUDNN_DATA_FLOAT, 103);
        auto inMeanTensor        = per_channel_tensor_create(CUDNN_DATA_FLOAT, 104);
        auto inVarTensor         = per_channel_tensor_create(CUDNN_DATA_FLOAT, 105);
        auto outMeanTensor       = per_channel_tensor_create(CUDNN_DATA_FLOAT, 106);
        auto outVarTensor        = per_channel_tensor_create(CUDNN_DATA_FLOAT, 107);
        auto savedMeanTensor     = per_channel_tensor_create(CUDNN_DATA_FLOAT, 108);
        auto savedInvVarTensor   = per_channel_tensor_create(CUDNN_DATA_FLOAT, 109);

        auto peerStatTensor      = peer_tensor_create(CUDNN_DATA_FLOAT, 202);

        int64_t epsilon_stride[4];
        generateStrides(epsilon, epsilon_stride, 4, CUDNN_TENSOR_NHWC);
        auto scalar_tensor_create = [&epsilon_stride, &epsilon](cudnnDataType_t type,
                                int64_t id) {
            return cudnn_frontend::TensorBuilder()
                   .setDim(4, epsilon)
                   .setStrides(4, epsilon_stride)
                   .setId(id)
                   .setAlignment(16)
                   .setDataType(type)
                   .setByValue(true)
                   .build();
        };

        auto epsilonTensor       = scalar_tensor_create(CUDNN_DATA_DOUBLE, 300);
        auto expDecayTensor      = scalar_tensor_create(CUDNN_DATA_DOUBLE, 301);

#if (CUDNN_VERSION >= 8500)
        // Batch normalization
        cudnnBackendNormMode_t normalizationMode = CUDNN_BATCH_NORM;

        // Forward training
        cudnnBackendNormFwdPhase_t phase = CUDNN_NORM_FWD_TRAINING;

        //Create a Finalize node
        auto batch_norm_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR)
                    .setNormalizationMode(normalizationMode)
                    .setNormFwdPhase(phase)
                    .setxDesc(xTensor)
                    .setScaleAndBias(scaleTensor, biasTensor)
                    .setPrevRunningMeanAndVar(inMeanTensor, inVarTensor)
                    .setNextRunningMeanAndVar(outMeanTensor, outVarTensor)
                    .setSavedMeanAndInvVar(savedMeanTensor, savedInvVarTensor)
                    .setEpsilonTensor(epsilonTensor)
                    .setExpDecayFactorTensor(expDecayTensor)
                    .addPeerStatTensor(peerStatTensor)
                    .setyDesc(yTensor)
                    .build();

        std::array<cudnn_frontend::Operation const*, 1> ops = {&batch_norm_op};
        auto opGraph = cudnn_frontend::OperationGraphBuilder().setHandle(handle_).setOperationGraph(ops.size(), ops.data()).build();
        std::cout << opGraph.describe() << std::endl;

        cudnn_frontend::EngineConfigList filtered_configs;
        auto statuses = 
            cudnn_frontend::get_heuristics_list<2>({"heuristics_instant" 
            , "heuristics_fallback"
            }, opGraph,::AllowAll, filtered_configs, true);
        
        std::cout << "get_heuristics_list Statuses: ";
        for (auto i = 0 ; i < statuses.size(); i++) {
            std::cout << cudnn_frontend::to_string(statuses[i]) << " ";
        }
        std::cout << std::endl;
        std::cout << "Filter config list has " << filtered_configs.size() << " configurations " << std::endl;

        auto plan_builder = [&filtered_configs, &opGraph, &handle_]() {
            for (auto i = 0; i < filtered_configs.size(); i++) {
                try {
                    auto plan = cudnn_frontend::ExecutionPlanBuilder().setHandle(handle_).setEngineConfig(filtered_configs[i], opGraph.getTag()).build();
		            return plan;
                } catch (cudnn_frontend::cudnnException &e) {
                    continue;
                }
            }
            return cudnn_frontend::ExecutionPlanBuilder().setHandle(handle_).setEngineConfig(filtered_configs[0], opGraph.getTag()).build();
        };

	    CHECK(filtered_configs.size() > 0);
        auto plan = plan_builder();
        std::cout << "Plan tag: " << plan.getTag() << std::endl;

        auto workspace_size = plan.getWorkspaceSize();
        std::cout << plan.describe() << std::endl;

        void* workspace_ptr = nullptr;
        if (workspace_size > 0) {
            checkCudaErr(cudaMalloc(&workspace_ptr, workspace_size));
        }

        void* data_ptrs[13] = {xDevPtr, yDevPtr, scaledevPtr, biasdevPtr, 
                               in_meandevPtr, in_vardevPtr, out_meandevPtr, out_vardevPtr,
                               saved_meandevPtr, saved_inv_vardevPtr, peer_devPtr,
                               &epsilon_val, &exponential_decay_factor};
        int64_t uids[13]    = {100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 202, 300, 301};
        auto variantPack  = cudnn_frontend::VariantPackBuilder()
                               .setWorkspacePointer(workspace_ptr)
                               .setDataPointers(13, data_ptrs)
                               .setUids(13, uids)
                               .build();
        std::cout << "variantPack " << variantPack.describe() << std::endl;
        cudnnStatus_t status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());

        checkCudaErr(cudaDeviceSynchronize());
        if (workspace_size > 0) {
            checkCudaErr(cudaFree(workspace_ptr));
        }

        cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error", status);

        std::cout << "Batch normalization forward run completed successfully" << std::endl;

#endif

    } catch (cudnn_frontend::cudnnException &e) {
        struct cudaDeviceProp prop;
        checkCudaErrors(cudaGetDeviceProperties(&prop, 0));
        if (prop.major == 8) {
            std::cout << "[ERROR] Exception " << e.what() << std::endl;
            CHECK(false);
        }
    }
}

void
run_batch_norm_backward(
    int64_t *perChannelSum, 
    int64_t *epsilon,
    int64_t *tensorDims,
    int64_t *peerDims,

    void *xDevPtr, 
    void *dyDevPtr, 
    void *scaledevPtr, 
    void *saved_meandevPtr, 
    void *saved_inv_vardevPtr, 
    void *peer_devPtr, 
    void *dxDevPtr, 
    void *dscaledevPtr, 
    void *dbiasdevPtr, 

    double epsilon_val)

{
    cudnnHandle_t handle_;
    std::cout << "================ Running Batch Norm Backward =======================" << std::endl;
    try {
        // Create the cudnn handle
        checkCudnnErr(cudnnCreate(&handle_));

        // Creates the necessary tensor descriptors
        int64_t tensor_stride[4];
        int64_t stride[4];
        int64_t peer_stride[4];

        generateStrides(tensorDims, tensor_stride, 4, CUDNN_TENSOR_NHWC);
        generateStrides(peerDims, peer_stride, 4, CUDNN_TENSOR_NHWC);

        auto tensor_create = [&tensor_stride, &tensorDims](cudnnDataType_t type,
                                int64_t id) {
            return cudnn_frontend::TensorBuilder()
                   .setDim(4, tensorDims)
                   .setStrides(4, tensor_stride)
                   .setId(id)
                   .setAlignment(16)
                   .setDataType(type)
                   .build();
        };

        auto peer_tensor_create = [&peer_stride, &tensorDims](cudnnDataType_t type,
                                int64_t id) {
            return cudnn_frontend::TensorBuilder()
                   .setDim(4, tensorDims)
                   .setStrides(4, peer_stride)
                   .setId(id)
                   .setAlignment(16)
                   .setDataType(type)
                   .build();
        };

        generateStrides(perChannelSum, stride, 4, CUDNN_TENSOR_NHWC);

        auto per_channel_tensor_create = [&stride, &perChannelSum](cudnnDataType_t type,
                                int64_t id) {
            return cudnn_frontend::TensorBuilder()
                   .setDim(4, perChannelSum)
                   .setStrides(4, stride)
                   .setId(id)
                   .setAlignment(16)
                   .setDataType(type)
                   .build();
        };

        auto xTensor             = tensor_create(CUDNN_DATA_HALF, 100);
        auto dyTensor            = tensor_create(CUDNN_DATA_HALF, 101);
        auto scaleTensor         = per_channel_tensor_create(CUDNN_DATA_FLOAT, 102);
        auto savedMeanTensor     = per_channel_tensor_create(CUDNN_DATA_FLOAT, 103);
        auto savedInvVarTensor   = per_channel_tensor_create(CUDNN_DATA_FLOAT, 104);
        auto peerStatTensor      = peer_tensor_create(CUDNN_DATA_FLOAT, 105);
        auto dxTensor            = tensor_create(CUDNN_DATA_HALF, 200);
        auto dScaleTensor        = per_channel_tensor_create(CUDNN_DATA_FLOAT, 201);
        auto dBiasTensor         = per_channel_tensor_create(CUDNN_DATA_FLOAT, 202);


        int64_t epsilon_stride[4];
        generateStrides(epsilon, epsilon_stride, 4, CUDNN_TENSOR_NHWC);
        auto scalar_tensor_create = [&epsilon_stride, &epsilon](cudnnDataType_t type,
                                int64_t id) {
            return cudnn_frontend::TensorBuilder()
                   .setDim(4, epsilon)
                   .setStrides(4, epsilon_stride)
                   .setId(id)
                   .setAlignment(16)
                   .setDataType(type)
                   .setByValue(true)
                   .build();
        };

        auto epsilonTensor       = scalar_tensor_create(CUDNN_DATA_DOUBLE, 300);

#if (CUDNN_VERSION >= 8500)
        // Batch normalization
        cudnnBackendNormMode_t normalizationMode = CUDNN_BATCH_NORM;

        //Create a Finalize node
        auto batch_norm_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR)
                    .setNormalizationMode(normalizationMode)
                    .setxDesc(xTensor)
                    .setSavedMeanAndInvVar(savedMeanTensor, savedInvVarTensor)
                    .setdyDesc(dyTensor)
                    .setScale(scaleTensor)
                    .setEpsilonTensor(epsilonTensor)
                    .setDScaleAndDBias(dScaleTensor, dBiasTensor)
                    .setdxDesc(dxTensor)
                    .addPeerStatTensor(peerStatTensor)
                    .build();

        std::array<cudnn_frontend::Operation const*, 1> ops = {&batch_norm_op};
        auto opGraph = cudnn_frontend::OperationGraphBuilder().setHandle(handle_).setOperationGraph(ops.size(), ops.data()).build();
        std::cout << opGraph.describe() << std::endl;

        cudnn_frontend::EngineConfigList filtered_configs;
        auto statuses = 
            cudnn_frontend::get_heuristics_list<2>({"heuristics_instant" 
            , "heuristics_fallback"
            }, opGraph,::AllowAll, filtered_configs, true);
        
        std::cout << "get_heuristics_list Statuses: ";
        for (auto i = 0 ; i < statuses.size(); i++) {
            std::cout << cudnn_frontend::to_string(statuses[i]) << " ";
        }
        std::cout << std::endl;
        std::cout << "Filter config list has " << filtered_configs.size() << " configurations " << std::endl;

        auto plan_builder = [&filtered_configs, &opGraph, &handle_]() {
            for (auto i = 0; i < filtered_configs.size(); i++) {
                try {
                    auto plan = cudnn_frontend::ExecutionPlanBuilder().setHandle(handle_).setEngineConfig(filtered_configs[i], opGraph.getTag()).build();
		            return plan;
                } catch (cudnn_frontend::cudnnException &e) {
                    continue;
                }
            }
            return cudnn_frontend::ExecutionPlanBuilder().setHandle(handle_).setEngineConfig(filtered_configs[0], opGraph.getTag()).build();
        };

	    CHECK(filtered_configs.size() > 0);
        auto plan = plan_builder();
        std::cout << "Plan tag: " << plan.getTag() << std::endl;

        auto workspace_size = plan.getWorkspaceSize();
        std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;

        void* workspace_ptr = nullptr;
        if (workspace_size > 0) {
            checkCudaErr(cudaMalloc(&workspace_ptr, workspace_size));
        }

        constexpr int var_pack_size = 10;
        void* data_ptrs[var_pack_size] = {xDevPtr, dyDevPtr, scaledevPtr,
                               saved_meandevPtr, saved_inv_vardevPtr, peer_devPtr,
                               &epsilon_val,
                               dxDevPtr, dscaledevPtr, dbiasdevPtr};
        int64_t uids[var_pack_size]    = {100, 101, 102, 103, 104, 105, 300, 200, 201, 202};
        auto variantPack  = cudnn_frontend::VariantPackBuilder()
                               .setWorkspacePointer(workspace_ptr)
                               .setDataPointers(var_pack_size, data_ptrs)
                               .setUids(var_pack_size, uids)
                               .build();
        std::cout << "variantPack " << variantPack.describe() << std::endl;
        cudnnStatus_t status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());

        checkCudaErr(cudaDeviceSynchronize());
        if (workspace_size > 0) {
            checkCudaErr(cudaFree(workspace_ptr));
        }

        cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error", status);

        std::cout << "Batch normalization backward run completed successfully" << std::endl;
#endif



    } catch (cudnn_frontend::cudnnException &e) {
        struct cudaDeviceProp prop;
        checkCudaErrors(cudaGetDeviceProperties(&prop, 0));
        if (prop.major == 8) { 
            std::cout << "[ERROR] Exception " << e.what() << std::endl;
            CHECK(false);
        }
    }
}