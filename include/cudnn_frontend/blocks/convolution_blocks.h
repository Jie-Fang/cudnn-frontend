#pragma once

#include <cudnn_frontend_ConvDesc.h>
#include <cudnn_frontend_Heuristics.h>
#include <cudnn_frontend_Logging.h>

#include <cudnn_frontend/blocks/helpers.h>
#include <cudnn_frontend/blocks/iblock.h>

namespace cudnn_frontend {

class ConvolutionBlock : public IBlock {
private:

protected:

public:
    convolution_properties params;

    ConvolutionBlock(convolution_properties const& input_params) {
        params = input_params;
    }

    Type getType() override final {
        return Type::CONVOLUTION;
    }

    int validate() override final {

        cudnn_frontend::generateStrides(params.input_dim, params.input_stride, params.dim_count, CUDNN_TENSOR_NHWC);
        cudnn_frontend::generateStrides(params.weight_dim, params.weight_stride, params.dim_count, CUDNN_TENSOR_NHWC);
        cudnn_frontend::generateStrides(params.output_dim, params.output_stride, params.dim_count, CUDNN_TENSOR_NHWC);

        return 0;
    }

    int createTensors() override final {
        
        getLogger() << "[cudnn_frontend] INFO: " << "Building ConvolutionBlock tensors..." << std::endl;

        auto input  = cudnn_frontend::TensorBuilder()
                        .setDim(params.dim_count, params.input_dim)
                        .setStrides(params.dim_count, params.input_stride)
                        .setId(params.uids_with_offset[convolution_properties::UIDs::INPUT_UID])
                        .setAlignment(16)
                        .setDataType(params.tensor_data_type)
                        .setVirtual(false)
                        .setByValue(false)
                        .build();
        tensors.emplace("input", std::make_shared<Tensor>(std::move(input)));

        auto weight = cudnn_frontend::TensorBuilder()
                        .setDim(params.dim_count, params.weight_dim)
                        .setStrides(params.dim_count, params.weight_stride)
                        .setId(params.uids_with_offset[convolution_properties::UIDs::WEIGHT_UID])
                        .setAlignment(16)
                        .setDataType(params.tensor_data_type)
                        .setVirtual(false)
                        .setByValue(false)
                        .build();
        tensors.emplace("weight", std::make_shared<Tensor>(std::move(weight)));

        auto output = cudnn_frontend::TensorBuilder()
                        .setDim(params.dim_count, params.output_dim)
                        .setStrides(params.dim_count, params.output_stride)
                        .setId(params.uids_with_offset[convolution_properties::UIDs::OUTPUT_UID])
                        .setAlignment(16)
                        .setDataType(params.tensor_data_type)
                        .setVirtual(false)
                        .setByValue(false)
                        .build();
        tensors.emplace("output", std::make_shared<Tensor>(std::move(output)));

        getLogger() << "[cudnn_frontend] INFO: " << "Built ConvolutionBlock tensors." << std::endl;

        return 0;
    }
    
    int createDescritpors() override final {
        return 0;
    }

    int createOperations() override final {

        getLogger() << "[cudnn_frontend] INFO: " << "Building ConvolutionBlock operations..." << std::endl;
        
        #ifndef NV_CUDNN_DISABLE_EXCEPTION
        try {
        #endif

        // convolution descriptor
        int64_t const tensor_dim_count = params.dim_count;
        int64_t const spatial_dim_count = tensor_dim_count - 2;
        int64_t const* conv_stride = params.stride;
        int64_t const* conv_padding = params.padding;
        int64_t const* conv_dilation = params.dilation;

        auto convolution_descriptor = cudnn_frontend::ConvDescBuilder()
                                                        .setComputeType(params.compute_type)
                                                        .setMathMode(CUDNN_CROSS_CORRELATION)
                                                        .setSpatialDimCount(spatial_dim_count)
                                                        .setSpatialStride(spatial_dim_count, conv_stride)
                                                        .setPrePadding(spatial_dim_count, conv_padding)
                                                        .setPostPadding(spatial_dim_count, conv_padding)
                                                        .setDilation(spatial_dim_count, conv_dilation)
                                                        .build();

        // Create the convolution operation.
        auto convolution_operation = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                                        .setxDesc(*(tensors.at("input")))
                                        .setwDesc(*(tensors.at("weight")))
                                        .setyDesc(*(tensors.at("output")))
                                        .setcDesc(convolution_descriptor)
                                        .setAlpha(1.f)
                                        .setBeta(0.f)
                                        .build();
        
    
        operations.emplace("conv", std::make_shared<Operation>(std::move(convolution_operation)));

        getLogger() << "[cudnn_frontend] INFO: " << "Built ConvolutionBlock operation." << std::endl;

        #ifndef NV_CUDNN_DISABLE_EXCEPTION
        } catch (cudnn_frontend::cudnnException &e) {
            throw cudnnException(e.what(), e.getCudnnStatus());
        }
        #endif
        
        return 0;
    }

    int partition(cudnnHandle_t& handle) override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Partioning ConvolutionBlock..." << std::endl;

        std::vector<Operation const*> operation_graph = {operations["conv"].get()};
        auto convolution_graph = cudnn_frontend::OperationGraphBuilder().setHandle(handle).setOperationGraph(operation_graph.size(), operation_graph.data()).build();
        operation_graphs.emplace("conv_graph", std::make_shared<OperationGraph>(std::move(convolution_graph)));

        getLogger() << "[cudnn_frontend] INFO: " << "Partitioned ConvolutionBlock." << std::endl;
        return 0;
    }

    int createExecutionPlan(cudnnHandle_t& handle) override final {
        getLogger() << "[cudnn_frontend] INFO: " << "ConvolutionBlock getting plan from heuristics..." << std::endl;

        cudnn_frontend::EngineConfigList filtered_configs;
        auto statuses = 
            cudnn_frontend::get_heuristics_list<2>({"heuristics_instant", "heuristics_fallback"}, *(operation_graphs["conv_graph"]), allowAllConfig, filtered_configs, true);
        
        getLogger() << "[cudnn_frontend] INFO: " << "get_heuristics_list statuses: ";
        for (size_t i = 0 ; i < statuses.size(); i++) {
            getLogger() << cudnn_frontend::to_string(statuses[i]) << " ";
        }
        getLogger() << std::endl;

        getLogger() << "[cudnn_frontend] INFO: " << "Filter config list has " << filtered_configs.size() << " configurations." << std::endl;

        for (size_t i = 0; i < filtered_configs.size(); i++) {
            getLogger() << "[cudnn_frontend] INFO: " << "Trying config: " << i << std::endl;

            #ifndef NV_CUDNN_DISABLE_EXCEPTION
            try {
            #endif

            auto plan = cudnn_frontend::ExecutionPlanBuilder()
                            .setHandle(handle)
                            .setEngineConfig(filtered_configs[i], operation_graphs["conv_graph"]->getTag())
                            .build();

            if (plan.get_status() != CUDNN_STATUS_SUCCESS) {
                getLogger() << "[cudnn_frontend] ERROR: " << "Config " << i << " failed with " << plan.get_error() << std::endl;
                continue; 
            }

            getLogger() << "[cudnn_frontend] INFO: " << "Config " << i << " succeeded! Plan has built!" << std::endl;
            getLogger() << "[cudnn_frontend] INFO: " << plan.describe() << std::endl;
            
            execution_plans.emplace("conv_plan", std::make_shared<ExecutionPlan>(std::move(plan)));
            return 0;

            #ifndef NV_CUDNN_DISABLE_EXCEPTION
            } catch (cudnn_frontend::cudnnException &e) {
                // The last config didn't work (E.g. all configs didn't work)
                if (i == filtered_configs.size() - 1) {
                    throw cudnnException(e.what(), e.getCudnnStatus());
                }
                continue;
            }
            #endif
        }

        return 1;
    }

    int build(cudnnHandle_t& handle) override final {

        validate();
        createTensors();
        createDescritpors();
        createOperations();
        partition(handle);
        createExecutionPlan(handle);

        return 0;
    }
    
    int execute(cudnnHandle_t& handle) override final {
        (void)handle;
        return 0;
    } 
};

} // namespace cudnn_frontend