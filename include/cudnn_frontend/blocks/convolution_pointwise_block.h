#pragma once

#include <cudnn_frontend_Heuristics.h>
#include <cudnn_frontend_Logging.h>

#include <cudnn_frontend/blocks/helpers.h>
#include <cudnn_frontend/blocks/iblock.h>
#include <cudnn_frontend/blocks/convolution_blocks.h>
#include <cudnn_frontend/blocks/pointwise_block.h>

namespace cudnn_frontend {

class ConvolutionPointwiseBlock : public IBlock {
private:

protected:

public:
    std::shared_ptr<ConvolutionBlock> conv_block;
    std::shared_ptr<PointwiseBlock> pointwise_block;

    ConvolutionPointwiseBlock(int64_t const& offset = 1) {
        conv_block = std::make_shared<ConvolutionBlock>(offset);
        pointwise_block = std::make_shared<PointwiseBlock>(offset + 200);
    }

    Type getType() override final {
        return Type::BLOCK;
    }

    int validate() override final {
        conv_block->validate();
        pointwise_block->validate();

        conv_block->tensor_props["Y"].is_virtual = true;
        pointwise_block->tensor_props["X"] = conv_block->tensor_props["Y"];

        return 0;
    }

    int createTensors() override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Building ConvolutionPointwiseBlock tensors..." << std::endl;
        conv_block->createTensors();
        pointwise_block->createTensors();
        getLogger() << "[cudnn_frontend] INFO: " << "Built ConvolutionPointwiseBlock tensors." << std::endl;

        return 0;
    }
    
    int createDescritpors() override final {
        return 0;
    }

    int createOperations() override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Building ConvolutionPointwiseBlock operations..." << std::endl;
        conv_block->createOperations();
        pointwise_block->createOperations();
        getLogger() << "[cudnn_frontend] INFO: " << "Built ConvolutionPointwiseBlock operation." << std::endl;

        return 0;
    }

    int partition(cudnnHandle_t& handle) override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Partioning ConvolutionPointwiseBlock..." << std::endl;

        std::vector<Operation const*> operation_graph = {conv_block->operations["conv"].get(), pointwise_block->operations["pointwise"].get()};
        auto conv_pointwise_graph = cudnn_frontend::OperationGraphBuilder().setHandle(handle).setOperationGraph(operation_graph.size(), operation_graph.data()).build();
        operation_graphs.emplace("conv_pointwise_graph", std::make_shared<OperationGraph>(std::move(conv_pointwise_graph)));

        getLogger() << "[cudnn_frontend] INFO: " << "Partitioned ConvolutionPointwiseBlock." << std::endl;
        return 0;
    }

    int createExecutionPlan(cudnnHandle_t& handle) override final {
        getLogger() << "[cudnn_frontend] INFO: " << "ConvolutionPointwiseBlock getting plan from heuristics..." << std::endl;

        cudnn_frontend::EngineConfigList filtered_configs;
        auto statuses = 
            cudnn_frontend::get_heuristics_list<2>({"heuristics_instant", "heuristics_fallback"}, *(operation_graphs["conv_pointwise_graph"]), allowAllConfig, filtered_configs, true);
        
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
                            .setEngineConfig(filtered_configs[i], operation_graphs["conv_pointwise_graph"]->getTag())
                            .build();

            if (plan.get_status() != CUDNN_STATUS_SUCCESS) {
                getLogger() << "[cudnn_frontend] ERROR: " << "Config " << i << " failed with " << plan.get_error() << std::endl;
                continue; 
            }

            getLogger() << "[cudnn_frontend] INFO: " << "Config " << i << " succeeded! Plan has built!" << std::endl;
            getLogger() << "[cudnn_frontend] INFO: " << plan.describe() << std::endl;
            
            execution_plans.emplace("conv_pointwise_plan", std::make_shared<ExecutionPlan>(std::move(plan)));
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