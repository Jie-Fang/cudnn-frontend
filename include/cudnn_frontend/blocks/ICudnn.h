#pragma once

#include <memory>
#include <vector>
#include <unordered_map>

#include <cudnn_frontend_Tensor.h>
#include <cudnn_frontend_Operation.h>
#include <cudnn_frontend_OperationGraph.h>
#include <cudnn_frontend_ExecutionPlan.h>
#include <cudnn_frontend_VariantPack.h>

namespace cudnn_frontend {

class ICudnn {

    friend class ConvolutionPointwiseBlock;

private:

protected:
    std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>> tensors;    
    std::unordered_map<std::string, std::shared_ptr<cudnn_frontend::Operation>> operations;

    std::vector<std::shared_ptr<OperationGraph>> operation_graphs;
    std::vector<std::shared_ptr<ExecutionPlan>> execution_plans;
    std::unordered_map<std::string, std::shared_ptr<VariantPack>> variant_packs;

    virtual int createTensors() = 0;
    
    virtual int createDescritpors() = 0;

    virtual int createOperations() = 0;

    virtual int createExecutionPlan(cudnnHandle_t& handle) {
         
         for(auto const& operation_graph: operation_graphs) {
            getLogger() << "[cudnn_frontend] INFO: " << " Getting plan from heuristics for " << operation_graph->getTag() << " ..." << std::endl;

            cudnn_frontend::EngineConfigList filtered_configs;
            auto statuses = 
                cudnn_frontend::get_heuristics_list<2>({"heuristics_instant", "heuristics_fallback"}, *operation_graph, allowAllConfig, filtered_configs, true);
            
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
                                .setEngineConfig(filtered_configs[i], operation_graph->getTag())
                                .build();

                if (plan.get_status() != CUDNN_STATUS_SUCCESS) {
                    getLogger() << "[cudnn_frontend] ERROR: " << "Config " << i << " failed with " << plan.get_error() << std::endl;
                    return 1; 
                }

                getLogger() << "[cudnn_frontend] INFO: " << "Config " << i << " succeeded! Plan has built!" << std::endl;
                getLogger() << "[cudnn_frontend] INFO: " << plan.describe() << std::endl;
                
                execution_plans.push_back(std::make_shared<ExecutionPlan>(std::move(plan)));
                getLogger() << "[cudnn_frontend] INFO: " << " Successfully built plan." << std::endl;
                return 0;
                #ifndef NV_CUDNN_DISABLE_EXCEPTION
                } catch (cudnn_frontend::cudnnException &e) {
                    // The last config didn't work (E.g. all configs didn't work)
                    if (i == filtered_configs.size() - 1) {
                        throw cudnnException(e.what(), e.getCudnnStatus());
                    }
                    return 1;
                }
                #endif
            }

            getLogger() << "[cudnn_frontend] INFO: " << " Failed to build plan." << std::endl;
        }

        return 0; 
    }
};

} // namespace cudnn_frontend