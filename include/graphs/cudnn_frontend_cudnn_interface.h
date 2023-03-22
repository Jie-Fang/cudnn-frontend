#pragma once

#include <memory>
#include <vector>
#include <unordered_map>

#include "cudnn_frontend_Tensor.h"
#include "cudnn_frontend_Operation.h"
#include "cudnn_frontend_OperationGraph.h"
#include "cudnn_frontend_ExecutionPlan.h"
#include "cudnn_frontend_VariantPack.h"

#include "graphs/cudnn_frontend_graph_properties.h"

namespace cudnn_frontend {

class ICudnn {
private:

protected:
    std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>> tensors;
    std::unordered_map<std::string, std::shared_ptr<cudnn_frontend::Operation>> operations;
    std::unordered_map<std::string, std::vector<int64_t>> tensors_in_operations;

    std::vector<std::unique_ptr<ExecutionPlan>> execution_plans;
    std::vector<std::vector<int64_t>> variant_pack_uids;

    error_t create_cudnn_tensor(std::shared_ptr<tensor_properties const> const& props) {

        auto const& dim = props->get_dim();
        getLogger() << "[cudnn_frontend] INFO: Tensor dims are ";
        for(auto sz: dim) getLogger() << sz << " ";

        auto tensor = cudnn_frontend::TensorBuilder()
                        .setDim(props->get_stride().size(), props->get_dim().data())
                        .setStrides(props->get_stride().size(), props->get_stride().data())
                        .setId(props->get_uid())
                        .setAlignment(16)
                        .setDataType(props->get_data_type())
                        .setVirtual(props->get_is_virtual())
                        .setByValue(props->get_is_pass_by_value())
                        .build();
        tensors.emplace(props->get_uid(), std::make_shared<Tensor>(std::move(tensor)));
        
        return error_t::OK;
    }

    error_t create_cudnn_execution_plan(std::vector<std::vector<std::string>> const& sub_graphs) {
        cudnnHandle_t handle;
        cudnnCreate(&handle);

        for(auto const& sub_graph: sub_graphs) {
            std::vector<Operation const*> cudnn_operations;
            for(auto const& operation_name: sub_graph) {
                cudnn_operations.push_back(operations.at(operation_name).get());
            }
            auto cudnn_operation_graph = cudnn_frontend::OperationGraphBuilder().setHandle(handle).setOperationGraph(cudnn_operations.size(), cudnn_operations.data()).build();

            getLogger() << "[cudnn_frontend] INFO: " << " Getting plan from heuristics for " << cudnn_operation_graph.getTag() << " ..." << std::endl;

            cudnn_frontend::EngineConfigList filtered_configs;
            auto statuses = 
                cudnn_frontend::get_heuristics_list<2>({"heuristics_instant", "heuristics_fallback"}, cudnn_operation_graph, allowAllConfig, filtered_configs, true);
            
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
                                .setEngineConfig(filtered_configs[i], cudnn_operation_graph.getTag())
                                .build();
                if (plan.get_status() != CUDNN_STATUS_SUCCESS) {
                    getLogger() << "[cudnn_frontend] ERROR: " << "Config " << i << " failed with " << plan.get_error() << std::endl;
                    // If last config, return error
                    // or else continue to the next config
                    if (i == filtered_configs.size() - 1) {
                        return error_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED;
                    }
                    continue;
                }
                getLogger() << "[cudnn_frontend] INFO: " << "Config " << i << " succeeded! Plan has built!" << std::endl;
                getLogger() << "[cudnn_frontend] INFO: " << plan.describe() << std::endl;
                
                execution_plans.push_back(std::make_unique<ExecutionPlan>(std::move(plan)));
                getLogger() << "[cudnn_frontend] INFO: " << " Successfully built plan." << std::endl;

                // Getting here means plan successfully built
                // move onto next operation graph
                break;
                
                #ifndef NV_CUDNN_DISABLE_EXCEPTION
                } catch (cudnn_frontend::cudnnException &e) {
                    // The last config didn't work (E.g. all configs didn't work)
                    getLogger() << "[cudnn_frontend] ERROR: " << "Config " << i << " failed with " << e.getCudnnStatus() << " " << e.what() << std::endl;
                    if (i == filtered_configs.size() - 1) {
                        return error_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED;
                    }
                    continue;
                }
                #endif
            }
            
            // Push variant pack tensors required for this operation graph
            std::vector<int64_t> variant_pack_for_operation_graph = {}; 
            for(auto const& operation_name: sub_graph) {
                auto const& temp = tensors_in_operations.at(operation_name);
                variant_pack_for_operation_graph.insert(std::end(variant_pack_for_operation_graph), std::begin(temp), std::end(temp));
            }
            variant_pack_uids.emplace_back(variant_pack_for_operation_graph);
        }
        
        cudnnDestroy(handle);

	    return error_t::OK;
    }

public:
    std::unordered_map<std::string, std::shared_ptr<Operation>> const &
    get_operations() {
        return operations;
    }
    
    error_t get_workspace_size(int64_t& workspace_size) const {
        int64_t current_workspace_size = 0;
        for(auto const& execution_plan: execution_plans) {
            current_workspace_size += execution_plan->getWorkspaceSize();
        }
        workspace_size = current_workspace_size;
        return error_t::OK;
    }

    error_t execute_cudnn_plans(std::unordered_map<int64_t, void*> const& tensor_uid_to_pointer_map, void * workspace_ptr) {
        cudnnHandle_t handle;
        cudnnCreate(&handle);

        for(size_t i = 0; i < execution_plans.size(); ++i) {
            auto const& execution_plan = execution_plans[i];
            auto const& variant_pack_uid = variant_pack_uids[i];

            getLogger() << "[cudnn_frontend] INFO: Executing " << execution_plan->getTag() << "..." << std::endl;

            std::vector<void *> device_ptrs;
            for(auto const& uid: variant_pack_uid) {
                device_ptrs.push_back(tensor_uid_to_pointer_map.at(uid));
            }
            auto variant_pack = VariantPackBuilder()
                                .setDataPointers(device_ptrs.size(), device_ptrs.data())
                                .setUids(variant_pack_uid.size(), variant_pack_uid.data())
                                .setWorkspacePointer(workspace_ptr)
                                .build();
            if (variant_pack.get_status() != CUDNN_STATUS_SUCCESS) {
                getLogger() << "[cudnn_frontend] ERROR: Variant pack creation failed with " << variant_pack.get_error() << std::endl;
                return error_t::INVALID_VARIANT_PACK; 
            }
            getLogger() << "[cudnn_frontend] INFO: Built variant pack for " << execution_plan->getTag() << "..." << std::endl;

            auto status = cudnnBackendExecute(handle, execution_plan->get_raw_desc(), variant_pack.get_raw_desc());
            if (status != CUDNN_STATUS_SUCCESS) {
                getLogger() << "[cudnn_frontend] ERROR: Graph execution failed." << std::endl;
                return error_t::GRAPH_EXECUTION_FAILED; 
            }
            getLogger() << "[cudnn_frontend] INFO: Executed " << execution_plan->getTag() << "." << std::endl;
        }

        cudnnDestroy(handle);

        return error_t::OK;
    }

};

} // namespace cudnn_frontend
