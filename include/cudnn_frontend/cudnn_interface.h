#pragma once

#include <memory>
#include <vector>
#include <unordered_map>

#include "../cudnn_frontend_Tensor.h"
#include "../cudnn_frontend_Operation.h"
#include "../cudnn_frontend_OperationGraph.h"
#include "../cudnn_frontend_EngineConfig.h"
#include "../cudnn_frontend_ExecutionPlan.h"
#include "../cudnn_frontend_VariantPack.h"

#include "graph_properties.h"
#include "graph_helpers.h"
#include "plans.h"

namespace cudnn_frontend {

namespace detail {
inline error_t
query_cudnn_heuristics_impl(std::shared_ptr<OperationGraph_v8> const& operation_graph,
                            cudnn_frontend::EngineConfigList& configs,
                            std::vector<HeurMode_t> const& modes) {
    auto const& operation_graph_tag = operation_graph->getTag();
    getLogger() << "[cudnn_frontend] INFO: "
                << " Getting plan from heuristics for " << operation_graph_tag << " ..." << std::endl;

    auto statuses = cudnn_frontend::get_heuristics_list(modes, *operation_graph, allowAllConfig, configs, true);

    getLogger() << "[cudnn_frontend] INFO: get_heuristics_list statuses: ";
    for (size_t i = 0; i < statuses.size(); i++) {
        getLogger() << cudnn_frontend::to_string(statuses[i]) << " ";
    }
    getLogger() << std::endl;

    getLogger() << "[cudnn_frontend] INFO: config list has " << configs.size() << " configurations." << std::endl;

    if (configs.empty()) {
        getLogger() << "[cudnn_frontend] ERROR: No valid engine configs returned from heuristics.";
        return {error_code_t::HEURISTIC_QUERY_FAILED, "No valid engine configs for " + operation_graph_tag};
    }
    return {error_code_t::OK, ""};
}

inline error_t
query_heuristics(std::vector<std::shared_ptr<OperationGraph_v8>> const& operation_graphs,
                 std::unordered_map<std::string, EngineConfigList>& op_graph_to_configs,
                 std::vector<HeurMode_t> const& modes) {
    for (auto const& operation_graph : operation_graphs) {
        cudnn_frontend::EngineConfigList configs;
        CHECK_CUDNN_FRONTEND_ERROR(detail::query_cudnn_heuristics_impl(operation_graph, configs, modes));
        
        cudnn_frontend::EngineConfigList good_configs;

        for (auto &engine_config : configs) {
            int64_t elem_count = 0;
            ManagedOpaqueDescriptor extractedEngine   = make_shared_backend_pointer(CUDNN_BACKEND_ENGINE_DESCRIPTOR);
            cudnnBackendDescriptor_t extractedEngine_ = extractedEngine->get_backend_descriptor();
            auto status = cudnnBackendGetAttribute(engine_config->get_backend_descriptor(),
                                                   CUDNN_ATTR_ENGINECFG_ENGINE,
                                                   CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                   1,
                                                   &elem_count,
                                                   &extractedEngine_);
            if(status == CUDNN_STATUS_SUCCESS) {
                good_configs.push_back(engine_config);
            }
        }

        getLogger() << "[cudnn_frontend] INFO: config list has " << good_configs.size() << " good configurations." << std::endl;
        op_graph_to_configs.emplace(operation_graph->getTag(), good_configs);
    }
    return {error_code_t::OK, ""};
}

inline error_t
create_cudnn_execution_plan(std::shared_ptr<ExecutionPlan>& plan,
                            ManagedOpaqueDescriptor& config,
                            std::string const& operation_graph_tag,
                            cudnnHandle_t handle) {
#ifndef NV_CUDNN_DISABLE_EXCEPTION
    try {
#endif
        auto built_plan = cudnn_frontend::ExecutionPlanBuilder()
                              .setHandle(handle)
                              .setEngineConfig(config, operation_graph_tag)
                              .build();
        if (built_plan.get_status() != CUDNN_STATUS_SUCCESS) {
            getLogger() << "[cudnn_frontend] ERROR: "
                        << "Config failed with " << built_plan.get_error() << std::endl;
            return {error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED, "Couldn't build plan from Config."};
        }

        getLogger() << "[cudnn_frontend] INFO: Config succeeded! Plan has built!\n";
        getLogger() << "[cudnn_frontend] INFO: " << built_plan.describe() << std::endl;
        plan = std::make_shared<ExecutionPlan>(std::move(built_plan));

#ifndef NV_CUDNN_DISABLE_EXCEPTION
    } catch (cudnn_frontend::cudnnException& e) {
        getLogger() << "[cudnn_frontend] ERROR: "
                    << "Config failed with " << e.getCudnnStatus() << " " << e.what() << std::endl;
        return {error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED, "Couldn't build plan from Config."};
    }
#endif

    return {error_code_t::OK, ""};
}

}  // namespace detail

class Execution_plan_list {
    std::string operation_tag;
    EngineConfigList engine_configs;
    std::vector<std::vector<cudnnBackendNumericalNote_t>> numeric_notes;
    std::vector<std::vector<cudnnBackendNumericalNote_t>> behavior_notes;

    std::vector<bool> filtered_indices;
    int64_t max_workspace_allowed = std::numeric_limits<int64_t>::max();

   public:
    std::vector<std::shared_ptr<ExecutionPlan>> execution_plans;

    void
    set_tag(std::string const& tag) {
        operation_tag = tag;
    }
    void
    set_engine_configs(EngineConfigList list) {
        engine_configs = list;
    }

    std::vector<std::shared_ptr<ExecutionPlan>>&
    get_execution_plans() {
        return execution_plans;
    }

    error_t
    query_properties() {
        numeric_notes.reserve(engine_configs.size());
        behavior_notes.reserve(engine_configs.size());
        filtered_indices.resize(engine_configs.size());
        for (auto& engine_config : engine_configs) {
            int64_t elem_count = 0;
            std::vector<cudnnBackendNumericalNote_t> numerics;
            std::vector<cudnnBackendNumericalNote_t> behavior;

            ManagedOpaqueDescriptor extractedEngine   = make_shared_backend_pointer(CUDNN_BACKEND_ENGINE_DESCRIPTOR);
            cudnnBackendDescriptor_t extractedEngine_ = extractedEngine->get_backend_descriptor();
            auto status = cudnnBackendGetAttribute(engine_config->get_backend_descriptor(),
                                                   CUDNN_ATTR_ENGINECFG_ENGINE,
                                                   CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                   1,
                                                   &elem_count,
                                                   &extractedEngine_);
            RETURN_CUDNN_FRONTEND_ERROR_IF((status != CUDNN_STATUS_SUCCESS), error_code_t::HEURISTIC_QUERY_FAILED, "Heuristic query Engine failed.");

            status = cudnnBackendGetAttribute(extractedEngine_,
                                              CUDNN_ATTR_ENGINE_NUMERICAL_NOTE,
                                              CUDNN_TYPE_NUMERICAL_NOTE,
                                              CUDNN_NUMERICAL_NOTE_TYPE_COUNT,
                                              &elem_count,
                                              nullptr);
            RETURN_CUDNN_FRONTEND_ERROR_IF((status != CUDNN_STATUS_SUCCESS), error_code_t::HEURISTIC_QUERY_FAILED, "Heuristic query Numerical Note failed");
            
            numerics.resize(static_cast<size_t>(elem_count));
            status = cudnnBackendGetAttribute(extractedEngine_,
                                              CUDNN_ATTR_ENGINE_NUMERICAL_NOTE,
                                              CUDNN_TYPE_NUMERICAL_NOTE,
                                              CUDNN_NUMERICAL_NOTE_TYPE_COUNT,
                                              &elem_count,
                                              numerics.data());
            RETURN_CUDNN_FRONTEND_ERROR_IF((status != CUDNN_STATUS_SUCCESS), error_code_t::HEURISTIC_QUERY_FAILED, "Heuristic query Numerical Note failed");
            status = cudnnBackendGetAttribute(extractedEngine_,
                                              CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE,
                                              CUDNN_TYPE_BEHAVIOR_NOTE,
                                              CUDNN_BEHAVIOR_NOTE_TYPE_COUNT,
                                              &elem_count,
                                              nullptr);
            RETURN_CUDNN_FRONTEND_ERROR_IF((status != CUDNN_STATUS_SUCCESS), error_code_t::HEURISTIC_QUERY_FAILED, "Heuristic query Behavior Note failed");

            behavior.resize(static_cast<size_t>(elem_count));
            status = cudnnBackendGetAttribute(extractedEngine_,
                                              CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE,
                                              CUDNN_TYPE_BEHAVIOR_NOTE,
                                              CUDNN_BEHAVIOR_NOTE_TYPE_COUNT,
                                              &elem_count,
                                              behavior.data());
            RETURN_CUDNN_FRONTEND_ERROR_IF((status != CUDNN_STATUS_SUCCESS), error_code_t::HEURISTIC_QUERY_FAILED, "Heuristic query Behavior Note failed");
            numeric_notes.emplace_back(numerics);
            behavior_notes.emplace_back(behavior);
        }
        return {error_code_t::OK, ""};
    }

    error_t
    filter_out_numeric_notes(std::vector<cudnnBackendNumericalNote_t> const& notes) {
        for (auto note : notes) {
            for (auto i = 0u; i < engine_configs.size(); i++) {
                if (std::find(numeric_notes[i].begin(), numeric_notes[i].end(), note) != numeric_notes[i].end()) {
                    filtered_indices[i] = true;
                }
            }
        }
        return {error_code_t::OK, ""};
    }

    error_t
    filter_out_behavior_notes(std::vector<cudnnBackendBehaviorNote_t> const& notes) {
        for (auto note : notes) {
            for (auto i = 0u; i < engine_configs.size(); i++) {
                if (std::find(behavior_notes[i].begin(), behavior_notes[i].end(), note) != behavior_notes[i].end()) {
                    filtered_indices[i] = true;
                }
            }
        }
        return {error_code_t::OK, ""};
    }

    void
    set_max_workspace_allowed(int64_t const workspace_allowed) {
        max_workspace_allowed = workspace_allowed;
    }

    EngineConfigList
    get_filtered_engine_configs() {
        EngineConfigList filtered_engine_configs;
        getLogger() << "[cudnn_frontend] INFO: "
                    << " Filtering engine_configs ..." << engine_configs.size() << std::endl;
        for (auto i = 0u; i < engine_configs.size(); i++) {
            if (filtered_indices[i] == false) {
                filtered_engine_configs.push_back(engine_configs[i]);
            }
        }
        getLogger() << "[cudnn_frontend] INFO: "
                    << " Filtered engine_configs ..." << filtered_engine_configs.size() << std::endl;
        return filtered_engine_configs;
    }

    error_t
    check_support(cudnnHandle_t handle) {
        auto const& configs = get_filtered_engine_configs();
        for (auto config : configs) {
            std::shared_ptr<ExecutionPlan> plan;
            auto const& fe_status = detail::create_cudnn_execution_plan(plan, config, operation_tag, handle);

            if (fe_status.is_good() && plan->getWorkspaceSize() <= max_workspace_allowed) {
                execution_plans.push_back(plan);
                return {error_code_t::OK, ""};
            }
        }

        return {error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED,
                "[cudnn_frontend] Error: No execution plans built successfully."};
    }

    error_t
    build_all_plans(cudnnHandle_t handle) {
        auto const& configs = get_filtered_engine_configs();
        for (auto config : configs) {
            std::shared_ptr<ExecutionPlan> plan;
            auto const& fe_status = detail::create_cudnn_execution_plan(plan, config, operation_tag, handle);

            if (fe_status.is_good() && plan->getWorkspaceSize() <= max_workspace_allowed) {
                execution_plans.push_back(plan);
            }
        }

        RETURN_CUDNN_FRONTEND_ERROR_IF(execution_plans.empty(),
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "No execution plans finalized successfully. Hence, not supported.");

        return {error_code_t::OK, ""};
    }

    int64_t
    get_max_workspace_size() {
        int64_t max_size = 0;
        for (auto& plan : execution_plans) {
            max_size = std::max(max_size, plan->getWorkspaceSize());
        }
        return max_size;
    }
};

class ICudnn {
   protected:
    using uid_t = int64_t;

    //// Store tensors and operations as they (probably?) need to be kept alive.
    //
    // The tensor mapping from fe::Tensor to be::Tensor.
    //
    // sub nodes share fe::Tensor. Example, in a conv-bias graph, conv output Y and bias input IN_0 are the same
    // fe::Tensor. But both sub ndoes need to work together to make sure only one be::Tensor is created. Hence this
    // uid_to_backend_tensors acts as the global registry for each sub node to use.
    //
    // Key cannot be fe::Tensor, or shared_ptr<fe::Tensor>, or underlying object address of fe::Tensor.
    // Hence using uid, as that uniquely identifies both types of tensors.
    std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>> uid_to_tensors;
    std::vector<cudnn_frontend::Operation> operations;

    std::vector<std::shared_ptr<OperationGraph_v8>> operation_graphs;
    std::vector<std::unordered_set<uid_t>> variant_pack_uids;

    Execution_plan_list plans;

    // TODO: Always returns OK. Can the status and error message be accessed from tensor descriptor?
    error_t
    create_cudnn_tensor(std::shared_ptr<graph::Tensor_attributes> const& props,
                        int64_t& uid,
                        std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors) {
        // Check whether tensor already created
        // TODO: Do not reply on uid being 0?
        if (props->get_uid() == 0) {
            // Make sure no other tensor somehow already has claimed uid.
            RETURN_CUDNN_FRONTEND_ERROR_IF(tensors.find(uid) != tensors.end(),
                                           error_code_t::ATTRIBUTE_NOT_SET,
                                           "Trying to assign same uid to possibily two different tensors.");
            props->set_uid(uid);
            uid++;

            if (auto ragged_offset_props = props->get_ragged_offset()) {
                CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(ragged_offset_props, uid, tensors));

                auto tensor = cudnn_frontend::TensorBuilder()
                                  .setDim(props->get_dim().size(), props->get_dim().data())
                                  .setStrides(props->get_stride().size(), props->get_stride().data())
                                  .setId(props->get_uid())
                                  .setAlignment(16)
                                  .setDataType(props->get_data_type())
                                  .setVirtual(props->get_is_virtual())
                                  .setByValue(props->get_is_pass_by_value())
                                  .setReorderType(props->get_reordering_type())
                                  .setRaggedOffset(tensors.at(ragged_offset_props->get_uid()))
                                  .build();
                tensors.emplace(props->get_uid(), std::make_shared<Tensor>(std::move(tensor)));
            } else {
                auto tensor = cudnn_frontend::TensorBuilder()
                                  .setDim(props->get_dim().size(), props->get_dim().data())
                                  .setStrides(props->get_stride().size(), props->get_stride().data())
                                  .setId(props->get_uid())
                                  .setAlignment(16)
                                  .setDataType(props->get_data_type())
                                  .setVirtual(props->get_is_virtual())
                                  .setByValue(props->get_is_pass_by_value())
                                  .setReorderType(props->get_reordering_type())
                                  .build();
                tensors.emplace(props->get_uid(), std::make_shared<Tensor>(std::move(tensor)));
            }
        } else {
            // Make sure tensor's uid is present in backend tensor registry.
            RETURN_CUDNN_FRONTEND_ERROR_IF(
                tensors.find(props->get_uid()) == tensors.end(),
                error_code_t::ATTRIBUTE_NOT_SET,
                "Backend tensor already not found for non-zero Id: " + std::to_string(props->get_uid()));

            getLogger() << "[cudnn_frontend] INFO: Backend tensor already created for Id: " +
                               std::to_string(props->get_uid())
                        << std::endl;
        }

        return {error_code_t::OK, ""};
    }

    error_t
    create_cudnn_operation_graphs(cudnnHandle_t handle) {
        std::vector<Operation const*> cudnn_operations;
        for (auto const& operation : operations) {
            cudnn_operations.push_back(&operation);
        }
        auto cudnn_operation_graph = cudnn_frontend::OperationGraphBuilder()
                                         .setHandle(handle)
                                         .setOperationGraph(cudnn_operations.size(), cudnn_operations.data())
                                         .build();

        operation_graphs.push_back(std::make_shared<OperationGraph_v8>(std::move(cudnn_operation_graph)));
        getLogger() << "[cudnn_frontend] INFO: Successfully built Operation Graphs." << std::endl;

        return {error_code_t::OK, ""};
    }

   public:
    int64_t
    get_cudnn_workspace_size_node() const {
        int64_t current_workspace_size = 0;
        for (auto const& execution_plan : plans.execution_plans) {
            current_workspace_size += execution_plan->getWorkspaceSize();
        }
        return current_workspace_size;
    }

    error_t
    execute_cudnn_plans(cudnnHandle_t handle,
                        std::unordered_map<uid_t, void*> const& tensor_uid_to_pointer_map,
                        void* workspace_ptr) {
        getLogger() << "[cudnn_frontend] INFO: Executing " << plans.execution_plans.size() << " Plans." << std::endl;

        for (size_t i = 0; i < plans.execution_plans.size(); ++i) {
            auto const& execution_plan   = plans.execution_plans[i];
            auto const& variant_pack_uid = variant_pack_uids[i];

            getLogger() << "[cudnn_frontend] INFO: Executing " << execution_plan->getTag() << "..." << std::endl;

            std::vector<void*> device_ptrs;
            std::vector<uid_t> uids;
            for (auto const& uid : variant_pack_uid) {
                auto search = tensor_uid_to_pointer_map.find(uid);
                RETURN_CUDNN_FRONTEND_ERROR_IF(search == tensor_uid_to_pointer_map.end(),
                                               error_code_t::INVALID_VARIANT_PACK,
                                               "Uid " + std::to_string(uid) + " does not exist in variant pack.");
                device_ptrs.push_back(tensor_uid_to_pointer_map.at(uid));
                uids.push_back(uid);
            }
            auto variant_pack = VariantPackBuilder()
                                    .setDataPointers(device_ptrs.size(), device_ptrs.data())
                                    .setUids(uids.size(), uids.data())
                                    .setWorkspacePointer(workspace_ptr)
                                    .build();
            if (variant_pack.get_status() != CUDNN_STATUS_SUCCESS) {
                std::string message = "[cudnn_frontend] ERROR: Variant pack creation failed with " +
                                      std::string(variant_pack.get_error());
                return {error_code_t::INVALID_VARIANT_PACK, message};
            }
            getLogger() << "[cudnn_frontend] INFO: Built variant pack for " << execution_plan->getTag() << "..."
                        << std::endl;

            auto status = cudnnBackendExecute(handle, execution_plan->get_raw_desc(), variant_pack.get_raw_desc());
            if (status != CUDNN_STATUS_SUCCESS) {
                std::string message = "[cudnn_frontend] ERROR: Graph execution failed.";
                return {error_code_t::GRAPH_EXECUTION_FAILED, message};
            }
            getLogger() << "[cudnn_frontend] INFO: Executed " << execution_plan->getTag() << "." << std::endl;
        }

        return {error_code_t::OK, ""};
    }
};


}  // namespace cudnn_frontend
