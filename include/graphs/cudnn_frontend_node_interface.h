#pragma once

#include <memory>
#include <vector>
#include <unordered_map>

#include <cuda_fp16.h>
#include <variant>

#include "cudnn_frontend_Tensor.h"
#include "cudnn_frontend_Operation.h"
#include "cudnn_frontend_OperationGraph.h"
#include "cudnn_frontend_ExecutionPlan.h"
#include "cudnn_frontend_VariantPack.h"

#include "graphs/cudnn_frontend_cudnn_interface.h"

#include "graphs/cudnn_frontend_graph_properties.h"

namespace cudnn_frontend {

namespace graph {

// Interface for all nodes to follow.
class INode: public ICudnn {
public:
    // A closed set of types that are allowed to be passed by value today
    using pass_by_values_t = std::variant<half, float>; 

    using allowed_nodes_t = std::variant<

                            >;

    std::string name;
private:
    detail::Context context;

    virtual error_t assignUids_() {
        return error_t::OK;
    };
    
    virtual error_t validate_node() const {
        return error_t::OK;
    };

    error_t assignUids() {
        CHECK_CUDNN_FRONTEND_ERROR(assignUids_());
        for(auto const& sub_node: sub_nodes) {
            auto status = sub_node->assignUids();
            if(status != error_t::OK) {
                getLogger() << "[cudnn_frontend] ERROR: " << status << " Failed to create tensors in " << name << std::endl;
                return status;
            }
        }
        return error_t::OK;
    }

    virtual error_t pass_by_value_tensors_(std::unordered_map<std::shared_ptr<Tensor>, pass_by_values_t>&) {
        return error_t::OK;
    }

    error_t gather_pass_by_value_tensors(std::unordered_map<std::shared_ptr<Tensor>, pass_by_values_t>& tensor_to_pass_by_value) {
        CHECK_CUDNN_FRONTEND_ERROR(pass_by_value_tensors_(tensor_to_pass_by_value));
        for(auto const& sub_node: sub_nodes) {
            auto status = sub_node->gather_pass_by_value_tensors(tensor_to_pass_by_value);
            if(status != error_t::OK) {
                getLogger() << "[cudnn_frontend] ERROR: " << status << " Failed to gather pass by value tensors in " << name << std::endl;
                return status;
            }
        }
        return error_t::OK;
    }  

protected:
    // Type of each node. Nodes can either be a composite (value COMPOSITE) or
    // one of the other primitive types. Primitives types are nothing but
    // cudnn operations.
    enum class Type {
        COMPOSITE
        , BATCHNORM
        , BN_FINALIZE
        , CONVOLUTION
        , DBN_WEIGHT
        , DGRAD
        , GENSTATS
        , MATMUL
        , POINTWISE
        , REDUCTION
        , RESAMPLE
        , RNG
        , SCALED_DOT_PRODUCT_ATTENTION
        , WGRAD
    };
    Type tag;
    
    virtual error_t createTensors() {
        for(auto const& sub_node: sub_nodes) {
            auto status = sub_node->createTensors();
            if(status != error_t::OK) {
                getLogger() << "[cudnn_frontend] ERROR: " << status << " Failed to create tensors in " << name << std::endl;
                return status;
            }
        }
        return error_t::OK;
    }

    virtual error_t createOperationGraphs(cudnnHandle_t) = 0;
    virtual error_t createExecutionPlans(cudnnHandle_t) = 0;

    virtual error_t createOperations() {
        for(auto const& sub_node: sub_nodes) {
            auto status = sub_node->createOperations();
            if(status != error_t::OK) {
                getLogger() << "[cudnn_frontend] ERROR: " << status << " Failed to create operation for " << name << " in " << name << std::endl;
                return status;
            }

            // Roll up operations to parent node, so that parent can too partition operation graphs.
            for (auto const &item : sub_node->get_operations()) {
                operations.emplace(item.first, item.second);
            }
            for (auto const &item : sub_node->tensors_in_operations) {
                tensors_in_operations.emplace(item.first, item.second);
            }
        }
        return error_t::OK;
    }
    
    std::vector<std::unique_ptr<INode>> sub_nodes;
public:
    virtual Type getType() = 0;

    detail::Context& get_context() {
        return context;
    }

    virtual error_t infer_properties() {
        for(auto const& sub_node: sub_nodes) {
            auto status = sub_node->infer_properties();
            if(status != error_t::OK) {
                getLogger() << "[cudnn_frontend] ERROR: " << status << " Failed to infer properties in " << name << std::endl;
                return status;
            }
        }
        return error_t::OK;
    }

    error_t validate() const {
        // First validate sub nodes
        for(auto const& sub_node: sub_nodes) {
            auto status = sub_node->validate();
            if(status != error_t::OK) {
                getLogger() << "[cudnn_frontend] ERROR: " << status << " Failed to validate in " << name << std::endl;
                return status;
            }
        }

        // Now validate self
        auto status = validate_node();
        if(status != error_t::OK) {
            getLogger() << "[cudnn_frontend] ERROR: " << status << " Failed to validate in " << name << std::endl;
            return status;
        }

        return error_t::OK;
    }

    error_t build(cudnnHandle_t handle) {
        auto status = infer_properties();
        if(status != error_t::OK) {
            getLogger() << "[cudnn_frontend] ERROR: " << status << " Failed to build in " << name << std::endl;
            return status;
        }

        status = validate();
        if(status != error_t::OK) {
            getLogger() << "[cudnn_frontend] ERROR: " << status << " Failed to build in " << name << std::endl;
            return status;
        }

        status = assignUids();
        if(status != error_t::OK) {
            getLogger() << "[cudnn_frontend] ERROR: " << status << " Failed to assignUids in " << name << std::endl;
            return status;
        }

        status = createTensors();
        if(status != error_t::OK) {
            getLogger() << "[cudnn_frontend] ERROR: " << status << " Failed to createTensors in " << name << std::endl;
            return status;
        }

        status = createOperations();
        if(status != error_t::OK) {
            getLogger() << "[cudnn_frontend] ERROR: " << status << " Failed to createOperations in " << name << std::endl;
            return status;
        }

        status = createOperationGraphs(handle);
        if(status != error_t::OK) {
            getLogger() << "[cudnn_frontend] ERROR: " << status << " Failed to createOperationGraphs in " << name << std::endl;
            return status;
        }

        status = query_heuristics();
        if(status != error_t::OK) {
            getLogger() << "[cudnn_frontend] ERROR: " << status << " Failed to query Heuristics in " << name << std::endl;
            return status;
        }

        status = createExecutionPlans(handle);
        if(status != error_t::OK) {
            getLogger() << "[cudnn_frontend] ERROR: " << status << " Failed to buildExecutionPlans in " << name << std::endl;
            return status;
        }

        return error_t::OK;
    }
    
    int64_t get_workspace_size() const {
        int64_t current_workspace_size = get_cudnn_workspace_size();
        for(auto const& sub_node: sub_nodes) {
            current_workspace_size += sub_node->get_cudnn_workspace_size();
        }
        return current_workspace_size;
    }

    error_t execute(cudnnHandle_t handle, std::unordered_map<std::shared_ptr<Tensor>, void*> const& tensor_to_pointer_map) {
        std::unordered_map<int64_t, void*> tensor_uid_to_pointer_map;
        void* workspace_ptr = nullptr;

        for (auto const &item : tensor_to_pointer_map) {
            // TODO: worksapce hack. FIX ME!!!
            if(item.first->get_name() == "workspace") {
                workspace_ptr = item.second;
            }
            else {
                tensor_uid_to_pointer_map.emplace(item.first->get_uid(), item.second);
            }
        }

        std::unordered_map<std::shared_ptr<Tensor>, pass_by_values_t> tensor_to_pass_by_value;
        auto status = gather_pass_by_value_tensors(tensor_to_pass_by_value);
        if(status != error_t::OK) {
            getLogger() << "[cudnn_frontend] ERROR: " << status << " Failed to gather_pass_by_value_tensors in " << name << std::endl;
            return status;
        }

        // Add pass_by_value data pointers to tensor_uid_to_pointer map
        // object lifetime is controlled by tensor_to_pass_by_value which means the pointer should stay valid during execute
        for(auto& [tensor, value]: tensor_to_pass_by_value) {
            void* value_ptr = nullptr;
            if((value_ptr = std::get_if<half>(&value))) {
                tensor_uid_to_pointer_map.emplace(tensor->get_uid(), value_ptr);
            }
            else if((value_ptr = std::get_if<float>(&value))) {
                tensor_uid_to_pointer_map.emplace(tensor->get_uid(), value_ptr);
            }
            else {
                status = error_t::INVALID_VARIANT_PACK;
                getLogger() << "[cudnn_frontend] ERROR: " << status << " Unexpected type for pass by value tensor in " << name << std::endl;
                return status;
            }
        }
        
        status = execute_cudnn_plans(handle, tensor_uid_to_pointer_map, workspace_ptr);
        if(status != error_t::OK) {
            getLogger() << "[cudnn_frontend] ERROR: " << status << " Execution failed in " << name << std::endl;
            return status;
        }
        
        return status;
    }

    INode(std::string const& name, detail::Context const& context) : name(name), context(context) {}

    virtual ~INode() {};
};

class Execution_plan_list {

    std::string operation_tag;
    EngineConfigList engine_configs;
    std::vector<std::vector<cudnnBackendNumericalNote_t>> numeric_notes;
    std::vector<std::vector<cudnnBackendNumericalNote_t>> behavior_notes;

    std::vector<std::shared_ptr<ExecutionPlan>>           execution_plans;
    std::shared_ptr<ExecutionPlan>           candidate = nullptr;

    std::vector<bool> filtered_indices;

    public:
    void set_tag(std::string const &tag) {operation_tag = tag;}
    void set_engine_configs(EngineConfigList list) {engine_configs= list;}

    std::shared_ptr<ExecutionPlan> const &
    get_candidate() const {return candidate;}

    error_t query_properties() {
        numeric_notes.reserve(engine_configs.size());
        behavior_notes.reserve(engine_configs.size());
        filtered_indices.resize(engine_configs.size());
        for (auto &engine_config : engine_configs) {
            int64_t elem_count                   = 0;
            std::vector<cudnnBackendNumericalNote_t> numerics;
            std::vector<cudnnBackendNumericalNote_t> behavior;

            ManagedOpaqueDescriptor extractedEngine = make_shared_backend_pointer(CUDNN_BACKEND_ENGINE_DESCRIPTOR);
            cudnnBackendDescriptor_t extractedEngine_ = extractedEngine->get_backend_descriptor();
            auto status = cudnnBackendGetAttribute(engine_config->get_backend_descriptor(),
                                        CUDNN_ATTR_ENGINECFG_ENGINE,
                                        CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                        1,
                                        &elem_count,
                                        &extractedEngine_);
            if (status != CUDNN_STATUS_SUCCESS) {return error_t::HEURISTIC_QUERY_FAILED;}

            status = cudnnBackendGetAttribute(extractedEngine_,
                                CUDNN_ATTR_ENGINE_NUMERICAL_NOTE,
                                CUDNN_TYPE_NUMERICAL_NOTE,
                                CUDNN_NUMERICAL_NOTE_TYPE_COUNT,
                                &elem_count,
                                nullptr);
            if (status != CUDNN_STATUS_SUCCESS) {return error_t::HEURISTIC_QUERY_FAILED;}
            numerics.resize(static_cast<size_t>(elem_count));
            status = cudnnBackendGetAttribute(extractedEngine_,
                                CUDNN_ATTR_ENGINE_NUMERICAL_NOTE,
                                CUDNN_TYPE_NUMERICAL_NOTE,
                                CUDNN_NUMERICAL_NOTE_TYPE_COUNT,
                                &elem_count,
                                numerics.data());
            if (status != CUDNN_STATUS_SUCCESS) {return error_t::HEURISTIC_QUERY_FAILED;}
#if (CUDNN_VERSION >= 8200)
            status = cudnnBackendGetAttribute(extractedEngine_,
                                    CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE,
                                    CUDNN_TYPE_BEHAVIOR_NOTE,
                                    CUDNN_BEHAVIOR_NOTE_TYPE_COUNT,
                                    &elem_count,
                                    nullptr);
            if (status != CUDNN_STATUS_SUCCESS) {return error_t::HEURISTIC_QUERY_FAILED;}
            behavior.resize(static_cast<size_t>(elem_count));
            status = cudnnBackendGetAttribute(extractedEngine_,
                                    CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE,
                                    CUDNN_TYPE_BEHAVIOR_NOTE,
                                    CUDNN_BEHAVIOR_NOTE_TYPE_COUNT,
                                    &elem_count,
                                    behavior.data());
            if (status != CUDNN_STATUS_SUCCESS) {return error_t::HEURISTIC_QUERY_FAILED;}
#endif
            numeric_notes.emplace_back(numerics);
            behavior_notes.emplace_back(behavior);
        }
        return error_t::OK;
    }

    error_t
    filter_by_numeric_notes(std::vector<cudnnBackendNumericalNote_t> const &notes) {
        for (auto note : notes) {
            for (auto i = 0u; i < engine_configs.size(); i++) {
                if (std::find(numeric_notes[i].begin(), numeric_notes[i].end(),note) != numeric_notes[i].end()) {
                    filtered_indices[i] = true;
                }
            }
        }
        return error_t::OK;
    }

    error_t
    filter_by_behavior_notes(std::vector<cudnnBackendBehaviorNote_t> const &notes) {
        for (auto note : notes) {
            for (auto i = 0u; i < engine_configs.size(); i++) {
                if (std::find(behavior_notes[i].begin(), behavior_notes[i].end(),note) != behavior_notes[i].end()) {
                    filtered_indices[i] = true;
                }
            }
        }
        return error_t::OK;
    }

    EngineConfigList
    get_filtered_engine_configs() {
        EngineConfigList filtered_engine_configs;
        getLogger() << "[cudnn_frontend] INFO: " << " Filtering engine_configs ..." << engine_configs.size() << std::endl;
        for (auto i = 0u; i < engine_configs.size(); i++) {
            if (filtered_indices[i] == false) {
                filtered_engine_configs.push_back(engine_configs[i]);
            }
        }
        getLogger() << "[cudnn_frontend] INFO: " << " Filtered engine_configs ..." << filtered_engine_configs.size() << std::endl;
        return filtered_engine_configs;
    }

    error_t
    build_plans(cudnnHandle_t handle) {
        auto configs = get_filtered_engine_configs();
        for (auto &config: configs) {
            #ifndef NV_CUDNN_DISABLE_EXCEPTION
            try {
            #endif
            auto plan = cudnn_frontend::ExecutionPlanBuilder()
                            .setHandle(handle)
                            .setEngineConfig(config, operation_tag)
                            .build();
            if (plan.get_status() != CUDNN_STATUS_SUCCESS) {
                getLogger() << "[cudnn_frontend] ERROR: " << "Config failed with " << plan.get_error() << std::endl;
                continue;
            }
            getLogger() << "[cudnn_frontend] INFO: " << "Config succeeded! Plan has built!" << std::endl;
            getLogger() << "[cudnn_frontend] INFO: " << plan.describe() << std::endl;

            execution_plans.push_back(std::make_shared<ExecutionPlan>(std::move(plan)));
            if (candidate == nullptr) {
                candidate = execution_plans.front();
            }

            #ifndef NV_CUDNN_DISABLE_EXCEPTION
            } catch (cudnn_frontend::cudnnException &e) {
                getLogger() << "[cudnn_frontend] ERROR: " << "Config failed with " << e.getCudnnStatus() << " " << e.what() << std::endl;
                continue;
            }
            #endif
        }

        if (candidate == nullptr) {
            return error_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED;
        }
        return error_t::OK;
    }

    int64_t
    get_max_workspace_size() {
        int64_t max_size = 0;
        for (auto &plan : execution_plans) {
            max_size = std::max(max_size, plan->getWorkspaceSize());
        }
        return max_size;
    }

    int64_t
    get_workspace_size() {
        if (candidate) {
            candidate->getWorkspaceSize();
        } else {
            return 0;
        }
        return 0;
    }

};

} // namespace graph

} // namespace cudnn_frontend