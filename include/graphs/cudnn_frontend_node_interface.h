#pragma once

#include <memory>
#include <vector>
#include <unordered_map>

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

protected:
    // Type of each node. Nodes can either be a composite (value COMPOSITE) or
    // one of the other primitive types. Primitives types are nothing but
    // cudnn operations.
    enum class Type {
        COMPOSITE
        , BATCHNORM
        , CONVOLUTION
        , MATMUL
        , POINTWISE
        , REDUCTION
        , RESAMPLE
    };
    Type tag;

    virtual Type getType() = 0;
    detail::Context context;

    virtual error_t createTensors() {
        for(auto const& sub_node: sub_nodes) {
            auto status = sub_node.second->createTensors();
            if(status != error_t::OK) {
                getLogger() << "[cudnn_frontend] ERROR: " << status << " Failed to create tensors in " << name << std::endl;
                return status;
            }
        }
        return error_t::OK;
    }

    virtual error_t createOperationGraphs(cudnnHandle_t) = 0;
    virtual error_t createExecutionPlans() = 0;

    virtual error_t createOperations() {
        for(auto const& sub_node: sub_nodes) {
            auto status = sub_node.second->createOperations();
            if(status != error_t::OK) {
                getLogger() << "[cudnn_frontend] ERROR: " << status << " Failed to create operation for " << sub_node.first << " in " << name << std::endl;
                return status;
            }

            // Roll up operations to parent node, so that parent can too partition operation graphs.
            for (auto const &item : sub_node.second->get_operations()) {
                operations.emplace(item.first, item.second);
            }
            for (auto const &item : sub_node.second->tensors_in_operations) {
                tensors_in_operations.emplace(item.first, item.second);
            }
        }
        return error_t::OK;
    }
    
public:
    std::string name;
    int offset = 1;

    // Tensors belonging to each node.
    // Connecting nodes can modify and delete tensors in this container.
    std::unordered_map<std::string, std::shared_ptr<graph::Tensor>> tensor_props;

    INode* parent_node;
    std::unordered_map <std::string, std::shared_ptr<INode>> sub_nodes;
    
    error_t set_intermediate_data_type(DataType_t const type) {
        context.set_intermediate_data_type(type);
        return error_t::OK;
    }

    error_t set_io_data_type(DataType_t const type) {
        context.set_io_data_type(type);
        return error_t::OK;
    }

    error_t set_compute_data_type(DataType_t const type) {
        context.set_compute_data_type(type);
        return error_t::OK;
    }

    virtual error_t infer_properties() {
        for(auto const& sub_node: sub_nodes) {
            auto status = sub_node.second->infer_properties();
            if(status != error_t::OK) {
                getLogger() << "[cudnn_frontend] ERROR: " << status << " Failed to infer properties in " << name << std::endl;
                return status;
            }
        }
        return error_t::OK;
    }

    virtual error_t validate() const {
        for(auto const& sub_node: sub_nodes) {
            auto status = sub_node.second->validate();
            if(status != error_t::OK) {
                getLogger() << "[cudnn_frontend] ERROR: " << status << " Failed to validate in " << name << std::endl;
                return status;
            }
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

        status = createExecutionPlans();
        if(status != error_t::OK) {
            getLogger() << "[cudnn_frontend] ERROR: " << status << " Failed to buildExecutionPlans in " << name << std::endl;
            return status;
        }

        return error_t::OK;
    }
    
    int64_t get_workspace_size() const {
        int64_t current_workspace_size = get_cudnn_workspace_size();
        for(auto const& sub_node: sub_nodes) {
            current_workspace_size += sub_node.second->get_cudnn_workspace_size();
        }
        return current_workspace_size;
    }

    error_t execute(std::unordered_map<std::string, void*> const& tensor_name_to_pointer_map) {
        std::unordered_map<int64_t, void*> tensor_uid_to_pointer_map;
        void* workspace_ptr = nullptr;

        for (auto const &item : tensor_name_to_pointer_map) {
            if(item.first == "workspace") {
                workspace_ptr = item.second;
            }
            else {
                tensor_uid_to_pointer_map.emplace(get_tensor_props(item.first)->get_uid(), item.second);
            }
        }
        
        auto status = execute_cudnn_plans(tensor_uid_to_pointer_map, workspace_ptr);
        if(status != error_t::OK) {
            getLogger() << "[cudnn_frontend] ERROR: " << status << " Execution failed in " << name << std::endl;
            return status;
        }
        
        return status;
    }

    INode(std::string const& name, int64_t const offset) : name(name), offset(offset), parent_node(nullptr) {}

    virtual ~INode() {};

    int insert_tensor(std::string const& name, graph::Tensor& properties) {
        tensor_props.emplace(name, std::make_shared<graph::Tensor>(properties));
        return 0;
    }


    std::shared_ptr<graph::Tensor> get_tensor_props(std::string const& name) const {
        if(tensor_props.count(name)) {
            return tensor_props.at(name);
        }
        if(parent_node == nullptr) {
            return nullptr;
        }
        // This optimization is not required right now.
        // And without it, this function can be qualified as const which helps during development.
        // tensor_props[name] = parent_node->get_tensor_props(name);
        return parent_node->get_tensor_props(name);
    }

    void fill_missing_context() {
        // If no parent_node, there is no context to fill missing properties with.
        if(parent_node == nullptr)
            return;

        parent_node->fill_missing_context();
        context.fill_missing_properties(parent_node->context);
    }
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

class CompositeNode : public INode {

protected:
    std::vector<std::string> operation_names;

    Type
    getType() override {
        return Type::COMPOSITE;
    }

    error_t createOperationGraphs(cudnnHandle_t handle) override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Partitioning CompositeNode..." << std::endl;

        // Currently just make one large graph of operations from all sub nodes.
        for (auto node : sub_nodes) {
            getLogger() << "Getting the operation from " << node.first << std::endl;
            for (auto &operation : node.second->get_operations()) {
                operation_names.push_back(operation.first);
            }
        }

        getLogger() << "Operation Graph has " << operation_names.size() << " operations." << std::endl;

        auto status = create_cudnn_operation_graphs({operation_names}, handle);
        if(status != error_t::OK) {
            getLogger() << "[cudnn_frontend] ERROR: " << status << " Failed to create execution plans for graph partitioning in CompositeNode." << std::endl;
            return status;
        }

        getLogger() << "[cudnn_frontend] INFO: Partitioned CompositeNode." << std::endl;
        return error_t::OK;
    }

    error_t createExecutionPlans() override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Creating Execution Plans..." << std::endl;

        // auto status = create_cudnn_execution_plan();
        // if(status != error_t::OK) {
        //     getLogger() << "[cudnn_frontend] ERROR: " << status << " Failed to create execution plans for graph partitioning in CompositeNode." << std::endl;
        //     return status;
        // }

        getLogger() << "[cudnn_frontend] INFO: Created Execution Plans." << std::endl;
        return error_t::OK;
    }

public:
    CompositeNode(std::string const& name, int64_t const offset) : INode(name, offset) {}

    ~CompositeNode() {};

    error_t
    set_executor(Execution_plan_list const &plan_list) {
        execution_plans.emplace_back(plan_list.get_candidate());
        return error_t::OK;
    }

    error_t
    get_engine_configs(HeurMode_t mode, Execution_plan_list &plan_list) {
        getLogger() << "[cudnn_frontend] INFO: Extracting engine configs." << std::endl;

        switch (mode) {
        case HeurMode_t::HEUR_MODE_A:
            if(mode_a_engine_configs.size() == 0){return error_t::HEURISTIC_QUERY_FAILED;}
            plan_list.set_tag(mode_a_engine_configs.begin()->first);
            plan_list.set_engine_configs(mode_a_engine_configs.begin()->second);
            break;
        case HeurMode_t::HEUR_MODE_B:
            if(mode_b_engine_configs.size() == 0){return error_t::HEURISTIC_QUERY_FAILED;}
            plan_list.set_tag(mode_b_engine_configs.begin()->first);
            plan_list.set_engine_configs(mode_b_engine_configs.begin()->second);
            break;
        case HeurMode_t::HEUR_MODE_FALLBACK:
            if(fallback_engine_configs.size() == 0){return error_t::HEURISTIC_QUERY_FAILED;}
            plan_list.set_tag(fallback_engine_configs.begin()->first);
            plan_list.set_engine_configs(fallback_engine_configs.begin()->second);
            break;
        }

        getLogger() << "[cudnn_frontend] INFO: Querying engine config properties." << std::endl;
        plan_list.query_properties();

        return error_t::OK;
    }

};

} // namespace graph

} // namespace cudnn_frontend