#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include <variant>
#include <limits>

#include <cuda_fp16.h>

#include "../cudnn_frontend_Tensor.h"
#include "../cudnn_frontend_Operation.h"
#include "../cudnn_frontend_OperationGraph.h"
#include "../cudnn_frontend_ExecutionPlan.h"
#include "../cudnn_frontend_VariantPack.h"

#include "cudnn_interface.h"

#include "graph_properties.h"

namespace cudnn_frontend {

namespace graph {

// Interface for all nodes to follow.
class INode : public ICudnn {
   public:
    // A closed set of types that are allowed to be passed by value today
    using pass_by_values_t = std::variant<half, float, void*>;

    // Stores workspace size in bytes required by FE node
    // It does NOT include cudnn backend workspace
    size_t workspace_size;

    detail::Context context;

   private:
    virtual error_t
    infer_properties_node() {
        return {error_code_t::OK, ""};
    };

    virtual error_t
    validate_node() const {
        return {error_code_t::OK, ""};
    };

    virtual int64_t
    get_fe_workspace_size_node() const {
        // Mostly no FE nodes have require workspace
        return 0;
    }

    int64_t
    get_cudnn_workspace_size() const {
        int64_t cudnn_workspace_size = get_cudnn_workspace_size_node();
        for (auto const& sub_node : sub_nodes) {
            cudnn_workspace_size += sub_node->get_cudnn_workspace_size();
        }
        return cudnn_workspace_size;
    }

    int64_t
    get_fe_workspace_size() const {
        int64_t fe_workspace_size = get_fe_workspace_size_node();
        for (auto const& sub_node : sub_nodes) {
            fe_workspace_size += sub_node->get_fe_workspace_size();
        }
        return fe_workspace_size;
    }

    virtual error_t
    pass_by_value_tensors_(cudnnHandle_t,
                           std::unordered_map<std::shared_ptr<Tensor_attributes>, pass_by_values_t>&,
                           void*) {
        return {error_code_t::OK, ""};
    }

    error_t
    gather_pass_by_value_tensors(
        cudnnHandle_t const& handle,
        std::unordered_map<std::shared_ptr<Tensor_attributes>, pass_by_values_t>& tensor_to_pass_by_value,
        void* fe_workspace) {
        void* node_workspace = fe_workspace;
        CHECK_CUDNN_FRONTEND_ERROR(pass_by_value_tensors_(handle, tensor_to_pass_by_value, node_workspace));
        node_workspace = static_cast<char*>(node_workspace) + get_fe_workspace_size_node();
        for (auto const& sub_node : sub_nodes) {
            CHECK_CUDNN_FRONTEND_ERROR(
                sub_node->gather_pass_by_value_tensors(handle, tensor_to_pass_by_value, node_workspace));
            node_workspace = static_cast<char*>(node_workspace) + sub_node->get_fe_workspace_size_node();
        }
        return {error_code_t::OK, ""};
    }

   protected:
    // Type of each node. Nodes can either be a composite (value COMPOSITE) or
    // one of the other primitive types. Primitives types are nothing but
    // cudnn operations.
    enum class Type {
        COMPOSITE,
        BATCHNORM,
        BATCHNORM_INFERENCE,
        BN_FINALIZE,
        CONVOLUTION,
        DBN,
        DBN_WEIGHT,
        DLN,
        DIN,
        DGRAD,
        DRMSNorm,
        GENSTATS,
        LAYERNORM,
        INSTANCENORM,
        MATMUL,
        POINTWISE,
        REDUCTION,
        RESAMPLE,
        RESHAPE,
        RMSNORM,
        RNG,
        SCALED_DOT_PRODUCT_ATTENTION,
        WGRAD
    };
    Type tag;

    // Creates cudnn tensors for each node (and its sub nodes)
    virtual error_t
    create_cudnn_tensors(int64_t& uid,
                         std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& uid_to_backend_tensors) {
        for (auto const& sub_node : sub_nodes) {
            CHECK_CUDNN_FRONTEND_ERROR(sub_node->create_cudnn_tensors(uid, uid_to_backend_tensors));
        }
        return {error_code_t::OK, ""};
    }

    // Creates cudnn operation for each node (and its sub nodes)
    // Only INode that map to a primitive cudnn operation need to specialize.
    virtual error_t
    create_cudnn_operations(
        std::unordered_set<uid_t>& uids_involved_in_operation,
        std::vector<cudnn_frontend::Operation>& backend_operations,
        std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& uid_to_backend_tensors) {
        for (auto const& sub_node : sub_nodes) {
            CHECK_CUDNN_FRONTEND_ERROR(sub_node->create_cudnn_operations(
                uids_involved_in_operation, backend_operations, uid_to_backend_tensors));
        }
        return {error_code_t::OK, ""};
    }

    // An implicitly topological-sorted vector of sub nodes.
    // The sorted order is a side effect of functional API.
    std::vector<std::unique_ptr<INode>> sub_nodes;

   public:
    virtual Type
    getType() = 0;

    error_t
    validate() {
        // validate self
        CHECK_CUDNN_FRONTEND_ERROR(validate_node());

        // infer_properties self
        CHECK_CUDNN_FRONTEND_ERROR(infer_properties_node());

        // validate sub nodes
        for (auto const& sub_node : sub_nodes) {
            CHECK_CUDNN_FRONTEND_ERROR(sub_node->validate());
        }

        return {error_code_t::OK, ""};
    }

    error_t
    build_operation_graph(cudnnHandle_t handle) {
        // Starting uid for operation graph.
        // Each time a backend tensor is created, uid will be incremented by 1, ensuring uniqueness.
        // TODO: Maybe just use uid_to_tensors size as uid each time?
        int64_t uid = 1;

        // Lower each sub node to cudnn backend.
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensors(uid, uid_to_tensors));

        // INode needs to keep track of all uids that an operation graph uses.
        // This is because cudnn backend will not accept extra tensors in variant pack.
        // But FE users provide 1 large list of tensors.
        // So internally FE assigns subset of the usre-provided tensor list to each operation graph.
        // Also, as uid in a variant pack have to be unique, keep a set of them.
        std::unordered_set<uid_t> uids_involved_in_operation;
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_operations(uids_involved_in_operation, operations, uid_to_tensors));

        // The method here fuses all operations. There will be 1 operation graph in total.
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_operation_graphs(handle));
        variant_pack_uids.push_back(std::move(uids_involved_in_operation));

        return {error_code_t::OK, ""};
    }

    int64_t
    get_workspace_size() const {
        // There are two workspaces:
        // - cudnn execution plan workspace
        // - FE node workspace (example: alibiSlope for fmha)
        return get_fe_workspace_size() + get_cudnn_workspace_size();
    }

    error_t
    execute(cudnnHandle_t handle,
            std::unordered_map<std::shared_ptr<Tensor_attributes>, void*> const& tensor_to_pointer_map,
            void* workspace) {
        std::unordered_map<int64_t, void*> tensor_uid_to_pointer_map;
        for (auto const& [tensor, pointer] : tensor_to_pointer_map) {
            tensor_uid_to_pointer_map.emplace(tensor->get_uid(), pointer);
        }

        std::unordered_map<std::shared_ptr<Tensor_attributes>, pass_by_values_t> tensor_to_pass_by_value;
        void* fe_workspace    = workspace;
        void* cudnn_workspace = static_cast<char*>(fe_workspace) + get_fe_workspace_size();

        CHECK_CUDNN_FRONTEND_ERROR(gather_pass_by_value_tensors(handle, tensor_to_pass_by_value, fe_workspace));

        // Add pass_by_value data pointers to tensor_uid_to_pointer map
        // object lifetime is controlled by tensor_to_pass_by_value which means the pointer should stay valid during
        // execute
        for (auto& [tensor, value] : tensor_to_pass_by_value) {
            if (half* half_value_ptr = std::get_if<half>(&value)) {
                tensor_uid_to_pointer_map.emplace(tensor->get_uid(), half_value_ptr);
            } else if (float* float_value_ptr = std::get_if<float>(&value)) {
                tensor_uid_to_pointer_map.emplace(tensor->get_uid(), float_value_ptr);
            } else if (void** void_value_ptr = std::get_if<void*>(&value)) {
                tensor_uid_to_pointer_map.emplace(tensor->get_uid(), *void_value_ptr);
            } else {
                RETURN_CUDNN_FRONTEND_ERROR_IF(
                    true, error_code_t::INVALID_VARIANT_PACK, "Unexpected type for pass by value tensor.");
            }
        }

        CHECK_CUDNN_FRONTEND_ERROR(execute_cudnn_plans(handle, tensor_uid_to_pointer_map, cudnn_workspace));

        return {error_code_t::OK, ""};
    }

    INode(detail::Context const& context) : context(context) {}

    virtual void
    serialize(json& j) const {
        j["nodes"];
        for (auto const& sub_node : sub_nodes) {
            json j_sub_node;
            sub_node->serialize(j_sub_node);
            j["nodes"].push_back(j_sub_node);
        }
    };

    virtual ~INode(){};
};

[[maybe_unused]] static void
to_json(json& j, const INode& p) {
    p.serialize(j);
}

}  // namespace graph

}  // namespace cudnn_frontend