#pragma once

#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend {

namespace graph {
class RoPENode : public NodeCRTP<RoPENode> {
   public:
    RoPE_attributes attributes;

    RoPENode(RoPE_attributes&& attributes_, detail::Context const& context)
        : NodeCRTP(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::ROPE;
    }

    error_t
    infer_properties_node() override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO: Inferencing properties for RoPE node " << attributes.name);

        attributes.fill_from_context(context);

        auto INPUT  = attributes.inputs[RoPE_attributes::input_names::INPUT];
        auto OUTPUT = attributes.outputs[RoPE_attributes::output_names::OUTPUT];

        // Output has same dims and strides as input
        if (OUTPUT->get_dim().empty()) {
            OUTPUT->set_dim(INPUT->get_dim());
        }
        if (OUTPUT->get_stride().empty()) {
            OUTPUT->set_stride(INPUT->get_stride());
        }

        return {error_code_t::OK, ""};
    }

    error_t
    pre_validate_node() const override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO: Validating RoPENode " << attributes.name);

        RETURN_CUDNN_FRONTEND_ERROR_IF(
            attributes.inputs.find(RoPE_attributes::input_names::INPUT) == attributes.inputs.end(),
            error_code_t::ATTRIBUTE_NOT_SET,
            "RoPE node requires INPUT tensor.");

        RETURN_CUDNN_FRONTEND_ERROR_IF(
            attributes.inputs.find(RoPE_attributes::input_names::COS) == attributes.inputs.end(),
            error_code_t::ATTRIBUTE_NOT_SET,
            "RoPE node requires COS tensor.");

        RETURN_CUDNN_FRONTEND_ERROR_IF(
            attributes.inputs.find(RoPE_attributes::input_names::SIN) == attributes.inputs.end(),
            error_code_t::ATTRIBUTE_NOT_SET,
            "RoPE node requires SIN tensor.");

        return {error_code_t::OK, ""};
    }

    error_t
    create_cudnn_operations(
        std::unordered_set<Tensor_attributes::uid_t>& uids_involved_in_operations,
        std::vector<std::shared_ptr<cudnn_frontend::Operation>>& operations,
        managed_backend_descriptor_t& raw_operations,
        std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors) const override final {
        CUDNN_FRONTEND_UNUSED(operations);
        CUDNN_FRONTEND_UNUSED(raw_operations);
        CUDNN_FRONTEND_UNUSED(tensors);

        // RoPE is an OSS-only operation. No cuDNN backend descriptor exists.
        // Register our tensor UIDs in the variant pack so the framework tracks them
        // for pointer resolution during execution.
        for (auto const& [name, tensor] : attributes.inputs) {
            if (tensor) {
                uids_involved_in_operations.insert(tensor->get_uid());
            }
        }
        for (auto const& [name, tensor] : attributes.outputs) {
            if (tensor) {
                uids_involved_in_operations.insert(tensor->get_uid());
            }
        }

        return {error_code_t::OK, ""};
    }

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    virtual void
    serialize(json& j) const override final {
        j = {{"tag", "ROPE"}, {"name", attributes.name}};
    }
#endif
};

}  // namespace graph

}  // namespace cudnn_frontend
