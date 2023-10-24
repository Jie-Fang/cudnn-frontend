#pragma once

#include "../../cudnn_frontend_ReductionDesc.h"
#include "../../cudnn_frontend_Logging.h"

#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend::graph {

class ReductionNode : public INode {
    Reduction_attributes attributes;

   public:
    ReductionNode(Reduction_attributes&& attributes_, detail::Context const& context)
        : INode(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::REDUCTION;
    }

    error_t
    validate_node() const override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Validating reduction node " << attributes.name << "..." << std::endl;

        RETURN_CUDNN_FRONTEND_ERROR_IF(
            attributes.inputs.find(Reduction_attributes::input_names::X) == attributes.inputs.end(),
            error_code_t::ATTRIBUTE_NOT_SET,
            "reduction input not set.");

        RETURN_CUDNN_FRONTEND_ERROR_IF(
            attributes.outputs.find(Reduction_attributes::output_names::Y) == attributes.outputs.end(),
            error_code_t::ATTRIBUTE_NOT_SET,
            "reduction Y not set.");

        return {error_code_t::OK, ""};
    }

    error_t
    infer_properties_node() override final {
        getLogger() << "[cudnn_frontend] INFO: Inferrencing properties for reduction node " << attributes.name << "..."
                    << std::endl;

        attributes.fill_from_context(context);
        CHECK_CUDNN_FRONTEND_ERROR(attributes.validate_inputs());

        // Only inferrencing from IN_0 to OUT_0 works today.
        auto x_tensor = attributes.inputs[Reduction_attributes::input_names::X];
        auto y_tensor = attributes.outputs[Reduction_attributes::output_names::Y];

        auto const& x_tensor_dim = x_tensor->get_dim();
        auto y_tensor_dim        = y_tensor->get_dim();
        // Only infer dims and strides if user did not set them
        if (y_tensor_dim.empty()) {
            y_tensor->set_dim(x_tensor_dim);
        }
        if (y_tensor->get_stride().empty()) {
            auto const& y_dim = y_tensor->get_dim();
            // Default to NHWC
            auto const& stride_order = detail::generate_NHWC_stride_order(y_dim.size());
            y_tensor->set_stride(detail::generate_stride(y_dim, stride_order));
        }

        return {error_code_t::OK, ""};
    }

    error_t
    create_cudnn_tensors(int64_t& uid,
                         std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors) override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Building ReductionNode tensors " << attributes.name << "..." << std::endl;

        for (auto const& [name, tensor] : attributes.inputs) {
            if (tensor) {
                CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(tensor, uid, tensors));
            }
        }
        for (auto const& [name, tensor] : attributes.outputs) {
            if (tensor) {
                CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(tensor, uid, tensors));
            }
        }
        return {error_code_t::OK, ""};
    }

    error_t
    create_cudnn_operations(
        std::unordered_set<uid_t>& uids_involved_in_operations,
        std::vector<cudnn_frontend::Operation_v8>& operations,
        std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors) override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Building ReductionNode operations " << attributes.name << "..." << std::endl;

#ifndef NV_CUDNN_DISABLE_EXCEPTION
        try {
#endif

            auto reduction_descriptor = cudnn_frontend::ReductionDescBuilder()
                                            .setComputeType(attributes.compute_data_type)
                                            .setReductionOp(attributes.get_mode().value())
                                            .build();

            auto reduction_operation =
                cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
                    .setxDesc(*(tensors.at(attributes.inputs[Reduction_attributes::input_names::X]->get_uid())))
                    .setyDesc(*(tensors.at(attributes.outputs[Reduction_attributes::output_names::Y]->get_uid())))
                    .setreductionDesc(reduction_descriptor)
                    .build();

            operations.push_back(std::move(reduction_operation));
            for (auto const& [name, tensor] : attributes.inputs) {
                if (tensor && tensor->get_is_virtual() == false) {
                    uids_involved_in_operations.insert(tensor->get_uid());
                }
            }
            for (auto const& [name, tensor] : attributes.outputs) {
                if (tensor && tensor->get_is_virtual() == false) {
                    uids_involved_in_operations.insert(tensor->get_uid());
                }
            }
#ifndef NV_CUDNN_DISABLE_EXCEPTION
        } catch (cudnn_frontend::cudnnException& e) {
            throw cudnnException(e.what(), e.getCudnnStatus());
        }
#endif

        return {error_code_t::OK, ""};
    }

    virtual void
    serialize(json& j) const override final {
        j = attributes;
    }
};

}  // namespace cudnn_frontend::graph