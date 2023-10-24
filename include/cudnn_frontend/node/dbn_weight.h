#pragma once

#include "../../cudnn_frontend_Heuristics.h"
#include "../../cudnn_frontend_Logging.h"

#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend {

namespace graph {

class DBNWeightNode : public INode {
    DBN_weight_attributes attributes;

   public:
    DBNWeightNode(DBN_weight_attributes&& attributes_, detail::Context const& context)
        : INode(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::DBN_WEIGHT;
    }

    error_t
    infer_properties_node() override final {
        getLogger() << "[cudnn_frontend] INFO: Inferencing properties for batchnorm finalize node " << attributes.name
                    << "..." << std::endl;

        attributes.fill_from_context(context);
        CHECK_CUDNN_FRONTEND_ERROR(attributes.validate_inputs());

        // TODO: Only inferencing from DY works today.
        auto DY                  = attributes.inputs[DBN_weight_attributes::input_names::DY];
        auto const dy_tensor_dim = DY->get_dim();

        auto X            = attributes.inputs[DBN_weight_attributes::input_names::X];
        auto x_tensor_dim = X->get_dim();
        // Only infer dims and strides if user did not set them
        if (x_tensor_dim.empty()) {
            x_tensor_dim.resize(dy_tensor_dim.size());
            X->set_dim(dy_tensor_dim);
        }
        if (X->get_stride().empty()) {
            auto const& X_dim = X->get_dim();
            // Default to NHWC
            auto const& stride_order = detail::generate_NHWC_stride_order(X_dim.size());
            X->set_stride(detail::generate_stride(X_dim, stride_order));
        }

        // Set channel length tensors
        auto infer_per_channel_tensors = [&dy_tensor_dim](std::shared_ptr<Tensor_attributes> const& T) {
            auto tensor_dim = T->get_dim();
            // Only infer dims and strides if user did not set them
            if (T->get_dim().empty()) {
                tensor_dim.resize(dy_tensor_dim.size(), 1);
                tensor_dim[1] = dy_tensor_dim[1];
                T->set_dim(tensor_dim);
            }
            if (T->get_stride().empty()) {
                auto const& T_dim = T->get_dim();
                // Default to NHWC
                auto const& stride_order = detail::generate_NHWC_stride_order(T_dim.size());
                T->set_stride(detail::generate_stride(T_dim, stride_order));
            }
        };
        infer_per_channel_tensors(attributes.outputs[DBN_weight_attributes::output_names::DBIAS]);
        infer_per_channel_tensors(attributes.outputs[DBN_weight_attributes::output_names::DSCALE]);
        infer_per_channel_tensors(attributes.outputs[DBN_weight_attributes::output_names::EQ_BIAS]);
        infer_per_channel_tensors(attributes.outputs[DBN_weight_attributes::output_names::EQ_SCALE_DY]);
        infer_per_channel_tensors(attributes.outputs[DBN_weight_attributes::output_names::EQ_SCALE_X]);

        return {error_code_t::OK, ""};
    }

    error_t
    create_cudnn_tensors(int64_t& uid,
                         std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors) override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Building DBNWeightNode tensors " << attributes.name << "..." << std::endl;

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
                    << "Building DBNWeightNode operations " << attributes.name << "..." << std::endl;

#ifndef NV_CUDNN_DISABLE_EXCEPTION
        try {
#endif

            // Create the batchnorm operation.
            auto batchnorm_operation =
                cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR)
                    .setComputeType(CUDNN_DATA_FLOAT)
                    .setEqScalesAndBias(
                        *(tensors.at(attributes.outputs[DBN_weight_attributes::output_names::EQ_SCALE_DY]->get_uid())),
                        *(tensors.at(attributes.outputs[DBN_weight_attributes::output_names::EQ_SCALE_X]->get_uid())),
                        *(tensors.at(attributes.outputs[DBN_weight_attributes::output_names::EQ_BIAS]->get_uid())))
                    .setSavedMeanAndInvVar(
                        *(tensors.at(attributes.inputs[DBN_weight_attributes::input_names::MEAN]->get_uid())),
                        *(tensors.at(attributes.inputs[DBN_weight_attributes::input_names::INV_VARIANCE]->get_uid())))
                    .setScale(*(tensors.at(attributes.inputs[DBN_weight_attributes::input_names::SCALE]->get_uid())))
                    .setxDesc(*(tensors.at(attributes.inputs[DBN_weight_attributes::input_names::X]->get_uid())))
                    .setdyDesc(*(tensors.at(attributes.inputs[DBN_weight_attributes::input_names::DY]->get_uid())))
                    .setDScaleAndDBias(
                        *(tensors.at(attributes.outputs[DBN_weight_attributes::output_names::DSCALE]->get_uid())),
                        *(tensors.at(attributes.outputs[DBN_weight_attributes::output_names::DBIAS]->get_uid())))
                    .build();

            operations.push_back(std::move(batchnorm_operation));
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

}  // namespace graph

}  // namespace cudnn_frontend