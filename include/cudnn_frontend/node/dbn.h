#pragma once

#include "../../cudnn_frontend_Heuristics.h"
#include "../../cudnn_frontend_Logging.h"

#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend {

namespace graph {

class DBNNode : public INode {
   public:
    Batchnorm_backward_attributes attributes;

    DBNNode(Batchnorm_backward_attributes&& attributes_, detail::Context const& context)
        : INode(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::DBN;
    }

    error_t
    validate_node() const override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Validating DBNNode " << attributes.name << "..." << std::endl;

        return {error_code_t::OK, ""};
    }

    error_t
    infer_properties_node() override final {
        getLogger() << "[cudnn_frontend] INFO: Inferencing properties for DBN node " << attributes.name << "..."
                    << std::endl;

        attributes.fill_from_context(context);
        CHECK_CUDNN_FRONTEND_ERROR(attributes.validate_inputs());

        // TODO: Only inferencing from X works today.
        auto X                  = attributes.inputs[Batchnorm_backward_attributes::input_names::X];
        auto const x_tensor_dim = X->get_dim();

        auto DX            = attributes.outputs[Batchnorm_backward_attributes::output_names::DX];
        auto dx_tensor_dim = DX->get_dim();
        // Only infer dims and strides if user did not set them
        if (dx_tensor_dim.empty()) {
            dx_tensor_dim.resize(x_tensor_dim.size());
            DX->set_dim(x_tensor_dim);
        }
        if (DX->get_stride().empty()) {
            auto const& DX_dim = DX->get_dim();
            // Default to NHWC
            auto const& stride_order = detail::generate_NHWC_stride_order(DX_dim.size());
            DX->set_stride(detail::generate_stride(DX_dim, stride_order));
        }

        // Set channel length tensors
        auto infer_per_channel_tensors = [&x_tensor_dim](std::shared_ptr<Tensor_attributes>& T) {
            auto tensor_dim = T->get_dim();
            // Only infer dims and strides if user did not set them
            if (tensor_dim.empty()) {
                tensor_dim.resize(x_tensor_dim.size(), 1);
                tensor_dim[1] = x_tensor_dim[1];
                T->set_dim(tensor_dim);
            }
            if (T->get_stride().empty()) {
                auto const& T_dim = T->get_dim();
                // Default to NHWC
                auto const& stride_order = detail::generate_NHWC_stride_order(T_dim.size());
                T->set_stride(detail::generate_stride(T_dim, stride_order));
            }
        };
        infer_per_channel_tensors(attributes.outputs[Batchnorm_backward_attributes::output_names::DSCALE]);
        infer_per_channel_tensors(attributes.outputs[Batchnorm_backward_attributes::output_names::DBIAS]);
        return {error_code_t::OK, ""};
    }

    error_t
    create_cudnn_tensors(int64_t& uid,
                         std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors) override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Building DBNNode tensors " << attributes.name << "..." << std::endl;

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

        // Special case in BN where peer stats is also an input but is not present in inputs map
        for (auto const& tensor : attributes.peer_stats) {
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
                    << "Building DBNNode operations " << attributes.name << "..." << std::endl;

#ifndef NV_CUDNN_DISABLE_EXCEPTION
        try {
#endif

            std::vector<cudnn_frontend::Tensor> peer_stats;
            for (auto const& peer_stat : attributes.peer_stats) {
                peer_stats.emplace_back(std::move(*(tensors.at(peer_stat->get_uid()))));
            }

            // Create the DBN operation.
            auto DBN_operation =
                cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_NORM_BACKWARD_DESCRIPTOR)
                    .setNormalizationMode(NormMode_t::BATCH_NORM)
                    .setxDesc(
                        *(tensors.at(attributes.inputs[Batchnorm_backward_attributes::input_names::X]->get_uid())))
                    .setdyDesc(
                        *(tensors.at(attributes.inputs[Batchnorm_backward_attributes::input_names::DY]->get_uid())))
                    .setScale(
                        *(tensors.at(attributes.inputs[Batchnorm_backward_attributes::input_names::SCALE]->get_uid())))
                    .setSavedMeanAndInvVar(
                        *(tensors.at(attributes.inputs[Batchnorm_backward_attributes::input_names::MEAN]->get_uid())),
                        *(tensors.at(
                            attributes.inputs[Batchnorm_backward_attributes::input_names::INV_VARIANCE]->get_uid())))
                    .setDScaleAndDBias(
                        *(tensors.at(
                            attributes.outputs[Batchnorm_backward_attributes::output_names::DSCALE]->get_uid())),
                        *(tensors.at(
                            attributes.outputs[Batchnorm_backward_attributes::output_names::DBIAS]->get_uid())))
                    // .setEpsilonTensor(*(tensors.at(epsilon->get_uid())))
                    .setdxDesc(
                        *(tensors.at(attributes.outputs[Batchnorm_backward_attributes::output_names::DX]->get_uid())))
                    .setPeerStatTensor(peer_stats)
                    .build();

            operations.push_back(std::move(DBN_operation));
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
            // Special case in BN where peer stats is also an input but is not present in inputs map
            for (auto const& tensor : attributes.peer_stats) {
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