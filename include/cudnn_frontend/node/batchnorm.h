#pragma once

#include "../../cudnn_frontend_Heuristics.h"
#include "../../cudnn_frontend_Logging.h"

#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend {

namespace graph {
class BatchNormNode : public INode {
   public:
    Batchnorm_attributes attributes;

    BatchNormNode(Batchnorm_attributes&& attributes_, detail::Context const& context)
        : INode(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::BATCHNORM;
    }

    error_t
    infer_properties_node() override final {
        getLogger() << "[cudnn_frontend] INFO: Inferencing properties for batchnorm node " << attributes.name << "..."
                    << std::endl;

        attributes.fill_from_context(context);

        auto X                  = attributes.inputs[Batchnorm_attributes::input_names::X];
        auto const x_tensor_dim = X->get_dim();

        auto Y            = attributes.outputs[Batchnorm_attributes::output_names::Y];
        auto y_tensor_dim = Y->get_dim();
        // Only infer dims and strides if user did not set them
        if (y_tensor_dim.empty()) {
            y_tensor_dim.resize(x_tensor_dim.size());
            Y->set_dim(x_tensor_dim);
        }
        if (Y->get_stride().empty()) {
            auto const& Y_dim = Y->get_dim();
            // Default to NHWC
            auto const& stride_order = detail::generate_NHWC_stride_order(Y_dim.size());
            Y->set_stride(detail::generate_stride(Y_dim, stride_order));
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
        infer_per_channel_tensors(attributes.outputs[Batchnorm_attributes::output_names::MEAN]);
        infer_per_channel_tensors(attributes.outputs[Batchnorm_attributes::output_names::INV_VARIANCE]);
        infer_per_channel_tensors(attributes.outputs[Batchnorm_attributes::output_names::NEXT_RUNNING_MEAN]);
        infer_per_channel_tensors(attributes.outputs[Batchnorm_attributes::output_names::NEXT_RUNNING_VAR]);
        infer_per_channel_tensors(attributes.inputs[Batchnorm_attributes::input_names::PREV_RUNNING_MEAN]);
        infer_per_channel_tensors(attributes.inputs[Batchnorm_attributes::input_names::PREV_RUNNING_VAR]);
        infer_per_channel_tensors(attributes.inputs[Batchnorm_attributes::input_names::SCALE]);
        infer_per_channel_tensors(attributes.inputs[Batchnorm_attributes::input_names::BIAS]);

        // Set scalar tensors
        auto infer_scalar_tensors = [&x_tensor_dim](std::shared_ptr<Tensor_attributes>& T) {
            auto tensor_dim = T->get_dim();
            // Only infer dims and strides if user did not set them
            if (tensor_dim.empty()) {
                tensor_dim.resize(x_tensor_dim.size(), 1);
                T->set_dim(tensor_dim);
            }
            if (T->get_stride().empty()) {
                auto const& T_dim = T->get_dim();
                // Default to NHWC
                auto const& stride_order = detail::generate_NHWC_stride_order(T_dim.size());
                T->set_stride(detail::generate_stride(T_dim, stride_order));
            }
        };
        infer_scalar_tensors(attributes.inputs[Batchnorm_attributes::input_names::EPSILON]);
        infer_scalar_tensors(attributes.inputs[Batchnorm_attributes::input_names::MOMENTUM]);

        for (auto const& peer_stat : attributes.peer_stats) {
            if (peer_stat->get_stride().empty()) {
                auto const& peer_stat_dim = peer_stat->get_dim();
                // Default to NHWC
                auto const& stride_order = detail::generate_NHWC_stride_order(peer_stat_dim.size());
                peer_stat->set_stride(detail::generate_stride(peer_stat_dim, stride_order));
            }
        }

        return {error_code_t::OK, ""};
    }

    error_t
    validate_node() const override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Validating BatchNormNode " << attributes.name << "..." << std::endl;

        // Norm forward phase should be set
        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.forward_phase == NormFwdPhase_t::NOT_SET,
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "Forward phase not set of batchnorm node.");

        return {error_code_t::OK, ""};
    }

    error_t
    create_cudnn_tensors(int64_t& uid,
                         std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors) override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Building BatchNormNode tensors " << attributes.name << "..." << std::endl;

        for (auto const& tensor : attributes.get_tensors()) {
            CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(tensor, uid, tensors));
        }
        return {error_code_t::OK, ""};
    }

    error_t
    create_cudnn_operations(
        std::unordered_set<uid_t>& uids_involved_in_operations,
        std::vector<cudnn_frontend::Operation_v8>& operations,
        std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors) override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Building BatchNormNode operations " << attributes.name << "..." << std::endl;

#ifndef NV_CUDNN_DISABLE_EXCEPTION
        try {
#endif

            std::vector<cudnn_frontend::Tensor> peer_stats;
            for (auto const& peer_stat : attributes.peer_stats) {
                peer_stats.emplace_back(std::move(*(tensors[peer_stat->get_uid()])));
            }

            auto batchnorm_operation =
                cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_NORM_FORWARD_DESCRIPTOR)
                    .setNormalizationMode(NormMode_t::BATCH_NORM)
                    .setNormFwdPhase(attributes.forward_phase)
                    .setxDesc(*(tensors[attributes.inputs[Batchnorm_attributes::input_names::X]->get_uid()]))
                    .setSavedMeanAndInvVar(
                        *(tensors[attributes.outputs[Batchnorm_attributes::output_names::MEAN]->get_uid()]),
                        *(tensors[attributes.outputs[Batchnorm_attributes::output_names::INV_VARIANCE]->get_uid()]))
                    .setScaleAndBias(*(tensors[attributes.inputs[Batchnorm_attributes::input_names::SCALE]->get_uid()]),
                                     *(tensors[attributes.inputs[Batchnorm_attributes::input_names::BIAS]->get_uid()]))
                    .setPrevRunningMeanAndVar(
                        *(tensors[attributes.inputs[Batchnorm_attributes::input_names::PREV_RUNNING_MEAN]->get_uid()]),
                        *(tensors[attributes.inputs[Batchnorm_attributes::input_names::PREV_RUNNING_VAR]->get_uid()]))
                    .setNextRunningMeanAndVar(
                        *(tensors[attributes.outputs[Batchnorm_attributes::output_names::NEXT_RUNNING_MEAN]
                                      ->get_uid()]),
                        *(tensors[attributes.outputs[Batchnorm_attributes::output_names::NEXT_RUNNING_VAR]->get_uid()]))
                    .setEpsilonTensor(
                        *(tensors[attributes.inputs[Batchnorm_attributes::input_names::EPSILON]->get_uid()]))
                    .setExpDecayFactorTensor(
                        *(tensors[attributes.inputs[Batchnorm_attributes::input_names::MOMENTUM]->get_uid()]))
                    .setyDesc(*(tensors[attributes.outputs[Batchnorm_attributes::output_names::Y]->get_uid()]))
                    .setPeerStatTensor(peer_stats)
                    .build();

            for (auto const& tensor : attributes.get_tensors()) {
                if (tensor->get_is_virtual() == false) {
                    uids_involved_in_operations.insert(tensor->get_uid());
                }
            }

            operations.push_back(std::move(batchnorm_operation));

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