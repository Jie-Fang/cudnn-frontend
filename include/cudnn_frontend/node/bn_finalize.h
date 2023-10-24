#pragma once

#include "../../cudnn_frontend_Heuristics.h"
#include "../../cudnn_frontend_Logging.h"

#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend {

namespace graph {

class BatchNormFinalizeNode : public INode {
    BN_finalize_attributes attributes;

   public:
    BatchNormFinalizeNode(BN_finalize_attributes&& attributes_, detail::Context const& context)
        : INode(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::BN_FINALIZE;
    }

    error_t
    infer_properties_node() override final {
        getLogger() << "[cudnn_frontend] INFO: Inferencing properties for batchnorm finalize node  " << attributes.name
                    << "..." << std::endl;

        attributes.fill_from_context(context);
        CHECK_CUDNN_FRONTEND_ERROR(attributes.validate_inputs());

        auto SUM                  = attributes.inputs[BN_finalize_attributes::input_names::SUM];
        auto const sum_tensor_dim = SUM->get_dim();

        // Set channel length tensors
        auto infer_per_channel_tensors = [&sum_tensor_dim](std::shared_ptr<Tensor_attributes>& T) {
            auto tensor_dim = T->get_dim();
            // Only infer dims and strides if user did not set them
            if (tensor_dim.empty()) {
                tensor_dim = sum_tensor_dim;
                T->set_dim(tensor_dim);
            }
            if (T->get_stride().empty()) {
                auto const& T_dim = T->get_dim();
                // Default to NHWC
                auto const& stride_order = detail::generate_NHWC_stride_order(T_dim.size());
                T->set_stride(detail::generate_stride(T_dim, stride_order));
            }
        };
        infer_per_channel_tensors(attributes.outputs[BN_finalize_attributes::output_names::EQ_BIAS]);
        infer_per_channel_tensors(attributes.outputs[BN_finalize_attributes::output_names::EQ_SCALE]);
        infer_per_channel_tensors(attributes.outputs[BN_finalize_attributes::output_names::MEAN]);
        infer_per_channel_tensors(attributes.outputs[BN_finalize_attributes::output_names::INV_VARIANCE]);
        infer_per_channel_tensors(attributes.outputs[BN_finalize_attributes::output_names::NEXT_RUNNING_MEAN]);
        infer_per_channel_tensors(attributes.outputs[BN_finalize_attributes::output_names::NEXT_RUNNING_VAR]);

        return {error_code_t::OK, ""};
    }

    error_t
    create_cudnn_tensors(int64_t& uid,
                         std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors) override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Building BatchNormFinalizeNode tensors " << attributes.name << "..." << std::endl;

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
                    << "Building BatchNormFinalizeNode operations " << attributes.name << "..." << std::endl;

#ifndef NV_CUDNN_DISABLE_EXCEPTION
        try {
#endif

            // Create the batchnorm operation.
            auto batchnorm_operation =
                cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR)
                    .setComputeType(CUDNN_DATA_FLOAT)
                    .setBNFinalizeMode(CUDNN_BN_FINALIZE_STATISTICS_TRAINING)
                    .setSumDesc(*(tensors.at(attributes.inputs[BN_finalize_attributes::input_names::SUM]->get_uid())))
                    .setSqSumDesc(
                        *(tensors.at(attributes.inputs[BN_finalize_attributes::input_names::SQ_SUM]->get_uid())))
                    .setEqScaleAndBias(
                        *(tensors.at(attributes.outputs[BN_finalize_attributes::output_names::EQ_SCALE]->get_uid())),
                        *(tensors.at(attributes.outputs[BN_finalize_attributes::output_names::EQ_BIAS]->get_uid())))
                    .setSavedMeanAndInvVar(
                        *(tensors.at(attributes.outputs[BN_finalize_attributes::output_names::MEAN]->get_uid())),
                        *(tensors.at(
                            attributes.outputs[BN_finalize_attributes::output_names::INV_VARIANCE]->get_uid())))
                    .setScaleAndBias(
                        *(tensors.at(attributes.inputs[BN_finalize_attributes::input_names::SCALE]->get_uid())),
                        *(tensors.at(attributes.inputs[BN_finalize_attributes::input_names::BIAS]->get_uid())))
                    .setPrevRunningMeanAndVar(
                        *(tensors.at(
                            attributes.inputs[BN_finalize_attributes::input_names::PREV_RUNNING_MEAN]->get_uid())),
                        *(tensors.at(
                            attributes.inputs[BN_finalize_attributes::input_names::PREV_RUNNING_VAR]->get_uid())))
                    .setNextRunningMeanAndVar(
                        *(tensors.at(
                            attributes.outputs[BN_finalize_attributes::output_names::NEXT_RUNNING_MEAN]->get_uid())),
                        *(tensors.at(
                            attributes.outputs[BN_finalize_attributes::output_names::NEXT_RUNNING_VAR]->get_uid())))
                    .setEpsilonTensor(
                        *(tensors.at(attributes.inputs[BN_finalize_attributes::input_names::EPSILON]->get_uid())))
                    .setExpDecayFactorTensor(
                        *(tensors.at(attributes.inputs[BN_finalize_attributes::input_names::MOMENTUM]->get_uid())))
                    .setAccumCountTensor(
                        *(tensors.at(attributes.inputs[BN_finalize_attributes::input_names::ACCUM_COUNT]->get_uid())))
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