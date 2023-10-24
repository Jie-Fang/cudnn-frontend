#pragma once

#include "../../cudnn_frontend_Heuristics.h"
#include "../../cudnn_frontend_Logging.h"

#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend {

namespace graph {
class RMSNormNode : public INode {
   public:
    Rmsnorm_attributes attributes;

    RMSNormNode(Rmsnorm_attributes&& attributes_, detail::Context const& context)
        : INode(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::RMSNORM;
    }

    error_t
    infer_properties_node() override final {
        getLogger() << "[cudnn_frontend] INFO: Inferencing properties for rmsnorm node " << attributes.name << "..."
                    << std::endl;

        attributes.fill_from_context(context);
        CHECK_CUDNN_FRONTEND_ERROR(attributes.validate_inputs());

        auto X                  = attributes.inputs[Rmsnorm_attributes::input_names::X];
        auto const x_tensor_dim = X->get_dim();

        auto Y            = attributes.outputs[Rmsnorm_attributes::output_names::Y];
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

        if (attributes.forward_phase == NormFwdPhase_t::TRAINING) {
            auto inv_var = attributes.outputs[Rmsnorm_attributes::output_names::INV_VARIANCE];
            if (auto inv_var_dim = inv_var->get_dim(); inv_var_dim.empty()) {
                inv_var_dim.resize(x_tensor_dim.size(), 1);
                inv_var_dim[0] = x_tensor_dim[0];
                inv_var->set_dim(inv_var_dim);
            }
            if (inv_var->get_stride().empty()) {
                auto const& inv_var_dim = inv_var->get_dim();
                // Default to NHWC
                auto const& stride_order = detail::generate_NHWC_stride_order(inv_var_dim.size());
                inv_var->set_stride(detail::generate_stride(inv_var_dim, stride_order));
            }
        }
        return {error_code_t::OK, ""};
    }

    error_t
    validate_node() const override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Validating RMSNormNode " << attributes.name << "..." << std::endl;

        // Norm forward phase should be set
        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.forward_phase == NormFwdPhase_t::NOT_SET,
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "Forward phase not set of rmsnorm node.");

        return {error_code_t::OK, ""};
    }

    error_t
    create_cudnn_tensors(int64_t& uid,
                         std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors) override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Building RMSNormNode tensors " << attributes.name << "..." << std::endl;

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
                    << "Building RMSNormNode operations " << attributes.name << "..." << std::endl;

#ifndef NV_CUDNN_DISABLE_EXCEPTION
        try {
#endif
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

            if (attributes.inputs[Rmsnorm_attributes::input_names::BIAS]) {
                if (attributes.forward_phase == NormFwdPhase_t::TRAINING) {
                    auto rmsnorm_operation =
                        cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_NORM_FORWARD_DESCRIPTOR)
                            .setNormalizationMode(NormMode_t::RMS_NORM)
                            .setNormFwdPhase(attributes.forward_phase)
                            .setxDesc(*(tensors.at(attributes.inputs[Rmsnorm_attributes::input_names::X]->get_uid())))
                            .setSavedInvVar(*(tensors.at(
                                attributes.outputs[Rmsnorm_attributes::output_names::INV_VARIANCE]->get_uid())))
                            .setScaleAndBias(
                                *(tensors.at(attributes.inputs[Rmsnorm_attributes::input_names::SCALE]->get_uid())),
                                *(tensors.at(attributes.inputs[Rmsnorm_attributes::input_names::BIAS]->get_uid())))
                            .setEpsilonTensor(
                                *(tensors.at(attributes.inputs[Rmsnorm_attributes::input_names::EPSILON]->get_uid())))
                            .setyDesc(*(tensors.at(attributes.outputs[Rmsnorm_attributes::output_names::Y]->get_uid())))
                            .build();
                    operations.push_back(std::move(rmsnorm_operation));
                } else {
                    auto rmsnorm_operation =
                        cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_NORM_FORWARD_DESCRIPTOR)
                            .setNormalizationMode(NormMode_t::RMS_NORM)
                            .setNormFwdPhase(attributes.forward_phase)
                            .setxDesc(*(tensors.at(attributes.inputs[Rmsnorm_attributes::input_names::X]->get_uid())))
                            .setScaleAndBias(
                                *(tensors.at(attributes.inputs[Rmsnorm_attributes::input_names::SCALE]->get_uid())),
                                *(tensors.at(attributes.inputs[Rmsnorm_attributes::input_names::BIAS]->get_uid())))
                            .setEpsilonTensor(
                                *(tensors.at(attributes.inputs[Rmsnorm_attributes::input_names::EPSILON]->get_uid())))
                            .setyDesc(*(tensors.at(attributes.outputs[Rmsnorm_attributes::output_names::Y]->get_uid())))
                            .build();
                    operations.push_back(std::move(rmsnorm_operation));
                }
            } else {
                if (attributes.forward_phase == NormFwdPhase_t::TRAINING) {
                    auto rmsnorm_operation =
                        cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_NORM_FORWARD_DESCRIPTOR)
                            .setNormalizationMode(NormMode_t::RMS_NORM)
                            .setNormFwdPhase(attributes.forward_phase)
                            .setxDesc(*(tensors.at(attributes.inputs[Rmsnorm_attributes::input_names::X]->get_uid())))
                            .setSavedInvVar(*(tensors.at(
                                attributes.outputs[Rmsnorm_attributes::output_names::INV_VARIANCE]->get_uid())))
                            .setScale(
                                *(tensors.at(attributes.inputs[Rmsnorm_attributes::input_names::SCALE]->get_uid())))
                            .setEpsilonTensor(
                                *(tensors.at(attributes.inputs[Rmsnorm_attributes::input_names::EPSILON]->get_uid())))
                            .setyDesc(*(tensors.at(attributes.outputs[Rmsnorm_attributes::output_names::Y]->get_uid())))
                            .build();
                    operations.push_back(std::move(rmsnorm_operation));
                } else {
                    auto rmsnorm_operation =
                        cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_NORM_FORWARD_DESCRIPTOR)
                            .setNormalizationMode(NormMode_t::RMS_NORM)
                            .setNormFwdPhase(attributes.forward_phase)
                            .setxDesc(*(tensors.at(attributes.inputs[Rmsnorm_attributes::input_names::X]->get_uid())))
                            .setScale(
                                *(tensors.at(attributes.inputs[Rmsnorm_attributes::input_names::SCALE]->get_uid())))
                            .setEpsilonTensor(
                                *(tensors.at(attributes.inputs[Rmsnorm_attributes::input_names::EPSILON]->get_uid())))
                            .setyDesc(*(tensors.at(attributes.outputs[Rmsnorm_attributes::output_names::Y]->get_uid())))
                            .build();
                    operations.push_back(std::move(rmsnorm_operation));
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

class DRMSNormNode : public INode {
   public:
    Rmsnorm_backward_attributes attributes;

    DRMSNormNode(Rmsnorm_backward_attributes&& attributes_, detail::Context const& context)
        : INode(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::DRMSNorm;
    }

    error_t
    validate_node() const override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Validating DRMSNormNode node " << attributes.name << "..." << std::endl;

        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.use_dbias.has_value() == false,
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "DRMSNormNode node needs has_bias(bool) to be called.");

        return {error_code_t::OK, ""};
    }

    error_t
    infer_properties_node() override final {
        getLogger() << "[cudnn_frontend] INFO: Inferencing properties for DRMSNorm node " << attributes.name << "..."
                    << std::endl;

        attributes.fill_from_context(context);
        CHECK_CUDNN_FRONTEND_ERROR(attributes.validate_inputs());

        // TODO: Only inferencing from X works today.
        auto X                  = attributes.inputs[Rmsnorm_backward_attributes::input_names::X];
        auto const x_tensor_dim = X->get_dim();

        auto DY            = attributes.inputs[Rmsnorm_backward_attributes::input_names::DY];
        auto dy_tensor_dim = DY->get_dim();

        // Only infer dims and strides if user did not set them
        if (dy_tensor_dim.empty()) {
            dy_tensor_dim.resize(x_tensor_dim.size());
            DY->set_dim(x_tensor_dim);
        }
        if (DY->get_stride().empty()) {
            auto const& DY_dim = DY->get_dim();
            // Default to NHWC
            auto const& stride_order = detail::generate_NHWC_stride_order(DY_dim.size());
            DY->set_stride(detail::generate_stride(DY_dim, stride_order));
        }

        auto DX            = attributes.outputs[Rmsnorm_backward_attributes::output_names::DX];
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

        auto scale_bias_dim = X->get_dim();
        scale_bias_dim[0]   = 1;

        // Set channel length tensors
        auto infer_scale_bias_tensors = [&scale_bias_dim](std::shared_ptr<Tensor_attributes>& T) {
            auto tensor_dim = T->get_dim();
            // Only infer dims and strides if user did not set them
            if (tensor_dim.empty()) {
                T->set_dim(scale_bias_dim);
            }
            if (T->get_stride().empty()) {
                auto const& T_dim = T->get_dim();
                // Default to NHWC
                auto const& stride_order = detail::generate_NHWC_stride_order(T_dim.size());
                T->set_stride(detail::generate_stride(T_dim, stride_order));
            }
        };

        infer_scale_bias_tensors(attributes.outputs[Rmsnorm_backward_attributes::output_names::DSCALE]);
        if (attributes.use_dbias.value()) {
            infer_scale_bias_tensors(attributes.outputs[Rmsnorm_backward_attributes::output_names::DBIAS]);
        }

        return {error_code_t::OK, ""};
    }

    error_t
    create_cudnn_tensors(int64_t& uid,
                         std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors) override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Building DRMSNormNode tensors " << attributes.name << "..." << std::endl;

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
                    << "Building DRMSNormNode operations " << attributes.name << "..." << std::endl;

#ifndef NV_CUDNN_DISABLE_EXCEPTION
        try {
#endif

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

            if (attributes.use_dbias.value()) {
                // Create the DRMSNorm operation.
                auto DRMSNorm_operation =
                    cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_NORM_BACKWARD_DESCRIPTOR)
                        .setNormalizationMode(NormMode_t::RMS_NORM)
                        .setxDesc(
                            *(tensors.at(attributes.inputs[Rmsnorm_backward_attributes::input_names::X]->get_uid())))
                        .setdyDesc(
                            *(tensors.at(attributes.inputs[Rmsnorm_backward_attributes::input_names::DY]->get_uid())))
                        .setScale(*(
                            tensors.at(attributes.inputs[Rmsnorm_backward_attributes::input_names::SCALE]->get_uid())))
                        .setSavedInvVar(*(tensors.at(
                            attributes.inputs[Rmsnorm_backward_attributes::input_names::INV_VARIANCE]->get_uid())))
                        .setDScaleAndDBias(
                            *(tensors.at(
                                attributes.outputs[Rmsnorm_backward_attributes::output_names::DSCALE]->get_uid())),
                            *(tensors.at(
                                attributes.outputs[Rmsnorm_backward_attributes::output_names::DBIAS]->get_uid())))
                        .setdxDesc(
                            *(tensors.at(attributes.outputs[Rmsnorm_backward_attributes::output_names::DX]->get_uid())))
                        .build();
                operations.push_back(std::move(DRMSNorm_operation));
            } else {
                // Create the DRMSNorm operation.
                auto DRMSNorm_operation =
                    cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_NORM_BACKWARD_DESCRIPTOR)
                        .setNormalizationMode(NormMode_t::RMS_NORM)
                        .setxDesc(
                            *(tensors.at(attributes.inputs[Rmsnorm_backward_attributes::input_names::X]->get_uid())))
                        .setdyDesc(
                            *(tensors.at(attributes.inputs[Rmsnorm_backward_attributes::input_names::DY]->get_uid())))
                        .setScale(*(
                            tensors.at(attributes.inputs[Rmsnorm_backward_attributes::input_names::SCALE]->get_uid())))
                        .setSavedInvVar(*(tensors.at(
                            attributes.inputs[Rmsnorm_backward_attributes::input_names::INV_VARIANCE]->get_uid())))
                        .setDScale(*(tensors.at(
                            attributes.outputs[Rmsnorm_backward_attributes::output_names::DSCALE]->get_uid())))
                        .setdxDesc(
                            *(tensors.at(attributes.outputs[Rmsnorm_backward_attributes::output_names::DX]->get_uid())))
                        .build();
                operations.push_back(std::move(DRMSNorm_operation));
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