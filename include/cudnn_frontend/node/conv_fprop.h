#pragma once

#include "../../cudnn_frontend_ConvDesc.h"
#include "../../cudnn_frontend_Heuristics.h"
#include "../../cudnn_frontend_Logging.h"

#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend::graph {

class ConvolutionNode : public INode {
   public:
    Conv_fprop_attributes attributes;

    ConvolutionNode(Conv_fprop_attributes&& attributes_, detail::Context const& context)
        : INode(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::CONVOLUTION;
    }

    error_t
    infer_properties_node() override final {
        getLogger() << "[cudnn_frontend] INFO: Inferrencing properties for conv node " << attributes.name << "..."
                    << std::endl;

        attributes.fill_from_context(context);

        // TODO: Only inferrencing from (X, W) -> Y works today.
        auto X = attributes.inputs[Conv_fprop_attributes::input_names::X];
        auto W = attributes.inputs[Conv_fprop_attributes::input_names::W];
        auto Y = attributes.outputs[Conv_fprop_attributes::output_names::Y];

        auto const x_tensor_dim = X->get_dim();
        auto const w_tensor_dim = W->get_dim();
        auto y_tensor_dim       = Y->get_dim();

        // Only infer dims and strides if user did not set them
        if (y_tensor_dim.empty()) {
            y_tensor_dim.resize(x_tensor_dim.size());
            auto const& padding  = attributes.get_padding();
            auto const& stride   = attributes.get_stride();
            auto const& dilation = attributes.get_dilation();
            // N
            y_tensor_dim[0] = x_tensor_dim[0];
            // PQ
            for (size_t dim = 2; dim < x_tensor_dim.size(); ++dim) {
                y_tensor_dim[dim] =
                    1 + (x_tensor_dim[dim] - dilation[dim - 2] * (w_tensor_dim[dim] - 1) - 1 + 2 * padding[dim - 2]) /
                            stride[dim - 2];
            }
            // K
            y_tensor_dim[1] = w_tensor_dim[0];
            Y->set_dim(y_tensor_dim);
        }
        if (Y->get_stride().empty()) {
            auto const& Y_dim = Y->get_dim();
            // Default to NHWC
            auto const& stride_order = detail::generate_NHWC_stride_order(Y_dim.size());
            Y->set_stride(detail::generate_stride(Y_dim, stride_order));
        }

        return {error_code_t::OK, ""};
    }

    error_t
    create_cudnn_tensors(int64_t& uid,
                         std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors) override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Building ConvolutionNode tensors " << attributes.name << "..." << std::endl;

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
                    << "Building ConvolutionNode operations " << attributes.name << "..." << std::endl;

#ifndef NV_CUDNN_DISABLE_EXCEPTION
        try {
#endif

            // convolution descriptor
            int64_t const spatial_dim_count = attributes.get_padding().size();
            auto convolution_descriptor     = cudnn_frontend::ConvDescBuilder()
                                              .setComputeType(attributes.compute_data_type)
                                              .setMathMode(CUDNN_CROSS_CORRELATION)
                                              .setSpatialDimCount(spatial_dim_count)
                                              .setSpatialStride(spatial_dim_count, attributes.get_stride().data())
                                              .setPrePadding(spatial_dim_count, attributes.get_padding().data())
                                              .setPostPadding(spatial_dim_count, attributes.get_padding().data())
                                              .setDilation(spatial_dim_count, attributes.get_dilation().data())
                                              .build();

            // Create the convolution operation.
            auto convolution_operation =
                cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                    .setxDesc(*(tensors.at(attributes.inputs[Conv_fprop_attributes::input_names::X]->get_uid())))
                    .setwDesc(*(tensors.at(attributes.inputs[Conv_fprop_attributes::input_names::W]->get_uid())))
                    .setyDesc(*(tensors.at(attributes.outputs[Conv_fprop_attributes::output_names::Y]->get_uid())))
                    .setcDesc(convolution_descriptor)
                    .setAlpha(1.f)
                    .setBeta(0.f)
                    .build();

            for (auto const& tensor : attributes.get_tensors()) {
                if (tensor->get_is_virtual() == false) {
                    uids_involved_in_operations.insert(tensor->get_uid());
                }
            }

            operations.push_back(std::move(convolution_operation));

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