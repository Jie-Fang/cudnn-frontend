#pragma once

#include <cudnn_frontend_ConvDesc.h>
#include <cudnn_frontend_Heuristics.h>
#include <cudnn_frontend_Logging.h>

#include "cudnn_frontend_graph_helpers.h"
#include "cudnn_frontend_node_interface.h"

namespace cudnn_frontend::graph {

class ConvolutionNode : public INode {
    std::shared_ptr<Convolution> options;
public:

    ConvolutionNode(std::string const& name, std::shared_ptr<Convolution> const options)  : INode (name), options(options) {}

    Type getType() override final {
        return Type::CONVOLUTION;
    }

    error_t infer_properties() override final {
        getLogger() << "[cudnn_frontend] INFO: Inferrencing properties for conv node named " << name << "." << std::endl;
        
        // Merge with ancestor's context
        fill_missing_context();

        options->fill_from_context(get_context());

        // TODO: Only inferrencing from (X, W) -> Y works today.
        auto X = options->inputs.X;
        auto W = options->inputs.W;
        auto Y = options->outputs.Y;
        
        auto const x_tensor_dim = X->get_dim();
        auto const w_tensor_dim = W->get_dim();
        auto y_tensor_dim = Y->get_dim();
        if(x_tensor_dim.size() != w_tensor_dim.size()) {
            auto status = error_t::SHAPE_DEDUCTION_FAILED;
            getLogger() << "[cudnn_frontend] ERROR: " << status << "  Tensor dimensionality mismatch at X and W ports of " << name << "." << std::endl;
            return status;
        }

        if(y_tensor_dim.empty()) {
            y_tensor_dim.resize(x_tensor_dim.size());
            auto const& padding = options->get_padding();
            auto const& stride = options->get_stride();
            auto const& dilation = options->get_dilation();
            // N
            y_tensor_dim[0] = x_tensor_dim[0];
            // PQ
            for(size_t dim = 2; dim < x_tensor_dim.size(); ++dim) {
                y_tensor_dim[dim] = 1 + (x_tensor_dim[dim] - dilation[dim-2]*(w_tensor_dim[dim]-1)-1 + 2*padding[dim - 2]) / stride[dim - 2];
            }
            // K
            y_tensor_dim[1] = w_tensor_dim[0];
            Y->set_dim(y_tensor_dim).generateStrides(CUDNN_TENSOR_NHWC);
        } else {
            if(x_tensor_dim.size() != y_tensor_dim.size()) {
                auto status = error_t::SHAPE_DEDUCTION_FAILED;
                getLogger() << "[cudnn_frontend] ERROR: " << status << " Tensor dimensionality mismatch at X and Y ports of " << name << "." << std::endl;
                return status;
            }
        }

        return error_t::OK;
    }

    error_t assignUids_() override final {
        options->inputs.X->set_uid(ICudnn::create_new_uid());
        options->inputs.W->set_uid(ICudnn::create_new_uid());
        options->outputs.Y->set_uid(ICudnn::create_new_uid());
        return error_t::OK;
    }

    error_t createTensors() override final {

        getLogger() << "[cudnn_frontend] INFO: " << "Building ConvolutionNode tensors..." << std::endl;

        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options->inputs.X));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options->inputs.W));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options->outputs.Y));

        getLogger() << "[cudnn_frontend] INFO: " << "Built ConvolutionNode tensors." << std::endl;

        return error_t::OK;
    }

    error_t createOperations() override final {

        getLogger() << "[cudnn_frontend] INFO: " << "Building ConvolutionNode operations..." << std::endl;
        
        #ifndef NV_CUDNN_DISABLE_EXCEPTION
        try {
        #endif

        // convolution descriptor
        int64_t const spatial_dim_count = options->get_padding().size();
        auto convolution_descriptor = cudnn_frontend::ConvDescBuilder()
                                                        .setComputeType(options->get_compute_data_type())
                                                        .setMathMode(CUDNN_CROSS_CORRELATION)
                                                        .setSpatialDimCount(spatial_dim_count)
                                                        .setSpatialStride(spatial_dim_count, options->get_stride().data())
                                                        .setPrePadding(spatial_dim_count, options->get_padding().data())
                                                        .setPostPadding(spatial_dim_count, options->get_padding().data())
                                                        .setDilation(spatial_dim_count, options->get_dilation().data())
                                                        .build();

        // Create the convolution operation.
        auto convolution_operation = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                                        .setxDesc(*(tensors.at(options->inputs.X->get_uid())))
                                        .setwDesc(*(tensors.at(options->inputs.W->get_uid())))
                                        .setyDesc(*(tensors.at(options->outputs.Y->get_uid())))
                                        .setcDesc(convolution_descriptor)
                                        .setAlpha(1.f)
                                        .setBeta(0.f)
                                        .build();
        operations.emplace(name, std::make_shared<Operation_v8>(std::move(convolution_operation)));

        // Push all real tensors as required for operation execution.
        auto const& tensors_involved_in_operation = {
            options->inputs.X
            , options->inputs.W
            , options->outputs.Y
        };
        for(auto const& tensor: tensors_involved_in_operation) {
            if(tensor && tensor->get_is_virtual() == false) {
                tensors_in_operations[name].emplace_back(tensor->get_uid());
            }
        }

        getLogger() << "[cudnn_frontend] INFO: " << "Built ConvolutionNode operation." << std::endl;

        #ifndef NV_CUDNN_DISABLE_EXCEPTION
        } catch (cudnn_frontend::cudnnException &e) {
            throw cudnnException(e.what(), e.getCudnnStatus());
        }
        #endif
        
        return error_t::OK;
    }

    error_t createOperationGraphs(cudnnHandle_t) override final {
        return error_t::OK;
    }

    error_t createExecutionPlans(cudnnHandle_t) override final {
        return error_t::OK;
    }
};

} // namespace cudnn_frontend::graph