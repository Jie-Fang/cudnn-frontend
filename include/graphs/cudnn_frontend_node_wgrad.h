#pragma once

#include <cudnn_frontend_ConvDesc.h>
#include <cudnn_frontend_Heuristics.h>
#include <cudnn_frontend_Logging.h>

#include "cudnn_frontend_graph_helpers.h"
#include "cudnn_frontend_node_interface.h"

namespace cudnn_frontend {

namespace graph {

class WgradNode : public INode {
private:

protected:

public:
    std::shared_ptr<Wgrad> props;

    WgradNode(std::string const& name, int64_t offset = 1)  : INode (name, offset) {}

    Type getType() override final {
        return Type::WGRAD;
    }

    error_t infer_properties() override final {
        getLogger() << "[cudnn_frontend] INFO: Inferrencing properties for conv node named " << name << "." << std::endl;
        props->update_uids(offset);

        // Merge with ancestor's context
        fill_missing_context();

        props->fill_from_context(get_context());

        // TODO: Only inferrencing from (X, DY) -> DW works today.
        auto x_tensor_prop = get_tensor_props(props->get_tensor_at_port(Wgrad::PORTS::X));
        auto dw_tensor_prop = get_tensor_props(props->get_tensor_at_port(Wgrad::PORTS::DW));
        auto dy_tensor_prop = get_tensor_props(props->get_tensor_at_port(Wgrad::PORTS::DY));
        
        auto const x_tensor_dim = x_tensor_prop->get_dim();
        auto const dy_tensor_dim = dy_tensor_prop->get_dim();
        auto dw_tensor_dim = dw_tensor_prop->get_dim();
        if(x_tensor_dim.size() != dy_tensor_dim.size()) {
            auto status = error_t::SHAPE_DEDUCTION_FAILED;
            getLogger() << "[cudnn_frontend] ERROR: " << status << "  Tensor dimensionality mismatch at X and DY ports of " << name << "." << std::endl;
            return status;
        }

        if(dw_tensor_dim.empty()) {
            dw_tensor_dim.resize(x_tensor_dim.size());
            auto const& padding = props->get_padding();
            auto const& stride = props->get_stride();
            auto const& dilation = props->get_dilation();
            // x NCHW
            // w KCRS
            // y NKPQ
            // K
            dw_tensor_dim[0] = dy_tensor_dim[1];
            // C
            dw_tensor_dim[1] = x_tensor_dim[1];
            // RS
            for(size_t dim = 2; dim < x_tensor_dim.size(); ++dim) {
                dw_tensor_dim[dim] = (x_tensor_dim[dim] + 2*padding[dim - 2] - (dy_tensor_dim[dim] - 1) * stride[dim - 2] - 1) / dilation[dim-2] + 1;
            }
            dw_tensor_prop->set_dim(dw_tensor_dim);
        } else {
            if(x_tensor_dim.size() != dw_tensor_dim.size()) {
            auto status = error_t::SHAPE_DEDUCTION_FAILED;
                getLogger() << "[cudnn_frontend] ERROR: " << status << " Tensor dimensionality mismatch at X and Y ports of " << name << "." << std::endl;
                return status;
            }
        }

        for(size_t i = 0; i < Wgrad::PORTS::COUNT; ++i) {
            auto tensor_prop = get_tensor_props(props->get_tensor_at_port(static_cast<Wgrad::PORTS>(i)));

            tensor_prop->fill_from_context(get_context());
        }

        return error_t::OK;
    }

    error_t validate() const override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Validating WgradNode..." << std::endl;

        // TODO: check all properties of this operation and its tensor are correct
        // Like do dim count match dim/stride
        // Do dim and corresponding stride match

        getLogger() << "[cudnn_frontend] INFO: " << "Validated WgradNode." << std::endl;
        return error_t::OK;
    }

    error_t createTensors() override final {

        getLogger() << "[cudnn_frontend] INFO: " << "Building WgradNode tensors..." << std::endl;

        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Wgrad::PORTS::X))));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Wgrad::PORTS::DW))));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Wgrad::PORTS::DY))));

        getLogger() << "[cudnn_frontend] INFO: " << "Built WgradNode tensors." << std::endl;

        return error_t::OK;
    }

    error_t createOperations() override final {

        getLogger() << "[cudnn_frontend] INFO: " << "Building WgradNode operations..." << std::endl;
        
        #ifndef NV_CUDNN_DISABLE_EXCEPTION
        try {
        #endif

        // wgrad descriptor
        int64_t const spatial_dim_count = props->get_padding().size();
        auto wgrad_descriptor = cudnn_frontend::ConvDescBuilder()
                                                        .setComputeType(props->get_compute_data_type())
                                                        .setMathMode(CUDNN_CROSS_CORRELATION)
                                                        .setSpatialDimCount(spatial_dim_count)
                                                        .setSpatialStride(spatial_dim_count, props->get_stride().data())
                                                        .setPrePadding(spatial_dim_count, props->get_padding().data())
                                                        .setPostPadding(spatial_dim_count, props->get_padding().data())
                                                        .setDilation(spatial_dim_count, props->get_dilation().data())
                                                        .build();

        // Create the wgrad operation.
        auto wgrad_operation = cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR)
                                        .setxDesc(*(tensors.at(props->uids[Wgrad::PORTS::X])))
                                        .setdwDesc(*(tensors.at(props->uids[Wgrad::PORTS::DW])))
                                        .setdyDesc(*(tensors.at(props->uids[Wgrad::PORTS::DY])))
                                        .setcDesc(wgrad_descriptor)
                                        .setAlpha(1.f)
                                        .setBeta(0.f)
                                        .build();
        operations.emplace(name, std::make_shared<Operation_v8>(std::move(wgrad_operation)));

        // Push all real tensors as required for operation execution.
        auto const& tensor_props_involved_in_operation = {
            get_tensor_props(props->get_tensor_at_port(Wgrad::PORTS::X))
            , get_tensor_props(props->get_tensor_at_port(Wgrad::PORTS::DW))
            , get_tensor_props(props->get_tensor_at_port(Wgrad::PORTS::DY))
        };
        for(auto const& tensor_props: tensor_props_involved_in_operation) {
            if(tensor_props->get_is_virtual() == false) {
                tensors_in_operations[name].emplace_back(tensor_props->get_uid());
            }
        }

        getLogger() << "[cudnn_frontend] INFO: " << "Built WgradNode operation." << std::endl;

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

} // namespace graph

} // namespace cudnn_frontend