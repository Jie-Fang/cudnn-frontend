#pragma once

#include <cudnn_frontend_ConvDesc.h>
#include <cudnn_frontend_Heuristics.h>
#include <cudnn_frontend_Logging.h>

#include "cudnn_frontend_graph_helpers.h"
#include "cudnn_frontend_node_interface.h"

namespace cudnn_frontend {

namespace graph {

class DgradNode : public INode {
private:

protected:

public:
    std::shared_ptr<Dgrad> props;

    DgradNode(std::string const& name, int64_t offset = 1)  : INode (name, offset) {}

    Type getType() override final {
        return Type::DGRAD;
    }

    int set_properties(std::string const& INode_name, std::shared_ptr<Dgrad> properties) {
        if(sub_nodes.size() != 0) {
            return 1;
        }
        if(INode_name != name) {
            return 1;
        }

        props = properties;
        return 0;
    }

    error_t infer_properties() override final {
        getLogger() << "[cudnn_frontend] INFO: Inferrencing properties for dgrad node named " << name << "." << std::endl;
        props->update_uids(offset);

        // Merge with ancestor's context
        fill_missing_context();

        props->fill_from_context(get_context());

        // TODO: Only inferrencing from (X, DY) -> DW works today.
        auto dx_tensor_prop = get_tensor_props(props->get_tensor_at_port(Dgrad::PORTS::DX));
        auto w_tensor_prop = get_tensor_props(props->get_tensor_at_port(Dgrad::PORTS::W));
        auto dy_tensor_prop = get_tensor_props(props->get_tensor_at_port(Dgrad::PORTS::DY));
        
        auto const w_tensor_dim = w_tensor_prop->get_dim();
        auto const dy_tensor_dim = dy_tensor_prop->get_dim();
        auto dx_tensor_dim = dx_tensor_prop->get_dim();
        if(w_tensor_dim.size() != dy_tensor_dim.size()) {
            auto status = error_t::SHAPE_DEDUCTION_FAILED;
            getLogger() << "[cudnn_frontend] ERROR: " << status << "  Tensor dimensionality mismatch at W and DY ports of " << name << "." << std::endl;
            return status;
        }

        if(dx_tensor_dim.empty()) {
            dx_tensor_dim.resize(w_tensor_dim.size());
            auto const& padding = props->get_padding();
            auto const& stride = props->get_stride();
            auto const& dilation = props->get_dilation();
            // x NCHW
            // w KCRS
            // y NKPQ
            // N
            dx_tensor_dim[0] = dy_tensor_dim[0];
            // C
            dx_tensor_dim[1] = w_tensor_dim[1];
            // HW
            for(size_t dim = 2; dim < w_tensor_dim.size(); ++dim) {
                dx_tensor_dim[dim] = (dy_tensor_dim[dim] - 1) * stride[dim - 2] - 2*padding[dim - 2] + 1 + dilation[dim-2]*(w_tensor_dim[dim]-1);
            }
            dx_tensor_prop->set_dim(dx_tensor_dim);
        } else {
            if(w_tensor_dim.size() != dx_tensor_dim.size()) {
            auto status = error_t::SHAPE_DEDUCTION_FAILED;
                getLogger() << "[cudnn_frontend] ERROR: " << status << " Tensor dimensionality mismatch at W and DX ports of " << name << "." << std::endl;
                return status;
            }
        }

        for(size_t i = 0; i < Dgrad::PORTS::COUNT; ++i) {
            auto tensor_prop = get_tensor_props(props->get_tensor_at_port(static_cast<Dgrad::PORTS>(i)));

            tensor_prop->fill_from_context(get_context());

            if(tensor_prop->is_uid_set)
                props->uids[i] = tensor_prop->get_uid();
            tensor_prop->set_properties_from_context(CUDNN_TENSOR_NHWC, props->uids[i]);
        }

        return error_t::OK;
    }

    error_t validate() const override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Validating DgradNode..." << std::endl;

        // TODO: check all properties of this operation and its tensor are correct
        // Like do dim count match dim/stride
        // Do dim and corresponding stride match

        getLogger() << "[cudnn_frontend] INFO: " << "Validated DgradNode." << std::endl;
        return error_t::OK;
    }

    error_t createTensors() override final {

        getLogger() << "[cudnn_frontend] INFO: " << "Building DgradNode tensors..." << std::endl;

        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Dgrad::PORTS::DX))));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Dgrad::PORTS::W))));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Dgrad::PORTS::DY))));

        getLogger() << "[cudnn_frontend] INFO: " << "Built DgradNode tensors." << std::endl;

        return error_t::OK;
    }

    error_t createOperations() override final {

        getLogger() << "[cudnn_frontend] INFO: " << "Building DgradNode operations..." << std::endl;
        
        #ifndef NV_CUDNN_DISABLE_EXCEPTION
        try {
        #endif

        // dgrad descriptor
        int64_t const spatial_dim_count = props->get_padding().size();
        auto dgrad_descriptor = cudnn_frontend::ConvDescBuilder()
                                                        .setComputeType(props->get_compute_data_type())
                                                        .setMathMode(CUDNN_CROSS_CORRELATION)
                                                        .setSpatialDimCount(spatial_dim_count)
                                                        .setSpatialStride(spatial_dim_count, props->get_stride().data())
                                                        .setPrePadding(spatial_dim_count, props->get_padding().data())
                                                        .setPostPadding(spatial_dim_count, props->get_padding().data())
                                                        .setDilation(spatial_dim_count, props->get_dilation().data())
                                                        .build();

        // Create the dgrad operation.
        auto dgrad_operation = cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR)
                                        .setdxDesc(*(tensors.at(props->uids[Dgrad::PORTS::DX])))
                                        .setwDesc(*(tensors.at(props->uids[Dgrad::PORTS::W])))
                                        .setdyDesc(*(tensors.at(props->uids[Dgrad::PORTS::DY])))
                                        .setcDesc(dgrad_descriptor)
                                        .setAlpha(1.f)
                                        .setBeta(0.f)
                                        .build();
        operations.emplace(name, std::make_shared<Operation_v8>(std::move(dgrad_operation)));

        // Push all real tensors as required for operation execution.
        auto const& tensor_props_involved_in_operation = {
            get_tensor_props(props->get_tensor_at_port(Dgrad::PORTS::DX))
            , get_tensor_props(props->get_tensor_at_port(Dgrad::PORTS::W))
            , get_tensor_props(props->get_tensor_at_port(Dgrad::PORTS::DY))
        };
        for(auto const& tensor_props: tensor_props_involved_in_operation) {
            if(tensor_props->get_is_virtual() == false) {
                tensors_in_operations[name].emplace_back(tensor_props->get_uid());
            }
        }

        getLogger() << "[cudnn_frontend] INFO: " << "Built DgradNode operation." << std::endl;

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