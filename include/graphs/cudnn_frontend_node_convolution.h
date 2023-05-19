#pragma once

#include <cudnn_frontend_ConvDesc.h>
#include <cudnn_frontend_Heuristics.h>
#include <cudnn_frontend_Logging.h>

#include "cudnn_frontend_graph_helpers.h"
#include "cudnn_frontend_node_interface.h"

namespace cudnn_frontend {

namespace graph {

class ConvolutionNode : public INode {
private:

protected:

public:
    std::shared_ptr<Convolution> props;

    ConvolutionNode(std::string const& name, int64_t offset = 1)  : INode (name, offset) {}

    Type getType() override final {
        return Type::CONVOLUTION;
    }

    int set_properties(std::string const& INode_name, std::shared_ptr<Convolution> properties) {
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
        getLogger() << "[cudnn_frontend] INFO: Inferrencing properties for conv node named " << name << "." << std::endl;
        props->update_uids(offset);

        // Merge with ancestor's context
        fill_missing_context();

        if(props->get_compute_data_type() == DataType_t::NOT_SET) {
            props->set_compute_data_type(context.get_compute_data_type());
        }

        // TODO: Only inferrencing from (X, W) -> Y works today.
        auto x_tensor_prop = get_tensor_props(props->get_tensor_at_port(Convolution::PORTS::X));
        auto w_tensor_prop = get_tensor_props(props->get_tensor_at_port(Convolution::PORTS::W));
        auto y_tensor_prop = get_tensor_props(props->get_tensor_at_port(Convolution::PORTS::Y));
        
        auto const x_tensor_dim = x_tensor_prop->get_dim();
        auto const w_tensor_dim = w_tensor_prop->get_dim();
        auto y_tensor_dim = y_tensor_prop->get_dim();
        if(x_tensor_dim.size() != w_tensor_dim.size()) {
            auto status = error_t::SHAPE_DEDUCTION_FAILED;
            getLogger() << "[cudnn_frontend] ERROR: " << status << "  Tensor dimensionality mismatch at X and W ports of " << name << "." << std::endl;
            return status;
        }
        
        if(y_tensor_dim.empty()) {
            y_tensor_dim.resize(x_tensor_dim.size());
            auto const& padding = props->get_padding();
            auto const& stride = props->get_stride();
            auto const& dilation = props->get_dilation();
            // N
            y_tensor_dim[0] = x_tensor_dim[0];
            // PQ
            for(size_t dim = 2; dim < x_tensor_dim.size(); ++dim) {        
                y_tensor_dim[dim] = 1 + (x_tensor_dim[dim] - dilation[dim-2]*(w_tensor_dim[dim]-1)-1 + 2*padding[dim - 2]) / stride[dim - 2];
            }
            // K
            y_tensor_dim[1] = w_tensor_dim[0];
            y_tensor_prop->set_dim(y_tensor_dim);
        } else {
            if(x_tensor_dim.size() != y_tensor_dim.size()) {
            auto status = error_t::SHAPE_DEDUCTION_FAILED;
                getLogger() << "[cudnn_frontend] ERROR: " << status << " Tensor dimensionality mismatch at X and Y ports of " << name << "." << std::endl;
                return status;
            }
        }

        for(size_t i = 0; i < Convolution::PORTS::COUNT; ++i) {
            auto tensor_prop = get_tensor_props(props->get_tensor_at_port(static_cast<Convolution::PORTS>(i)));

            if(!(tensor_prop->is_data_type_set)) {
                if(tensor_prop->get_is_virtual()) {
                    tensor_prop->set_data_type(context.get_intermediate_data_type());
                }    
                else {
                    tensor_prop->set_data_type(context.get_io_data_type());
                }
            }

            if(tensor_prop->is_uid_set)
                props->uids[i] = tensor_prop->get_uid();
            tensor_prop->set_properties_from_context(CUDNN_TENSOR_NHWC, props->uids[i]);
        }

        return error_t::OK;
    }
    
    error_t validate() const override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Validating ConvolutionNode..." << std::endl;

        // TODO: check all properties of this operation and its tensor are correct
        // Like do dim count match dim/stride
        // Do dim and corresponding stride match

        getLogger() << "[cudnn_frontend] INFO: " << "Validated ConvolutionNode." << std::endl;
        return error_t::OK;
    }

    error_t createTensors() override final {

        getLogger() << "[cudnn_frontend] INFO: " << "Building ConvolutionNode tensors..." << std::endl;

        create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Convolution::PORTS::X)));
        create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Convolution::PORTS::W)));
        create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Convolution::PORTS::Y)));

        getLogger() << "[cudnn_frontend] INFO: " << "Built ConvolutionNode tensors." << std::endl;

        return error_t::OK;
    }
    
    error_t createOperations() override final {

        getLogger() << "[cudnn_frontend] INFO: " << "Building ConvolutionNode operations..." << std::endl;
        
        #ifndef NV_CUDNN_DISABLE_EXCEPTION
        try {
        #endif

        // convolution descriptor
        int64_t const spatial_dim_count = props->get_padding().size();
        auto convolution_descriptor = cudnn_frontend::ConvDescBuilder()
                                                        .setComputeType(props->get_compute_data_type())
                                                        .setMathMode(CUDNN_CROSS_CORRELATION)
                                                        .setSpatialDimCount(spatial_dim_count)
                                                        .setSpatialStride(spatial_dim_count, props->get_stride().data())
                                                        .setPrePadding(spatial_dim_count, props->get_padding().data())
                                                        .setPostPadding(spatial_dim_count, props->get_padding().data())
                                                        .setDilation(spatial_dim_count, props->get_dilation().data())
                                                        .build();

        // Create the convolution operation.
        auto convolution_operation = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                                        .setxDesc(*(tensors.at(props->uids[Convolution::PORTS::X])))
                                        .setwDesc(*(tensors.at(props->uids[Convolution::PORTS::W])))
                                        .setyDesc(*(tensors.at(props->uids[Convolution::PORTS::Y])))
                                        .setcDesc(convolution_descriptor)
                                        .setAlpha(1.f)
                                        .setBeta(0.f)
                                        .build();
        operations.emplace(name, std::make_shared<Operation_v8>(std::move(convolution_operation)));
        
        // Push all real tensors as required for operation execution.
        auto const& tensor_props_involved_in_operation = {
            get_tensor_props(props->get_tensor_at_port(Convolution::PORTS::X))
            , get_tensor_props(props->get_tensor_at_port(Convolution::PORTS::W))
            , get_tensor_props(props->get_tensor_at_port(Convolution::PORTS::Y))
        };
        for(auto const& tensor_props: tensor_props_involved_in_operation) {
            if(tensor_props->get_is_virtual() == false) {
                tensors_in_operations[name].emplace_back(tensor_props->get_uid());
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

    error_t partition() override final {
        getLogger() << "[cudnn_frontend] INFO: Partitioning ConvolutionNode..." << std::endl;

        auto status = create_cudnn_execution_plan({{name}});
        if(status != error_t::OK) {
            getLogger() << "[cudnn_frontend] ERROR: " << status << " Failed to create execution plans for graph partitioning in ConvolutionNode." << std::endl;
            return status;
        }

        getLogger() << "[cudnn_frontend] INFO: Partitioned ConvolutionNode." << std::endl;
        return error_t::OK;
    }
};

} // namespace graph

} // namespace cudnn_frontend