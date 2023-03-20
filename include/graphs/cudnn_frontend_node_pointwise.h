#pragma once

#include <cudnn_frontend_PointWiseDesc.h>
#include <cudnn_frontend_Heuristics.h>
#include <cudnn_frontend_Logging.h>

#include "cudnn_frontend_graph_helpers.h"
#include "cudnn_frontend_node_interface.h"

namespace cudnn_frontend {

class PointwiseNode : public INode {
private:

protected:

public:
    std::shared_ptr<pointwise_properties> props;

    PointwiseNode(std::string const& name, int64_t const offset = 1)  : INode (name, offset) {}

    Type getType() override final {
        return Type::POINTWISE;
    }

    int set_properties(std::string const& INode_name, std::shared_ptr<pointwise_properties> properties) {
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
        getLogger() << "[cudnn_frontend] INFO: Inferrencing properties for pointwise node named " << name << "." << std::endl;

        props->update_uids(offset);

        // TODO: Only inferrencing from (X, B) -> Y works today.
        auto x_tensor_prop = get_tensor_props(props->get_port_name(pointwise_properties::PORTS::X));
        auto y_tensor_prop = get_tensor_props(props->get_port_name(pointwise_properties::PORTS::Y));
        
        auto const& x_tensor_dim = x_tensor_prop->get_dim();
        auto& y_tensor_dim = y_tensor_prop->get_dim();        
        if(y_tensor_dim.empty()) {
            y_tensor_dim.resize(x_tensor_dim.size());
            for(size_t dim = 0; dim < x_tensor_dim.size(); ++dim) {        
                y_tensor_dim[dim] = x_tensor_dim[dim];
            }
        } else {
            if(x_tensor_dim.size() != y_tensor_dim.size()) {
            auto status = error_t::SHAPE_DEDUCTION_FAILED;
                getLogger() << "[cudnn_frontend] ERROR: " << status << " Tensor dimensionality mismatch at X and Y ports of " << name << "." << std::endl;
                return status;
            }
        }

        for(size_t i = 0; i < pointwise_properties::PORTS::COUNT; ++i) {
            auto tensor_prop = get_tensor_props(props->get_port_name(static_cast<pointwise_properties::PORTS>(i)));
            if(tensor_prop == nullptr)
                continue;

            // Users still do not set tensor uids
            // But there might be a case that a previous node when setting its properties set the shared tensor prop's uid.
            // In such a case, do not use the independently initialized uids in props, rather update props to actual tensor uid.
            if(tensor_prop->is_uid_set)
                props->uids[i] = tensor_prop->get_uid();
            tensor_prop->set_properties_from_context(CUDNN_TENSOR_NHWC, props->get_tensor_data_type(), props->uids[i]);
        }
        return error_t::OK;
    }

    error_t validate() const override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Validating PointwiseNode..." << std::endl;

        getLogger() << "[cudnn_frontend] INFO: " << "Validated PointwiseNode." << std::endl;
        return error_t::OK;
    }

    error_t createTensors() override final {
        
        getLogger() << "[cudnn_frontend] INFO: " << "Building PointwiseNode tensors X:" << std::endl;
        create_cudnn_tensor(get_tensor_props(props->get_port_name(pointwise_properties::PORTS::X)));

        getLogger() << "[cudnn_frontend] INFO: " << "Building PointwiseNode tensors Y:" << std::endl;
        create_cudnn_tensor(get_tensor_props(props->get_port_name(pointwise_properties::PORTS::Y)));

        if(props->get_mode() == PointwiseMode_t::ADD || props->get_mode() == PointwiseMode_t::MUL) {
            getLogger() << "[cudnn_frontend] INFO: " << "Building PointwiseNode tensors B:" << std::endl;
            create_cudnn_tensor(get_tensor_props(props->get_port_name(pointwise_properties::PORTS::B)));
        }

        return error_t::OK;
    }


    error_t createOperations() override final {

        getLogger() << "[cudnn_frontend] INFO: " << "Building PointwiseNode operations..." << std::endl;
        
        #ifndef NV_CUDNN_DISABLE_EXCEPTION
        try {
        #endif

        auto pointwise_descriptor = cudnn_frontend::PointwiseDescBuilder()
                                                        .setComputeType(props->get_compute_type())
                                                        .setMode(props->get_mode())
                                                        .build();
        auto const port_count = get_pointwise_mode_port_count(props->get_mode());
        if(port_count == 3) {
            auto pointwise_operation = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                                            .setxDesc(*(tensors.at(props->uids[pointwise_properties::PORTS::X])))
                                            .setbDesc(*(tensors.at(props->uids[pointwise_properties::PORTS::B])))
                                            .setyDesc(*(tensors.at(props->uids[pointwise_properties::PORTS::Y])))
                                            .setpwDesc(pointwise_descriptor)
                                            .build();
            operations.emplace(name, std::make_shared<Operation>(std::move(pointwise_operation)));
                
            // Push all real tensors as required for operation execution.
            auto const& tensor_props_involved_in_operation = {
                get_tensor_props(props->get_port_name(pointwise_properties::PORTS::X))
                , get_tensor_props(props->get_port_name(pointwise_properties::PORTS::B))
                , get_tensor_props(props->get_port_name(pointwise_properties::PORTS::Y))
            };
            for(auto const& tensor_props: tensor_props_involved_in_operation) {
                if(tensor_props->get_is_virtual() == false) {
                    tensors_in_operations[name].emplace_back(tensor_props->get_uid());
                }
            }
        }
        else if(port_count == 2) {
            auto pointwise_operation = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                                            .setxDesc(*(tensors.at(props->uids[pointwise_properties::PORTS::X])))
                                            .setyDesc(*(tensors.at(props->uids[pointwise_properties::PORTS::Y])))
                                            .setpwDesc(pointwise_descriptor)
                                            .build();
            operations.emplace(name, std::make_shared<Operation>(std::move(pointwise_operation)));
   
            // Push all real tensors as required for operation execution.
            auto const& tensor_props_involved_in_operation = {
                get_tensor_props(props->get_port_name(pointwise_properties::PORTS::X))
                , get_tensor_props(props->get_port_name(pointwise_properties::PORTS::Y))
            };
            for(auto const& tensor_props: tensor_props_involved_in_operation) {
                if(tensor_props->get_is_virtual() == false) {
                    tensors_in_operations[name].emplace_back(tensor_props->get_uid());
                }
            }
        }

        getLogger() << "[cudnn_frontend] INFO: " << "Built PointwiseNode operation." << std::endl;

        #ifndef NV_CUDNN_DISABLE_EXCEPTION
        } catch (cudnn_frontend::cudnnException &e) {
            throw cudnnException(e.what(), e.getCudnnStatus());
        }
        #endif
        
        return error_t::OK;
    }

    error_t partition() override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Partitioning PointwiseNode..." << std::endl;
        
        auto status = create_cudnn_execution_plan({{name}});
        if(status != error_t::OK) {
            getLogger() << "[cudnn_frontend] ERROR: " << status << " Failed to create execution plans for graph partitioning in PointwiseNode." << std::endl;
            return status;
        }

        getLogger() << "[cudnn_frontend] INFO: " << "Partitioned PointwiseNode." << std::endl;
        return error_t::OK;
    }
};

} // namespace cudnn_frontend