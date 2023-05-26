#pragma once

#include <cudnn_frontend_PointWiseDesc.h>
#include <cudnn_frontend_Heuristics.h>
#include <cudnn_frontend_Logging.h>

#include "cudnn_frontend_graph_helpers.h"
#include "cudnn_frontend_node_interface.h"

namespace cudnn_frontend {

namespace graph {

class PointwiseNode : public INode {
private:

protected:

public:
    std::shared_ptr<Pointwise> props;

    PointwiseNode(std::string const& name, int64_t const offset = 1)  : INode (name, offset) {}

    Type getType() override final {
        return Type::POINTWISE;
    }

    int set_properties(std::string const& INode_name, std::shared_ptr<Pointwise> properties) {
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
        // Merge with ancestor's context
        fill_missing_context();

        if(props->get_compute_data_type() == DataType_t::NOT_SET) {
            props->set_compute_data_type(context.get_compute_data_type());
        }
        
        // TODO: Only inferrencing from (X, B) -> Y works today.
        auto x_tensor_prop = get_tensor_props(props->get_tensor_at_port(Pointwise::PORTS::IN_0));
        auto y_tensor_prop = get_tensor_props(props->get_tensor_at_port(Pointwise::PORTS::OUT_0));
        
        auto const& x_tensor_dim = x_tensor_prop->get_dim();
        auto y_tensor_dim = y_tensor_prop->get_dim();
        if(y_tensor_dim.empty()) {
            y_tensor_prop->set_dim(x_tensor_dim);
        } else {
            if(x_tensor_dim.size() != y_tensor_dim.size()) {
            auto status = error_t::SHAPE_DEDUCTION_FAILED;
                getLogger() << "[cudnn_frontend] ERROR: " << status << " Tensor dimensionality mismatch at X and Y ports of " << name << "." << std::endl;
                return status;
            }
        }

        for(size_t i = 0; i < Pointwise::PORTS::COUNT; ++i) {
            auto tensor_prop = get_tensor_props(props->get_tensor_at_port(static_cast<Pointwise::PORTS>(i)));
            if(tensor_prop == nullptr)
                continue;

            tensor_prop->fill_from_context(get_context());

            // Users still do not set tensor uids
            // But there might be a case that a previous node when setting its properties set the shared tensor prop's uid.
            // In such a case, do not use the independently initialized uids in props, rather update props to actual tensor uid.
            if(tensor_prop->is_uid_set)
                props->uids[i] = tensor_prop->get_uid();
            tensor_prop->set_properties_from_context(CUDNN_TENSOR_NHWC, props->uids[i]);
        }
        return error_t::OK;
    }

    error_t validate() const override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Validating PointwiseNode..." << std::endl;

        auto status = error_t::OK;

        // Ensure that ports are matched to tensors in accordance with port count.
        // X and Y should always be present.
        auto X = get_tensor_props(props->get_tensor_at_port(Pointwise::PORTS::IN_0));
        if(X == nullptr) {
            status = error_t::ATTRIBUTE_NOT_SET;
            getLogger() << "[cudnn_frontend] ERROR: " << status << " X port of pointwise node named " << name << " not mapped to a tensor." << std::endl;
            return status;
        }

        auto Y = get_tensor_props(props->get_tensor_at_port(Pointwise::PORTS::OUT_0));
        if(Y == nullptr) {
            status = error_t::ATTRIBUTE_NOT_SET;
            getLogger() << "[cudnn_frontend] ERROR: " << status << " Y port of pointwise node named " << name << " not mapped to a tensor." << std::endl;
            return status;
        }

        auto const port_count = get_pointwise_mode_port_count(props->get_mode());
        if(port_count == 3) {
            auto B = get_tensor_props(props->get_tensor_at_port(Pointwise::PORTS::IN_1));
            if(B == nullptr) {
                status = error_t::ATTRIBUTE_NOT_SET;
                getLogger() << "[cudnn_frontend] ERROR: " << status << " B port of pointwise node named " << name << " not mapped to a tensor." << std::endl;
                return status;
            }
        }

        getLogger() << "[cudnn_frontend] INFO: " << "Validated PointwiseNode." << std::endl;
        return status;
    }

    error_t createTensors() override final {

        getLogger() << "[cudnn_frontend] INFO: " << "Building PointwiseNode " << name << " tensors X:" << std::endl;
        create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Pointwise::PORTS::IN_0)));

        getLogger() << "[cudnn_frontend] INFO: " << "Building PointwiseNode " << name << " tensors Y:" << std::endl;
        create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Pointwise::PORTS::OUT_0)));
        
        auto const port_count = get_pointwise_mode_port_count(props->get_mode());
        if(port_count >= 3) {
            getLogger() << "[cudnn_frontend] INFO: " << "Building PointwiseNode " << name << " tensors B:" << std::endl;
            create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Pointwise::PORTS::IN_1)));
        }
        if(port_count >= 4) {
            getLogger() << "[cudnn_frontend] INFO: " << "Building PointwiseNode " << name << " tensors T:" << std::endl;
            create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Pointwise::PORTS::IN_2)));
        }

        return error_t::OK;
    }


    error_t createOperations() override final {

        getLogger() << "[cudnn_frontend] INFO: " << "Building PointwiseNode operations..." << std::endl;
        
        #ifndef NV_CUDNN_DISABLE_EXCEPTION
        try {
        #endif

        auto pointwise_descriptor = cudnn_frontend::PointwiseDescBuilder()
                                                        .setAxis(props->get_axis().value_or(-1))
                                                        .setComputeType(props->get_compute_data_type())
                                                        .setMode(props->get_mode())
                                                        .build();

        auto const port_count = get_pointwise_mode_port_count(props->get_mode());
        if(port_count == 4) {
            auto pointwise_operation = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                                            .setxDesc(*(tensors.at(props->uids[Pointwise::PORTS::IN_0])))
                                            .setbDesc(*(tensors.at(props->uids[Pointwise::PORTS::IN_1])))
                                            .settDesc(*(tensors.at(props->uids[Pointwise::PORTS::IN_2])))
                                            .setyDesc(*(tensors.at(props->uids[Pointwise::PORTS::OUT_0])))
                                            .setpwDesc(pointwise_descriptor)
                                            .build();
            operations.emplace(name, std::make_shared<Operation_v8>(std::move(pointwise_operation)));

            // Push all real tensors as required for operation execution.
            auto const& tensor_props_involved_in_operation = {
                get_tensor_props(props->get_tensor_at_port(Pointwise::PORTS::IN_0))
                , get_tensor_props(props->get_tensor_at_port(Pointwise::PORTS::IN_1))
                , get_tensor_props(props->get_tensor_at_port(Pointwise::PORTS::IN_2))
                , get_tensor_props(props->get_tensor_at_port(Pointwise::PORTS::OUT_0))
            };
            auto& tensors_in_operation = tensors_in_operations[name];
            for(auto const& tensor_props: tensor_props_involved_in_operation) {
                if(tensor_props->get_is_virtual() == false) {
                    tensors_in_operation.emplace_back(tensor_props->get_uid());
                }
            }
        }
        else if(port_count == 3) {
            auto pointwise_operation = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                                            .setxDesc(*(tensors.at(props->uids[Pointwise::PORTS::IN_0])))
                                            .setbDesc(*(tensors.at(props->uids[Pointwise::PORTS::IN_1])))
                                            .setyDesc(*(tensors.at(props->uids[Pointwise::PORTS::OUT_0])))
                                            .setpwDesc(pointwise_descriptor)
                                            .build();
            operations.emplace(name, std::make_shared<Operation_v8>(std::move(pointwise_operation)));

            // Push all real tensors as required for operation execution.
            auto const& tensor_props_involved_in_operation = {
                get_tensor_props(props->get_tensor_at_port(Pointwise::PORTS::IN_0))
                , get_tensor_props(props->get_tensor_at_port(Pointwise::PORTS::IN_1))
                , get_tensor_props(props->get_tensor_at_port(Pointwise::PORTS::OUT_0))
            };
            auto& tensors_in_operation = tensors_in_operations[name];
            for(auto const& tensor_props: tensor_props_involved_in_operation) {
                if(tensor_props->get_is_virtual() == false) {
                    tensors_in_operation.emplace_back(tensor_props->get_uid());
                }
            }
        }
        else if(port_count == 2) {
            auto pointwise_operation = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                                            .setxDesc(*(tensors.at(props->uids[Pointwise::PORTS::IN_0])))
                                            .setyDesc(*(tensors.at(props->uids[Pointwise::PORTS::OUT_0])))
                                            .setpwDesc(pointwise_descriptor)
                                            .build();
            operations.emplace(name, std::make_shared<Operation_v8>(std::move(pointwise_operation)));

            // Push all real tensors as required for operation execution.
            auto const& tensor_props_involved_in_operation = {
                get_tensor_props(props->get_tensor_at_port(Pointwise::PORTS::IN_0))
                , get_tensor_props(props->get_tensor_at_port(Pointwise::PORTS::OUT_0))
            };
            auto& tensors_in_operation = tensors_in_operations[name];
            for(auto const& tensor_props: tensor_props_involved_in_operation) {
                if(tensor_props->get_is_virtual() == false) {
                    tensors_in_operation.emplace_back(tensor_props->get_uid());
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

    error_t createOperationGraphs(cudnnHandle_t) override final {
        return error_t::OK;
    }

    error_t createExecutionPlans(cudnnHandle_t) override final {
        return error_t::OK;
    }
};

} // namespace graph

} // namespace cudnn_frontend