#pragma once

#include <cudnn_frontend_PointWiseDesc.h>
#include <cudnn_frontend_Heuristics.h>
#include <cudnn_frontend_Logging.h>

#include "cudnn_frontend_graph_helpers.h"
#include "cudnn_frontend_node_interface.h"

namespace cudnn_frontend::graph {

class PointwiseNode : public INode {
    std::shared_ptr<Pointwise> options;
public:

    PointwiseNode(std::string const& name, std::shared_ptr<Pointwise> const options, int64_t const offset = 1)  : INode (name, offset), options(options) {}

    Type getType() override final {
        return Type::POINTWISE;
    }

    error_t infer_properties() override final {
        getLogger() << "[cudnn_frontend] INFO: Inferrencing properties for pointwise node named " << name << "." << std::endl;
        
        // Merge with ancestor's context
        fill_missing_context();
            
        // Use context to fill in missing options
        options->fill_from_context(get_context());
        
        // Only inferrencing from IN_0 to OUT_0 works today.
        auto in_0_tensor = options->inputs.IN_0;
        auto out_0_tensor = options->outputs.OUT_0;
        
        auto const& in_0_tensor_dim = in_0_tensor->get_dim();
        auto out_0_tensor_dim = out_0_tensor->get_dim();
        if(out_0_tensor_dim.empty()) {
            out_0_tensor->set_dim(in_0_tensor_dim).generateStrides(CUDNN_TENSOR_NHWC);
        } else {
            if(out_0_tensor_dim.size() != in_0_tensor_dim.size()) {
            auto status = error_t::SHAPE_DEDUCTION_FAILED;
                getLogger() << "[cudnn_frontend] ERROR: " << status << " Tensor dimensionality mismatch at X and Y ports of " << name << "." << std::endl;
                return status;
            }
        }

        // TODO: gather all tensors and assign them uids at once using a counter. So no need to keep uids in properties.
        // But for the time being doing it here manually.
        if(in_0_tensor->is_uid_set == false) {
            in_0_tensor->set_uid(offset + 1);
        }
        if(options->inputs.IN_1 && options->inputs.IN_1->is_uid_set == false) {
            options->inputs.IN_1->set_uid(offset + 2);
        }
        if(options->inputs.IN_2 && options->inputs.IN_2->is_uid_set == false) {
            options->inputs.IN_2->set_uid(offset + 3);
        }
        if(out_0_tensor->is_uid_set == false) {
            out_0_tensor->set_uid(offset + 4);
        }

        return error_t::OK;
    }

    error_t validate() const override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Validating PointwiseNode..." << std::endl;

        auto status = error_t::OK;

        // Ensure that ports are matched to tensors in accordance with port count.
        // X and Y should always be present.
        auto X = options->inputs.IN_0;
        if(X == nullptr) {
            status = error_t::ATTRIBUTE_NOT_SET;
            getLogger() << "[cudnn_frontend] ERROR: " << status << " X port of pointwise node named " << name << " not mapped to a tensor." << std::endl;
            return status;
        }

        auto Y = options->outputs.OUT_0;
        if(Y == nullptr) {
            status = error_t::ATTRIBUTE_NOT_SET;
            getLogger() << "[cudnn_frontend] ERROR: " << status << " Y port of pointwise node named " << name << " not mapped to a tensor." << std::endl;
            return status;
        }

        auto const port_count = get_pointwise_mode_port_count(options->get_mode().value());
        if(port_count >= 3) {
            auto B = options->inputs.IN_1;
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
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options->inputs.IN_0));

        getLogger() << "[cudnn_frontend] INFO: " << "Building PointwiseNode " << name << " tensors Y:" << std::endl;
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options->outputs.OUT_0));
        
        auto const port_count = get_pointwise_mode_port_count(options->get_mode().value());
        if(port_count >= 3) {
            getLogger() << "[cudnn_frontend] INFO: " << "Building PointwiseNode " << name << " tensors B:" << std::endl;
            CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options->inputs.IN_1));
        }
        if(port_count >= 4) {
            getLogger() << "[cudnn_frontend] INFO: " << "Building PointwiseNode " << name << " tensors T:" << std::endl;
            CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options->inputs.IN_2));
        }

        return error_t::OK;
    }


    error_t createOperations() override final {

        getLogger() << "[cudnn_frontend] INFO: " << "Building PointwiseNode operations..." << std::endl;
        
        #ifndef NV_CUDNN_DISABLE_EXCEPTION
        try {
        #endif

        auto pointwise_descriptor = cudnn_frontend::PointwiseDescBuilder()
                                                        .setAxis(options->get_axis().value_or(-1))
                                                        .setComputeType(options->get_compute_data_type())
                                                        .setMode(options->get_mode().value())
                                                        .build();

        auto const port_count = get_pointwise_mode_port_count(options->get_mode().value());
        if(port_count == 4) {
            auto pointwise_operation = cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_POINTWISE_DESCRIPTOR)
                                            .setxDesc(*(tensors.at(options->inputs.IN_0->get_uid())))
                                            .setbDesc(*(tensors.at(options->inputs.IN_1->get_uid())))
                                            .settDesc(*(tensors.at(options->inputs.IN_2->get_uid())))
                                            .setyDesc(*(tensors.at(options->outputs.OUT_0->get_uid())))
                                            .setpwDesc(pointwise_descriptor)
                                            .build();
            operations.emplace(name, std::make_shared<Operation_v8>(std::move(pointwise_operation)));
        }
        else if(port_count == 3) {
            if(options->get_mode() == PointwiseMode_t::RELU_BWD) {
                auto pointwise_operation = cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_POINTWISE_DESCRIPTOR)
                                                .setdyDesc(*(tensors.at(options->inputs.IN_0->get_uid())))
                                                .setxDesc(*(tensors.at(options->inputs.IN_1->get_uid())))
                                                .setdxDesc(*(tensors.at(options->outputs.OUT_0->get_uid())))
                                                .setpwDesc(pointwise_descriptor)
                                                .build();
                operations.emplace(name, std::make_shared<Operation_v8>(std::move(pointwise_operation)));
            }
            else {
                auto pointwise_operation = cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_POINTWISE_DESCRIPTOR)
                                                .setxDesc(*(tensors.at(options->inputs.IN_0->get_uid())))
                                                .setbDesc(*(tensors.at(options->inputs.IN_1->get_uid())))
                                                .setyDesc(*(tensors.at(options->outputs.OUT_0->get_uid())))
                                                .setpwDesc(pointwise_descriptor)
                                                .build();
                operations.emplace(name, std::make_shared<Operation_v8>(std::move(pointwise_operation)));
            }
        }
        else if(port_count == 2) {
            auto pointwise_operation = cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_POINTWISE_DESCRIPTOR)
                                            .setxDesc(*(tensors.at(options->inputs.IN_0->get_uid())))
                                            .setyDesc(*(tensors.at(options->outputs.OUT_0->get_uid())))
                                            .setpwDesc(pointwise_descriptor)
                                            .build();
            operations.emplace(name, std::make_shared<Operation_v8>(std::move(pointwise_operation)));
        }

        // Push all real tensors as required for operation execution.
        auto const& tensors_involved_in_operation = {
            options->inputs.IN_0
            , options->inputs.IN_1
            , options->inputs.IN_2
            , options->outputs.OUT_0
        };
        auto& tensors_in_operation = tensors_in_operations[name];
        for(auto const& tensor: tensors_involved_in_operation) {
            if(tensor && tensor->get_is_virtual() == false) {
                tensors_in_operation.emplace_back(tensor->get_uid());
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

} // namespace cudnn_frontend::graph