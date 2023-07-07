#pragma once

#include <cudnn_frontend_PointWiseDesc.h>
#include <cudnn_frontend_Heuristics.h>
#include <cudnn_frontend_Logging.h>

#include "cudnn_frontend_graph_helpers.h"
#include "cudnn_frontend_node_interface.h"

namespace cudnn_frontend::graph {

class PointwiseNode : public INode {
public:
    Pointwise_attributes options;

    PointwiseNode(std::string const& name, Pointwise_attributes&& options_, detail::Context const& context)  : INode (name, context), options(std::move(options_)) {}
    
    Type getType() override final {
        return Type::POINTWISE;
    }

    error_t infer_properties_node() override final {
        getLogger() << "[cudnn_frontend] INFO: Inferrencing properties for pointwise node named " << name << "." << std::endl;
        
        options.fill_from_context(context);
        
        // Only inferrencing from IN_0 to OUT_0 works today.
        auto in_0_tensor = options.inputs.IN_0;
        auto out_0_tensor = options.outputs.OUT_0;
        
        auto out_0_tensor_dim = out_0_tensor->get_dim();
        if(out_0_tensor_dim.empty()) {
            out_0_tensor->set_dim(in_0_tensor->get_dim()).set_stride(in_0_tensor->get_stride());
        } else {
            if(out_0_tensor_dim.size() != in_0_tensor->get_dim().size()) {
                auto status = error_code_t::SHAPE_DEDUCTION_FAILED;
                std::string message = "[cudnn_frontend] ERROR: Tensor dimensionality mismatch at X and Y ports of " + name;
                return {status, message};
            }
        }

        return {error_code_t::OK, ""};
    }

    error_t validate_node() override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Validating PointwiseNode..." << std::endl;

        // Ensure that ports are matched to tensors in accordance with port count.
        // X and Y should always be present.
        auto X = options.inputs.IN_0;
        if(X == nullptr) {
            auto status = error_code_t::ATTRIBUTE_NOT_SET;
            std::string message = "[cudnn_frontend] ERROR:  X port of pointwise node named " + name + " not mapped to a tensor.";
            return {status, message};
        }

        auto Y = options.outputs.OUT_0;
        if(Y == nullptr) {
            auto status = error_code_t::ATTRIBUTE_NOT_SET;
            std::string message = "[cudnn_frontend] ERROR:  Y port of pointwise node named " + name + " not mapped to a tensor.";
            return {status, message};
        }

        auto const port_count = get_pointwise_mode_port_count(options.get_mode().value());
        if(port_count >= 3) {
            auto B = options.inputs.IN_1;
            if(B == nullptr) {
                auto status = error_code_t::ATTRIBUTE_NOT_SET;
                std::string message = "[cudnn_frontend] ERROR:  B port of pointwise node named " + name + " not mapped to a tensor.";
                return {status, message};
            }
        }

        getLogger() << "[cudnn_frontend] INFO: " << "Validated PointwiseNode." << std::endl;
        return {error_code_t::OK, ""};
    }

    error_t assign_uids_node() override final {
        options.inputs.IN_0->set_uid(ICudnn::create_new_uid());
        if(options.inputs.IN_1)options.inputs.IN_1->set_uid(ICudnn::create_new_uid());
        if(options.inputs.IN_2)options.inputs.IN_2->set_uid(ICudnn::create_new_uid());
        options.outputs.OUT_0->set_uid(ICudnn::create_new_uid());
        return {error_code_t::OK, ""};
    }

    error_t createTensors() override final {

        getLogger() << "[cudnn_frontend] INFO: " << "Building PointwiseNode " << name << " tensors X:" << std::endl;
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.IN_0));
        
        auto const port_count = get_pointwise_mode_port_count(options.get_mode().value());
        if(port_count >= 3) {
            getLogger() << "[cudnn_frontend] INFO: " << "Building PointwiseNode " << name << " tensors B:" << std::endl;
            CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.IN_1));
        }
        if(port_count >= 4) {
            getLogger() << "[cudnn_frontend] INFO: " << "Building PointwiseNode " << name << " tensors T:" << std::endl;
            CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.IN_2));
        }

        getLogger() << "[cudnn_frontend] INFO: " << "Building PointwiseNode " << name << " tensors Y:" << std::endl;
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.outputs.OUT_0));

        return {error_code_t::OK, ""};
    }


    error_t createOperations() override final {

        getLogger() << "[cudnn_frontend] INFO: " << "Building PointwiseNode operations..." << std::endl;
        
        #ifndef NV_CUDNN_DISABLE_EXCEPTION
        try {
        #endif

        auto pointwise_descriptor = cudnn_frontend::PointwiseDescBuilder()
                                                        .setAxis(options.get_axis().value_or(-1))
                                                        .setComputeType(options.get_compute_data_type())
                                                        .setMode(options.get_mode().value())
                                                        .build();

        auto const port_count = get_pointwise_mode_port_count(options.get_mode().value());
        if(port_count == 4) {
            auto pointwise_operation = cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_POINTWISE_DESCRIPTOR)
                                            .setxDesc(*(tensors.at(options.inputs.IN_0->get_uid())))
                                            .setbDesc(*(tensors.at(options.inputs.IN_1->get_uid())))
                                            .settDesc(*(tensors.at(options.inputs.IN_2->get_uid())))
                                            .setyDesc(*(tensors.at(options.outputs.OUT_0->get_uid())))
                                            .setpwDesc(pointwise_descriptor)
                                            .build();
            operations.emplace(name, std::make_shared<Operation_v8>(std::move(pointwise_operation)));
        }
        else if(port_count == 3) {
            if(options.get_mode() == PointwiseMode_t::RELU_BWD) {
                auto pointwise_operation = cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_POINTWISE_DESCRIPTOR)
                                                .setdyDesc(*(tensors.at(options.inputs.IN_0->get_uid())))
                                                .setxDesc(*(tensors.at(options.inputs.IN_1->get_uid())))
                                                .setdxDesc(*(tensors.at(options.outputs.OUT_0->get_uid())))
                                                .setpwDesc(pointwise_descriptor)
                                                .build();
                operations.emplace(name, std::make_shared<Operation_v8>(std::move(pointwise_operation)));
            }
            else {
                auto pointwise_operation = cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_POINTWISE_DESCRIPTOR)
                                                .setxDesc(*(tensors.at(options.inputs.IN_0->get_uid())))
                                                .setbDesc(*(tensors.at(options.inputs.IN_1->get_uid())))
                                                .setyDesc(*(tensors.at(options.outputs.OUT_0->get_uid())))
                                                .setpwDesc(pointwise_descriptor)
                                                .build();
                operations.emplace(name, std::make_shared<Operation_v8>(std::move(pointwise_operation)));
            }
        }
        else if(port_count == 2) {
            auto pointwise_operation = cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_POINTWISE_DESCRIPTOR)
                                            .setxDesc(*(tensors.at(options.inputs.IN_0->get_uid())))
                                            .setyDesc(*(tensors.at(options.outputs.OUT_0->get_uid())))
                                            .setpwDesc(pointwise_descriptor)
                                            .build();
            operations.emplace(name, std::make_shared<Operation_v8>(std::move(pointwise_operation)));
        }

        // Push all real tensors as required for operation execution.
        auto const& tensors_involved_in_operation = {
            options.inputs.IN_0
            , options.inputs.IN_1
            , options.inputs.IN_2
            , options.outputs.OUT_0
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
        
        return {error_code_t::OK, ""};
    }

    error_t createOperationGraphs(cudnnHandle_t) override final {
        return {error_code_t::OK, ""};
    }
    
    virtual void serialize(json& j) const override final {
        j = options;
    }
};

} // namespace cudnn_frontend::graph