#pragma once

#include <cudnn_frontend_MatMulDesc.h>
#include <cudnn_frontend_Heuristics.h>
#include <cudnn_frontend_Logging.h>

#include "cudnn_frontend_graph_helpers.h"
#include "cudnn_frontend_node_interface.h"

namespace cudnn_frontend::graph {

    class MatmulNode : public INode {
        Matmul options;
    public:

        MatmulNode(std::string const& name, Matmul&& options_, detail::Context const& context)  : INode (name, context), options(std::move(options_)) {
            options.fill_from_context(get_context());
        }
        
        Type getType() override final {
            return Type::MATMUL;
        }

        error_t infer_properties_node() override final {
            getLogger() << "[cudnn_frontend] INFO: Inferrencing properties for matmul node named " << name << "." << std::endl;

            // Only inferrencing from (A, B) -> C works today.
            auto a_tensor_prop = options.inputs.A;
            auto b_tensor_prop = options.inputs.B;
            auto c_tensor_prop = options.outputs.C;
            
            auto const a_tensor_dim = a_tensor_prop->get_dim();
            auto const b_tensor_dim = b_tensor_prop->get_dim();
            auto c_tensor_dim = c_tensor_prop->get_dim();
            if(a_tensor_dim.size() != b_tensor_dim.size()) {
                auto status = error_t::SHAPE_DEDUCTION_FAILED;
                getLogger() << "[cudnn_frontend] ERROR: " << status << "  Tensor dimensions mismatch at A and B ports of " << name << " " << a_tensor_dim.size() << ":" << b_tensor_dim.size() << "." << std::endl;
                return status;
            }
            
            if(c_tensor_dim.empty()) {
                c_tensor_dim.resize(a_tensor_dim.size());
                c_tensor_dim[0] = a_tensor_dim[0]; // B
                c_tensor_dim[1] = a_tensor_dim[1]; // M
                c_tensor_dim[2] = b_tensor_dim[2]; // N
                c_tensor_prop->set_dim(c_tensor_dim).generateStrides(CUDNN_TENSOR_NHWC);
            } else {
                if(a_tensor_dim.size() != c_tensor_dim.size()) {
                    auto status = error_t::SHAPE_DEDUCTION_FAILED;
                    getLogger() << "[cudnn_frontend] ERROR: " << status << " Tensor dimensionality mismatch at A and C ports of " << name << "." << std::endl;
                    return status;
                }
            }

            return error_t::OK;
        }

        error_t assign_uids_node() override final {
            options.inputs.A->set_uid(ICudnn::create_new_uid());
            options.inputs.B->set_uid(ICudnn::create_new_uid());
            if(options.inputs.M_override)options.inputs.M_override->set_uid(ICudnn::create_new_uid());
            if(options.inputs.N_override)options.inputs.N_override->set_uid(ICudnn::create_new_uid());
            if(options.inputs.K_override)options.inputs.K_override->set_uid(ICudnn::create_new_uid());
            options.outputs.C->set_uid(ICudnn::create_new_uid());
            return error_t::OK;
        }

        error_t createTensors() override final {

            getLogger() << "[cudnn_frontend] INFO: " << "Building MatmulNode tensors at node name " << name << std::endl;

            CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.A));
            CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.B));
            CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.outputs.C));
            
            for(auto const& tensor: {options.inputs.M_override, options.inputs.N_override, options.inputs.K_override}) {
                if(tensor) {
                    CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(tensor));
                }
            }
            getLogger() << "[cudnn_frontend] INFO: " << "Built MatmulNode tensors at node name " << name << std::endl;

            return error_t::OK;
        }
        
        error_t createOperations() override final {

            getLogger() << "[cudnn_frontend] INFO: " << "Building MatmulNode operations for node name " << name << std::endl;
            
            #ifndef NV_CUDNN_DISABLE_EXCEPTION
            try {
            #endif

            // matmul descriptor
            auto matmul_descriptor = cudnn_frontend::MatMulDescBuilder()
                                                            .setComputeType(options.get_compute_data_type())
                                                            .build();

            if(options.inputs.N_override) {
                // Create the matmul operation.
                auto matmul_operation = cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_MATMUL_DESCRIPTOR)
                                                .setaMatDesc(*tensors.at(options.inputs.A->get_uid()))
                                                .setbMatDesc(*tensors.at(options.inputs.B->get_uid()))
                                                .setcMatDesc(*tensors.at(options.outputs.C->get_uid()))
                                                .setmatmulDesc(matmul_descriptor)
                                                .setmOverrideDesc(*tensors.at(options.inputs.M_override->get_uid()))
                                                .setnOverrideDesc(*tensors.at(options.inputs.N_override->get_uid()))
                                                .build();
                operations.emplace(name, std::make_shared<Operation_v8>(std::move(matmul_operation)));
            }
            else if(options.inputs.K_override) {
                // Create the matmul operation.
                auto matmul_operation = cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_MATMUL_DESCRIPTOR)
                                                .setaMatDesc(*tensors.at(options.inputs.A->get_uid()))
                                                .setbMatDesc(*tensors.at(options.inputs.B->get_uid()))
                                                .setcMatDesc(*tensors.at(options.outputs.C->get_uid()))
                                                .setmatmulDesc(matmul_descriptor)
                                                .setmOverrideDesc(*tensors.at(options.inputs.M_override->get_uid()))
                                                .setkOverrideDesc(*tensors.at(options.inputs.K_override->get_uid()))
                                                .build();
                operations.emplace(name, std::make_shared<Operation_v8>(std::move(matmul_operation)));
            }
            else {
                // Create the matmul operation.
                auto matmul_operation = cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_MATMUL_DESCRIPTOR)
                                                .setaMatDesc(*tensors.at(options.inputs.A->get_uid()))
                                                .setbMatDesc(*tensors.at(options.inputs.B->get_uid()))
                                                .setcMatDesc(*tensors.at(options.outputs.C->get_uid()))
                                                .setmatmulDesc(matmul_descriptor)
                                                .build();
                operations.emplace(name, std::make_shared<Operation_v8>(std::move(matmul_operation)));
            }
        
            // Push all real tensors as required for operation execution.
            auto const& tensors_involved_in_operation = {
                options.inputs.A
                , options.inputs.B
                , options.inputs.M_override
                , options.inputs.N_override
                , options.inputs.K_override
                , options.outputs.C
            };
            for(auto const& tensor: tensors_involved_in_operation) {
                if(tensor && tensor->get_is_virtual() == false) {
                    tensors_in_operations[name].emplace_back(tensor->get_uid());
                }
            }

            getLogger() << "[cudnn_frontend] INFO: " << "Built MatmulNode operation for node name " << name << std::endl;

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