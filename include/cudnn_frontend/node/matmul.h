#pragma once

#include "../../cudnn_frontend_MatMulDesc.h"
#include "../../cudnn_frontend_Heuristics.h"
#include "../../cudnn_frontend_Logging.h"

#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend::graph {

class MatmulNode : public INode {
    Matmul_attributes attributes;

   public:
    MatmulNode(Matmul_attributes&& attributes_, detail::Context const& context)
        : INode(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::MATMUL;
    }

    error_t
    validate_node() const override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Validating matmul node " << attributes.name << "..." << std::endl;

        CUDNN_FE_VALIDATE_INPUT_TENSOR(Matmul_attributes::input_names::A);
        CUDNN_FE_VALIDATE_INPUT_TENSOR(Matmul_attributes::input_names::B);
        CUDNN_FE_VALIDATE_OUTPUT_TENSOR(Matmul_attributes::output_names::C);

        return {error_code_t::OK, ""};
    }

    error_t
    infer_properties_node() override final {
        getLogger() << "[cudnn_frontend] INFO: Inferrencing properties for matmul node " << attributes.name << "..."
                    << std::endl;

        attributes.fill_from_context(context);
        CHECK_CUDNN_FRONTEND_ERROR(attributes.validate_inputs());

        // Only inferrencing from (A, B) -> C works today.
        auto a_tensor = attributes.inputs[Matmul_attributes::input_names::A];
        auto b_tensor = attributes.inputs[Matmul_attributes::input_names::B];
        auto c_tensor = attributes.outputs[Matmul_attributes::output_names::C];

        auto const a_tensor_dim = a_tensor->get_dim();
        auto const b_tensor_dim = b_tensor->get_dim();
        auto c_tensor_dim       = c_tensor->get_dim();

        // Only infer dims and strides if user did not set them
        if (c_tensor_dim.empty()) {
            c_tensor_dim.resize(a_tensor_dim.size());
            if (a_tensor_dim.size() == 4) {
                c_tensor_dim[0] = a_tensor_dim[0];  // B
                c_tensor_dim[1] = a_tensor_dim[1];  // H
                c_tensor_dim[2] = a_tensor_dim[2];  // M
                c_tensor_dim[3] = b_tensor_dim[3];  // N
            } else {
                c_tensor_dim[0] = a_tensor_dim[0];  // B
                c_tensor_dim[1] = a_tensor_dim[1];  // M
                c_tensor_dim[2] = b_tensor_dim[2];  // N
            }
            c_tensor->set_dim(c_tensor_dim);
        }
        if (c_tensor->get_stride().empty()) {
            auto const& c_dim = c_tensor->get_dim();
            // Default to Col major
            auto const& stride_order = detail::generate_row_major_stride_order(c_dim.size());
            c_tensor->set_stride(detail::generate_stride(c_dim, stride_order));
        }

        return {error_code_t::OK, ""};
    }

    error_t
    create_cudnn_tensors(int64_t& uid,
                         std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors) override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Building MatmulNode tensors " << attributes.name << "..." << std::endl;

        for (auto const& [name, tensor] : attributes.inputs) {
            if (tensor) {
                CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(tensor, uid, tensors));
            }
        }
        for (auto const& [name, tensor] : attributes.outputs) {
            if (tensor) {
                CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(tensor, uid, tensors));
            }
        }

        return {error_code_t::OK, ""};
    }

    error_t
    create_cudnn_operations(
        std::unordered_set<uid_t>& uids_involved_in_operations,
        std::vector<cudnn_frontend::Operation_v8>& operations,
        std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors) override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Building MatmulNode operations " << attributes.name << "..." << std::endl;

#ifndef NV_CUDNN_DISABLE_EXCEPTION
        try {
#endif

            // matmul descriptor
            auto matmul_descriptor =
                cudnn_frontend::MatMulDescBuilder().setComputeType(attributes.compute_data_type).build();

            if (attributes.inputs[Matmul_attributes::input_names::N_override]) {
                // Create the matmul operation.
                auto matmul_operation =
                    cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_MATMUL_DESCRIPTOR)
                        .setaMatDesc(*tensors.at(attributes.inputs[Matmul_attributes::input_names::A]->get_uid()))
                        .setbMatDesc(*tensors.at(attributes.inputs[Matmul_attributes::input_names::B]->get_uid()))
                        .setcMatDesc(*tensors.at(attributes.outputs[Matmul_attributes::output_names::C]->get_uid()))
                        .setmatmulDesc(matmul_descriptor)
                        .setmOverrideDesc(
                            *tensors.at(attributes.inputs[Matmul_attributes::input_names::M_override]->get_uid()))
                        .setnOverrideDesc(
                            *tensors.at(attributes.inputs[Matmul_attributes::input_names::N_override]->get_uid()))
                        .build();

                operations.push_back(std::move(matmul_operation));
            } else if (attributes.inputs[Matmul_attributes::input_names::K_override]) {
                // Create the matmul operation.
                auto matmul_operation =
                    cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_MATMUL_DESCRIPTOR)
                        .setaMatDesc(*tensors.at(attributes.inputs[Matmul_attributes::input_names::A]->get_uid()))
                        .setbMatDesc(*tensors.at(attributes.inputs[Matmul_attributes::input_names::B]->get_uid()))
                        .setcMatDesc(*tensors.at(attributes.outputs[Matmul_attributes::output_names::C]->get_uid()))
                        .setmatmulDesc(matmul_descriptor)
                        .setmOverrideDesc(
                            *tensors.at(attributes.inputs[Matmul_attributes::input_names::M_override]->get_uid()))
                        .setkOverrideDesc(
                            *tensors.at(attributes.inputs[Matmul_attributes::input_names::K_override]->get_uid()))
                        .build();
                operations.push_back(std::move(matmul_operation));
            } else {
                // Create the matmul operation.
                auto matmul_operation =
                    cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_MATMUL_DESCRIPTOR)
                        .setaMatDesc(*tensors.at(attributes.inputs[Matmul_attributes::input_names::A]->get_uid()))
                        .setbMatDesc(*tensors.at(attributes.inputs[Matmul_attributes::input_names::B]->get_uid()))
                        .setcMatDesc(*tensors.at(attributes.outputs[Matmul_attributes::output_names::C]->get_uid()))
                        .setmatmulDesc(matmul_descriptor)
                        .build();

                operations.push_back(std::move(matmul_operation));
            }

            for (auto const& [name, tensor] : attributes.inputs) {
                if (tensor && tensor->get_is_virtual() == false) {
                    uids_involved_in_operations.insert(tensor->get_uid());
                }
            }
            for (auto const& [name, tensor] : attributes.outputs) {
                if (tensor && tensor->get_is_virtual() == false) {
                    uids_involved_in_operations.insert(tensor->get_uid());
                }
            }

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