#pragma once

#include "../../cudnn_frontend_Rng.h"
#include "../../cudnn_frontend_Logging.h"

#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend::graph {

class RngNode : public INode {
    Rng_attributes attributes;

   public:
    RngNode(Rng_attributes&& attributes_, detail::Context const& context)
        : INode(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::RNG;
    }

    error_t
    validate_node() const override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Validating RngNode " << attributes.name << "..." << std::endl;

        RETURN_CUDNN_FRONTEND_ERROR_IF(
            attributes.outputs.find(Rng_attributes::output_names::Y) == attributes.outputs.end(),
            error_code_t::ATTRIBUTE_NOT_SET,
            "rng output not set.");

        return {error_code_t::OK, ""};
    }

    error_t
    create_cudnn_tensors(int64_t& uid,
                         std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors) override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Building RngNode tensors " << attributes.name << "..." << std::endl;

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
    infer_properties_node() override final {
        getLogger() << "[cudnn_frontend] INFO: Inferrencing properties for rng node " << attributes.name << "..."
                    << std::endl;

        auto y_tensor = attributes.outputs[Rng_attributes::output_names::Y];

        attributes.fill_from_context(context);
        CHECK_CUDNN_FRONTEND_ERROR(attributes.validate_inputs());

        // If user does not set shape and layout of the generated tensor,
        // Get it from node attributes
        // If layout is not set, generate the strides from layout

        if (y_tensor->get_dim().empty() && attributes.get_dim().size()) {
            y_tensor->set_dim(attributes.dim);
        }

        if (y_tensor->get_stride().empty()) {
            if (attributes.get_stride().size()) {
                y_tensor->set_stride(attributes.get_stride());
            } else {
                auto const& y_dim = y_tensor->get_dim();
                // Default to NHWC
                auto const& stride_order = detail::generate_NHWC_stride_order(y_dim.size());
                y_tensor->set_stride(detail::generate_stride(y_dim, stride_order));
            }
        }

        if (y_tensor->get_dim().empty() || y_tensor->get_stride().empty()) {
            return {error_code_t::SHAPE_DEDUCTION_FAILED, "RNG node output shape deduction failed"};
        }

        return {error_code_t::OK, ""};
    }

    error_t
    create_cudnn_operations(
        std::unordered_set<uid_t>& uids_involved_in_operations,
        std::vector<cudnn_frontend::Operation_v8>& operations,
        std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors) override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Building RngNode operations " << attributes.name << "..." << std::endl;

#ifndef NV_CUDNN_DISABLE_EXCEPTION
        try {
#endif

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

            if (attributes.get_distribution() == RngDistribution_t::BERNOULLI) {
                auto rng_descriptor = cudnn_frontend::RngDescBuilder()
                                          .setRngDistribution(attributes.get_distribution())
                                          .setBernoulliDistProbability(attributes.get_bernoulli_probability().value())
                                          .build();

                if (attributes.inputs[Rng_attributes::input_names::Seed]) {
                    auto Rng_operation =
                        cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_RNG_DESCRIPTOR)
                            .setyDesc(*(tensors.at(attributes.outputs[Rng_attributes::output_names::Y]->get_uid())))
                            .setRngDesc(rng_descriptor)
                            .setSeedDesc(*(tensors.at(attributes.inputs[Rng_attributes::input_names::Seed]->get_uid())))
                            .setOffsetDesc(
                                *(tensors.at(attributes.inputs[Rng_attributes::input_names::Offset]->get_uid())))
                            .build();
                    operations.push_back(std::move(Rng_operation));

                } else {
                    auto Rng_operation =
                        cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_RNG_DESCRIPTOR)
                            .setyDesc(*(tensors.at(attributes.outputs[Rng_attributes::output_names::Y]->get_uid())))
                            .setRngDesc(rng_descriptor)
                            .setSeed(attributes.get_seed().value())
                            .build();
                    operations.push_back(std::move(Rng_operation));
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