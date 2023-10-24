#pragma once

#include "../../cudnn_frontend_Heuristics.h"
#include "../../cudnn_frontend_Logging.h"

#include "../graph_helpers.h"
#include "../node_interface.h"

#include "pointwise.h"
#include "reduction.h"

namespace cudnn_frontend::graph {

class SoftmaxNode : public INode {
   public:
    Softmax_attributes attributes;

    SoftmaxNode(Softmax_attributes&& attributes_, detail::Context const& context)
        : INode(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::COMPOSITE;
    }

    error_t
    validate_node() const override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Validating SoftmaxNode " << attributes.name << "..." << std::endl;

        RETURN_CUDNN_FRONTEND_ERROR_IF(
            attributes.use_stats.has_value() == false, error_code_t::ATTRIBUTE_NOT_SET, "use_stats attribute not set.");
        return {error_code_t::OK, ""};
    }

    error_t
    infer_properties_node() override final {
        getLogger() << "[cudnn_frontend] INFO: Inferrencing properties for Softmax node " << attributes.name << "."
                    << std::endl;

        attributes.fill_from_context(context);

        // Fill properties of virtual tensors
        auto const& p_dim = attributes.inputs[Softmax_attributes::input_names::P]->get_dim();
        auto b            = p_dim[0];
        auto h            = p_dim[1];
        auto s_q          = p_dim[2];

        // Lower attributes to max attributes
        auto max_output = std::make_shared<Tensor_attributes>();
        max_output
            ->set_is_virtual(true)
            // Reduction today has no dim inferencing logic today. Hence, hardcoding output dim here.
            .set_dim({b, h, s_q, 1})
            .set_stride({h * s_q, s_q, 1, 1});

        auto max_attributes = Reduction_attributes();
        max_attributes.set_name("max");
        max_attributes.set_mode(ReductionMode_t::MAX);
        max_attributes.inputs[Reduction_attributes::input_names::X] =
            attributes.inputs[Softmax_attributes::input_names::P];
        max_attributes.outputs[Reduction_attributes::output_names::Y] = max_output;
        auto max_node = std::make_unique<ReductionNode>(std::move(max_attributes), context);
        sub_nodes.emplace_back(std::move(max_node));

        // Lower attributes to sub attributes
        auto sub_output = std::make_shared<Tensor_attributes>();
        sub_output->set_is_virtual(true);

        Pointwise_attributes sub_attributes;
        sub_attributes.set_name("sub");
        sub_attributes.set_mode(PointwiseMode_t::SUB);
        sub_attributes.inputs[Pointwise_attributes::input_names::IN_0] =
            attributes.inputs[Softmax_attributes::input_names::P];
        sub_attributes.inputs[Pointwise_attributes::input_names::IN_1]    = max_output;
        sub_attributes.outputs[Pointwise_attributes::output_names::OUT_0] = sub_output;
        auto sub_node = std::make_unique<PointwiseNode>(std::move(sub_attributes), context);
        sub_nodes.emplace_back(std::move(sub_node));

        // Lower attributes to exp attributes
        auto exp_output = std::make_shared<Tensor_attributes>();
        exp_output->set_is_virtual(true);

        Pointwise_attributes exp_attributes;
        exp_attributes.set_name("exp");
        exp_attributes.set_mode(PointwiseMode_t::EXP);
        exp_attributes.inputs[Pointwise_attributes::input_names::IN_0]    = sub_output;
        exp_attributes.outputs[Pointwise_attributes::output_names::OUT_0] = exp_output;
        auto exp_node = std::make_unique<PointwiseNode>(std::move(exp_attributes), context);
        sub_nodes.emplace_back(std::move(exp_node));

        // Lower attributes to reduce sum attributes
        auto sum_output = std::make_shared<Tensor_attributes>();
        sum_output
            ->set_is_virtual(true)
            // Reduction today has no dim inferencing logic today. Hence, hardcoding output dim here.
            .set_dim({b, h, s_q, 1})
            .set_stride({h * s_q, s_q, 1, 1});

        auto sum_attributes = Reduction_attributes();
        sum_attributes.set_name("sum");
        sum_attributes.set_mode(ReductionMode_t::ADD);
        sum_attributes.inputs[Reduction_attributes::input_names::X]   = exp_output;
        sum_attributes.outputs[Reduction_attributes::output_names::Y] = sum_output;
        auto sum_node = std::make_unique<ReductionNode>(std::move(sum_attributes), context);
        sub_nodes.emplace_back(std::move(sum_node));

        // Another path to add when in flash attention mode.
        if (attributes.use_stats.value()) {
            // Lower attributes to log attributes
            auto log_output = std::make_shared<Tensor_attributes>();
            log_output->set_is_virtual(true);

            auto log_attributes = Pointwise_attributes();
            log_attributes.set_name("log");
            log_attributes.set_mode(PointwiseMode_t::LOG);
            log_attributes.inputs[Pointwise_attributes::input_names::IN_0]    = sum_output;
            log_attributes.outputs[Pointwise_attributes::output_names::OUT_0] = log_output;
            auto log_node = std::make_unique<PointwiseNode>(std::move(log_attributes), context);
            sub_nodes.emplace_back(std::move(log_node));

            // Lower attributes to add attributes
            auto add_attributes = Pointwise_attributes();
            add_attributes.set_name("add");
            add_attributes.set_mode(PointwiseMode_t::ADD);
            add_attributes.inputs[Pointwise_attributes::input_names::IN_0] = max_output;
            add_attributes.inputs[Pointwise_attributes::input_names::IN_1] = log_output;
            add_attributes.outputs[Pointwise_attributes::output_names::OUT_0] =
                attributes.outputs[Softmax_attributes::output_names::Stats];
            auto add_node = std::make_unique<PointwiseNode>(std::move(add_attributes), context);
            sub_nodes.emplace_back(std::move(add_node));
        }

        // Lower attributes to div attributes
        auto div_attributes = Pointwise_attributes();
        div_attributes.set_name("div");
        div_attributes.set_mode(PointwiseMode_t::DIV);
        div_attributes.inputs[Pointwise_attributes::input_names::IN_0] = exp_output;
        div_attributes.inputs[Pointwise_attributes::input_names::IN_1] = sum_output;
        div_attributes.outputs[Pointwise_attributes::output_names::OUT_0] =
            attributes.outputs[Softmax_attributes::output_names::S];
        auto div_node = std::make_unique<PointwiseNode>(std::move(div_attributes), context);
        sub_nodes.emplace_back(std::move(div_node));

        return {error_code_t::OK, ""};
    }
};
}  // namespace cudnn_frontend::graph