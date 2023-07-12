#pragma once

#include <cudnn_frontend_Heuristics.h>
#include <cudnn_frontend_Logging.h>

#include "cudnn_frontend_graph_helpers.h"
#include "cudnn_frontend_node_interface.h"

#include "graphs/cudnn_frontend_node_pointwise.h"
#include "graphs/cudnn_frontend_node_reduction.h"

namespace cudnn_frontend::graph {

    class SoftmaxNode : public INode {
        // TODO: Just storing the virtual tensors here, but look for potentially better ideas?
        // Otherwise this node is not owner of it anymore
        std::shared_ptr<Tensor_attributes> MAX;
        std::shared_ptr<Tensor_attributes> P_MAX;
        std::shared_ptr<Tensor_attributes> E;
        std::shared_ptr<Tensor_attributes> SUM;
        std::shared_ptr<Tensor_attributes> LOG;

    public:
        Softmax_attributes options;

        SoftmaxNode(std::string const& name, Softmax_attributes&& options_, detail::Context const& context)  : INode (name, context), options(std::move(options_)) {            
            // A dummy/virtual underlying tensor
            MAX = std::make_shared<Tensor_attributes>();
            MAX->set_is_virtual(true);
            P_MAX = std::make_shared<Tensor_attributes>();
            P_MAX->set_is_virtual(true);
            E = std::make_shared<Tensor_attributes>();
            E->set_is_virtual(true);
            SUM = std::make_shared<Tensor_attributes>();
            SUM->set_is_virtual(true);
            if(options.use_stats) {
                LOG = std::make_shared<Tensor_attributes>();
                LOG->set_is_virtual(true); 
            }

            // Lower options to max options
            auto max_options = Reduction_attributes("max");
            max_options.set_mode(ReductionMode_t::MAX);
            max_options.inputs.X = options.inputs.P;
            max_options.outputs.Y = MAX;
            auto max_node = std::make_unique<ReductionNode>(max_options.get_name(), std::move(max_options), context);
            sub_nodes.emplace_back(std::move(max_node));

            // Lower options to sub options
            auto sub_options = Pointwise_attributes("sub");
            sub_options.set_mode(PointwiseMode_t::SUB);
            sub_options.inputs.IN_0 = options.inputs.P;
            sub_options.inputs.IN_1 = MAX;
            sub_options.outputs.OUT_0 = P_MAX;
            auto sub_node = std::make_unique<PointwiseNode>(sub_options.get_name(), std::move(sub_options), context);
            sub_nodes.emplace_back(std::move(sub_node));

            // Lower options to exp options
            auto exp_options = Pointwise_attributes("exp");
            exp_options.set_mode(PointwiseMode_t::EXP);
            exp_options.inputs.IN_0 = P_MAX;
            exp_options.outputs.OUT_0 = E;
            auto exp_node = std::make_unique<PointwiseNode>(exp_options.get_name(), std::move(exp_options), context);
            sub_nodes.emplace_back(std::move(exp_node));

            // Lower options to reduce sum options
            auto sum_options = Reduction_attributes("sum");
            sum_options.set_mode(ReductionMode_t::ADD);
            sum_options.inputs.X = E;
            sum_options.outputs.Y = SUM;
            auto sum_node = std::make_unique<ReductionNode>(sum_options.get_name(), std::move(sum_options), context);
            sub_nodes.emplace_back(std::move(sum_node));

            // Another path to add when in flash attention mode.
            if(options.use_stats) {
                // Lower options to log options
                auto log_options = Pointwise_attributes("log");
                log_options.set_mode(PointwiseMode_t::LOG);
                log_options.inputs.IN_0 = SUM;
                log_options.outputs.OUT_0 = LOG;
                auto log_node = std::make_unique<PointwiseNode>(log_options.get_name(), std::move(log_options), context);
                sub_nodes.emplace_back(std::move(log_node));

                // Lower options to add options
                auto add_options = Pointwise_attributes("add_stats");
                add_options.set_mode(PointwiseMode_t::ADD);
                add_options.inputs.IN_0 = MAX;
                add_options.inputs.IN_1 = LOG;
                add_options.outputs.OUT_0 = options.outputs.Stats;
                auto add_node = std::make_unique<PointwiseNode>(add_options.get_name(), std::move(add_options), context);
                sub_nodes.emplace_back(std::move(add_node));
            }

            // Lower options to div options
            auto div_options = Pointwise_attributes("div");
            div_options.set_mode(PointwiseMode_t::DIV);
            div_options.inputs.IN_0 = E;
            div_options.inputs.IN_1 = SUM;
            div_options.outputs.OUT_0 = options.outputs.S;
            // Softmax output only non-virutal when non-flash version and in forward training mode
            div_options.outputs.OUT_0->set_is_virtual((!options.is_inference) || (options.use_stats));
            auto div_node = std::make_unique<PointwiseNode>(div_options.get_name(), std::move(div_options), context);
            sub_nodes.emplace_back(std::move(div_node));
        }

        Type getType() override final {
            return Type::COMPOSITE;
        }

        error_t infer_properties_node() override final {
            getLogger() << "[cudnn_frontend] INFO: Inferrencing properties for Softmax node named " << name << "." << std::endl;
        
            options.fill_from_context(context);

            // Fill properties of virtual tensors
            auto const& p_dim = options.inputs.P->get_dim();
            auto b = p_dim[0];
            auto h = p_dim[1];
            auto s_q = p_dim[2];
            auto s_kv = p_dim[3];
            // MAX
            MAX->set_dim({b, h, s_q, 1})
             .set_stride({h * s_q, s_q, 1, 1})
             .fill_from_context(context);
            // P_MAX
            P_MAX->set_dim({b, h, s_q, s_kv})
             .set_stride({h * s_q * s_kv, s_q * s_kv, s_kv, 1})
             .fill_from_context(context);
            // E
            E->set_dim({b, h, s_q, s_kv})
             .set_stride({h * s_q * s_kv, s_q * s_kv, s_kv, 1})
             .fill_from_context(context);
            // SUM
            SUM->set_dim({b, h, s_q, 1})
             .set_stride({h * s_q, s_q, 1, 1})
             .fill_from_context(context);

            // Infer dims and strides for output tensor as matmul node has no context of mha
            options.outputs.S
                ->set_dim({b, h, s_q, s_kv})
                .set_stride({h * s_q * s_kv, s_q * s_kv, s_kv, 1})
                .fill_from_context(context);

            return {error_code_t::OK, ""};
        }

        error_t createOperationGraphs(cudnnHandle_t) override final {
            return {error_code_t::OK, ""};
        }
    };
} // namespace cudnn_frontend::graph