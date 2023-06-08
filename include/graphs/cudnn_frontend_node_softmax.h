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
        std::shared_ptr<Tensor> MAX;
        std::shared_ptr<Tensor> P_MAX;
        std::shared_ptr<Tensor> E;
        std::shared_ptr<Tensor> SUM;

        Softmax options;
    public:

        SoftmaxNode(std::string const& name, Softmax&& options_, detail::Context const& context)  : INode (name, context), options(std::move(options_)) {
            options.fill_from_context(get_context());
            
            // A dummy/virtual underlying tensor
            MAX = std::make_shared<Tensor>("MAX");
            MAX->set_is_virtual(true);
            P_MAX = std::make_shared<Tensor>("P_MAX");
            P_MAX->set_is_virtual(true);
            E = std::make_shared<Tensor>("E");
            E->set_is_virtual(true);
            SUM = std::make_shared<Tensor>("SUM");
            SUM->set_is_virtual(true);

            // Lower options to max options
            auto max_options = Reduction("max");
            max_options.set_mode(ReductionMode_t::MAX);
            max_options.inputs.X = options.inputs.P;
            max_options.outputs.Y = MAX;
            auto max_node = std::make_unique<ReductionNode>(max_options.get_name(), std::move(max_options), get_context());
            sub_nodes.emplace_back(std::move(max_node));

            // Lower options to sub options
            auto sub_options = Pointwise("sub");
            sub_options.set_mode(PointwiseMode_t::SUB);
            sub_options.inputs.IN_0 = options.inputs.P;
            sub_options.inputs.IN_1 = MAX;
            sub_options.outputs.OUT_0 = P_MAX;
            auto sub_node = std::make_unique<PointwiseNode>(sub_options.get_name(), std::move(sub_options), get_context());
            sub_nodes.emplace_back(std::move(sub_node));

            // Lower options to exp options
            auto exp_options = Pointwise("exp");
            exp_options.set_mode(PointwiseMode_t::EXP);
            exp_options.inputs.IN_0 = P_MAX;
            exp_options.outputs.OUT_0 = E;
            auto exp_node = std::make_unique<PointwiseNode>(exp_options.get_name(), std::move(exp_options), get_context());
            sub_nodes.emplace_back(std::move(exp_node));

            // Lower options to sum options
            auto sum_options = Reduction("sum");
            sum_options.set_mode(ReductionMode_t::ADD);
            sum_options.inputs.X = E;
            sum_options.outputs.Y = SUM;
            auto sum_node = std::make_unique<ReductionNode>(sum_options.get_name(), std::move(sum_options), get_context());
            sub_nodes.emplace_back(std::move(sum_node));

            // Lower options to div options
            auto div_options = Pointwise("div");
            div_options.set_mode(PointwiseMode_t::DIV);
            div_options.inputs.IN_0 = E;
            div_options.inputs.IN_1 = SUM;
            div_options.outputs.OUT_0 = options.outputs.S;
            auto div_node = std::make_unique<PointwiseNode>(div_options.get_name(), std::move(div_options), get_context());
            sub_nodes.emplace_back(std::move(div_node));
        }

        Type getType() override final {
            return Type::COMPOSITE;
        }

        error_t infer_properties() override final {
            getLogger() << "[cudnn_frontend] INFO: Inferrencing properties for Softmax node named " << name << "." << std::endl;

            // Fill properties of virtual tensors
            auto const& p_dim = options.inputs.P->get_dim();
            auto b = p_dim[0];
            auto h = p_dim[1];
            auto s_q = p_dim[2];
            auto s_kv = p_dim[3];
            // MAX
            MAX->set_dim({b, h, s_q, 1})
             .set_stride({h * s_q, s_q, 1, 1})
             .fill_from_context(get_context());
            // P_MAX
            P_MAX->set_dim({b, h, s_q, s_kv})
             .set_stride({h * s_q * s_kv, s_q * s_kv, s_kv, 1})
             .fill_from_context(get_context());
            // E
            E->set_dim({b, h, s_q, s_kv})
             .set_stride({h * s_q * s_kv, s_q * s_kv, s_kv, 1})
             .fill_from_context(get_context());
            // SUM
            SUM->set_dim({b, h, s_q, 1})
             .set_stride({h * s_q, s_q, 1, 1})
             .fill_from_context(get_context());

            // Infer dims and strides for output tensor as matmul node has no context of mha
            options.outputs.S
                ->set_dim({b, h, s_q, s_kv})
                .set_stride({h * s_q * s_kv, s_q * s_kv, s_kv, 1})
                .fill_from_context(get_context());

            // TODO: do away this redundant code by tweaking global infer_properties
            for(auto const& sub_node: sub_nodes) {
                CHECK_CUDNN_FRONTEND_ERROR(sub_node->infer_properties());
            }


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