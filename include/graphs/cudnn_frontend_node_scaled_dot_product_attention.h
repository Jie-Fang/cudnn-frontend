#pragma once

#include <cudnn_frontend_Heuristics.h>
#include <cudnn_frontend_Logging.h>

#include "cudnn_frontend_graph_helpers.h"
#include "cudnn_frontend_node_interface.h"

#include "graphs/cudnn_frontend_node_matmul.h"
#include "graphs/cudnn_frontend_node_softmax.h"

namespace cudnn_frontend::graph {

    class ScaledDotProductAttentionNode : public INode {
        // TODO: Just storing the virtual tensors here, but look for potentially better ideas?
        // Otherwise this node is not owner of it anymore
        std::shared_ptr<Tensor> P;
        std::shared_ptr<Tensor> S;

        std::shared_ptr<Scaled_dot_product_attention> options;
    public:

        ScaledDotProductAttentionNode(std::string const& name, std::shared_ptr<Scaled_dot_product_attention> const options, int64_t const offset = 1)  : INode (name, offset), options(options) {
            // A dummy underlying tensor whose properties will be filled in infer_properties()
            P = std::make_shared<Tensor>("P");
            S = (options->get_is_inference() ? std::make_shared<Tensor>("S") : options->outputs.S); // Use tensor provided by Graph when real S

            // Lower options to bmm1 options
            auto bmm1_options = std::make_shared<Matmul>("bmm1");
            bmm1_options->inputs.A = options->inputs.Q;
            bmm1_options->inputs.B = options->inputs.K;
            bmm1_options->inputs.M_override = options->inputs.SEQ_LEN_Q;
            bmm1_options->inputs.N_override = options->inputs.SEQ_LEN_K;
            bmm1_options->outputs.C = P;
            auto bmm1_node = std::make_shared<MatMulNode>(bmm1_options->get_name(), bmm1_options, offset+100);
            sub_nodes.emplace(bmm1_options->get_name(), bmm1_node);
            bmm1_node->parent_node = this;

            // Lower options to softmax options
            auto softmax_options = std::make_shared<Softmax>("softmax");
            softmax_options->set_is_inference(options->get_is_inference());
            softmax_options->inputs.P = bmm1_options->outputs.C;
            softmax_options->outputs.S = S;
            auto softmax_node = std::make_shared<SoftmaxNode>(softmax_options->get_name(), softmax_options, offset+200);
            sub_nodes.emplace(softmax_options->get_name(), softmax_node);
            softmax_node->parent_node = this;

            // Lower options to bmm2 options
            auto bmm2_options = std::make_shared<Matmul>("bmm2");
            bmm2_options->inputs.A = softmax_options->outputs.S; // connect them bmms
            bmm2_options->inputs.B = options->inputs.V;
            bmm2_options->inputs.M_override = options->inputs.SEQ_LEN_Q;
            bmm2_options->inputs.K_override = options->inputs.SEQ_LEN_K;
            bmm2_options->outputs.C = options->outputs.O;
            auto bmm2_node = std::make_shared<MatMulNode>(bmm2_options->get_name(), bmm2_options, offset + 300);
            sub_nodes.emplace(bmm2_options->get_name(), bmm2_node);
            bmm2_node->parent_node = this;
        }

        Type getType() override final {
            return Type::COMPOSITE;
        }

        error_t infer_properties() override final {
            getLogger() << "[cudnn_frontend] INFO: Inferrencing properties for Scaled_dot_product_attention node named " << name << "." << std::endl;

            // Merge with ancestor's context
            fill_missing_context();

            options->fill_from_context(get_context());

            // TODO: gather all tensors and assign them uids at once using a counter. So no need to keep uids in properties.
            // But for the time being doing it here manually.
            options->inputs.Q->set_uid(offset + 1);
            options->inputs.K->set_uid(offset + 2);
            options->inputs.V->set_uid(offset + 3);
            options->inputs.SEQ_LEN_Q->set_uid(offset + 4);
            options->inputs.SEQ_LEN_K->set_uid(offset + 5);
            options->outputs.O->set_uid(offset + 6);

            // Fill properties of virtual tensors
            auto const& q_dim = options->inputs.Q->get_dim();
            auto b = q_dim[0];
            auto h = q_dim[1];
            auto s_q = q_dim[2];
            auto d = q_dim[3];
            // P
            auto const& k_dim = options->inputs.K->get_dim();
            auto s_kv = k_dim[3];
            P->set_dim({b, h, s_q, s_kv})
             .set_stride({h * s_q * s_kv, s_q * s_kv, s_kv, 1})
             .set_is_virtual(true)
             .fill_from_context(get_context());
            
            // TODO: Remove once sub_nodes is a vector which is implicitly in sorted order.
            // Only exists to satisfy bmm2 validation
            S->set_dim({b, h, s_q, s_kv})
             .set_stride({h * s_q * s_kv, s_q * s_kv, s_kv, 1});

            // Infer dims and strides for output tensor as matmul node has no context of mha
            // TODO: Rethink whether mha node needs to set it?
            options->outputs.O->set_dim({b,h,s_q,d}).set_stride({s_q*h*d,d,h*d,1});

            // TODO: do away this redundant code by tweaking global infer_properties
            for(auto const& sub_node: sub_nodes) {
                CHECK_CUDNN_FRONTEND_ERROR(sub_node.second->infer_properties());
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