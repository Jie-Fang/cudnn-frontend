#pragma once

#include <cudnn_frontend_Heuristics.h>
#include <cudnn_frontend_Logging.h>

#include "cudnn_frontend_graph_helpers.h"
#include "cudnn_frontend_node_interface.h"

#include "graphs/cudnn_frontend_node_matmul.h"
#include "graphs/cudnn_frontend_node_softmax.h"

namespace cudnn_frontend::graph {

    class ScaledDotProductAttentionNode : public INode {
        std::shared_ptr<Tensor> P;

        std::shared_ptr<Scaled_dot_product_attention> options;
    public:

        ScaledDotProductAttentionNode(std::string const& name, std::shared_ptr<Scaled_dot_product_attention> const options, int64_t const offset = 1)  : INode (name, offset), options(options) {
            std::shared_ptr<Tensor> last_output;

            // Lower options to bmm1 options
            auto bmm1_options = std::make_shared<Matmul>("bmm1");
            bmm1_options->inputs.A = options->inputs.Q;
            bmm1_options->inputs.B = options->inputs.K;
            bmm1_options->inputs.M_override = options->inputs.SEQ_LEN_Q;
            bmm1_options->inputs.N_override = options->inputs.SEQ_LEN_K;
            last_output = bmm1_options->outputs.C = P = std::make_shared<Tensor>("P"); // A dummy underlying tensor whose properties will be filled in infer_properties()
            bmm1_options->outputs.C->set_is_virtual(true);
            auto bmm1_node = std::make_shared<MatMulNode>(bmm1_options->get_name(), bmm1_options, offset+100);
            sub_nodes.emplace_back(bmm1_node);
            bmm1_node->parent_node = this;
            
            if(options->inputs.Bias) {
                // Lower options to add options
                auto add_options = std::make_shared<Pointwise>("bias");
                auto add_node = std::make_shared<PointwiseNode>(add_options->get_name(), add_options, offset+20);
                sub_nodes.emplace_back(add_node);
                add_node->parent_node = this;
                add_options->set_mode(PointwiseMode_t::ADD);
                add_options->inputs.IN_0 = bmm1_options->outputs.C;
                add_options->inputs.IN_1 = options->inputs.Bias;
                last_output = add_options->outputs.OUT_0 = std::make_shared<Tensor>("after_bias");
                add_options->outputs.OUT_0->set_is_virtual(true);
            }

            // Lower options to softmax options
            auto softmax_options = std::make_shared<Softmax>("softmax");
            softmax_options->inputs.P = last_output;
            // Use tensor provided by Graph when real S
            if(options->get_is_inference()) {
                last_output = softmax_options->outputs.S = std::make_shared<Tensor>("S");
                softmax_options->outputs.S->set_is_virtual(true);
            }
            else {
                last_output = softmax_options->outputs.S = options->outputs.S;
                
                // Requirement by cudnn backend as output is a special swizzled format.
                last_output->set_reordering_type(cudnn_frontend::TensorReordering_t::F16x16);
            }
            auto softmax_node = std::make_shared<SoftmaxNode>(softmax_options->get_name(), softmax_options, offset+200);
            sub_nodes.emplace_back(softmax_node);
            softmax_node->parent_node = this;

            // Lower options to bmm2 options
            auto bmm2_options = std::make_shared<Matmul>("bmm2");
            // // Requirement by cudnn backend to take in bmm2 aType as i/o type.
            last_output->set_data_type(DataType_t::HALF);
            bmm2_options->inputs.A = last_output;
            bmm2_options->inputs.B = options->inputs.V;
            bmm2_options->inputs.M_override = options->inputs.SEQ_LEN_Q;
            bmm2_options->inputs.K_override = options->inputs.SEQ_LEN_K;
            bmm2_options->outputs.C = options->outputs.O;
            auto bmm2_node = std::make_shared<MatMulNode>(bmm2_options->get_name(), bmm2_options, offset + 300);
            sub_nodes.emplace_back(bmm2_node);
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
             .fill_from_context(get_context());
            // Infer dims and strides for output tensor as matmul node has no context of mha
            // TODO: Rethink whether mha node needs to set it?
            options->outputs.O->set_dim({b,h,s_q,d}).set_stride({s_q*h*d,d,h*d,1});

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