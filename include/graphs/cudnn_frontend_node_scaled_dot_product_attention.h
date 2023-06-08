#pragma once

#include <cudnn_frontend_Heuristics.h>
#include <cudnn_frontend_Logging.h>

#include "cudnn_frontend_graph_helpers.h"
#include "cudnn_frontend_node_interface.h"

#include "graphs/cudnn_frontend_node_matmul.h"
#include "graphs/cudnn_frontend_node_softmax.h"

namespace cudnn_frontend::graph {

    class ScaledDotProductAttentionNode : public INode {
        std::shared_ptr<Tensor> rng_output;
        std::shared_ptr<Tensor> P;
        std::shared_ptr<Tensor> scale;
        std::shared_ptr<Tensor> dropout_scale;

        std::shared_ptr<Scaled_dot_product_attention> options;
    public:

        ScaledDotProductAttentionNode(std::string const& name, std::shared_ptr<Scaled_dot_product_attention> const options, detail::Context const& context)  : INode (name, context), options(options) {
            options->fill_from_context(get_context());
            
            std::shared_ptr<Tensor> last_output;

            // User does not create tensor for scale k, so create it internally
            // Data type is i/o type
            scale = std::make_shared<Tensor>("scale_k");
            scale->set_dim({1,1,1,1}).set_stride({1,1,1,1}).set_is_pass_by_value(true);
            dropout_scale = std::make_shared<Tensor>("dropout_scale");
            dropout_scale->set_dim({1,1,1,1}).set_stride({1,1,1,1}).set_is_pass_by_value(true);
            
            // Lower options to scale options
            auto scale_options = std::make_shared<Pointwise>("scale_k");
            scale_options->set_mode(PointwiseMode_t::MUL);
            scale_options->inputs.IN_0 = options->inputs.K;
            scale_options->inputs.IN_1 = scale;
            last_output = scale_options->outputs.OUT_0 = std::make_shared<Tensor>("after_scale_k");
            scale_options->outputs.OUT_0->set_is_virtual(true);
            auto scale_node = std::make_shared<PointwiseNode>(scale_options->get_name(), scale_options, get_context());
            sub_nodes.emplace_back(scale_node);

            // Lower options to bmm1 options
            auto bmm1_options = std::make_shared<Matmul>("bmm1");
            bmm1_options->inputs.A = options->inputs.Q;
            // Requirement by cudnn backend to take in bmm1 bType as i/o type.
            last_output->set_data_type(DataType_t::HALF);
            bmm1_options->inputs.B = last_output;
            bmm1_options->inputs.M_override = options->inputs.SEQ_LEN_Q;
            bmm1_options->inputs.N_override = options->inputs.SEQ_LEN_K;
            last_output = bmm1_options->outputs.C = P = std::make_shared<Tensor>("P"); // A dummy underlying tensor whose properties will be filled in infer_properties()
            bmm1_options->outputs.C->set_is_virtual(true);
            auto bmm1_node = std::make_shared<MatmulNode>(bmm1_options->get_name(), bmm1_options, get_context());
            sub_nodes.emplace_back(bmm1_node);
            
            if(options->inputs.Bias) {
                // Lower options to add options
                auto add_options = std::make_shared<Pointwise>("bias");
                add_options->set_mode(PointwiseMode_t::ADD);
                add_options->inputs.IN_0 = bmm1_options->outputs.C;
                add_options->inputs.IN_1 = options->inputs.Bias;
                last_output = add_options->outputs.OUT_0 = std::make_shared<Tensor>("after_bias");
                add_options->outputs.OUT_0->set_is_virtual(true);
                auto add_node = std::make_shared<PointwiseNode>(add_options->get_name(), add_options, get_context());
                sub_nodes.emplace_back(add_node);
            }

            // Lower options to softmax options
            auto softmax_options = std::make_shared<Softmax>("softmax");
            softmax_options->inputs.P = last_output;
            // Use tensor provided by Graph when real S
            if(options->get_is_inference()) {
                last_output = softmax_options->outputs.S = std::make_shared<Tensor>("S");
                softmax_options->outputs.S->set_is_virtual(true);
                auto softmax_node = std::make_shared<SoftmaxNode>(softmax_options->get_name(), softmax_options, get_context());
                sub_nodes.emplace_back(softmax_node);
            }
            else {
                // Two cases for training: dropout present or not
                bool const dropout_present = options->get_dropout_probability().has_value() || options->inputs.Dropout_mask;
                if(dropout_present) {
                    last_output = softmax_options->outputs.S = std::make_shared<Tensor>("S");
                    softmax_options->outputs.S->set_is_virtual(true);
                    auto softmax_node = std::make_shared<SoftmaxNode>(softmax_options->get_name(), softmax_options, get_context());
                    sub_nodes.emplace_back(softmax_node);

                    if(options->get_dropout_probability().has_value()) {
                        // Lower options to rng options
                        auto rng_options = std::make_shared<Rng>("rng");
                        rng_options->set_distribution(RngDistribution_t::BERNOULLI)
                            .set_seed(options->get_seed())
                            .set_bernoulli_probability(options->get_dropout_probability().value());
                        last_output = rng_options->outputs.Y = rng_output = std::make_shared<Tensor>("after_rng");
                        rng_options->outputs.Y->set_is_virtual(true);
                        auto rng_node = std::make_shared<RngNode>(rng_options->get_name(), rng_options, get_context());
                        sub_nodes.emplace_back(rng_node);
                    }
                    else {
                        last_output = options->inputs.Dropout_mask;
                    }

                    // Lower options to mask options
                    auto mask_options = std::make_shared<Pointwise>("mask");
                    mask_options->set_mode(PointwiseMode_t::MUL);
                    mask_options->inputs.IN_0 = softmax_options->outputs.S;
                    mask_options->inputs.IN_1 = last_output;
                    last_output = mask_options->outputs.OUT_0 = options->outputs.S;
                    auto mask_node = std::make_shared<PointwiseNode>(mask_options->get_name(), mask_options, get_context());
                    sub_nodes.emplace_back(mask_node);
                        
                }
                else {
                    last_output = softmax_options->outputs.S = options->outputs.S;
                    auto softmax_node = std::make_shared<SoftmaxNode>(softmax_options->get_name(), softmax_options, get_context());
                    sub_nodes.emplace_back(softmax_node);
                }

                // Requirement by cudnn backend as output is a special swizzled format.
                last_output->set_reordering_type(cudnn_frontend::TensorReordering_t::F16x16);
            }

            // Inference or not, dropout or not, always put a scale.
            // Default value 1.f. Will have no perf impact
            // Lower options to dropout_scale options
            auto dropout_scale_options = std::make_shared<Pointwise>("dropout_scale");
            dropout_scale_options->set_mode(PointwiseMode_t::MUL);
            dropout_scale_options->inputs.IN_0 = last_output;
            dropout_scale_options->inputs.IN_1 = dropout_scale;
            last_output = dropout_scale_options->outputs.OUT_0 = std::make_shared<Tensor>("after_dropout_scale");
            dropout_scale_options->outputs.OUT_0->set_is_virtual(true);
            auto dropout_scale_node = std::make_shared<PointwiseNode>(dropout_scale_options->get_name(), dropout_scale_options, get_context());
            sub_nodes.emplace_back(dropout_scale_node);

            // Lower options to bmm2 options
            auto bmm2_options = std::make_shared<Matmul>("bmm2");
            // Requirement by cudnn backend to take in bmm2 aType as i/o type.
            last_output->set_data_type(DataType_t::HALF);
            bmm2_options->inputs.A = last_output;
            bmm2_options->inputs.B = options->inputs.V;
            bmm2_options->inputs.M_override = options->inputs.SEQ_LEN_Q;
            bmm2_options->inputs.K_override = options->inputs.SEQ_LEN_K;
            bmm2_options->outputs.C = options->outputs.O;
            auto bmm2_node = std::make_shared<MatmulNode>(bmm2_options->get_name(), bmm2_options, get_context());
            sub_nodes.emplace_back(bmm2_node);
        }

        Type getType() override final {
            return Type::COMPOSITE;
        }

        error_t infer_properties() override final {
            getLogger() << "[cudnn_frontend] INFO: Inferrencing properties for Scaled_dot_product_attention node named " << name << "." << std::endl;

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
             
            // rng_output
            // kickstarting rng Y and subsequant MUL infer_properties
            if(rng_output) {
                rng_output->set_dim({b, h, s_q, s_kv})
                .set_stride({h * s_q * s_kv, s_q * s_kv, s_kv, 1});
            }

            // Infer dims and strides for output tensor as matmul node has no context of mha
            // TODO: Rethink whether mha node needs to set it?
            options->outputs.O->set_dim({b,h,s_q,d}).set_stride({s_q*h*d,d,h*d,1});

            // Compute dropout scale
            if(options->get_dropout_probability().has_value()) {
                auto const p = options->get_dropout_probability().value();
                options->set_dropout_scale(1.f / (1.f - p));
            }

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
    
        virtual error_t pass_by_value_tensors_(std::unordered_map<std::shared_ptr<Tensor>, pass_by_values_t>& tensor_to_pass_by_value) override {
            half scale_value = options->get_scale_k();
            tensor_to_pass_by_value.emplace(scale, scale_value);
            
            half dropout_scale_value = options->get_dropout_scale();
            tensor_to_pass_by_value.emplace(dropout_scale, dropout_scale_value);

            return error_t::OK;
        }

    };

} // namespace cudnn_frontend::graph