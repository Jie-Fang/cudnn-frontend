#pragma once

#include <cudnn_frontend_Heuristics.h>
#include <cudnn_frontend_Logging.h>

#include "cudnn_frontend_graph_helpers.h"
#include "cudnn_frontend_node_interface.h"

#include "graphs/cudnn_frontend_node_matmul.h"
#include "graphs/cudnn_frontend_node_pointwise.h"
#include "graphs/cudnn_frontend_node_rng.h"
#include "graphs/cudnn_frontend_node_softmax.h"

namespace cudnn_frontend::graph {

    class ScaledDotProductAttentionNode : public INode {
        std::shared_ptr<Tensor> rng_output;
        std::shared_ptr<Tensor> P;
        std::shared_ptr<Tensor> scale;
        std::shared_ptr<Tensor> dropout_scale;
        std::shared_ptr<Tensor> negative_inf;

        Scaled_dot_product_attention options;
    public:

        ScaledDotProductAttentionNode(std::string const& name, Scaled_dot_product_attention&& options_, detail::Context const& context)  : INode (name, context), options(std::move(options_)) {
            options.fill_from_context(get_context());
            
            std::shared_ptr<Tensor> last_output;

            // User does not create tensor for scale k, so create it internally
            // Data type is i/o type
            scale = std::make_shared<Tensor>("scale_k");
            scale->set_dim({1,1,1,1}).set_stride({1,1,1,1}).set_is_pass_by_value(true);
            dropout_scale = std::make_shared<Tensor>("dropout_scale");
            dropout_scale->set_dim({1,1,1,1}).set_stride({1,1,1,1}).set_is_pass_by_value(true);
            negative_inf = std::make_shared<Tensor>("negative_inf");
            negative_inf->set_dim({1,1,1,1}).set_stride({1,1,1,1}).set_is_pass_by_value(true).set_data_type(DataType_t::FLOAT);
            
            // Lower options to scale options
            auto scale_options = Pointwise("scale_k");
            scale_options.set_mode(PointwiseMode_t::MUL);
            scale_options.inputs.IN_0 = options.inputs.K;
            scale_options.inputs.IN_1 = scale;
            last_output = scale_options.outputs.OUT_0 = std::make_shared<Tensor>("after_scale_k");
            scale_options.outputs.OUT_0->set_is_virtual(true);
            auto scale_node = std::make_unique<PointwiseNode>(scale_options.get_name(), std::move(scale_options), get_context());
            sub_nodes.emplace_back(std::move(scale_node));

            // Lower options to bmm1 options
            auto bmm1_options = Matmul("bmm1");
            bmm1_options.inputs.A = options.inputs.Q;
            // Requirement by cudnn backend to take in bmm1 bType as i/o type.
            last_output->set_data_type(DataType_t::HALF);
            bmm1_options.inputs.B = last_output;
            bmm1_options.inputs.M_override = options.inputs.SEQ_LEN_Q;
            bmm1_options.inputs.N_override = options.inputs.SEQ_LEN_K;
            last_output = bmm1_options.outputs.C = P = std::make_shared<Tensor>("P"); // A dummy underlying tensor whose properties will be filled in infer_properties()
            bmm1_options.outputs.C->set_is_virtual(true);
            auto bmm1_node = std::make_unique<MatmulNode>(bmm1_options.get_name(), std::move(bmm1_options), get_context());
            sub_nodes.emplace_back(std::move(bmm1_node));
            
            if(options.inputs.Bias) {
                // Lower options to add options
                Pointwise add_options("bias");
                add_options.set_mode(PointwiseMode_t::ADD);
                add_options.inputs.IN_0 = last_output;
                add_options.inputs.IN_1 = options.inputs.Bias;
                last_output = add_options.outputs.OUT_0 = std::make_shared<Tensor>("after_bias");
                add_options.outputs.OUT_0->set_is_virtual(true);
                auto add_node = std::make_unique<PointwiseNode>(add_options.get_name(), std::move(add_options), get_context());
                sub_nodes.emplace_back(std::move(add_node));
            }

            if(options.get_padding_masking()) {
                // Lower options to generate row index options
                Pointwise row_index_options("row_index");
                row_index_options.set_mode(PointwiseMode_t::GEN_INDEX).set_axis(2);
                row_index_options.inputs.IN_0 = last_output;
                auto row_index = row_index_options.outputs.OUT_0 = std::make_shared<Tensor>("row_index");
                row_index_options.outputs.OUT_0->set_is_virtual(true);
                auto row_index_node = std::make_unique<PointwiseNode>(row_index_options.get_name(), std::move(row_index_options), get_context());
                sub_nodes.emplace_back(std::move(row_index_node));

                // Lower options to generate col index options
                Pointwise col_index_options("col_index");
                col_index_options.set_mode(PointwiseMode_t::GEN_INDEX).set_axis(3);
                col_index_options.inputs.IN_0 = last_output;
                auto col_index = col_index_options.outputs.OUT_0 = std::make_shared<Tensor>("col_index");
                col_index_options.outputs.OUT_0->set_is_virtual(true);
                auto col_index_node = std::make_unique<PointwiseNode>(col_index_options.get_name(), std::move(col_index_options), get_context());
                sub_nodes.emplace_back(std::move(col_index_node));

                // Lower options to less than row options
                Pointwise less_than_row_options("less_than_row");
                less_than_row_options.set_mode(PointwiseMode_t::CMP_LT);
                less_than_row_options.inputs.IN_0 = row_index;
                less_than_row_options.inputs.IN_1 = options.inputs.SEQ_LEN_Q;
                auto less_than_row = less_than_row_options.outputs.OUT_0 = std::make_shared<Tensor>("less_than_row");
                less_than_row_options.outputs.OUT_0->set_is_virtual(true);
                auto less_than_row_node = std::make_unique<PointwiseNode>(less_than_row_options.get_name(), std::move(less_than_row_options), get_context());
                sub_nodes.emplace_back(std::move(less_than_row_node));
                
                // Lower options to less than col options
                Pointwise less_than_col_options("less_than_col");
                less_than_col_options.set_mode(PointwiseMode_t::CMP_LT);
                less_than_col_options.inputs.IN_0 = col_index;
                less_than_col_options.inputs.IN_1 = options.inputs.SEQ_LEN_K;
                auto less_than_col = less_than_col_options.outputs.OUT_0 = std::make_shared<Tensor>("less_than_col");
                less_than_col_options.outputs.OUT_0->set_is_virtual(true);
                auto less_than_col_node = std::make_unique<PointwiseNode>(less_than_col_options.get_name(), std::move(less_than_col_options), get_context());
                sub_nodes.emplace_back(std::move(less_than_col_node));

                // Lower options to logical and options
                Pointwise logical_and_options("padding_logical_and");
                logical_and_options.set_mode(PointwiseMode_t::LOGICAL_AND).set_compute_data_type(DataType_t::BOOLEAN);
                logical_and_options.inputs.IN_0 = less_than_row;
                logical_and_options.inputs.IN_1 = less_than_col;
                auto mask = logical_and_options.outputs.OUT_0 = std::make_shared<Tensor>("padding_logical_and");
                logical_and_options.outputs.OUT_0->set_is_virtual(true);
                auto logical_and_node = std::make_unique<PointwiseNode>(logical_and_options.get_name(), std::move(logical_and_options), get_context());
                sub_nodes.emplace_back(std::move(logical_and_node));

                if(options.get_causal_masking()) {
                    // Lower options to greater than options
                    Pointwise greater_than_options("greater_than");
                    greater_than_options.set_mode(PointwiseMode_t::CMP_GE);
                    greater_than_options.inputs.IN_0 = row_index;
                    greater_than_options.inputs.IN_1 = col_index;
                    auto row_greater_col = greater_than_options.outputs.OUT_0 = std::make_shared<Tensor>("greater_than");
                    greater_than_options.outputs.OUT_0->set_is_virtual(true);
                    auto greater_than_node = std::make_unique<PointwiseNode>(greater_than_options.get_name(), std::move(greater_than_options), get_context());
                    sub_nodes.emplace_back(std::move(greater_than_node));

                    // Lower options to logical and options
                    Pointwise logical_and_options("causal_logical_and");
                    logical_and_options.set_mode(PointwiseMode_t::LOGICAL_AND).set_compute_data_type(DataType_t::BOOLEAN);
                    logical_and_options.inputs.IN_0 = mask;
                    logical_and_options.inputs.IN_1 = row_greater_col;
                    mask = logical_and_options.outputs.OUT_0 = std::make_shared<Tensor>("causal_logical_and");
                    logical_and_options.outputs.OUT_0->set_is_virtual(true);
                    auto logical_and_node = std::make_unique<PointwiseNode>(logical_and_options.get_name(), std::move(logical_and_options), get_context());
                    sub_nodes.emplace_back(std::move(logical_and_node));
                }

                // Lower options to binary select options
                Pointwise binary_select_options("binary_select");
                binary_select_options.set_mode(PointwiseMode_t::BINARY_SELECT);
                binary_select_options.inputs.IN_0 = last_output;
                binary_select_options.inputs.IN_1 = negative_inf;
                binary_select_options.inputs.IN_2 = mask;
                last_output = binary_select_options.outputs.OUT_0 = std::make_shared<Tensor>("binary_select");
                binary_select_options.outputs.OUT_0->set_is_virtual(true);
                auto binary_select_node = std::make_unique<PointwiseNode>(binary_select_options.get_name(), std::move(binary_select_options), get_context());
                sub_nodes.emplace_back(std::move(binary_select_node));

            }

            // Lower options to softmax options
            auto softmax_options = Softmax("softmax");
            softmax_options.inputs.P = last_output;
            // Use tensor provided by Graph when real S
            if(options.get_is_inference()) {
                last_output = softmax_options.outputs.S = std::make_shared<Tensor>("S");
                softmax_options.outputs.S->set_is_virtual(true);
                auto softmax_node = std::make_unique<SoftmaxNode>(softmax_options.get_name(), std::move(softmax_options), get_context());
                sub_nodes.emplace_back(std::move(softmax_node));
            }
            else {
                // Two cases for training: dropout present or not
                bool const dropout_present = options.get_dropout_probability().has_value() || options.inputs.Dropout_mask;
                if(dropout_present) {
                    last_output = softmax_options.outputs.S = std::make_shared<Tensor>("S");
                    softmax_options.outputs.S->set_is_virtual(true);
                    auto softmax_node = std::make_unique<SoftmaxNode>(softmax_options.get_name(), std::move(softmax_options), get_context());
                    sub_nodes.emplace_back(std::move(softmax_node));

                    std::shared_ptr<Tensor> mask_output;
                    if(options.get_dropout_probability().has_value()) {
                        // Lower options to rng options
                        auto rng_options = Rng("rng");
                        rng_options.set_distribution(RngDistribution_t::BERNOULLI)
                            .set_seed(options.get_seed())
                            .set_bernoulli_probability(options.get_dropout_probability().value());
                        mask_output = rng_options.outputs.Y = rng_output = std::make_shared<Tensor>("after_rng");
                        rng_options.outputs.Y->set_is_virtual(true);
                        auto rng_node = std::make_unique<RngNode>(rng_options.get_name(), std::move(rng_options), get_context());
                        sub_nodes.emplace_back(std::move(rng_node));
                    }
                    else {
                        mask_output = options.inputs.Dropout_mask;
                    }

                    // Lower options to mask options
                    auto mask_options = Pointwise("mask");
                    mask_options.set_mode(PointwiseMode_t::MUL);
                    mask_options.inputs.IN_0 = last_output;
                    mask_options.inputs.IN_1 = mask_output;
                    last_output = mask_options.outputs.OUT_0 = options.outputs.S;
                    auto mask_node = std::make_unique<PointwiseNode>(mask_options.get_name(), std::move(mask_options), get_context());
                    sub_nodes.emplace_back(std::move(mask_node));
                        
                }
                else {
                    last_output = softmax_options.outputs.S = options.outputs.S;
                    auto softmax_node = std::make_unique<SoftmaxNode>(softmax_options.get_name(), std::move(softmax_options), get_context());
                    sub_nodes.emplace_back(std::move(softmax_node));
                }

                // Requirement by cudnn backend as output is a special swizzled format.
                last_output->set_reordering_type(cudnn_frontend::TensorReordering_t::F16x16);
            }

            // Inference or not, dropout or not, always put a scale.
            // Default value 1.f. Will have no perf impact
            // Lower options to dropout_scale options
            auto dropout_scale_options = Pointwise("dropout_scale");
            dropout_scale_options.set_mode(PointwiseMode_t::MUL);
            dropout_scale_options.inputs.IN_0 = last_output;
            dropout_scale_options.inputs.IN_1 = dropout_scale;
            last_output = dropout_scale_options.outputs.OUT_0 = std::make_shared<Tensor>("after_dropout_scale");
            dropout_scale_options.outputs.OUT_0->set_is_virtual(true);
            auto dropout_scale_node = std::make_unique<PointwiseNode>(dropout_scale_options.get_name(), std::move(dropout_scale_options), get_context());
            sub_nodes.emplace_back(std::move(dropout_scale_node));

            // Lower options to bmm2 options
            auto bmm2_options = Matmul("bmm2");
            // Requirement by cudnn backend to take in bmm2 aType as i/o type.
            last_output->set_data_type(DataType_t::HALF);
            bmm2_options.inputs.A = last_output;
            bmm2_options.inputs.B = options.inputs.V;
            bmm2_options.inputs.M_override = options.inputs.SEQ_LEN_Q;
            bmm2_options.inputs.K_override = options.inputs.SEQ_LEN_K;
            bmm2_options.outputs.C = options.outputs.O;
            auto bmm2_node = std::make_unique<MatmulNode>(bmm2_options.get_name(), std::move(bmm2_options), get_context());
            sub_nodes.emplace_back(std::move(bmm2_node));
        }

        Type getType() override final {
            return Type::COMPOSITE;
        }

        error_t infer_properties() override final {
            getLogger() << "[cudnn_frontend] INFO: Inferrencing properties for Scaled_dot_product_attention node named " << name << "." << std::endl;

            // Fill properties of virtual tensors
            auto const& q_dim = options.inputs.Q->get_dim();
            auto b = q_dim[0];
            auto h = q_dim[1];
            auto s_q = q_dim[2];
            auto d = q_dim[3];
            // P
            auto const& k_dim = options.inputs.K->get_dim();
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
            options.outputs.O->set_dim({b,h,s_q,d}).set_stride({s_q*h*d,d,h*d,1});

            // Compute dropout scale
            if(options.get_dropout_probability().has_value()) {
                auto const p = options.get_dropout_probability().value();
                options.set_dropout_scale(1.f / (1.f - p));
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
            half scale_value = options.get_scale_k();
            tensor_to_pass_by_value.emplace(scale, scale_value);
            
            half dropout_scale_value = options.get_dropout_scale();
            tensor_to_pass_by_value.emplace(dropout_scale, dropout_scale_value);
            
            float negative_inf_value = std::numeric_limits<float>::min();
            tensor_to_pass_by_value.emplace(negative_inf, negative_inf_value);

            return error_t::OK;
        }

    };

} // namespace cudnn_frontend::graph