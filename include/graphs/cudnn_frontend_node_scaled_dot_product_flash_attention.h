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

    class ScaledDotProductFlashAttentionNode : public INode {
        std::shared_ptr<Tensor_attributes> rng_output;
        std::shared_ptr<Tensor_attributes> dropout_scale;
        std::shared_ptr<Tensor_attributes> negative_inf;

    public:
        Scaled_dot_product_flash_attention_attributes options;

        ScaledDotProductFlashAttentionNode(std::string const& name, Scaled_dot_product_flash_attention_attributes&& options_, detail::Context const& context)  : INode (name, context), options(std::move(options_)) {}

        Type getType() override final {
            return Type::COMPOSITE;
        }
        
        error_t validate_node() const override final {
            getLogger() << "[cudnn_frontend] INFO: " << "Validating ScaledDotProductFlashAttentionNode..." << std::endl;

            if(options.is_inference.has_value() == false) {
                auto status = error_code_t::ATTRIBUTE_NOT_SET;
                std::string message = "[cudnn_frontend] ERROR: is_infernece attribute not set.";
                return {status, message};
            }

            if(options.dropout_probability.has_value() && options.dropout_probability.value() == 1) {
                auto status = error_code_t::ATTRIBUTE_NOT_SET;
                std::string message = "[cudnn_frontend] ERROR: Dropout probability cannot be 1 as corresponding scale wont be well formed.";
                return {status, message};
            }

            if(context.get_intermediate_data_type() == DataType_t::NOT_SET) {
                auto status = error_code_t::ATTRIBUTE_NOT_SET;
                std::string message = "[cudnn_frontend] ERROR: Intermediate tensor data type needs to be set as internal tensors require it.";
                return {status, message};
            }

            getLogger() << "[cudnn_frontend] INFO: " << "Validated ScaledDotProductFlashAttentionNode." << std::endl;
            return {error_code_t::OK, ""};
        }

        error_t infer_properties_node() override final {
            getLogger() << "[cudnn_frontend] INFO: Inferrencing properties for Scaled_dot_product_flash_attention node named " << name << "." << std::endl;

            options.fill_from_context(context);

            // Gather dims to fill properties of virtual tensors
            auto const& q_dim = options.inputs.Q->get_dim();
            auto b = q_dim[0];
            auto h = q_dim[1];
            auto s_q = q_dim[2];
            auto d = q_dim[3];
            auto const& k_dim = options.inputs.K->get_dim();
            auto s_kv = k_dim[3];

            std::shared_ptr<Tensor_attributes> last_output;
            
            // Lower options to bmm1 options
            auto bmm1_output = std::make_shared<Tensor_attributes>();
            bmm1_output->set_is_virtual(true)
                // Setting dims and strides as pointwise op wont have knowledge of how to do it for mha.
                .set_dim({b, h, s_q, s_kv})
                .set_stride({h * s_q * s_kv, s_q * s_kv, s_kv, 1});

            auto bmm1_options = Matmul_attributes("bmm1");
            bmm1_options.inputs.A = options.inputs.Q;
            bmm1_options.inputs.B = options.inputs.K;
            last_output = bmm1_options.outputs.C = bmm1_output;
            auto bmm1_node = std::make_unique<MatmulNode>(bmm1_options.get_name(), std::move(bmm1_options), context);
            sub_nodes.emplace_back(std::move(bmm1_node));
            
            // Optional scale
            if(options.inputs.Scale_k) {
                // Lower options to scale options
                auto scale_k_output = std::make_shared<Tensor_attributes>();
                scale_k_output->set_is_virtual(true);

                auto scale_options = Pointwise_attributes("scale_k");
                scale_options.set_mode(PointwiseMode_t::MUL);
                scale_options.inputs.IN_0 = last_output;
                scale_options.inputs.IN_1 = options.inputs.Scale_k;
                last_output = scale_options.outputs.OUT_0 = scale_k_output;
                auto scale_node = std::make_unique<PointwiseNode>(scale_options.get_name(), std::move(scale_options), context);
                sub_nodes.emplace_back(std::move(scale_node));
            }

            if(options.causal_mask) {
                // Lower options to generate row index options
                auto row_index_output = std::make_shared<Tensor_attributes>();
                row_index_output->set_is_virtual(true);

                Pointwise_attributes row_index_options("row_index");
                row_index_options.set_mode(PointwiseMode_t::GEN_INDEX).set_axis(2);
                row_index_options.inputs.IN_0 = last_output;
                row_index_options.outputs.OUT_0 = row_index_output;
                auto row_index_node = std::make_unique<PointwiseNode>(row_index_options.get_name(), std::move(row_index_options), context);
                sub_nodes.emplace_back(std::move(row_index_node));

                // Lower options to generate col index options
                auto col_index_output = std::make_shared<Tensor_attributes>();
                col_index_output->set_is_virtual(true);

                Pointwise_attributes col_index_options("col_index");
                col_index_options.set_mode(PointwiseMode_t::GEN_INDEX).set_axis(3);
                col_index_options.inputs.IN_0 = last_output;
                col_index_options.outputs.OUT_0 = col_index_output;
                auto col_index_node = std::make_unique<PointwiseNode>(col_index_options.get_name(), std::move(col_index_options), context);
                sub_nodes.emplace_back(std::move(col_index_node));

                // Lower options to greater than options
                auto row_greater_than_col_output = std::make_shared<Tensor_attributes>();
                row_greater_than_col_output->set_is_virtual(true)
                    // Hard coding data type
                    .set_data_type(DataType_t::BOOLEAN);

                Pointwise_attributes greater_than_options("greater_than");
                greater_than_options.set_mode(PointwiseMode_t::CMP_GE).set_compute_data_type(DataType_t::BOOLEAN);
                greater_than_options.inputs.IN_0 = row_index_output;
                greater_than_options.inputs.IN_1 = col_index_output;
                greater_than_options.outputs.OUT_0 = row_greater_than_col_output;
                auto greater_than_node = std::make_unique<PointwiseNode>(greater_than_options.get_name(), std::move(greater_than_options), context);
                sub_nodes.emplace_back(std::move(greater_than_node));

                // Lower options to binary select options
                negative_inf = std::make_shared<Tensor_attributes>();
                negative_inf->set_dim({1,1,1,1})
                    .set_stride({1,1,1,1})
                    .set_is_pass_by_value(true)
                    // Hard code data type float as FE itself will place FLOAT_MIN in variant pack later
                    .set_data_type(DataType_t::FLOAT);

                auto causal_mask_output = std::make_shared<Tensor_attributes>();
                causal_mask_output->set_is_virtual(true);

                Pointwise_attributes binary_select_options("binary_select");
                binary_select_options.set_mode(PointwiseMode_t::BINARY_SELECT);
                binary_select_options.inputs.IN_0 = last_output;
                binary_select_options.inputs.IN_1 = negative_inf;
                binary_select_options.inputs.IN_2 = row_greater_than_col_output;
                last_output = binary_select_options.outputs.OUT_0 = causal_mask_output;
                auto binary_select_node = std::make_unique<PointwiseNode>(binary_select_options.get_name(), std::move(binary_select_options), context);
                sub_nodes.emplace_back(std::move(binary_select_node));
            }

            // Lower options to softmax options
            auto softmax_output = std::make_shared<Tensor_attributes>();
            softmax_output->set_is_virtual(true);

            auto softmax_options = Softmax_attributes("softmax");
            softmax_options.use_stats = true; // As this is flash attention
            softmax_options.is_inference = options.is_inference;
            softmax_options.inputs.P = last_output;
            last_output = softmax_options.outputs.S = softmax_output;
            softmax_options.outputs.Stats = options.outputs.Stats;
            auto softmax_node = std::make_unique<SoftmaxNode>(softmax_options.get_name(), std::move(softmax_options), context);
            sub_nodes.emplace_back(std::move(softmax_node));

            // Lower options to rng options
            auto rng_output = std::make_shared<Tensor_attributes>();
            rng_output->set_is_virtual(true)
                // Hard coding dims and strides as rng output can no inputs to infer it from.
                .set_dim({b, h, s_q, s_kv})
                .set_stride({h * s_q * s_kv, s_q * s_kv, s_kv, 1});

            auto rng_options = Rng_attributes("rng");
            rng_options.set_distribution(RngDistribution_t::BERNOULLI)
                .set_bernoulli_probability(options.dropout_probability.value());
            rng_options.inputs.Seed = options.inputs.Seed;
            rng_options.inputs.Offset = options.inputs.Offset;
            rng_options.outputs.Y = rng_output;
            auto rng_node = std::make_unique<RngNode>(rng_options.get_name(), std::move(rng_options), context);
            sub_nodes.emplace_back(std::move(rng_node));

            // Lower options to mask options
            auto dropout_mask_output = std::make_shared<Tensor_attributes>();
            dropout_mask_output->set_is_virtual(true);

            auto mask_options = Pointwise_attributes("mask");
            mask_options.set_mode(PointwiseMode_t::MUL);
            mask_options.inputs.IN_0 = last_output;
            mask_options.inputs.IN_1 = rng_output;
            last_output = mask_options.outputs.OUT_0 = dropout_mask_output;
            auto mask_node = std::make_unique<PointwiseNode>(mask_options.get_name(), std::move(mask_options), context);
            sub_nodes.emplace_back(std::move(mask_node));

            // Inference or not, dropout or not, always put a scale.
            // Default value 1.f. Will have no perf impact
            // Lower options to dropout_scale options
            auto dropout_scale_output = std::make_shared<Tensor_attributes>();
            dropout_scale_output->set_is_virtual(true)
                // Requirement by cudnn backend to take in bmm2 aType as i/o type.
                .set_data_type(options.inputs.Q->get_data_type());

            dropout_scale = std::make_shared<Tensor_attributes>();
            dropout_scale->set_dim({1,1,1,1})
                .set_stride({1,1,1,1})
                .set_is_pass_by_value(true)
                // Hard code data type float as FE itself will place value in variant pack later
                .set_data_type(DataType_t::FLOAT);


            auto dropout_scale_options = Pointwise_attributes("dropout_scale");
            dropout_scale_options.set_mode(PointwiseMode_t::MUL);
            dropout_scale_options.inputs.IN_0 = last_output;
            dropout_scale_options.inputs.IN_1 = dropout_scale;
            last_output = dropout_scale_options.outputs.OUT_0 = dropout_scale_output;
            auto dropout_scale_node = std::make_unique<PointwiseNode>(dropout_scale_options.get_name(), std::move(dropout_scale_options), context);
            sub_nodes.emplace_back(std::move(dropout_scale_node));

            // Lower options to bmm2 options
            auto bmm2_options = Matmul_attributes("bmm2");
            bmm2_options.inputs.A = last_output;
            bmm2_options.inputs.B = options.inputs.V;
            bmm2_options.outputs.C = options.outputs.O;
            auto bmm2_node = std::make_unique<MatmulNode>(bmm2_options.get_name(), std::move(bmm2_options), context);
            sub_nodes.emplace_back(std::move(bmm2_node));

            // Set dims and strides if user did not
            if(options.outputs.O->get_dim().empty()) {
                // TODO: mha node needs to set it?
                options.outputs.O->set_dim({b,h,s_q,d}).set_stride({h*d,d,b*h*d,1});
            }

            return {error_code_t::OK, ""};
        }

        error_t createOperationGraphs(cudnnHandle_t) override final {
            return {error_code_t::OK, ""};
        }
    
        virtual error_t pass_by_value_tensors_(std::unordered_map<std::shared_ptr<Tensor_attributes>, pass_by_values_t>& tensor_to_pass_by_value) override {            
            options.dropout_scale = (1.f / (options.dropout_probability.value()));
            tensor_to_pass_by_value.emplace(dropout_scale, options.dropout_scale);
            
            float negative_inf_value = std::numeric_limits<float>::min();
            tensor_to_pass_by_value.emplace(negative_inf, negative_inf_value);

            return {error_code_t::OK, ""};
        }

    };

} // namespace cudnn_frontend::graph