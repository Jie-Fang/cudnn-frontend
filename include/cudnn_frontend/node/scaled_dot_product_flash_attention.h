#pragma once

#include "../../cudnn_frontend_Heuristics.h"
#include "../../cudnn_frontend_Logging.h"

#include "../graph_helpers.h"
#include "../node_interface.h"

#include "matmul.h"
#include "pointwise.h"
#include "rng.h"
#include "softmax.h"

namespace cudnn_frontend::graph {

class ScaledDotProductFlashAttentionNode : public INode {
    std::shared_ptr<Tensor_attributes> rng_output;
    std::shared_ptr<Tensor_attributes> dropout_scale;
    std::shared_ptr<Tensor_attributes> negative_inf_causal;
    std::shared_ptr<Tensor_attributes> negative_inf_padding;
    std::shared_ptr<Tensor_attributes> alibi_slopes;

   public:
    Scaled_dot_product_flash_attention_attributes attributes;

    ScaledDotProductFlashAttentionNode(Scaled_dot_product_flash_attention_attributes&& attributes_,
                                       detail::Context const& context)
        : INode(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::COMPOSITE;
    }

    error_t
    validate_node() const override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Validating ScaledDotProductFlashAttentionNode " << attributes.name << "..." << std::endl;

        CUDNN_FE_VALIDATE_INPUT_TENSOR(Scaled_dot_product_flash_attention_attributes::input_names::Q);
        CUDNN_FE_VALIDATE_INPUT_TENSOR(Scaled_dot_product_flash_attention_attributes::input_names::K);
        CUDNN_FE_VALIDATE_INPUT_TENSOR(Scaled_dot_product_flash_attention_attributes::input_names::V);

        CUDNN_FE_VALIDATE_OUTPUT_TENSOR(Scaled_dot_product_flash_attention_attributes::output_names::O); 


        #define CUDNN_FE_VALIDATE_STRIDE(port, port_map) \
                { auto const& t    = port_map.find(port); \
                RETURN_CUDNN_FRONTEND_ERROR_IF(t->second->get_stride().back() != 1, error_code_t::GRAPH_NOT_SUPPORTED,\
                     "The stride for the last dimension corresponding to the embedding size per head should be 1 for " + std::string(#port)); }

        CUDNN_FE_VALIDATE_STRIDE(Scaled_dot_product_flash_attention_attributes::input_names::Q, attributes.inputs); 
        CUDNN_FE_VALIDATE_STRIDE(Scaled_dot_product_flash_attention_attributes::input_names::K, attributes.inputs); 
        CUDNN_FE_VALIDATE_STRIDE(Scaled_dot_product_flash_attention_attributes::input_names::V, attributes.inputs); 
        CUDNN_FE_VALIDATE_STRIDE(Scaled_dot_product_flash_attention_attributes::output_names::O, attributes.outputs); 

        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.is_inference.has_value() == false,
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "is_infernece attribute not set");

        auto const& dropout_mask =
            attributes.inputs.find(Scaled_dot_product_flash_attention_attributes::input_names::Dropout_mask);
        bool const has_dropout_mask = (dropout_mask != attributes.inputs.end()) && (dropout_mask->second != nullptr);
        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.dropout_probability.has_value() && has_dropout_mask,
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "Using both, custom dropout mask and internal-mask generation using dropout "
                                       "probability, is ill-formed.");

        RETURN_CUDNN_FRONTEND_ERROR_IF(
            attributes.dropout_probability.has_value() && attributes.dropout_probability.value() == 1.0,
            error_code_t::ATTRIBUTE_NOT_SET,
            "Dropout probability cannot be 1 as corresponding scale wont be well formed.");

        RETURN_CUDNN_FRONTEND_ERROR_IF(context.get_intermediate_data_type() == DataType_t::NOT_SET,
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "Intermediate tensor data type needs to be set as internal tensors require it.");

        auto const& seq_len_q =
            attributes.inputs.find(Scaled_dot_product_flash_attention_attributes::input_names::SEQ_LEN_Q);
        bool const has_seq_len_q = (seq_len_q != attributes.inputs.end()) && (seq_len_q->second != nullptr);

        auto const& seq_len_kv =
            attributes.inputs.find(Scaled_dot_product_flash_attention_attributes::input_names::SEQ_LEN_KV);
        bool const has_seq_len_kv = (seq_len_kv != attributes.inputs.end()) && (seq_len_kv->second != nullptr);
        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.padding_mask && (!has_seq_len_q || !has_seq_len_kv),
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "Padding mask requires seq_len_q and seq_len_kv to be set.");

        RETURN_CUDNN_FRONTEND_ERROR_IF((!attributes.padding_mask) && (has_seq_len_q || has_seq_len_kv),
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "seq_len_q and seq_len_kv needs to be set only if padding mask is enabled.");

        auto const& attn_scale =
            attributes.inputs.find(Scaled_dot_product_flash_attention_attributes::input_names::Attn_scale);
        bool const has_attn_scale = (attn_scale != attributes.inputs.end()) && (attn_scale->second != nullptr);
        RETURN_CUDNN_FRONTEND_ERROR_IF(has_attn_scale && attributes.attn_scale_value.has_value(),
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "attn_scale with tensor and value cannot be set at the same time.");

        return {error_code_t::OK, ""};
    }

    error_t
    infer_properties_node() override final {
        getLogger() << "[cudnn_frontend] INFO: Inferrencing properties for Scaled_dot_product_flash_attention node  "
                    << attributes.name << "..." << std::endl;

        // DO NOT REMOVE
        // input data type is needed for:
        // - aType of bmm2
        // - dropout scale in pre 8.9.3
        attributes.fill_from_context(context);
        CHECK_CUDNN_FRONTEND_ERROR(attributes.validate_inputs());

        // Gather dims to fill properties of virtual tensors
        auto const& q_dim = attributes.inputs[Scaled_dot_product_flash_attention_attributes::input_names::Q]->get_dim();
        auto b            = q_dim[0];
        auto h            = q_dim[1];
        auto s_q          = q_dim[2];
        auto const& k_dim = attributes.inputs[Scaled_dot_product_flash_attention_attributes::input_names::K]->get_dim();
        auto s_kv         = k_dim[2];
        auto const& v_dim = attributes.inputs[Scaled_dot_product_flash_attention_attributes::input_names::V]->get_dim();
        auto d_v          = v_dim[3];

        // cuDNN frontend API attention requires Q, K, V where
        // Q = {b, h, s_q, d_qk}
        // K = {b, h, s_kv, d_qk}
        // V = {b, h, s_kv, d_v}
        // but cuDNN backend API attention requires Q, KT, V
        // Q = {b, h, s_q, d_qk}
        // KT = {b, h, d_qk, s_kv}
        // V = {b, h, s_kv, d_v}
        // So the code below maps the K->KT
        std::vector<int64_t> temp_vec;

        temp_vec = attributes.inputs[Scaled_dot_product_flash_attention_attributes::input_names::K]->get_dim();
        std::swap(temp_vec[2], temp_vec[3]);
        attributes.inputs[Scaled_dot_product_flash_attention_attributes::input_names::K]->set_dim(temp_vec);

        temp_vec = attributes.inputs[Scaled_dot_product_flash_attention_attributes::input_names::K]->get_stride();
        std::swap(temp_vec[2], temp_vec[3]);
        attributes.inputs[Scaled_dot_product_flash_attention_attributes::input_names::K]->set_stride(temp_vec);

        std::shared_ptr<Tensor_attributes> last_output;

        // Lower attributes to bmm1 attributes
        auto bmm1_output = std::make_shared<Tensor_attributes>();
        bmm1_output
            ->set_is_virtual(true)
            // Setting dims and strides as pointwise op wont have knowledge of how to do it for mha.
            .set_dim({b, h, s_q, s_kv})
            .set_stride({h * s_q * s_kv, s_q * s_kv, s_kv, 1});

        Matmul_attributes bmm1_attributes;
        bmm1_attributes.set_name("bmm1");
        bmm1_attributes.inputs[Matmul_attributes::input_names::A] =
            attributes.inputs[Scaled_dot_product_flash_attention_attributes::input_names::Q];
        bmm1_attributes.inputs[Matmul_attributes::input_names::B] =
            attributes.inputs[Scaled_dot_product_flash_attention_attributes::input_names::K];
        bmm1_attributes.inputs[Matmul_attributes::input_names::M_override] =
            attributes.inputs[Scaled_dot_product_flash_attention_attributes::input_names::SEQ_LEN_Q];
        bmm1_attributes.inputs[Matmul_attributes::input_names::N_override] =
            attributes.inputs[Scaled_dot_product_flash_attention_attributes::input_names::SEQ_LEN_KV];
        last_output = bmm1_attributes.outputs[Matmul_attributes::output_names::C] = bmm1_output;
        auto bmm1_node = std::make_unique<MatmulNode>(std::move(bmm1_attributes), context);
        sub_nodes.emplace_back(std::move(bmm1_node));

        // Optional scale
        if (attributes.attn_scale_value.has_value()) {
            attributes.inputs[Scaled_dot_product_flash_attention_attributes::input_names::Attn_scale] =
                std::make_shared<Tensor_attributes>();
            attributes.inputs[Scaled_dot_product_flash_attention_attributes::input_names::Attn_scale]
                ->set_dim({1, 1, 1, 1})
                .set_stride({1, 1, 1, 1})
                .set_data_type(DataType_t::FLOAT)
                .set_is_pass_by_value(true);
        }
        if (attributes.inputs[Scaled_dot_product_flash_attention_attributes::input_names::Attn_scale]) {
            // Lower attributes to scale attributes
            auto attn_scale_output = std::make_shared<Tensor_attributes>();
            attn_scale_output->set_is_virtual(true);

            Pointwise_attributes scale_attributes;
            scale_attributes.set_name("attn_scale");
            scale_attributes.set_mode(PointwiseMode_t::MUL);
            scale_attributes.inputs[Pointwise_attributes::input_names::IN_0] = last_output;
            scale_attributes.inputs[Pointwise_attributes::input_names::IN_1] =
                attributes.inputs[Scaled_dot_product_flash_attention_attributes::input_names::Attn_scale];
            last_output = scale_attributes.outputs[Pointwise_attributes::output_names::OUT_0] = attn_scale_output;
            auto scale_node = std::make_unique<PointwiseNode>(std::move(scale_attributes), context);
            sub_nodes.emplace_back(std::move(scale_node));
        }

        // Optional bias
        if (attributes.inputs[Scaled_dot_product_flash_attention_attributes::input_names::Bias]) {
            // Lower attributes to add attributes
            auto bias_output = std::make_shared<Tensor_attributes>();
            bias_output->set_is_virtual(true);

            Pointwise_attributes add_attributes;
            add_attributes.set_name("bias");
            add_attributes.set_mode(PointwiseMode_t::ADD);
            add_attributes.inputs[Pointwise_attributes::input_names::IN_0] = last_output;
            add_attributes.inputs[Pointwise_attributes::input_names::IN_1] =
                attributes.inputs[Scaled_dot_product_flash_attention_attributes::input_names::Bias];
            last_output = add_attributes.outputs[Pointwise_attributes::output_names::OUT_0] = bias_output;
            auto add_node = std::make_unique<PointwiseNode>(std::move(add_attributes), context);
            sub_nodes.emplace_back(std::move(add_node));
        }

        if (attributes.alibi_mask) {
            // Lower attributes to generate row index attributes
            auto row_index_output = std::make_shared<Tensor_attributes>();
            row_index_output->set_is_virtual(true).set_data_type(DataType_t::INT32);

            Pointwise_attributes row_index_attributes;
            row_index_attributes.set_name("gen_row_index")
                .set_mode(PointwiseMode_t::GEN_INDEX)
                .set_axis(2)
                .set_compute_data_type(DataType_t::INT32);
            row_index_attributes.inputs[Pointwise_attributes::input_names::IN_0]    = last_output;
            row_index_attributes.outputs[Pointwise_attributes::output_names::OUT_0] = row_index_output;
            auto row_index_node = std::make_unique<PointwiseNode>(std::move(row_index_attributes), context);
            sub_nodes.emplace_back(std::move(row_index_node));

            // Lower attributes to generate col index attributes
            auto col_index_output = std::make_shared<Tensor_attributes>();
            col_index_output->set_is_virtual(true).set_data_type(DataType_t::INT32);

            Pointwise_attributes col_index_attributes;
            col_index_attributes.set_name("gen_col_index")
                .set_mode(PointwiseMode_t::GEN_INDEX)
                .set_axis(3)
                .set_compute_data_type(DataType_t::INT32);
            col_index_attributes.inputs[Pointwise_attributes::input_names::IN_0]    = last_output;
            col_index_attributes.outputs[Pointwise_attributes::output_names::OUT_0] = col_index_output;
            auto col_index_node = std::make_unique<PointwiseNode>(std::move(col_index_attributes), context);
            sub_nodes.emplace_back(std::move(col_index_node));

            // Lower attributes to sub attributes
            auto sub_output = std::make_shared<Tensor_attributes>();
            sub_output->set_is_virtual(true).set_data_type(DataType_t::INT32);

            Pointwise_attributes sub_attributes;
            sub_attributes.set_name("sub").set_mode(PointwiseMode_t::SUB).set_compute_data_type(DataType_t::INT32);
            sub_attributes.inputs[Pointwise_attributes::input_names::IN_0]    = col_index_output;
            sub_attributes.inputs[Pointwise_attributes::input_names::IN_1]    = row_index_output;
            sub_attributes.outputs[Pointwise_attributes::output_names::OUT_0] = sub_output;
            auto sub_node = std::make_unique<PointwiseNode>(std::move(sub_attributes), context);
            sub_nodes.emplace_back(std::move(sub_node));

            // Multiply by alibi slope
            alibi_slopes = std::make_shared<Tensor_attributes>();
            alibi_slopes->set_dim({1, h, 1, 1})
                .set_stride({h, 1, 1, 1})
                // Hard code data type float as FE itself will compute and place in variant pack later
                .set_data_type(DataType_t::FLOAT);

            auto alibi_mask = std::make_shared<Tensor_attributes>();
            alibi_mask->set_is_virtual(true);

            Pointwise_attributes mul_attributes;
            mul_attributes.set_name("mul").set_mode(PointwiseMode_t::MUL);
            mul_attributes.inputs[Pointwise_attributes::input_names::IN_0]    = sub_output;
            mul_attributes.inputs[Pointwise_attributes::input_names::IN_1]    = alibi_slopes;
            mul_attributes.outputs[Pointwise_attributes::output_names::OUT_0] = alibi_mask;
            auto mul_node = std::make_unique<PointwiseNode>(std::move(mul_attributes), context);
            sub_nodes.emplace_back(std::move(mul_node));

            // Add alibi_mask
            auto add_output = std::make_shared<Tensor_attributes>();
            add_output->set_is_virtual(true);

            Pointwise_attributes add_attributes;
            add_attributes.set_name("add").set_mode(PointwiseMode_t::ADD);
            add_attributes.inputs[Pointwise_attributes::input_names::IN_0] = last_output;
            add_attributes.inputs[Pointwise_attributes::input_names::IN_1] = alibi_mask;
            last_output = add_attributes.outputs[Pointwise_attributes::output_names::OUT_0] = add_output;
            auto add_node = std::make_unique<PointwiseNode>(std::move(add_attributes), context);
            sub_nodes.emplace_back(std::move(add_node));
        }

        if (attributes.padding_mask) {
            // Lower attributes to generate row index attributes
            auto row_index_output = std::make_shared<Tensor_attributes>();
            row_index_output->set_is_virtual(true).set_data_type(DataType_t::INT32);

            Pointwise_attributes row_index_attributes;
            row_index_attributes.set_name("gen_row_index")
                .set_mode(PointwiseMode_t::GEN_INDEX)
                .set_axis(2)
                .set_compute_data_type(DataType_t::INT32);
            row_index_attributes.inputs[Pointwise_attributes::input_names::IN_0]    = last_output;
            row_index_attributes.outputs[Pointwise_attributes::output_names::OUT_0] = row_index_output;
            auto row_index_node = std::make_unique<PointwiseNode>(std::move(row_index_attributes), context);
            sub_nodes.emplace_back(std::move(row_index_node));

            // Lower attributes to generate col index attributes
            auto col_index_output = std::make_shared<Tensor_attributes>();
            col_index_output->set_is_virtual(true).set_data_type(DataType_t::INT32);

            Pointwise_attributes col_index_attributes;
            col_index_attributes.set_name("gen_col_index")
                .set_mode(PointwiseMode_t::GEN_INDEX)
                .set_axis(3)
                .set_compute_data_type(DataType_t::INT32);
            col_index_attributes.inputs[Pointwise_attributes::input_names::IN_0]    = last_output;
            col_index_attributes.outputs[Pointwise_attributes::output_names::OUT_0] = col_index_output;
            auto col_index_node = std::make_unique<PointwiseNode>(std::move(col_index_attributes), context);
            sub_nodes.emplace_back(std::move(col_index_node));

            // Create operation for row index less than seq_q
            auto row_less_seq_q_output = std::make_shared<Tensor_attributes>();
            row_less_seq_q_output->set_is_virtual(true).set_data_type(DataType_t::INT32);

            Pointwise_attributes row_less_seq_q_attributes;
            row_less_seq_q_attributes.set_name("row_less_seq_q")
                .set_mode(PointwiseMode_t::CMP_LT)
                .set_compute_data_type(DataType_t::INT32);
            row_less_seq_q_attributes.inputs[Pointwise_attributes::input_names::IN_0] = row_index_output;
            row_less_seq_q_attributes.inputs[Pointwise_attributes::input_names::IN_1] =
                attributes.inputs[Scaled_dot_product_flash_attention_attributes::input_names::SEQ_LEN_Q];
            row_less_seq_q_attributes.outputs[Pointwise_attributes::output_names::OUT_0] = row_less_seq_q_output;
            auto row_less_seq_q_node = std::make_unique<PointwiseNode>(std::move(row_less_seq_q_attributes), context);
            sub_nodes.emplace_back(std::move(row_less_seq_q_node));

            // Create operation for col index less than seq_q
            auto col_less_seq_kv_output = std::make_shared<Tensor_attributes>();
            col_less_seq_kv_output->set_is_virtual(true).set_data_type(DataType_t::INT32);

            Pointwise_attributes col_less_seq_kv_attributes;
            col_less_seq_kv_attributes.set_name("col_less_seq_kv")
                .set_mode(PointwiseMode_t::CMP_LT)
                .set_compute_data_type(DataType_t::INT32);
            col_less_seq_kv_attributes.inputs[Pointwise_attributes::input_names::IN_0] = col_index_output;
            col_less_seq_kv_attributes.inputs[Pointwise_attributes::input_names::IN_1] =
                attributes.inputs[Scaled_dot_product_flash_attention_attributes::input_names::SEQ_LEN_KV];
            col_less_seq_kv_attributes.outputs[Pointwise_attributes::output_names::OUT_0] = col_less_seq_kv_output;
            auto col_less_seq_kv_node = std::make_unique<PointwiseNode>(std::move(col_less_seq_kv_attributes), context);
            sub_nodes.emplace_back(std::move(col_less_seq_kv_node));

            // Create operation logical AND that creates padding mask
            auto logical_and_output = std::make_shared<Tensor_attributes>();
            logical_and_output->set_is_virtual(true).set_data_type(DataType_t::BOOLEAN);

            Pointwise_attributes logical_and_attributes;
            logical_and_attributes.set_name("logical_and")
                .set_mode(PointwiseMode_t::LOGICAL_AND)
                .set_compute_data_type(DataType_t::BOOLEAN);
            logical_and_attributes.inputs[Pointwise_attributes::input_names::IN_0]    = row_less_seq_q_output;
            logical_and_attributes.inputs[Pointwise_attributes::input_names::IN_1]    = col_less_seq_kv_output;
            logical_and_attributes.outputs[Pointwise_attributes::output_names::OUT_0] = logical_and_output;
            auto logical_and_node = std::make_unique<PointwiseNode>(std::move(logical_and_attributes), context);
            sub_nodes.emplace_back(std::move(logical_and_node));

            // Lower attributes to binary select attributes
            negative_inf_padding = std::make_shared<Tensor_attributes>();
            negative_inf_padding->set_dim({1, 1, 1, 1})
                .set_stride({1, 1, 1, 1})
                .set_is_pass_by_value(true)
                // Hard code data type float as FE itself will place FLOAT_MIN in variant pack later
                .set_data_type(DataType_t::FLOAT);

            auto padding_mask_output = std::make_shared<Tensor_attributes>();
            padding_mask_output->set_is_virtual(true);

            Pointwise_attributes binary_select_attributes;
            binary_select_attributes.set_name("binary_select");
            binary_select_attributes.set_mode(PointwiseMode_t::BINARY_SELECT);
            binary_select_attributes.inputs[Pointwise_attributes::input_names::IN_0] = last_output;
            binary_select_attributes.inputs[Pointwise_attributes::input_names::IN_1] = negative_inf_padding;
            binary_select_attributes.inputs[Pointwise_attributes::input_names::IN_2] = logical_and_output;
            last_output = binary_select_attributes.outputs[Pointwise_attributes::output_names::OUT_0] =
                padding_mask_output;
            auto binary_select_node = std::make_unique<PointwiseNode>(std::move(binary_select_attributes), context);
            sub_nodes.emplace_back(std::move(binary_select_node));
        }

        if (attributes.causal_mask) {
            // Lower attributes to generate row index attributes
            auto row_index_output = std::make_shared<Tensor_attributes>();
            row_index_output->set_is_virtual(true);

            Pointwise_attributes row_index_attributes;
            row_index_attributes.set_name("gen_row_index");
            row_index_attributes.set_mode(PointwiseMode_t::GEN_INDEX).set_axis(2);
            row_index_attributes.inputs[Pointwise_attributes::input_names::IN_0]    = last_output;
            row_index_attributes.outputs[Pointwise_attributes::output_names::OUT_0] = row_index_output;
            auto row_index_node = std::make_unique<PointwiseNode>(std::move(row_index_attributes), context);
            sub_nodes.emplace_back(std::move(row_index_node));

            // Lower attributes to generate col index attributes
            auto col_index_output = std::make_shared<Tensor_attributes>();
            col_index_output->set_is_virtual(true);

            Pointwise_attributes col_index_attributes;
            col_index_attributes.set_name("gen_col_index");
            col_index_attributes.set_mode(PointwiseMode_t::GEN_INDEX).set_axis(3);
            col_index_attributes.inputs[Pointwise_attributes::input_names::IN_0]    = last_output;
            col_index_attributes.outputs[Pointwise_attributes::output_names::OUT_0] = col_index_output;
            auto col_index_node = std::make_unique<PointwiseNode>(std::move(col_index_attributes), context);
            sub_nodes.emplace_back(std::move(col_index_node));

            // Lower attributes to greater than attributes
            auto row_greater_than_col_output = std::make_shared<Tensor_attributes>();
            row_greater_than_col_output
                ->set_is_virtual(true)
                // Hard coding data type
                .set_data_type(DataType_t::BOOLEAN);

            Pointwise_attributes greater_than_attributes;
            greater_than_attributes.set_name("row_greater_than_col");
            greater_than_attributes.set_mode(PointwiseMode_t::CMP_GE).set_compute_data_type(DataType_t::BOOLEAN);
            greater_than_attributes.inputs[Pointwise_attributes::input_names::IN_0]    = row_index_output;
            greater_than_attributes.inputs[Pointwise_attributes::input_names::IN_1]    = col_index_output;
            greater_than_attributes.outputs[Pointwise_attributes::output_names::OUT_0] = row_greater_than_col_output;
            auto greater_than_node = std::make_unique<PointwiseNode>(std::move(greater_than_attributes), context);
            sub_nodes.emplace_back(std::move(greater_than_node));

            // Lower attributes to binary select attributes
            negative_inf_causal = std::make_shared<Tensor_attributes>();
            negative_inf_causal->set_dim({1, 1, 1, 1})
                .set_stride({1, 1, 1, 1})
                .set_is_pass_by_value(true)
                // Hard code data type float as FE itself will place FLOAT_MIN in variant pack later
                .set_data_type(DataType_t::FLOAT);

            auto causal_mask_output = std::make_shared<Tensor_attributes>();
            causal_mask_output->set_is_virtual(true);

            Pointwise_attributes binary_select_attributes;
            binary_select_attributes.set_name("binary_select");
            binary_select_attributes.set_mode(PointwiseMode_t::BINARY_SELECT);
            binary_select_attributes.inputs[Pointwise_attributes::input_names::IN_0] = last_output;
            binary_select_attributes.inputs[Pointwise_attributes::input_names::IN_1] = negative_inf_causal;
            binary_select_attributes.inputs[Pointwise_attributes::input_names::IN_2] = row_greater_than_col_output;
            last_output = binary_select_attributes.outputs[Pointwise_attributes::output_names::OUT_0] =
                causal_mask_output;
            auto binary_select_node = std::make_unique<PointwiseNode>(std::move(binary_select_attributes), context);
            sub_nodes.emplace_back(std::move(binary_select_node));
        }

        // Lower attributes to softmax attributes
        auto softmax_output = std::make_shared<Tensor_attributes>();
        softmax_output->set_is_virtual(true);

        // Create a virtual output for stats if inference step otherwise output.Stats is already set
        auto softmax_stats = attributes.outputs[Scaled_dot_product_flash_attention_attributes::output_names::Stats];
        if (attributes.is_inference.value() == true) {
            softmax_stats = std::make_shared<Tensor_attributes>();
            softmax_stats->set_is_virtual(true);
        }

        Softmax_attributes softmax_attributes;
        softmax_attributes.set_name("softmax");
        softmax_attributes.use_stats                                  = true;  // As this is flash attention
        softmax_attributes.inputs[Softmax_attributes::input_names::P] = last_output;
        last_output = softmax_attributes.outputs[Softmax_attributes::output_names::S] = softmax_output;
        softmax_attributes.outputs[Softmax_attributes::output_names::Stats]           = softmax_stats;
        auto softmax_node = std::make_unique<SoftmaxNode>(std::move(softmax_attributes), context);
        sub_nodes.emplace_back(std::move(softmax_node));

        // Two cases for training: dropout present or not
        bool dropout_present = false;
        if (attributes.dropout_probability.has_value()) {
            dropout_present = true;
            // Special case: Skip dropout when 0.0 probability. Only do for 8.9.3 and up as rng isn't optional earlier.
            if (cudnnGetVersion() > 8902 && attributes.dropout_probability.value() == 0.0) {
                dropout_present = false;
            }
        } else if (attributes.inputs[Scaled_dot_product_flash_attention_attributes::input_names::Dropout_mask]) {
            dropout_present = true;
        }

        if (dropout_present) {
            // Lower attributes to rng attributes
            auto rng_output = std::make_shared<Tensor_attributes>();
            rng_output
                ->set_is_virtual(true)
                // Hard coding dims and strides as rng output can no inputs to infer it from.
                .set_dim({b, h, s_q, s_kv})
                .set_stride({h * s_q * s_kv, s_q * s_kv, s_kv, 1});

            Rng_attributes rng_attributes;
            rng_attributes.set_name("rng");
            rng_attributes.set_distribution(RngDistribution_t::BERNOULLI)
                .set_bernoulli_probability(1.0 -
                                           attributes.dropout_probability.value());  // As user sets dropout probability
            rng_attributes.inputs[Rng_attributes::input_names::Seed] =
                attributes.inputs[Scaled_dot_product_flash_attention_attributes::input_names::Seed];
            rng_attributes.inputs[Rng_attributes::input_names::Offset] =
                attributes.inputs[Scaled_dot_product_flash_attention_attributes::input_names::Offset];
            rng_attributes.outputs[Rng_attributes::output_names::Y] = rng_output;
            auto rng_node = std::make_unique<RngNode>(std::move(rng_attributes), context);
            sub_nodes.emplace_back(std::move(rng_node));

            // Lower attributes to mask attributes
            auto dropout_mask_output = std::make_shared<Tensor_attributes>();
            dropout_mask_output->set_is_virtual(true);

            Pointwise_attributes mask_attributes;
            mask_attributes.set_name("dropout_mask_mul");
            mask_attributes.set_mode(PointwiseMode_t::MUL);
            mask_attributes.inputs[Pointwise_attributes::input_names::IN_0] = last_output;
            mask_attributes.inputs[Pointwise_attributes::input_names::IN_1] = rng_output;
            last_output = mask_attributes.outputs[Pointwise_attributes::output_names::OUT_0] = dropout_mask_output;
            auto mask_node = std::make_unique<PointwiseNode>(std::move(mask_attributes), context);
            sub_nodes.emplace_back(std::move(mask_node));

            // Lower attributes to dropout_scale attributes
            auto dropout_scale_output = std::make_shared<Tensor_attributes>();
            dropout_scale_output->set_is_virtual(true);

            dropout_scale = std::make_shared<Tensor_attributes>();
            dropout_scale->set_dim({1, 1, 1, 1})
                .set_stride({1, 1, 1, 1})
                .set_is_pass_by_value(true)
// Hard code data type input type as FE itself will place value in variant pack later
#if CUDNN_VERSION < 8903
                .set_data_type(
                    attributes.inputs[Scaled_dot_product_flash_attention_attributes::input_names::Q]->get_data_type());
#else
                .set_data_type(DataType_t::FLOAT);
#endif

            Pointwise_attributes dropout_scale_attributes;
            dropout_scale_attributes.set_name("dropout_scale");
            dropout_scale_attributes.set_mode(PointwiseMode_t::MUL);
            dropout_scale_attributes.inputs[Pointwise_attributes::input_names::IN_0] = last_output;
            dropout_scale_attributes.inputs[Pointwise_attributes::input_names::IN_1] = dropout_scale;
            last_output = dropout_scale_attributes.outputs[Pointwise_attributes::output_names::OUT_0] =
                dropout_scale_output;
            auto dropout_scale_node = std::make_unique<PointwiseNode>(std::move(dropout_scale_attributes), context);
            sub_nodes.emplace_back(std::move(dropout_scale_node));
        }

        // Lower attributes to bmm2 attributes
        // Requirement by cudnn backend to take in bmm2 aType as i/o type.
        last_output->set_data_type(
            attributes.inputs[Scaled_dot_product_flash_attention_attributes::input_names::Q]->get_data_type());

        Matmul_attributes bmm2_attributes;
        bmm2_attributes.set_name("bmm2");
        bmm2_attributes.inputs[Matmul_attributes::input_names::A] = last_output;
        bmm2_attributes.inputs[Matmul_attributes::input_names::B] =
            attributes.inputs[Scaled_dot_product_flash_attention_attributes::input_names::V];
        bmm2_attributes.outputs[Matmul_attributes::output_names::C] =
            attributes.outputs[Scaled_dot_product_flash_attention_attributes::output_names::O];
        bmm2_attributes.inputs[Matmul_attributes::input_names::M_override] =
            attributes.inputs[Scaled_dot_product_flash_attention_attributes::input_names::SEQ_LEN_Q];
        bmm2_attributes.inputs[Matmul_attributes::input_names::K_override] =
            attributes.inputs[Scaled_dot_product_flash_attention_attributes::input_names::SEQ_LEN_KV];
        auto bmm2_node = std::make_unique<MatmulNode>(std::move(bmm2_attributes), context);
        sub_nodes.emplace_back(std::move(bmm2_node));

        // Set dims if user did not
        if (attributes.outputs[Scaled_dot_product_flash_attention_attributes::output_names::O]->get_dim().empty()) {
            attributes.outputs[Scaled_dot_product_flash_attention_attributes::output_names::O]->set_dim(
                {b, h, s_q, d_v});
        }
        if (attributes.outputs[Scaled_dot_product_flash_attention_attributes::output_names::O]->get_stride().empty()) {
            auto const O_dim =
                attributes.outputs[Scaled_dot_product_flash_attention_attributes::output_names::O]->get_dim();
            attributes.outputs[Scaled_dot_product_flash_attention_attributes::output_names::O]->set_stride(
                {O_dim[3] * O_dim[2] * O_dim[1], O_dim[3] * O_dim[2], O_dim[3], 1});
        }

        return {error_code_t::OK, ""};
    }

    virtual int64_t
    get_fe_workspace_size_node() const override final {
        auto const& q   = attributes.inputs.find(Scaled_dot_product_flash_attention_attributes::input_names::Q);
        int64_t const h = q->second->get_dim()[1];
        return h * sizeof(float);
    }

    virtual error_t
    pass_by_value_tensors_(
        cudnnHandle_t handle,
        std::unordered_map<std::shared_ptr<Tensor_attributes>, pass_by_values_t>& tensor_to_pass_by_value,
        void* node_workspace) override {
        if (attributes.dropout_probability.has_value() && attributes.dropout_probability.value() != 0.0) {
#if CUDNN_VERSION < 8903
            half dropout_scale_value = (1.0f / (1.0f - attributes.dropout_probability.value()));
#else
            float dropout_scale_value = (1.0f / (1.0f - attributes.dropout_probability.value()));
#endif
            tensor_to_pass_by_value.emplace(dropout_scale, dropout_scale_value);
        }

        if (attributes.padding_mask) {
            float negative_inf_value = std::numeric_limits<float>::lowest();
            tensor_to_pass_by_value.emplace(negative_inf_padding, negative_inf_value);
        }

        if (attributes.causal_mask) {
            float negative_inf_value = std::numeric_limits<float>::lowest();
            tensor_to_pass_by_value.emplace(negative_inf_causal, negative_inf_value);
        }

        if (attributes.alibi_mask) {
            int64_t const h =
                attributes.inputs[Scaled_dot_product_flash_attention_attributes::input_names::Q]->get_dim()[1];
            auto h_alibi_slopes_vector = detail::get_abili_slope(h);

            cudaStream_t stream;
            CHECK_CUDNN_ERROR(cudnnGetStream(handle, &stream));
            CHECK_CUDA_ERROR(cudaMemcpyAsync(
                node_workspace, h_alibi_slopes_vector.data(), h * sizeof(float), cudaMemcpyHostToDevice, stream));
            tensor_to_pass_by_value.emplace(alibi_slopes, node_workspace);
        }

        if (attributes.attn_scale_value.has_value()) {
            tensor_to_pass_by_value.emplace(
                attributes.inputs[Scaled_dot_product_flash_attention_attributes::input_names::Attn_scale],
                attributes.attn_scale_value.value());
        }

        return {error_code_t::OK, ""};
    }
};

class ScaledDotProductFlashAttentionBackwardNode : public INode {
   private:
    // non-virtual node cpu tensors
    std::shared_ptr<Tensor_attributes> one_tensor;
    std::shared_ptr<Tensor_attributes> negative_inf_padding;
    std::shared_ptr<Tensor_attributes> negative_inf_causal;

    // non-virtual node gpu tensors
    std::shared_ptr<Tensor_attributes> dQ_accum;
    int64_t dQ_accum_size = 0;
    std::shared_ptr<Tensor_attributes> softmax_sum;
    int64_t softmax_sum_size = 0;
    std::shared_ptr<Tensor_attributes> alibi_slopes;
    int64_t alibi_slopes_size = 0;

   public:
    Scaled_dot_product_flash_attention_backward_attributes attributes;

    ScaledDotProductFlashAttentionBackwardNode(Scaled_dot_product_flash_attention_backward_attributes&& attributes_,
                                               detail::Context const& context)
        : INode(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::COMPOSITE;
    }

    error_t
    validate_node() const override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Validating ScaledDotProductFlashAttentionBackwardNode" << attributes.name << "..." << std::endl;

        auto const& q = attributes.inputs.find(Scaled_dot_product_flash_attention_backward_attributes::input_names::Q);
        bool const has_q = (q != attributes.inputs.end()) && (q->second != nullptr);
        RETURN_CUDNN_FRONTEND_ERROR_IF(!has_q, error_code_t::ATTRIBUTE_NOT_SET, "Tensor input q not set");
        auto const& k = attributes.inputs.find(Scaled_dot_product_flash_attention_backward_attributes::input_names::K);
        bool const has_k = (k != attributes.inputs.end()) && (k->second != nullptr);
        RETURN_CUDNN_FRONTEND_ERROR_IF(!has_k, error_code_t::ATTRIBUTE_NOT_SET, "Tensor input k not set");
        auto const& v = attributes.inputs.find(Scaled_dot_product_flash_attention_backward_attributes::input_names::V);
        bool const has_v = (v != attributes.inputs.end()) && (v->second != nullptr);
        RETURN_CUDNN_FRONTEND_ERROR_IF(!has_v, error_code_t::ATTRIBUTE_NOT_SET, "Tensor input v not set");
        auto const& o = attributes.inputs.find(Scaled_dot_product_flash_attention_backward_attributes::input_names::O);
        bool const has_o = (o != attributes.inputs.end()) && (o->second != nullptr);
        RETURN_CUDNN_FRONTEND_ERROR_IF(!has_o, error_code_t::ATTRIBUTE_NOT_SET, "Tensor input o not set");
        auto const& dO =
            attributes.inputs.find(Scaled_dot_product_flash_attention_backward_attributes::input_names::dO);
        bool const has_dO = (dO != attributes.inputs.end()) && (dO->second != nullptr);
        RETURN_CUDNN_FRONTEND_ERROR_IF(!has_dO, error_code_t::ATTRIBUTE_NOT_SET, "Tensor input dO not set");
        auto const& Stats =
            attributes.inputs.find(Scaled_dot_product_flash_attention_backward_attributes::input_names::Stats);
        bool const has_Stats = (Stats != attributes.inputs.end()) && (Stats->second != nullptr);
        RETURN_CUDNN_FRONTEND_ERROR_IF(!has_Stats, error_code_t::ATTRIBUTE_NOT_SET, "Tensor input Stats not set");
        auto const& dq =
            attributes.outputs.find(Scaled_dot_product_flash_attention_backward_attributes::output_names::dQ);
        bool const has_dq = (dq != attributes.outputs.end()) && (dq->second != nullptr);
        RETURN_CUDNN_FRONTEND_ERROR_IF(!has_dq, error_code_t::ATTRIBUTE_NOT_SET, "Tensor input dq not set");
        auto const& dk =
            attributes.outputs.find(Scaled_dot_product_flash_attention_backward_attributes::output_names::dK);
        bool const has_dk = (dk != attributes.outputs.end()) && (dk->second != nullptr);
        RETURN_CUDNN_FRONTEND_ERROR_IF(!has_dk, error_code_t::ATTRIBUTE_NOT_SET, "Tensor input dk not set");
        auto const& dv =
            attributes.outputs.find(Scaled_dot_product_flash_attention_backward_attributes::output_names::dV);
        bool const has_dv = (dv != attributes.outputs.end()) && (dv->second != nullptr);
        RETURN_CUDNN_FRONTEND_ERROR_IF(!has_dv, error_code_t::ATTRIBUTE_NOT_SET, "Tensor input dv not set");

        RETURN_CUDNN_FRONTEND_ERROR_IF(q->second->get_stride().back() != 1 || k->second->get_stride().back() != 1 ||
                                           v->second->get_stride().back() != 1 || o->second->get_stride().back() != 1 ||
                                           dq->second->get_stride().back() != 1 ||
                                           dv->second->get_stride().back() != 1 ||
                                           dk->second->get_stride().back() != 1 || dO->second->get_stride().back() != 1,
                                       error_code_t::GRAPH_NOT_SUPPORTED,
                                       "The stride for the last dimension corresponding to the hidden size per head"
                                       " should be 1");

        auto const& dropout_mask =
            attributes.inputs.find(Scaled_dot_product_flash_attention_backward_attributes::input_names::Dropout_mask);
        bool const has_dropout_mask = (dropout_mask != attributes.inputs.end()) && (dropout_mask->second != nullptr);
        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.dropout_probability.has_value() && has_dropout_mask,
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "Using both, custom dropout mask and internal-mask generation using dropout "
                                       "probability, is ill-formed.");

        RETURN_CUDNN_FRONTEND_ERROR_IF(
            attributes.dropout_probability.has_value() && attributes.dropout_probability.value() == 1.0,
            error_code_t::ATTRIBUTE_NOT_SET,
            "Dropout probability cannot be 1 as corresponding scale wont be well formed.");

        auto const& seq_len_q =
            attributes.inputs.find(Scaled_dot_product_flash_attention_backward_attributes::input_names::SEQ_LEN_Q);
        bool const has_seq_len_q = (seq_len_q != attributes.inputs.end()) && (seq_len_q->second != nullptr);

        auto const& seq_len_kv =
            attributes.inputs.find(Scaled_dot_product_flash_attention_backward_attributes::input_names::SEQ_LEN_KV);
        bool const has_seq_len_kv = (seq_len_kv != attributes.inputs.end()) && (seq_len_kv->second != nullptr);

        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.padding_mask && (!has_seq_len_q || !has_seq_len_kv),
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "Padding mask requires seq_len_q and seq_len_kv to be set.");

        RETURN_CUDNN_FRONTEND_ERROR_IF((!attributes.padding_mask) && (has_seq_len_q || has_seq_len_kv),
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "seq_len_q and seq_len_kv needs to be set only if padding mask is enabled.");

        auto const& attn_scale =
            attributes.inputs.find(Scaled_dot_product_flash_attention_backward_attributes::input_names::Attn_scale);
        bool const has_attn_scale = (attn_scale != attributes.inputs.end()) && (attn_scale->second != nullptr);
        RETURN_CUDNN_FRONTEND_ERROR_IF(has_attn_scale && attributes.attn_scale_value.has_value(),
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "attn_scale with tensor and value cannot be set at the same time.");

        RETURN_CUDNN_FRONTEND_ERROR_IF(context.get_intermediate_data_type() == DataType_t::NOT_SET,
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "Intermediate tensor data type needs to be set as internal tensors require it.");

        return {error_code_t::OK, ""};
    }

    error_t
    infer_properties_node() override final {
        getLogger() << "[cudnn_frontend] INFO: Inferrencing properties for ScaledDotProductFlashAttentionBackwardNode "
                    << attributes.name << "..." << std::endl;

        attributes.fill_from_context(context);
        CHECK_CUDNN_FRONTEND_ERROR(attributes.validate_inputs());

        // Gather dims to fill properties of virtual tensors
        auto const& q_dim =
            attributes.inputs[Scaled_dot_product_flash_attention_backward_attributes::input_names::Q]->get_dim();
        auto b   = q_dim[0];
        auto h   = q_dim[1];
        auto s_q = q_dim[2];
        auto d   = q_dim[3];
        auto const& k_dim =
            attributes.inputs[Scaled_dot_product_flash_attention_backward_attributes::input_names::K]->get_dim();
        auto s_kv = k_dim[2];

        // cuDNN frontend API attention requires Q, K, V where
        // Q = {b, h, s_q, d}
        // K = {b, h, s_kv, d}
        // V = {b, h, s_kv, d}
        // but cuDNN backend API attention requires Q, KT, VT
        // Q = {b, h, s_q, d}
        // KT = {b, h, d, s_kv}
        // VT = {b, h, d, s_kv}
        // So the code below maps the K->KT and V->VT
        std::vector<int64_t> temp_vec;

        temp_vec = attributes.inputs[Scaled_dot_product_flash_attention_backward_attributes::input_names::K]->get_dim();
        std::swap(temp_vec[2], temp_vec[3]);
        attributes.inputs[Scaled_dot_product_flash_attention_backward_attributes::input_names::K]->set_dim(temp_vec);

        temp_vec =
            attributes.inputs[Scaled_dot_product_flash_attention_backward_attributes::input_names::K]->get_stride();
        std::swap(temp_vec[2], temp_vec[3]);
        attributes.inputs[Scaled_dot_product_flash_attention_backward_attributes::input_names::K]->set_stride(temp_vec);

        temp_vec = attributes.inputs[Scaled_dot_product_flash_attention_backward_attributes::input_names::V]->get_dim();
        std::swap(temp_vec[2], temp_vec[3]);
        attributes.inputs[Scaled_dot_product_flash_attention_backward_attributes::input_names::V]->set_dim(temp_vec);

        temp_vec =
            attributes.inputs[Scaled_dot_product_flash_attention_backward_attributes::input_names::V]->get_stride();
        std::swap(temp_vec[2], temp_vec[3]);
        attributes.inputs[Scaled_dot_product_flash_attention_backward_attributes::input_names::V]->set_stride(temp_vec);

        std::shared_ptr<Tensor_attributes> last_output, exp_softmax_output, dp_scaled_output, rng_output;

        // --------------Initialize and create tensors before creating nodes--------------------

        // one_tensor is needed for non-dropout graphs
        // one_tensor is passed by the node
        one_tensor = make_tensor_(false, {1, 1, 1, 1});
        one_tensor->set_is_pass_by_value(true).set_data_type(DataType_t::FLOAT);

        // alibi_slopes is passed by the node
        if (attributes.alibi_mask) {
            alibi_slopes = make_tensor_(false, {1, h, 1, 1});
            alibi_slopes->set_is_pass_by_value(false).set_data_type(DataType_t::FLOAT);
            alibi_slopes_size = h * sizeof(float);
        }

        // negative_inf_padding is passed by the node
        if (attributes.padding_mask) {
            negative_inf_padding = make_tensor_(false, {1, 1, 1, 1});
            negative_inf_padding->set_is_pass_by_value(true).set_data_type(DataType_t::FLOAT);
        }

        // negative_inf_causal is passed by the node
        if (attributes.causal_mask) {
            negative_inf_causal = make_tensor_(false, {1, 1, 1, 1});
            negative_inf_causal->set_is_pass_by_value(true).set_data_type(DataType_t::FLOAT);
        }

        bool is_dropout_prob = (attributes.dropout_probability.has_value());
        bool is_dropout_mask =
            (attributes.inputs[Scaled_dot_product_flash_attention_backward_attributes::input_names::Dropout_mask] !=
             nullptr);

        // if dropout_prob is used, then the node passes scale and scale inverse
        // if dropout_mask is used, then the user passes scale and scale_inverse
        if (is_dropout_prob) {
            attributes.inputs[Scaled_dot_product_flash_attention_backward_attributes::input_names::Dropout_scale] =
                make_tensor_(true, {1, 1, 1, 1});
            attributes.inputs[Scaled_dot_product_flash_attention_backward_attributes::input_names::Dropout_scale]
                ->set_is_pass_by_value(true)
                .set_data_type(DataType_t::FLOAT);
            attributes.inputs[Scaled_dot_product_flash_attention_backward_attributes::input_names::Dropout_scale_inv] =
                make_tensor_(true, {1, 1, 1, 1});
            attributes.inputs[Scaled_dot_product_flash_attention_backward_attributes::input_names::Dropout_scale_inv]
                ->set_is_pass_by_value(true)
                .set_data_type(DataType_t::FLOAT);
        }

        // WAR non-virtual dQ_accum is required if it is not
        // cudnn verision >= 8.9.5
        // device version >= hopper
        // sizeof(dp tensor) <= max_dp_workspace
        // non-virtual dQ_accum is passed by the node
        bool war_use_non_virtual_dQAccum = true;

        if (cudnnGetVersion() >= 8905) {
            struct cudaDeviceProp prop;
            CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, 0));
            if (prop.major >= 9) {
                // default upper limit for workspace 256MB
                int64_t max_dp_workspace_bytes = 256 * 1024 * 1024;

                // allow setting the upper limit with envvars
                char* env_dp_workspace_limit_char = std::getenv("CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT");
                if (env_dp_workspace_limit_char) {
                    try {
                        std::string env_dp_workspace_limit_str(env_dp_workspace_limit_char);
                        int64_t env_dp_workspace_limit = static_cast<int64_t>(std::stoll(env_dp_workspace_limit_str));
                        max_dp_workspace_bytes         = std::max(max_dp_workspace_bytes, env_dp_workspace_limit);
                    } catch (...) {
                        RETURN_CUDNN_FRONTEND_ERROR_IF(true,
                                                       error_code_t::ATTRIBUTE_NOT_SET,
                                                       "Invalid argument for CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT "
                                                       "(int64_t; in bytes)");
                    }
                }

                int64_t workspace_s_q               = ((s_q + 64 - 1) / 64) * 64;
                int64_t workspace_s_kv              = ((s_kv + 64 - 1) / 64) * 64;
                int64_t required_dp_workspace_bytes = b * h * workspace_s_q * workspace_s_kv * 2;

                if (required_dp_workspace_bytes <= max_dp_workspace_bytes) {
                    war_use_non_virtual_dQAccum = false;
                }
            }
        }

        if (war_use_non_virtual_dQAccum) {
            dQ_accum = make_tensor_(false, {b, h, s_q, d});
            dQ_accum->set_data_type(DataType_t::FLOAT).set_reordering_type(TensorReordering_t::F16x16);
            dQ_accum_size = b * h * s_q * d * sizeof(float);
        }

        // non-virtual softmax_sum is required for below cuDNN 8.9.5
        // non-virtual softmax_sum is passed by the node
        if (cudnnGetVersion() < 8905) {
            softmax_sum = make_tensor_(false, {b, h, s_q, 1});
            softmax_sum->set_data_type(DataType_t::FLOAT);
            softmax_sum_size = b * h * s_q * sizeof(float);
        }

        // --------------RNG node--------------------

        if (is_dropout_prob) {
            Rng_attributes rng_attr;
            rng_attr.set_distribution(RngDistribution_t::BERNOULLI);
            rng_attr.set_bernoulli_probability(1.0f - attributes.dropout_probability.value());
            rng_attr.inputs[Rng_attributes::input_names::Seed] =
                attributes.inputs[Scaled_dot_product_flash_attention_backward_attributes::input_names::Seed];
            rng_attr.inputs[Rng_attributes::input_names::Offset] =
                attributes.inputs[Scaled_dot_product_flash_attention_backward_attributes::input_names::Offset];
            rng_attr.outputs[Rng_attributes::output_names::Y] = rng_output = make_tensor_(true, {b, h, s_q, s_kv});
            sub_nodes.emplace_back(std::make_unique<RngNode>(std::move(rng_attr), context));
        } else if (is_dropout_mask) {
            rng_output =
                attributes.inputs[Scaled_dot_product_flash_attention_backward_attributes::input_names::Dropout_mask];
        }

        // --------------"dO * o => softmax_sum" chain--------------------

        // pointwise mul: dO * O
        Pointwise_attributes pw_mul_dO_O_attr;
        pw_mul_dO_O_attr.set_name("pw_mul_dO_O");
        pw_mul_dO_O_attr.set_mode(PointwiseMode_t::MUL);
        pw_mul_dO_O_attr.inputs[Pointwise_attributes::input_names::IN_0] =
            attributes.inputs[Scaled_dot_product_flash_attention_backward_attributes::input_names::dO];
        pw_mul_dO_O_attr.inputs[Pointwise_attributes::input_names::IN_1] =
            attributes.inputs[Scaled_dot_product_flash_attention_backward_attributes::input_names::O];
        pw_mul_dO_O_attr.outputs[Pointwise_attributes::output_names::OUT_0] = last_output =
            make_tensor_(true, {b, h, s_q, d});
        sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(pw_mul_dO_O_attr), context));

        // reduction add: dO * O
        Reduction_attributes reduction_add_dO_O_attr;
        reduction_add_dO_O_attr.set_name("reduction_add_dO_O");
        reduction_add_dO_O_attr.set_mode(ReductionMode_t::ADD);
        reduction_add_dO_O_attr.inputs[Reduction_attributes::input_names::X]   = last_output;
        reduction_add_dO_O_attr.outputs[Reduction_attributes::output_names::Y] = last_output =
            make_tensor_(true, {b, h, s_q, 1});
        sub_nodes.emplace_back(std::make_unique<ReductionNode>(std::move(reduction_add_dO_O_attr), context));

        // pointwise mul: dropout_scale inverse
        Pointwise_attributes pw_mul_dropout_scale_inv_attr;
        pw_mul_dropout_scale_inv_attr.set_name("pw_mul_dropout_scale_inv");
        pw_mul_dropout_scale_inv_attr.set_mode(PointwiseMode_t::MUL);
        pw_mul_dropout_scale_inv_attr.inputs[Pointwise_attributes::input_names::IN_0] = last_output;
        if (attributes.inputs[Scaled_dot_product_flash_attention_backward_attributes::input_names::Dropout_scale_inv]) {
            pw_mul_dropout_scale_inv_attr.inputs[Pointwise_attributes::input_names::IN_1] =
                attributes
                    .inputs[Scaled_dot_product_flash_attention_backward_attributes::input_names::Dropout_scale_inv];
        } else {
            // WAR dropout scale inverse is needed for non-dropout graphs
            pw_mul_dropout_scale_inv_attr.inputs[Pointwise_attributes::input_names::IN_1] = one_tensor;
        }
        if (softmax_sum) {
            pw_mul_dropout_scale_inv_attr.outputs[Pointwise_attributes::output_names::OUT_0] = softmax_sum;
        } else {
            pw_mul_dropout_scale_inv_attr.outputs[Pointwise_attributes::output_names::OUT_0] = softmax_sum =
                make_tensor_(true, {b, h, s_q, 1});
        }
        sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(pw_mul_dropout_scale_inv_attr), context));

        // --------------"Q @ KT => exp_softmax => dV" chain--------------------

        // matmul: Q * K^T
        Matmul_attributes matmul_Q_KT_attr;
        matmul_Q_KT_attr.set_name("matmul_Q_KT");
        matmul_Q_KT_attr.inputs[Matmul_attributes::input_names::A] =
            attributes.inputs[Scaled_dot_product_flash_attention_backward_attributes::input_names::Q];
        matmul_Q_KT_attr.inputs[Matmul_attributes::input_names::B] =
            attributes.inputs[Scaled_dot_product_flash_attention_backward_attributes::input_names::K];
        matmul_Q_KT_attr.inputs[Matmul_attributes::input_names::M_override] =
            attributes.inputs[Scaled_dot_product_flash_attention_backward_attributes::input_names::SEQ_LEN_Q];
        matmul_Q_KT_attr.inputs[Matmul_attributes::input_names::N_override] =
            attributes.inputs[Scaled_dot_product_flash_attention_backward_attributes::input_names::SEQ_LEN_KV];
        matmul_Q_KT_attr.outputs[Matmul_attributes::output_names::C] = last_output =
            make_tensor_(true, {b, h, s_q, s_kv});
        sub_nodes.emplace_back(std::make_unique<MatmulNode>(std::move(matmul_Q_KT_attr), context));

        if (attributes.attn_scale_value.has_value()) {
            attributes.inputs[Scaled_dot_product_flash_attention_backward_attributes::input_names::Attn_scale] =
                std::make_shared<Tensor_attributes>();
            attributes.inputs[Scaled_dot_product_flash_attention_backward_attributes::input_names::Attn_scale]
                ->set_dim({1, 1, 1, 1})
                .set_stride({1, 1, 1, 1})
                .set_data_type(DataType_t::FLOAT)
                .set_is_pass_by_value(true);
        }
        // pointwise mul: P bmmScale
        if (attributes.inputs[Scaled_dot_product_flash_attention_backward_attributes::input_names::Attn_scale]) {
            Pointwise_attributes pw_mul_S_bmm_scale_attr;
            pw_mul_S_bmm_scale_attr.set_name("pw_mul_S_bmm_scale");
            pw_mul_S_bmm_scale_attr.set_mode(PointwiseMode_t::MUL);
            pw_mul_S_bmm_scale_attr.inputs[Pointwise_attributes::input_names::IN_0] = last_output;
            pw_mul_S_bmm_scale_attr.inputs[Pointwise_attributes::input_names::IN_1] =
                attributes.inputs[Scaled_dot_product_flash_attention_backward_attributes::input_names::Attn_scale];
            pw_mul_S_bmm_scale_attr.outputs[Pointwise_attributes::output_names::OUT_0] = last_output =
                make_tensor_(true, {b, h, s_q, s_kv});
            sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(pw_mul_S_bmm_scale_attr), context));
        }

        // pointwise add: bias
        if (attributes.inputs[Scaled_dot_product_flash_attention_backward_attributes::input_names::Bias]) {
            Pointwise_attributes add_bias_attr;
            add_bias_attr.set_name("add_bias");
            add_bias_attr.set_mode(PointwiseMode_t::ADD);
            add_bias_attr.inputs[Pointwise_attributes::input_names::IN_0] = last_output;
            add_bias_attr.inputs[Pointwise_attributes::input_names::IN_1] =
                attributes.inputs[Scaled_dot_product_flash_attention_backward_attributes::input_names::Bias];
            add_bias_attr.outputs[Pointwise_attributes::output_names::OUT_0] = last_output =
                make_tensor_(true, {b, h, s_q, s_kv});
            sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(add_bias_attr), context));
        }

        // alibi mask DAG
        if (attributes.alibi_mask) {
            std::shared_ptr<Tensor_attributes> row_idx_output    = make_tensor_(true, {b, h, s_q, s_kv});
            std::shared_ptr<Tensor_attributes> col_idx_output    = make_tensor_(true, {b, h, s_q, s_kv});
            std::shared_ptr<Tensor_attributes> sub_idx_output    = make_tensor_(true, {b, h, s_q, s_kv});
            std::shared_ptr<Tensor_attributes> alibi_mask_output = make_tensor_(true, {b, h, s_q, s_kv});
            row_idx_output->set_data_type(DataType_t::INT32);
            col_idx_output->set_data_type(DataType_t::INT32);
            sub_idx_output->set_data_type(DataType_t::INT32);

            Pointwise_attributes gen_row_idx_attr;
            gen_row_idx_attr.set_name("gen_row_idx_alibi");
            gen_row_idx_attr.set_mode(PointwiseMode_t::GEN_INDEX).set_axis(2).set_compute_data_type(DataType_t::INT32);
            gen_row_idx_attr.inputs[Pointwise_attributes::input_names::IN_0]    = last_output;
            gen_row_idx_attr.outputs[Pointwise_attributes::output_names::OUT_0] = row_idx_output;
            sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(gen_row_idx_attr), context));

            Pointwise_attributes gen_col_idx_attr;
            gen_col_idx_attr.set_name("gen_col_idx_alibi");
            gen_col_idx_attr.set_mode(PointwiseMode_t::GEN_INDEX).set_axis(3).set_compute_data_type(DataType_t::INT32);
            gen_col_idx_attr.inputs[Pointwise_attributes::input_names::IN_0]    = last_output;
            gen_col_idx_attr.outputs[Pointwise_attributes::output_names::OUT_0] = col_idx_output;
            sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(gen_col_idx_attr), context));

            Pointwise_attributes sub_col_row_attr;
            sub_col_row_attr.set_name("sub_col_row_alibi");
            sub_col_row_attr.set_mode(PointwiseMode_t::SUB).set_compute_data_type(DataType_t::INT32);
            sub_col_row_attr.inputs[Pointwise_attributes::input_names::IN_0]    = col_idx_output;
            sub_col_row_attr.inputs[Pointwise_attributes::input_names::IN_1]    = row_idx_output;
            sub_col_row_attr.outputs[Pointwise_attributes::output_names::OUT_0] = sub_idx_output;
            sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(sub_col_row_attr), context));

            Pointwise_attributes mul_dist_slope_attr;
            mul_dist_slope_attr.set_name("mul_dist_slope_alibi");
            mul_dist_slope_attr.set_mode(PointwiseMode_t::MUL);
            mul_dist_slope_attr.inputs[Pointwise_attributes::input_names::IN_0]    = sub_idx_output;
            mul_dist_slope_attr.inputs[Pointwise_attributes::input_names::IN_1]    = alibi_slopes;
            mul_dist_slope_attr.outputs[Pointwise_attributes::output_names::OUT_0] = alibi_mask_output;
            sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(mul_dist_slope_attr), context));

            Pointwise_attributes add_alibi_attr;
            add_alibi_attr.set_name("add_alibi");
            add_alibi_attr.set_mode(PointwiseMode_t::ADD);
            add_alibi_attr.inputs[Pointwise_attributes::input_names::IN_0]    = last_output;
            add_alibi_attr.inputs[Pointwise_attributes::input_names::IN_1]    = alibi_mask_output;
            add_alibi_attr.outputs[Pointwise_attributes::output_names::OUT_0] = last_output =
                make_tensor_(true, {b, h, s_q, s_kv});
            sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(add_alibi_attr), context));
        }

        if (attributes.padding_mask) {
            std::shared_ptr<Tensor_attributes> row_idx_output      = make_tensor_(true, {b, h, s_q, s_kv});
            std::shared_ptr<Tensor_attributes> row_mask_output     = make_tensor_(true, {b, h, s_q, s_kv});
            std::shared_ptr<Tensor_attributes> col_idx_output      = make_tensor_(true, {b, h, s_q, s_kv});
            std::shared_ptr<Tensor_attributes> col_mask_output     = make_tensor_(true, {b, h, s_q, s_kv});
            std::shared_ptr<Tensor_attributes> padding_mask_output = make_tensor_(true, {b, h, s_q, s_kv});
            row_idx_output->set_data_type(DataType_t::INT32);
            row_mask_output->set_data_type(DataType_t::BOOLEAN);
            col_idx_output->set_data_type(DataType_t::INT32);
            col_mask_output->set_data_type(DataType_t::BOOLEAN);
            padding_mask_output->set_data_type(DataType_t::BOOLEAN);

            Pointwise_attributes gen_row_idx_attr;
            gen_row_idx_attr.set_name("gen_row_idx_alibi");
            gen_row_idx_attr.set_mode(PointwiseMode_t::GEN_INDEX).set_axis(2).set_compute_data_type(DataType_t::INT32);
            gen_row_idx_attr.inputs[Pointwise_attributes::input_names::IN_0]    = last_output;
            gen_row_idx_attr.outputs[Pointwise_attributes::output_names::OUT_0] = row_idx_output;
            sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(gen_row_idx_attr), context));

            Pointwise_attributes gen_col_idx_attr;
            gen_col_idx_attr.set_name("gen_col_idx_alibi");
            gen_col_idx_attr.set_mode(PointwiseMode_t::GEN_INDEX).set_axis(3).set_compute_data_type(DataType_t::INT32);
            gen_col_idx_attr.inputs[Pointwise_attributes::input_names::IN_0]    = last_output;
            gen_col_idx_attr.outputs[Pointwise_attributes::output_names::OUT_0] = col_idx_output;
            sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(gen_col_idx_attr), context));

            Pointwise_attributes lt_row_sq_attr;
            lt_row_sq_attr.set_name("lt_row_sq_causal");
            lt_row_sq_attr.set_mode(PointwiseMode_t::CMP_LT).set_compute_data_type(DataType_t::BOOLEAN);
            lt_row_sq_attr.inputs[Pointwise_attributes::input_names::IN_0] = row_idx_output;
            lt_row_sq_attr.inputs[Pointwise_attributes::input_names::IN_1] =
                attributes.inputs[Scaled_dot_product_flash_attention_backward_attributes::input_names::SEQ_LEN_Q];
            lt_row_sq_attr.outputs[Pointwise_attributes::output_names::OUT_0] = row_mask_output;
            sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(lt_row_sq_attr), context));

            Pointwise_attributes lt_col_skv_attr;
            lt_col_skv_attr.set_name("lt_col_skv_causal");
            lt_col_skv_attr.set_mode(PointwiseMode_t::CMP_LT).set_compute_data_type(DataType_t::BOOLEAN);
            lt_col_skv_attr.inputs[Pointwise_attributes::input_names::IN_0] = col_idx_output;
            lt_col_skv_attr.inputs[Pointwise_attributes::input_names::IN_1] =
                attributes.inputs[Scaled_dot_product_flash_attention_backward_attributes::input_names::SEQ_LEN_KV];
            lt_col_skv_attr.outputs[Pointwise_attributes::output_names::OUT_0] = col_mask_output;
            sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(lt_col_skv_attr), context));

            Pointwise_attributes and_row_col_mask_attr;
            and_row_col_mask_attr.set_name("and_row_col_mask");
            and_row_col_mask_attr.set_mode(PointwiseMode_t::LOGICAL_AND).set_compute_data_type(DataType_t::BOOLEAN);
            and_row_col_mask_attr.inputs[Pointwise_attributes::input_names::IN_0]    = row_mask_output;
            and_row_col_mask_attr.inputs[Pointwise_attributes::input_names::IN_1]    = col_mask_output;
            and_row_col_mask_attr.outputs[Pointwise_attributes::output_names::OUT_0] = padding_mask_output;
            sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(and_row_col_mask_attr), context));

            Pointwise_attributes select_padding_attr;
            select_padding_attr.set_name("select_causal");
            select_padding_attr.set_mode(PointwiseMode_t::BINARY_SELECT);
            select_padding_attr.inputs[Pointwise_attributes::input_names::IN_0]    = last_output;
            select_padding_attr.inputs[Pointwise_attributes::input_names::IN_1]    = negative_inf_padding;
            select_padding_attr.inputs[Pointwise_attributes::input_names::IN_2]    = padding_mask_output;
            select_padding_attr.outputs[Pointwise_attributes::output_names::OUT_0] = last_output =
                make_tensor_(true, {b, h, s_q, s_kv});
            sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(select_padding_attr), context));
        }

        // Causal Mask DAG
        if (attributes.causal_mask) {
            std::shared_ptr<Tensor_attributes> row_idx_output     = make_tensor_(true, {b, h, s_q, s_kv});
            std::shared_ptr<Tensor_attributes> col_idx_output     = make_tensor_(true, {b, h, s_q, s_kv});
            std::shared_ptr<Tensor_attributes> causal_mask_output = make_tensor_(true, {b, h, s_q, s_kv});
            row_idx_output->set_data_type(DataType_t::INT32);
            col_idx_output->set_data_type(DataType_t::INT32);
            causal_mask_output->set_data_type(DataType_t::BOOLEAN);

            Pointwise_attributes gen_row_idx_attr;
            gen_row_idx_attr.set_name("gen_row_idx_causal");
            gen_row_idx_attr.set_mode(PointwiseMode_t::GEN_INDEX).set_axis(2).set_compute_data_type(DataType_t::INT32);
            gen_row_idx_attr.inputs[Pointwise_attributes::input_names::IN_0]    = last_output;
            gen_row_idx_attr.outputs[Pointwise_attributes::output_names::OUT_0] = row_idx_output;
            sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(gen_row_idx_attr), context));

            Pointwise_attributes gen_col_idx_attr;
            gen_col_idx_attr.set_name("gen_col_idx_causal");
            gen_col_idx_attr.set_mode(PointwiseMode_t::GEN_INDEX).set_axis(3).set_compute_data_type(DataType_t::INT32);
            gen_col_idx_attr.inputs[Pointwise_attributes::input_names::IN_0]    = last_output;
            gen_col_idx_attr.outputs[Pointwise_attributes::output_names::OUT_0] = col_idx_output;
            sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(gen_col_idx_attr), context));

            Pointwise_attributes gt_row_col_attr;
            gt_row_col_attr.set_name("gt_row_col_causal");
            gt_row_col_attr.set_mode(PointwiseMode_t::CMP_GE).set_compute_data_type(DataType_t::BOOLEAN);
            gt_row_col_attr.inputs[Pointwise_attributes::input_names::IN_0]    = row_idx_output;
            gt_row_col_attr.inputs[Pointwise_attributes::input_names::IN_1]    = col_idx_output;
            gt_row_col_attr.outputs[Pointwise_attributes::output_names::OUT_0] = causal_mask_output;
            sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(gt_row_col_attr), context));

            Pointwise_attributes select_causal_attr;
            select_causal_attr.set_name("select_causal");
            select_causal_attr.set_mode(PointwiseMode_t::BINARY_SELECT);
            select_causal_attr.inputs[Pointwise_attributes::input_names::IN_0]    = last_output;
            select_causal_attr.inputs[Pointwise_attributes::input_names::IN_1]    = negative_inf_causal;
            select_causal_attr.inputs[Pointwise_attributes::input_names::IN_2]    = causal_mask_output;
            select_causal_attr.outputs[Pointwise_attributes::output_names::OUT_0] = last_output =
                make_tensor_(true, {b, h, s_q, s_kv});
            sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(select_causal_attr), context));
        }

        // pointwise subtract S
        Pointwise_attributes pw_subtract_s_attr;
        pw_subtract_s_attr.set_name("pw_subtract_s");
        pw_subtract_s_attr.set_mode(PointwiseMode_t::SUB);
        pw_subtract_s_attr.inputs[Pointwise_attributes::input_names::IN_0] = last_output;
        pw_subtract_s_attr.inputs[Pointwise_attributes::input_names::IN_1] =
            attributes.inputs[Scaled_dot_product_flash_attention_backward_attributes::input_names::Stats];
        pw_subtract_s_attr.outputs[Pointwise_attributes::output_names::OUT_0] = last_output =
            make_tensor_(true, {b, h, s_q, s_kv});
        sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(pw_subtract_s_attr), context));

        // pointwise exp softmax
        Pointwise_attributes exp_attr;
        exp_attr.set_name("exp_softmax");
        exp_attr.set_mode(PointwiseMode_t::EXP);
        exp_attr.inputs[Pointwise_attributes::input_names::IN_0]    = last_output;
        exp_attr.outputs[Pointwise_attributes::output_names::OUT_0] = last_output = exp_softmax_output =
            make_tensor_(true, {b, h, s_q, s_kv});
        last_output->set_data_type(context.get_io_data_type());
        sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(exp_attr), context));

        // pointwise dropout mask mul
        if (is_dropout_prob || is_dropout_mask) {
            Pointwise_attributes mask_attr;
            mask_attr.set_name("dropout_mask_mul");
            mask_attr.set_mode(PointwiseMode_t::MUL);
            mask_attr.inputs[Pointwise_attributes::input_names::IN_0]    = last_output;
            mask_attr.inputs[Pointwise_attributes::input_names::IN_1]    = rng_output;
            mask_attr.outputs[Pointwise_attributes::output_names::OUT_0] = last_output =
                make_tensor_(true, {b, h, s_q, s_kv});
            sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(mask_attr), context));
        }

        // pointwise dropout scale
        if (attributes.inputs[Scaled_dot_product_flash_attention_backward_attributes::input_names::Dropout_scale]) {
            Pointwise_attributes pw_mul_dropout_scale;
            pw_mul_dropout_scale.set_name("pw_mul_dropout_scale");
            pw_mul_dropout_scale.set_mode(PointwiseMode_t::MUL);
            pw_mul_dropout_scale.inputs[Pointwise_attributes::input_names::IN_0] = last_output;
            pw_mul_dropout_scale.inputs[Pointwise_attributes::input_names::IN_1] =
                attributes.inputs[Scaled_dot_product_flash_attention_backward_attributes::input_names::Dropout_scale];
            pw_mul_dropout_scale.outputs[Pointwise_attributes::output_names::OUT_0] = last_output =
                make_tensor_(true, {b, h, s_q, s_kv});
            sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(pw_mul_dropout_scale), context));
        }

        // reshape: transpose S
        Reshape_attributes transpose_s_attr;
        transpose_s_attr.set_name("transpose_s");
        transpose_s_attr.inputs[Reshape_attributes::input_names::X]   = last_output;
        transpose_s_attr.outputs[Reshape_attributes::output_names::Y] = last_output =
            make_tensor_(true, {b, h, s_kv, s_q}, {h * s_q * s_kv, s_q * s_kv, 1, s_kv});
        last_output->set_data_type(context.get_io_data_type());
        sub_nodes.emplace_back(std::make_unique<ReshapeNode>(std::move(transpose_s_attr), context));

        // matmul: S^T * dO
        Matmul_attributes matmul_ST_dO_attr;
        matmul_ST_dO_attr.set_name("matmul_ST_dO");
        matmul_ST_dO_attr.inputs[Matmul_attributes::input_names::A] = last_output;
        matmul_ST_dO_attr.inputs[Matmul_attributes::input_names::B] =
            attributes.inputs[Scaled_dot_product_flash_attention_backward_attributes::input_names::dO];
        matmul_ST_dO_attr.inputs[Matmul_attributes::input_names::M_override] =
            attributes.inputs[Scaled_dot_product_flash_attention_backward_attributes::input_names::SEQ_LEN_KV];
        matmul_ST_dO_attr.inputs[Matmul_attributes::input_names::K_override] =
            attributes.inputs[Scaled_dot_product_flash_attention_backward_attributes::input_names::SEQ_LEN_Q];
        matmul_ST_dO_attr.outputs[Matmul_attributes::output_names::C] =
            attributes.outputs[Scaled_dot_product_flash_attention_backward_attributes::output_names::dV];
        sub_nodes.emplace_back(std::make_unique<MatmulNode>(std::move(matmul_ST_dO_attr), context));

        // --------------"dO @ VT => dp_scaled_output => dK" chain--------------------

        // matmul: dO * V^T
        Matmul_attributes matmul_dO_VT_attr;
        matmul_dO_VT_attr.set_name("matmul_dO_VT");
        matmul_dO_VT_attr.inputs[Matmul_attributes::input_names::A] =
            attributes.inputs[Scaled_dot_product_flash_attention_backward_attributes::input_names::dO];
        matmul_dO_VT_attr.inputs[Matmul_attributes::input_names::B] =
            attributes.inputs[Scaled_dot_product_flash_attention_backward_attributes::input_names::V];
        matmul_dO_VT_attr.inputs[Matmul_attributes::input_names::M_override] =
            attributes.inputs[Scaled_dot_product_flash_attention_backward_attributes::input_names::SEQ_LEN_Q];
        matmul_dO_VT_attr.inputs[Matmul_attributes::input_names::N_override] =
            attributes.inputs[Scaled_dot_product_flash_attention_backward_attributes::input_names::SEQ_LEN_KV];
        matmul_dO_VT_attr.outputs[Matmul_attributes::output_names::C] = last_output =
            make_tensor_(true, {b, h, s_q, s_kv});
        sub_nodes.emplace_back(std::make_unique<MatmulNode>(std::move(matmul_dO_VT_attr), context));

        // pointwise mul: dS * mask
        Pointwise_attributes pw_mul_dS_mask_attr;
        pw_mul_dS_mask_attr.set_name("pw_mul_dS_mask");
        pw_mul_dS_mask_attr.set_mode(PointwiseMode_t::MUL);
        pw_mul_dS_mask_attr.inputs[Pointwise_attributes::input_names::IN_0] = last_output;
        if (is_dropout_prob || is_dropout_mask) {
            pw_mul_dS_mask_attr.inputs[Pointwise_attributes::input_names::IN_1] = rng_output;
        } else {
            pw_mul_dS_mask_attr.inputs[Pointwise_attributes::input_names::IN_1] = one_tensor;
        }
        pw_mul_dS_mask_attr.outputs[Pointwise_attributes::output_names::OUT_0] = last_output =
            make_tensor_(true, {b, h, s_q, s_kv});
        sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(pw_mul_dS_mask_attr), context));

        // pointwise: subtract ds
        Pointwise_attributes pw_subtract_ds_attr;
        pw_subtract_ds_attr.set_name("pw_subtract_ds");
        pw_subtract_ds_attr.set_mode(PointwiseMode_t::SUB);
        pw_subtract_ds_attr.inputs[Pointwise_attributes::input_names::IN_0]    = last_output;
        pw_subtract_ds_attr.inputs[Pointwise_attributes::input_names::IN_1]    = softmax_sum;
        pw_subtract_ds_attr.outputs[Pointwise_attributes::output_names::OUT_0] = last_output =
            make_tensor_(true, {b, h, s_q, s_kv});
        sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(pw_subtract_ds_attr), context));

        // pointwise: mul dP
        Pointwise_attributes pw_mul_dP_attr;
        pw_mul_dP_attr.set_name("pw_mul_dP");
        pw_mul_dP_attr.set_mode(PointwiseMode_t::MUL);
        pw_mul_dP_attr.inputs[Pointwise_attributes::input_names::IN_0]    = last_output;
        pw_mul_dP_attr.inputs[Pointwise_attributes::input_names::IN_1]    = exp_softmax_output;
        pw_mul_dP_attr.outputs[Pointwise_attributes::output_names::OUT_0] = last_output =
            make_tensor_(true, {b, h, s_q, s_kv});
        sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(pw_mul_dP_attr), context));

        // pointwise: mul dP_dropout_scale
        if (attributes.inputs[Scaled_dot_product_flash_attention_backward_attributes::input_names::Dropout_scale]) {
            Pointwise_attributes pw_mul_dP_dropout_scale_attr;
            pw_mul_dP_dropout_scale_attr.set_name("pw_mul_dP_dropout_scale");
            pw_mul_dP_dropout_scale_attr.set_mode(PointwiseMode_t::MUL);
            pw_mul_dP_dropout_scale_attr.inputs[Pointwise_attributes::input_names::IN_0] = last_output;
            pw_mul_dP_dropout_scale_attr.inputs[Pointwise_attributes::input_names::IN_1] =
                attributes.inputs[Scaled_dot_product_flash_attention_backward_attributes::input_names::Dropout_scale];
            pw_mul_dP_dropout_scale_attr.outputs[Pointwise_attributes::output_names::OUT_0] = last_output =
                make_tensor_(true, {b, h, s_q, s_kv});
            sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(pw_mul_dP_dropout_scale_attr), context));
        }

        // pointwise: mul dP_bmmScale
        if (attributes.inputs[Scaled_dot_product_flash_attention_backward_attributes::input_names::Attn_scale]) {
            Pointwise_attributes pw_mul_dP_bmm_scale_attr;
            pw_mul_dP_bmm_scale_attr.set_name("pw_mul_dP_bmm_scale");
            pw_mul_dP_bmm_scale_attr.set_mode(PointwiseMode_t::MUL);
            pw_mul_dP_bmm_scale_attr.inputs[Pointwise_attributes::input_names::IN_0] = last_output;
            pw_mul_dP_bmm_scale_attr.inputs[Pointwise_attributes::input_names::IN_1] =
                attributes.inputs[Scaled_dot_product_flash_attention_backward_attributes::input_names::Attn_scale];
            pw_mul_dP_bmm_scale_attr.outputs[Pointwise_attributes::output_names::OUT_0] = last_output =
                make_tensor_(true, {b, h, s_q, s_kv});
            sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(pw_mul_dP_bmm_scale_attr), context));
        }

        dp_scaled_output = last_output;

        // tranpose dP
        Reshape_attributes transpose_dP_attr;
        transpose_dP_attr.set_name("transpose_dP");
        transpose_dP_attr.inputs[Reshape_attributes::input_names::X]   = last_output;
        transpose_dP_attr.outputs[Reshape_attributes::output_names::Y] = last_output =
            make_tensor_(true, {b, h, s_kv, s_q}, {h * s_q * s_kv, s_q * s_kv, 1, s_kv});
        sub_nodes.emplace_back(std::make_unique<ReshapeNode>(std::move(transpose_dP_attr), context));

        // matmul: dP^T * Q
        Matmul_attributes matmul_dPT_Q_attr;
        matmul_dPT_Q_attr.set_name("matmul_dPT_Q");
        matmul_dPT_Q_attr.inputs[Matmul_attributes::input_names::A] = last_output;
        matmul_dPT_Q_attr.inputs[Matmul_attributes::input_names::B] =
            attributes.inputs[Scaled_dot_product_flash_attention_backward_attributes::input_names::Q];
        matmul_dPT_Q_attr.outputs[Matmul_attributes::output_names::C] =
            attributes.outputs[Scaled_dot_product_flash_attention_backward_attributes::output_names::dK];
        matmul_dPT_Q_attr.inputs[Matmul_attributes::input_names::M_override] =
            attributes.inputs[Scaled_dot_product_flash_attention_backward_attributes::input_names::SEQ_LEN_KV];
        matmul_dPT_Q_attr.inputs[Matmul_attributes::input_names::K_override] =
            attributes.inputs[Scaled_dot_product_flash_attention_backward_attributes::input_names::SEQ_LEN_Q];
        sub_nodes.emplace_back(std::make_unique<MatmulNode>(std::move(matmul_dPT_Q_attr), context));

        // --------------"dp_scaled @ K => dQ" chain--------------------

        auto const& kt_dim =
            attributes.inputs[Scaled_dot_product_flash_attention_backward_attributes::input_names::K]->get_dim();
        auto const& kt_stride =
            attributes.inputs[Scaled_dot_product_flash_attention_backward_attributes::input_names::K]->get_stride();

        // transpose KT
        Reshape_attributes transpose_K_attr;
        transpose_K_attr.set_name("transpose_K");
        transpose_K_attr.inputs[Reshape_attributes::input_names::X] =
            attributes.inputs[Scaled_dot_product_flash_attention_backward_attributes::input_names::K];
        transpose_K_attr.outputs[Reshape_attributes::output_names::Y] = last_output =
            make_tensor_(true,
                         {kt_dim[0], kt_dim[1], kt_dim[3], kt_dim[2]},
                         {kt_stride[0], kt_stride[1], kt_stride[3], kt_stride[2]});
        sub_nodes.emplace_back(std::make_unique<ReshapeNode>(std::move(transpose_K_attr), context));

        // matmul: dP * K
        Matmul_attributes matmul_dP_K_attr;
        matmul_dP_K_attr.set_name("matmul_dP_K");
        matmul_dP_K_attr.inputs[Matmul_attributes::input_names::A] = dp_scaled_output;
        matmul_dP_K_attr.inputs[Matmul_attributes::input_names::B] = last_output;
        if (dQ_accum) {
            matmul_dP_K_attr.outputs[Matmul_attributes::output_names::C] = dQ_accum;
        } else {
            matmul_dP_K_attr.outputs[Matmul_attributes::output_names::C] =
                attributes.outputs[Scaled_dot_product_flash_attention_backward_attributes::output_names::dQ];
        }
        matmul_dP_K_attr.inputs[Matmul_attributes::input_names::M_override] =
            attributes.inputs[Scaled_dot_product_flash_attention_backward_attributes::input_names::SEQ_LEN_Q];
        matmul_dP_K_attr.inputs[Matmul_attributes::input_names::K_override] =
            attributes.inputs[Scaled_dot_product_flash_attention_backward_attributes::input_names::SEQ_LEN_KV];
        sub_nodes.emplace_back(std::make_unique<MatmulNode>(std::move(matmul_dP_K_attr), context));

        if (dQ_accum) {
            Pointwise_attributes pw_identity_dQ_attr;
            pw_identity_dQ_attr.set_name("pw_identity_dQ");
            pw_identity_dQ_attr.set_mode(PointwiseMode_t::IDENTITY);
            pw_identity_dQ_attr.inputs[Pointwise_attributes::input_names::IN_0] = dQ_accum;
            pw_identity_dQ_attr.outputs[Pointwise_attributes::output_names::OUT_0] =
                attributes.outputs[Scaled_dot_product_flash_attention_backward_attributes::output_names::dQ];
            sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(pw_identity_dQ_attr), context));
        }

        return {error_code_t::OK, ""};
    }

    virtual int64_t
    get_fe_workspace_size_node() const override final {
        // set in infer_properties_node()
        return alibi_slopes_size + dQ_accum_size + softmax_sum_size;
    }

    error_t
    pass_by_value_tensors_(
        cudnnHandle_t handle,
        std::unordered_map<std::shared_ptr<Tensor_attributes>, pass_by_values_t>& tensor_to_pass_by_value,
        void* node_workspace) override {
        if (one_tensor) {
            tensor_to_pass_by_value.emplace(one_tensor, 1.0f);
        }

        if (attributes.attn_scale_value.has_value()) {
            tensor_to_pass_by_value.emplace(
                attributes.inputs[Scaled_dot_product_flash_attention_backward_attributes::input_names::Attn_scale],
                attributes.attn_scale_value.value());
        }

        if (attributes.alibi_mask) {
            int64_t const h =
                attributes.inputs[Scaled_dot_product_flash_attention_backward_attributes::input_names::Q]->get_dim()[1];
            auto alibi_slopes_vec = detail::get_abili_slope(h);

            cudaStream_t stream;
            CHECK_CUDNN_ERROR(cudnnGetStream(handle, &stream));
            CHECK_CUDA_ERROR(cudaMemcpyAsync(
                node_workspace, alibi_slopes_vec.data(), h * sizeof(float), cudaMemcpyHostToDevice, stream));
            tensor_to_pass_by_value.emplace(alibi_slopes, node_workspace);
            node_workspace = static_cast<char*>(node_workspace) + alibi_slopes_size;
        }

        if (attributes.padding_mask) {
            float negative_inf_value = std::numeric_limits<float>::lowest();
            tensor_to_pass_by_value.emplace(negative_inf_padding, negative_inf_value);
        }

        if (attributes.causal_mask) {
            float negative_inf_value = std::numeric_limits<float>::lowest();
            tensor_to_pass_by_value.emplace(negative_inf_causal, negative_inf_value);
        }

        if (attributes.dropout_probability.has_value()) {
            float dropout_scale_value     = 1.0f / (1.0f - attributes.dropout_probability.value());
            float dropout_scale_inv_value = (1.0f - attributes.dropout_probability.value());
            tensor_to_pass_by_value.emplace(
                attributes.inputs[Scaled_dot_product_flash_attention_backward_attributes::input_names::Dropout_scale],
                dropout_scale_value);
            tensor_to_pass_by_value.emplace(
                attributes
                    .inputs[Scaled_dot_product_flash_attention_backward_attributes::input_names::Dropout_scale_inv],
                dropout_scale_inv_value);
        }

        if (dQ_accum) {
            cudaStream_t stream;
            CHECK_CUDNN_ERROR(cudnnGetStream(handle, &stream));
            CHECK_CUDA_ERROR(cudaMemsetAsync(node_workspace, 0, dQ_accum_size, stream));
            tensor_to_pass_by_value.emplace(dQ_accum, node_workspace);
            node_workspace = static_cast<char*>(node_workspace) + dQ_accum_size;
        }

        if (softmax_sum) {
            // There is no requirement for softmax_sum to be memset to 0
            tensor_to_pass_by_value.emplace(softmax_sum, node_workspace);
        }

        return {error_code_t::OK, ""};
    }

   private:
    inline std::shared_ptr<Tensor_attributes>
    make_tensor_(bool is_virtual) {
        auto tensor = std::make_shared<Tensor_attributes>();
        tensor->set_is_virtual(is_virtual);
        return tensor;
    }

    inline std::shared_ptr<Tensor_attributes>
    make_tensor_(bool is_virtual, std::vector<int64_t> const& dim) {
        std::vector<int64_t> stride(dim.size());
        int64_t prod = 1;
        for (int i = (int)dim.size() - 1; i >= 0; --i) {
            stride[i] = prod;
            prod *= dim[i];
        }
        auto tensor = std::make_shared<Tensor_attributes>();
        tensor->set_is_virtual(is_virtual).set_dim(dim).set_stride(stride);
        return tensor;
    }

    inline std::shared_ptr<Tensor_attributes>
    make_tensor_(bool is_virtual, std::vector<int64_t> const& dim, std::vector<int64_t> const& stride) {
        auto tensor = std::make_shared<Tensor_attributes>();
        tensor->set_is_virtual(is_virtual).set_dim(dim).set_stride(stride);
        return tensor;
    }
};

}  // namespace cudnn_frontend::graph