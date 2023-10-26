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

class SDPA_FP8_Node : public INode {
    std::shared_ptr<Tensor_attributes> dropout_scale;
    std::shared_ptr<Tensor_attributes> KT;

   public:
    SDPA_FP8_attributes attributes;

    SDPA_FP8_Node(SDPA_FP8_attributes&& attributes_, detail::Context const& context)
        : INode(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::COMPOSITE;
    }

    error_t
    validate_node() const override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Validating SDPA_FP8_Node " << attributes.name << "..." << std::endl;

        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.is_inference.has_value() == false,
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "is_infernece attribute not set");

        auto const& dropout_mask    = attributes.inputs.find(SDPA_FP8_attributes::input_names::Dropout_mask);
        bool const has_dropout_mask = (dropout_mask != attributes.inputs.end()) && (dropout_mask->second != nullptr);
        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.dropout_probability.has_value() && has_dropout_mask,
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "Using both, custom dropout mask and internal-mask generation using dropout "
                                       "probability, is ill-formed.");

        RETURN_CUDNN_FRONTEND_ERROR_IF(
            attributes.dropout_probability.has_value() && attributes.dropout_probability.value() == 1.0,
            error_code_t::ATTRIBUTE_NOT_SET,
            "Dropout probability cannot be 1 as corresponding scale wont be well formed.");

        auto const& seq_len_q    = attributes.inputs.find(SDPA_FP8_attributes::input_names::SEQ_LEN_Q);
        bool const has_seq_len_q = (seq_len_q != attributes.inputs.end()) && (seq_len_q->second != nullptr);

        auto const& seq_len_kv    = attributes.inputs.find(SDPA_FP8_attributes::input_names::SEQ_LEN_KV);
        bool const has_seq_len_kv = (seq_len_kv != attributes.inputs.end()) && (seq_len_kv->second != nullptr);
        RETURN_CUDNN_FRONTEND_ERROR_IF(attributes.padding_mask && (!has_seq_len_q || !has_seq_len_kv),
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "Padding mask requires seq_len_q and seq_len_kv to be set.");

        // RETURN_CUDNN_FRONTEND_ERROR_IF((!attributes.padding_mask) && (has_seq_len_q || has_seq_len_kv),
        //                                error_code_t::ATTRIBUTE_NOT_SET,
        //                                "seq_len_q and seq_len_kv needs to be set only if padding mask is enabled.");

        return {error_code_t::OK, ""};
    }

    error_t
    infer_properties_node() override final {
        getLogger() << "[cudnn_frontend] INFO: Inferrencing properties for sdpa_fp8 node  " << attributes.name << "..."
                    << std::endl;

        attributes.fill_from_context(context);
        CHECK_CUDNN_FRONTEND_ERROR(attributes.validate_inputs());

        auto q_dim = attributes.inputs[SDPA_FP8_attributes::input_names::Q]->get_dim();
        auto k_dim = attributes.inputs[SDPA_FP8_attributes::input_names::K]->get_dim();
        auto v_dim = attributes.inputs[SDPA_FP8_attributes::input_names::V]->get_dim();

        if (attributes.inputs[SDPA_FP8_attributes::input_names::ragged_offset_QKV] != nullptr) {
            attributes.inputs[SDPA_FP8_attributes::input_names::Q]->set_ragged_offset(
                attributes.inputs[SDPA_FP8_attributes::input_names::ragged_offset_QKV]);
            attributes.inputs[SDPA_FP8_attributes::input_names::K]->set_ragged_offset(
                attributes.inputs[SDPA_FP8_attributes::input_names::ragged_offset_QKV]);
            attributes.inputs[SDPA_FP8_attributes::input_names::V]->set_ragged_offset(
                attributes.inputs[SDPA_FP8_attributes::input_names::ragged_offset_QKV]);
        }

        auto b    = q_dim[0];
        auto h    = q_dim[1];
        auto s_q  = q_dim[2];
        auto d_q  = q_dim[3];
        auto s_kv = k_dim[2];

        std::shared_ptr<Tensor_attributes> last_output;
        std::shared_ptr<Tensor_attributes> branch_output;

        // Reshape K
        auto transpose_k_attr = Reshape_attributes().set_name("K^T");
        last_output = KT    = reshape(attributes.inputs[SDPA_FP8_attributes::input_names::K], transpose_k_attr);
        auto const K_stride = attributes.inputs[SDPA_FP8_attributes::input_names::K]->get_stride();
        last_output->set_name("Kt")
            .set_is_virtual(false)
            .set_dim({b, h, d_q, s_kv})
            .set_stride({K_stride[0], K_stride[1], K_stride[3], K_stride[2]});

        if (auto const& ragged_offset_QKV = attributes.inputs[SDPA_FP8_attributes::input_names::ragged_offset_QKV]) {
            last_output->set_ragged_offset(ragged_offset_QKV);
        }
        ///////////////// TODO /////////////////////
        // transpose_k_attr.outputs.Y->set_is_virtual(true); (Need to fix a hack in cudnn backend to enable this)

        auto const& seq_len_q = attributes.inputs[SDPA_FP8_attributes::input_names::SEQ_LEN_Q];
        // auto const& seq_len_kv = attributes.inputs[SDPA_FP8_attributes::input_names::SEQ_LEN_KV];
        auto bmm1_attributes =
            Matmul_attributes().set_name("Q*K^T").set_m_override(seq_len_q).set_n_override(seq_len_q);
        last_output = matmul(attributes.inputs[SDPA_FP8_attributes::input_names::Q], last_output, bmm1_attributes);
        last_output->set_name("QKt");

        // Optional scale
        if (attributes.inputs[SDPA_FP8_attributes::input_names::Attn_scale]) {
            auto scale_attributes = Pointwise_attributes().set_name("attn_scale").set_mode(PointwiseMode_t::MUL);
            last_output           = pointwise(
                last_output, attributes.inputs[SDPA_FP8_attributes::input_names::Attn_scale], scale_attributes);
            last_output->set_name("QKt_as");
        }

        // descale_Q
        auto descale_q_attributes = Pointwise_attributes().set_name("descale_Q").set_mode(PointwiseMode_t::MUL);
        last_output               = pointwise(
            last_output, attributes.inputs[SDPA_FP8_attributes::input_names::descale_Q], descale_q_attributes);
        last_output->set_name("QKt_as_ds").set_data_type(DataType_t::FLOAT);

        // descale_K
        auto descale_k_attributes = Pointwise_attributes().set_name("descale_K").set_mode(PointwiseMode_t::MUL);
        last_output               = pointwise(
            last_output, attributes.inputs[SDPA_FP8_attributes::input_names::descale_K], descale_k_attributes);
        last_output->set_name("QKt_as_ds_ds").set_data_type(DataType_t::FLOAT);

        // This is required to avoid intermediate node output datatype being set to global context output datatype
        // softmax_output->set_data_type(DataType_t::FLOAT);

        // Lower options to softmax options
        auto softmax_output = std::make_shared<Tensor_attributes>();
        softmax_output->set_name("S").set_is_virtual(true);

        auto softmax_attributes = Softmax_attributes().set_name("softmax").has_M_Zinv(true).has_stats(false);
        // Special non-functional-style call. Needed because output already created and provided to user.
        softmax(last_output,
                softmax_attributes,
                softmax_output,
                attributes.outputs[SDPA_FP8_attributes::output_names::M],
                attributes.outputs[SDPA_FP8_attributes::output_names::Zinv]);
        last_output = softmax_output;

        // AMax_S
        auto const& amax_S     = attributes.outputs[SDPA_FP8_attributes::output_names::AMax_S];
        auto amax_S_attributes = Reduction_attributes().set_name("AMax_S").set_mode(ReductionMode_t::AMAX);
        // Special non-functional-style call. Needed because output already created and provided to user.
        reduction(last_output, amax_S_attributes, amax_S);
        /////////// TODO //////////////////
        // Check and assert on output dimensions?
        amax_S->set_dim({1, 1, 1, 1}).set_stride({1, 1, 1, 1}).set_is_virtual(false);

        // Two cases for training: dropout present or not
        // Special case: Skip dropout when 0.0 probability
        bool dropout_present =
            (attributes.dropout_probability.has_value() && attributes.dropout_probability.value() != 0.0);
        dropout_present = dropout_present || attributes.inputs[SDPA_FP8_attributes::input_names::Dropout_mask];

        if (dropout_present) {
            auto rng_attributes =
                Rng_attributes()
                    .set_name("rng")
                    .set_distribution(RngDistribution_t::BERNOULLI)
                    .set_bernoulli_probability(
                        1.0 - attributes.dropout_probability.value());  // As user sets dropout probability
            auto const& rng_output = rng(attributes.inputs[SDPA_FP8_attributes::input_names::Seed],
                                         attributes.inputs[SDPA_FP8_attributes::input_names::Offset],
                                         rng_attributes);
            rng_output->set_name("rng_nums")
                .set_dim({b, h, s_q, s_kv})
                .set_stride({h * s_q * s_kv, s_q * s_kv, s_kv, 1});

            auto mask_attributes = Pointwise_attributes().set_name("dropout_mask_mul").set_mode(PointwiseMode_t::MUL);
            last_output          = pointwise(last_output, rng_output, mask_attributes);
            last_output->set_name("S_drp");

            dropout_scale = std::make_shared<Tensor_attributes>();
            dropout_scale->set_dim({1, 1, 1, 1})
                .set_stride({1, 1, 1, 1})
                .set_is_pass_by_value(true)
                .set_data_type(DataType_t::FLOAT);

            auto dropout_scale_attributes =
                Pointwise_attributes().set_name("dropout_scale").set_mode(PointwiseMode_t::MUL);
            last_output = pointwise(last_output, dropout_scale, dropout_scale_attributes);
            last_output->set_name("S_drp_s");
        }

        // scale_S
        auto scale_s_attributes = Pointwise_attributes().set_name("scale_S").set_mode(PointwiseMode_t::MUL);
        last_output =
            pointwise(last_output, attributes.inputs[SDPA_FP8_attributes::input_names::scale_S], scale_s_attributes);
        last_output->set_name("S_s").set_data_type(DataType_t::FP8_E4M3);

        auto bmm2_attributes = Matmul_attributes().set_name("bmm2").set_m_override(seq_len_q).set_k_override(seq_len_q);
        last_output = matmul(last_output, attributes.inputs[SDPA_FP8_attributes::input_names::V], bmm2_attributes);
        last_output->set_name("SV");

        // descale_S
        auto descale_s_attributes = Pointwise_attributes().set_name("descale_S").set_mode(PointwiseMode_t::MUL);
        ///////////////// TODO /////////////////////////////////
        last_output =
            pointwise(last_output, attributes.inputs[SDPA_FP8_attributes::input_names::scale_S], descale_s_attributes);
        // This above wrong! Should be attributes.inputs[SDPA_FP8_attributes::input_names::scale_dS]. Decide if dS
        // should be allocated in extra workspace or trouble the user to provide and maintain yet another unecessary
        // detail
        last_output->set_name("SV_ds").set_data_type(DataType_t::FLOAT);

        // descale_V
        auto descale_v_attributes = Pointwise_attributes().set_name("descale_V").set_mode(PointwiseMode_t::MUL);
        last_output               = pointwise(
            last_output, attributes.inputs[SDPA_FP8_attributes::input_names::descale_V], descale_v_attributes);
        last_output->set_name("SV_ds_ds").set_data_type(DataType_t::FLOAT);

        // AMax_O
        auto const& amax_O     = attributes.outputs[SDPA_FP8_attributes::output_names::AMax_O];
        auto amax_O_attributes = Reduction_attributes().set_name("AMax_O").set_mode(ReductionMode_t::AMAX);
        // Special non-functional-style call. Needed because output already created and provided to user.
        reduction(last_output, amax_O_attributes, amax_O);
        /////////// TODO //////////////////
        // Check and assert on output dimensions?
        amax_O->set_dim({1, 1, 1, 1}).set_stride({1, 1, 1, 1}).set_is_virtual(false);

        // scale_O
        auto const& O           = attributes.outputs[SDPA_FP8_attributes::output_names::O];
        auto scale_O_attributes = Pointwise_attributes().set_name("scale_O").set_mode(PointwiseMode_t::MUL);
        // Special non-functional-style call. Needed because output already created and provided to user.
        pointwise(last_output, attributes.inputs[SDPA_FP8_attributes::input_names::scale_O], scale_O_attributes, O);

        if (auto const& ragged_offset_O = attributes.inputs[SDPA_FP8_attributes::input_names::ragged_offset_O]) {
            O->set_ragged_offset(ragged_offset_O);
        }

        return {error_code_t::OK, ""};
    }

    virtual int64_t
    get_fe_workspace_size_node() const override final {
        auto const& q   = attributes.inputs.find(SDPA_FP8_attributes::input_names::Q);
        int64_t const h = q->second->get_dim()[1];
        return h * sizeof(float);
    }

    virtual error_t
    pass_by_value_tensors_(
        cudnnHandle_t,
        std::unordered_map<std::shared_ptr<Tensor_attributes>, void*> const& tensor_to_pointer_map,
        std::unordered_map<std::shared_ptr<Tensor_attributes>, pass_by_values_t>& tensor_to_pass_by_value,
        void*) override {
        if (attributes.dropout_probability.has_value()) {
            float dropout_scale_value = (1.f / (1.0f - attributes.dropout_probability.value()));

            tensor_to_pass_by_value.emplace(dropout_scale, dropout_scale_value);
        }

        // sdpa_fp8 creates a non virtual KT.
        // User does not know about it and should not provide device pointer for it.
        // But the backend will ask for it.
        void* k_ptr = tensor_to_pointer_map.at(attributes.inputs[SDPA_FP8_attributes::input_names::K]);
        tensor_to_pass_by_value.emplace(KT, k_ptr);

        return {error_code_t::OK, ""};
    }
};

}  // namespace cudnn_frontend::graph