#pragma once

#include "../../cudnn_frontend_Heuristics.h"
#include "../../cudnn_frontend_Logging.h"

#include "../cudnn_frontend_graph_helpers.h"
#include "../cudnn_frontend_node_interface.h"

#include "matmul.h"
#include "pointwise.h"
#include "rng.h"
#include "softmax.h"

namespace cudnn_frontend::graph {

class ScaledDotProductFlashAttentionBackwardNode : public INode {
   private:
    std::shared_ptr<Tensor_attributes> negative_inf_causal;
    // one_tensor is needed for non-dropout graphs
    std::shared_ptr<Tensor_attributes> one_tensor;

    // non-virtual node workspace tensors
    std::shared_ptr<Tensor_attributes> dQ_accum;
    int64_t dQ_accum_size = 0;
    std::shared_ptr<Tensor_attributes> softmax_sum;
    int64_t softmax_sum_size = 0;

   public:
    Scaled_dot_product_flash_attention_backward_attributes options;

    ScaledDotProductFlashAttentionBackwardNode(Scaled_dot_product_flash_attention_backward_attributes&& options_,
                                               detail::Context const& context)
        : INode(context), options(std::move(options_)) {}

    Type
    getType() override final {
        return Type::COMPOSITE;
    }

    error_t
    validate_node() const override final {
        getLogger() << "[cudnn_frontend] INFO: "
                    << "Validating ScaledDotProductFlashAttentionBackwardNode" << options.name << "..." << std::endl;

        if (options.dropout_probability.has_value() && options.inputs.Dropout_mask) {
            auto status = error_code_t::ATTRIBUTE_NOT_SET;
            std::string message =
                "[cudnn_frontend] ERROR: Using both, custom dropout mask and internal-mask generation using dropout "
                "probability, is ill-formed.";
            getLogger() << message << std::endl;
            return {status, message};
        }

        if (options.dropout_probability.has_value() && options.dropout_probability.value() == 1.0) {
            auto status = error_code_t::ATTRIBUTE_NOT_SET;
            std::string message =
                "[cudnn_frontend] ERROR: Dropout probability cannot be 1 as corresponding scale wont be well formed.";
            getLogger() << message << std::endl;
            return {status, message};
        }

        if (context.get_intermediate_data_type() == DataType_t::NOT_SET) {
            auto status = error_code_t::ATTRIBUTE_NOT_SET;
            std::string message =
                "[cudnn_frontend] ERROR: Intermediate tensor data type needs to be set as internal tensors require it.";
            return {status, message};
        }

        return {error_code_t::OK, ""};
    }

    error_t
    infer_properties_node() override final {
        getLogger() << "[cudnn_frontend] INFO: Inferrencing properties for ScaledDotProductFlashAttentionBackwardNode "
                    << options.name << "..." << std::endl;

        options.fill_from_context(context);

        // Gather dims to fill properties of virtual tensors
        auto const& q_dim = options.inputs.Q->get_dim();
        auto b            = q_dim[0];
        auto h            = q_dim[1];
        auto s_q          = q_dim[2];
        auto d            = q_dim[3];
        auto const& k_dim = options.inputs.K->get_dim();
        auto s_kv         = k_dim[3];

        std::shared_ptr<Tensor_attributes> last_output, exp_softmax_output, dp_scaled_output, rng_output;

        // --------------Initialize and create tensors before creating nodes--------------------

        // one_tensor is needed for non-dropout graphs
        one_tensor = std::make_shared<Tensor_attributes>();
        one_tensor->set_dim({1, 1, 1, 1})
                .set_stride({1, 1, 1, 1})
                .set_is_pass_by_value(true)
                .set_data_type(DataType_t::FLOAT);

        // create tensors internal to the node
        if (options.causal_mask) {
            negative_inf_causal = std::make_shared<Tensor_attributes>();
            negative_inf_causal->set_dim({1, 1, 1, 1})
                                .set_stride({1, 1, 1, 1})
                                .set_is_pass_by_value(true)
                                .set_data_type(DataType_t::FLOAT);
        }

        bool is_dropout_prob = (options.dropout_probability.has_value());
        bool is_dropout_mask = (options.inputs.Dropout_mask != nullptr);

        // if dropout_prob is used, then the node creates scale and scale inverse
        // if dropout_mask is used, then the user creates scale and scale_inverse
        if (is_dropout_prob) {
            options.inputs.Dropout_scale = make_tensor_(true, {1, 1, 1, 1});
            options.inputs.Dropout_scale->set_data_type(DataType_t::FLOAT).set_is_pass_by_value(true);
            options.inputs.Dropout_scale_inv = make_tensor_(true, {1, 1, 1, 1});
            options.inputs.Dropout_scale_inv->set_data_type(DataType_t::FLOAT).set_is_pass_by_value(true);
        }

        // non-virtual dQAccum is required for below cuDNN 8.9.5
        if (CUDNN_VERSION < 8905 || (CUDNN_VERSION == 8905 && (b * h * s_q * d * sizeof(float) <= (1 << 30)))) {
            dQ_accum = make_tensor_(false, {b, h, s_q, d});
            dQ_accum->set_data_type(DataType_t::FLOAT).set_reordering_type(TensorReordering_t::F16x16);
            dQ_accum_size = b * h * s_q * d * sizeof(float);
        }

        // non-virtual softmax_sum is required for below cuDNN 8.9.5
        if (CUDNN_VERSION < 8905) {
            softmax_sum = make_tensor_(false, {b, h, s_q, 1});
            softmax_sum->set_data_type(DataType_t::FLOAT);
            softmax_sum_size = b * h * s_q * sizeof(float);
        }

        // --------------RNG node--------------------

        if (is_dropout_prob) {
            Rng_attributes rng_attr;
            rng_attr.set_distribution(RngDistribution_t::BERNOULLI);
            rng_attr.set_bernoulli_probability(1.0f - options.dropout_probability.value());
            rng_attr.inputs.Seed   = options.inputs.Seed;
            rng_attr.inputs.Offset = options.inputs.Offset;
            rng_attr.outputs.Y     = rng_output = make_tensor_(true,  {b, h, s_q, s_kv});
            sub_nodes.emplace_back(std::make_unique<RngNode>(std::move(rng_attr), context));
        } else if (is_dropout_mask) {
            rng_output = options.inputs.Dropout_mask;
        }

        // --------------"dO * o => softmax_sum" chain--------------------

        // pointwise mul: dO * O
        Pointwise_attributes pw_mul_dO_O_attr;
        pw_mul_dO_O_attr.set_name("pw_mul_dO_O");
        pw_mul_dO_O_attr.set_mode(PointwiseMode_t::MUL);
        pw_mul_dO_O_attr.inputs.IN_0   = options.inputs.dO;
        pw_mul_dO_O_attr.inputs.IN_1   = options.inputs.O;
        pw_mul_dO_O_attr.outputs.OUT_0 = last_output = make_tensor_(true, {b, h, s_q, d});
        sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(pw_mul_dO_O_attr), context));

        // reduction add: dO * O
        Reduction_attributes reduction_add_dO_O_attr;
        reduction_add_dO_O_attr.set_name("reduction_add_dO_O");
        reduction_add_dO_O_attr.set_mode(ReductionMode_t::ADD);
        reduction_add_dO_O_attr.inputs.X  = last_output;
        reduction_add_dO_O_attr.outputs.Y = last_output = make_tensor_(true, {b, h, s_q, 1});
        sub_nodes.emplace_back(std::make_unique<ReductionNode>(std::move(reduction_add_dO_O_attr), context));

        // pointwise mul: dropout_scale inverse
        Pointwise_attributes pw_mul_dropout_scale_inv_attr;
        pw_mul_dropout_scale_inv_attr.set_name("pw_mul_dropout_scale_inv");
        pw_mul_dropout_scale_inv_attr.set_mode(PointwiseMode_t::MUL);
        pw_mul_dropout_scale_inv_attr.inputs.IN_0   = last_output;
        if (options.inputs.Dropout_scale_inv != nullptr) {
            pw_mul_dropout_scale_inv_attr.inputs.IN_1 = options.inputs.Dropout_scale_inv;
        } else {
            // WAR dropout scale inverse is needed for non-dropout graphs
            pw_mul_dropout_scale_inv_attr.inputs.IN_1 = one_tensor;
        }
        if (softmax_sum != nullptr) {
            pw_mul_dropout_scale_inv_attr.outputs.OUT_0 = softmax_sum;
        } else {
            pw_mul_dropout_scale_inv_attr.outputs.OUT_0 = softmax_sum = make_tensor_(true, {b, h, s_q, 1});
        }
        sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(pw_mul_dropout_scale_inv_attr), context));

        // --------------"Q @ KT => exp_softmax => dV" chain--------------------

        // matmul: Q * K^T
        Matmul_attributes matmul_Q_KT_attr;
        matmul_Q_KT_attr.set_name("matmul_Q_KT");
        matmul_Q_KT_attr.inputs.A  = options.inputs.Q;
        matmul_Q_KT_attr.inputs.B  = options.inputs.K;
        matmul_Q_KT_attr.outputs.C = last_output = make_tensor_(true, {b, h, s_q, s_kv});
        sub_nodes.emplace_back(std::make_unique<MatmulNode>(std::move(matmul_Q_KT_attr), context));

        // pointwise mul: P bmmScale
        if (options.inputs.Attn_scale != nullptr) {
            Pointwise_attributes pw_mul_S_bmm_scale_attr;
            pw_mul_S_bmm_scale_attr.set_name("pw_mul_S_bmm_scale");
            pw_mul_S_bmm_scale_attr.set_mode(PointwiseMode_t::MUL);
            pw_mul_S_bmm_scale_attr.inputs.IN_0   = last_output;
            pw_mul_S_bmm_scale_attr.inputs.IN_1   = options.inputs.Attn_scale;
            pw_mul_S_bmm_scale_attr.outputs.OUT_0 = last_output = make_tensor_(true, {b, h, s_q, s_kv});
            sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(pw_mul_S_bmm_scale_attr), context));
        }

        // Causal Mask DAG
        if (options.causal_mask) {
            std::shared_ptr<Tensor_attributes> row_index_output  = make_tensor_(true, {b, h, s_q, s_kv});
            std::shared_ptr<Tensor_attributes> col_index_output  = make_tensor_(true, {b, h, s_q, s_kv});
            std::shared_ptr<Tensor_attributes> row_gt_col_output = make_tensor_(true, {b, h, s_q, s_kv});
            row_gt_col_output->set_data_type(DataType_t::BOOLEAN);

            // Lower options to generate row index options
            Pointwise_attributes row_index_attr;
            row_index_attr.set_name("gen_row_index");
            row_index_attr.set_mode(PointwiseMode_t::GEN_INDEX).set_axis(2);
            row_index_attr.inputs.IN_0   = last_output;
            row_index_attr.outputs.OUT_0 = row_index_output;
            sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(row_index_attr), context));

            Pointwise_attributes col_index_attr;
            col_index_attr.set_name("gen_col_index");
            col_index_attr.set_mode(PointwiseMode_t::GEN_INDEX).set_axis(3);
            col_index_attr.inputs.IN_0   = last_output;
            col_index_attr.outputs.OUT_0 = col_index_output;
            sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(col_index_attr), context));

            Pointwise_attributes greater_than_attr;
            greater_than_attr.set_name("row_greater_than_col");
            greater_than_attr.set_mode(PointwiseMode_t::CMP_GE).set_compute_data_type(DataType_t::BOOLEAN);
            greater_than_attr.inputs.IN_0   = row_index_output;
            greater_than_attr.inputs.IN_1   = col_index_output;
            greater_than_attr.outputs.OUT_0 = row_gt_col_output;
            sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(greater_than_attr), context));

            Pointwise_attributes binary_select_attr;
            binary_select_attr.set_name("binary_select");
            binary_select_attr.set_mode(PointwiseMode_t::BINARY_SELECT);
            binary_select_attr.inputs.IN_0   = last_output;
            binary_select_attr.inputs.IN_1   = negative_inf_causal;
            binary_select_attr.inputs.IN_2   = row_gt_col_output;
            binary_select_attr.outputs.OUT_0 = last_output = make_tensor_(true, {b, h, s_q, s_kv});
            sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(binary_select_attr), context));
        }

        // pointwise subtract S
        Pointwise_attributes pw_subtract_s_attr;
        pw_subtract_s_attr.set_name("pw_subtract_s");
        pw_subtract_s_attr.set_mode(PointwiseMode_t::SUB);
        pw_subtract_s_attr.inputs.IN_0   = last_output;
        pw_subtract_s_attr.inputs.IN_1   = options.inputs.Stats;
        pw_subtract_s_attr.outputs.OUT_0 = last_output = make_tensor_(true, {b, h, s_q, s_kv});
        sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(pw_subtract_s_attr), context));

        // pointwise exp softmax
        Pointwise_attributes exp_attr;
        exp_attr.set_name("exp_softmax");
        exp_attr.set_mode(PointwiseMode_t::EXP);
        exp_attr.inputs.IN_0   = last_output;
        exp_attr.outputs.OUT_0 = last_output = exp_softmax_output = make_tensor_(true, {b, h, s_q, s_kv});
        last_output->set_data_type(context.get_io_data_type());
        sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(exp_attr), context));

        // pointwise dropout mask mul
        if (is_dropout_prob || is_dropout_mask) {
            Pointwise_attributes mask_attr;
            mask_attr.set_name("dropout_mask_mul");
            mask_attr.set_mode(PointwiseMode_t::MUL);
            mask_attr.inputs.IN_0   = last_output;
            mask_attr.inputs.IN_1   = rng_output;
            mask_attr.outputs.OUT_0 = last_output = make_tensor_(true,  {b, h, s_q, s_kv});
            sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(mask_attr), context));
        }

        // pointwise dropout scale
        if (options.inputs.Dropout_scale != nullptr) {
            Pointwise_attributes pw_mul_dropout_scale;
            pw_mul_dropout_scale.set_name("pw_mul_dropout_scale");
            pw_mul_dropout_scale.set_mode(PointwiseMode_t::MUL);
            pw_mul_dropout_scale.inputs.IN_0   = last_output;
            pw_mul_dropout_scale.inputs.IN_1   = options.inputs.Dropout_scale;
            pw_mul_dropout_scale.outputs.OUT_0 = last_output = make_tensor_(true,  {b, h, s_q, s_kv});
            sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(pw_mul_dropout_scale), context));
        }

        // reshape: transpose S
        Reshape_attributes transpose_s_attr;
        transpose_s_attr.set_name("transpose_s");
        transpose_s_attr.inputs.X  = last_output;
        transpose_s_attr.outputs.Y = last_output = make_tensor_(true, {b, h, s_kv, s_q}, {h * s_q * s_kv, s_q * s_kv, 1, s_kv});
        sub_nodes.emplace_back(std::make_unique<ReshapeNode>(std::move(transpose_s_attr), context));

        // matmul: S^T * dO
        Matmul_attributes matmul_ST_dO_attr;
        matmul_ST_dO_attr.set_name("matmul_ST_dO");
        matmul_ST_dO_attr.inputs.A  = last_output;
        matmul_ST_dO_attr.inputs.B  = options.inputs.dO;
        matmul_ST_dO_attr.outputs.C = options.outputs.dV;
        sub_nodes.emplace_back(std::make_unique<MatmulNode>(std::move(matmul_ST_dO_attr), context));

        // --------------"dO @ VT => dp_scaled_output => dK" chain--------------------

        // matmul: dO * V^T
        Matmul_attributes matmul_dO_VT_attr;
        matmul_dO_VT_attr.set_name("matmul_dO_VT");
        matmul_dO_VT_attr.inputs.A  = options.inputs.dO;
        matmul_dO_VT_attr.inputs.B  = options.inputs.V;
        matmul_dO_VT_attr.outputs.C = last_output = make_tensor_(true, {b, h, s_q, s_kv});
        sub_nodes.emplace_back(std::make_unique<MatmulNode>(std::move(matmul_dO_VT_attr), context));

        // pointwise mul: dS * mask
        Pointwise_attributes pw_mul_dS_mask_attr;
        pw_mul_dS_mask_attr.set_name("pw_mul_dS_mask");
        pw_mul_dS_mask_attr.set_mode(PointwiseMode_t::MUL);
        pw_mul_dS_mask_attr.inputs.IN_0 = last_output;
        if (is_dropout_prob || is_dropout_mask) {
            pw_mul_dS_mask_attr.inputs.IN_1 = rng_output;
        } else {
            pw_mul_dS_mask_attr.inputs.IN_1 = one_tensor;
        }
        pw_mul_dS_mask_attr.outputs.OUT_0 = last_output = make_tensor_(true, {b, h, s_q, s_kv});
        sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(pw_mul_dS_mask_attr), context));

        // pointwise: subtract ds
        Pointwise_attributes pw_subtract_ds_attr;
        pw_subtract_ds_attr.set_name("pw_subtract_ds");
        pw_subtract_ds_attr.set_mode(PointwiseMode_t::SUB);
        pw_subtract_ds_attr.inputs.IN_0   = last_output;
        pw_subtract_ds_attr.inputs.IN_1   = softmax_sum;
        pw_subtract_ds_attr.outputs.OUT_0 = last_output = make_tensor_(true, {b, h, s_q, s_kv});
        sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(pw_subtract_ds_attr), context));

        // pointwise: mul dP
        Pointwise_attributes pw_mul_dP_attr;
        pw_mul_dP_attr.set_name("pw_mul_dP");
        pw_mul_dP_attr.set_mode(PointwiseMode_t::MUL);
        pw_mul_dP_attr.inputs.IN_0   = last_output;
        pw_mul_dP_attr.inputs.IN_1   = exp_softmax_output;
        pw_mul_dP_attr.outputs.OUT_0 = last_output = make_tensor_(true, {b, h, s_q, s_kv});
        sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(pw_mul_dP_attr), context));

        // pointwise: mul dP_dropout_scale
        if (options.inputs.Dropout_scale != nullptr) {
            Pointwise_attributes pw_mul_dP_dropout_scale_attr;
            pw_mul_dP_dropout_scale_attr.set_name("pw_mul_dP_dropout_scale");
            pw_mul_dP_dropout_scale_attr.set_mode(PointwiseMode_t::MUL);
            pw_mul_dP_dropout_scale_attr.inputs.IN_0   = last_output;
            pw_mul_dP_dropout_scale_attr.inputs.IN_1   = options.inputs.Dropout_scale;
            pw_mul_dP_dropout_scale_attr.outputs.OUT_0 = last_output = make_tensor_(true, {b, h, s_q, s_kv});
            sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(pw_mul_dP_dropout_scale_attr), context));
        }

        // pointwise: mul dP_bmmScale
        if (options.inputs.Attn_scale != nullptr) {
            Pointwise_attributes pw_mul_dP_bmm_scale_attr;
            pw_mul_dP_bmm_scale_attr.set_name("pw_mul_dP_bmm_scale");
            pw_mul_dP_bmm_scale_attr.set_mode(PointwiseMode_t::MUL);
            pw_mul_dP_bmm_scale_attr.inputs.IN_0   = last_output;
            pw_mul_dP_bmm_scale_attr.inputs.IN_1   = options.inputs.Attn_scale;
            pw_mul_dP_bmm_scale_attr.outputs.OUT_0 = last_output = make_tensor_(true, {b, h, s_q, s_kv});
            sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(pw_mul_dP_bmm_scale_attr), context));
        }

        dp_scaled_output = last_output;

        // tranpose dP
        Reshape_attributes transpose_dP_attr;
        transpose_dP_attr.set_name("transpose_dP");
        transpose_dP_attr.inputs.X  = last_output;
        transpose_dP_attr.outputs.Y = last_output = make_tensor_(true, {b, h, s_kv, s_q}, {h * s_q * s_kv, s_q * s_kv, 1, s_kv});
        sub_nodes.emplace_back(std::make_unique<ReshapeNode>(std::move(transpose_dP_attr), context));

        // matmul: dP^T * Q
        Matmul_attributes matmul_dP_Q_attr;
        matmul_dP_Q_attr.set_name("matmul_dP_Q");
        matmul_dP_Q_attr.inputs.A  = last_output;
        matmul_dP_Q_attr.inputs.B  = options.inputs.Q;
        matmul_dP_Q_attr.outputs.C = options.outputs.dK;
        sub_nodes.emplace_back(std::make_unique<MatmulNode>(std::move(matmul_dP_Q_attr), context));

        // --------------"dp_scaled_output @ KT => dQ" chain--------------------

        // transpose K
        Reshape_attributes transpose_K_attr;
        transpose_K_attr.set_name("transpose_K");
        transpose_K_attr.inputs.X  = options.inputs.K;
        transpose_K_attr.outputs.Y = last_output = make_tensor_(true, {b, h, s_kv, d});
        sub_nodes.emplace_back(std::make_unique<ReshapeNode>(std::move(transpose_K_attr), context));

        // matmul: dP * K
        Matmul_attributes matmul_dP_K_attr;
        matmul_dP_K_attr.set_name("matmul_dP_K");
        matmul_dP_K_attr.inputs.A  = dp_scaled_output;
        matmul_dP_K_attr.inputs.B  = last_output;
        if (dQ_accum != nullptr) {
            matmul_dP_K_attr.outputs.C = dQ_accum;
        } else {
            matmul_dP_K_attr.outputs.C = options.outputs.dQ;
        }
        sub_nodes.emplace_back(std::make_unique<MatmulNode>(std::move(matmul_dP_K_attr), context));

        if (dQ_accum != nullptr) {
            Pointwise_attributes pw_identity_dQ_attr;
            pw_identity_dQ_attr.set_name("pw_identity_dQ");
            pw_identity_dQ_attr.set_mode(PointwiseMode_t::IDENTITY);
            pw_identity_dQ_attr.inputs.IN_0   = dQ_accum;
            pw_identity_dQ_attr.outputs.OUT_0 = options.outputs.dQ;
            sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(pw_identity_dQ_attr), context));
        }

        return {error_code_t::OK, ""};
    }

    virtual int64_t
    get_fe_workspace_size_node() const override final {
        // set in infer_properties_node()
        return dQ_accum_size + softmax_sum_size;
    }

    error_t
    pass_by_value_tensors_(
        std::unordered_map<std::shared_ptr<Tensor_attributes>, pass_by_values_t>& tensor_to_pass_by_value,
        void* node_workspace) override {

        if (options.causal_mask) {
            float negative_inf_value = std::numeric_limits<float>::min();
            tensor_to_pass_by_value.emplace(negative_inf_causal, negative_inf_value);
        }

        if (options.dropout_probability.has_value()) {
            float dropout_scale_value = 1.f / (1 - options.dropout_probability.value());
            float dropout_scale_inv_value = (1 - options.dropout_probability.value());
            tensor_to_pass_by_value.emplace(options.inputs.Dropout_scale, dropout_scale_value);
            tensor_to_pass_by_value.emplace(options.inputs.Dropout_scale_inv, dropout_scale_inv_value);
        }

        // one_tensor is needed for non-dropout graphs
        if (one_tensor != nullptr) {
            tensor_to_pass_by_value.emplace(one_tensor, 1.0f);
        }

        if (dQ_accum != nullptr) {
            cudaMemset(node_workspace, 0, dQ_accum_size);
            tensor_to_pass_by_value.emplace(dQ_accum, node_workspace);
            node_workspace = static_cast<char*>(node_workspace) + dQ_accum_size;
        }

        if (softmax_sum != nullptr) {
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
        for (int i = dim.size() - 1; i >= 0; --i) {
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