#pragma once

#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <unordered_map>
#include <vector>

#include "context.h"

#include "../cudnn_frontend_utils.h"

namespace cudnn_frontend {

namespace graph {
// simple structure to hold all properties of a tensor.
// Each property has a getter setter.
class Tensor_attributes {
    template <typename>
    friend class Attributes;

    std::string name;
    DataType_t data_type               = DataType_t::NOT_SET;
    std::vector<int64_t> dim           = {};
    std::vector<int64_t> stride        = {};
    bool is_virtual                    = false;
    bool is_pass_by_value              = false;
    TensorReordering_t reordering_type = TensorReordering_t::NONE;
    int64_t uid                        = 0;

   public:
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Tensor_attributes,
                                   name,
                                   data_type,
                                   dim,
                                   stride,
                                   is_virtual,
                                   is_pass_by_value,
                                   reordering_type,
                                   uid)

    Tensor_attributes() = default;

    std::string
    get_name() const {
        return name;
    }

    auto
    set_name(std::string const& value) -> Tensor_attributes& {
        name = value;
        return *this;
    }

    DataType_t
    get_data_type() const {
        return data_type;
    }

    auto
    set_data_type(DataType_t const value) -> Tensor_attributes& {
        data_type = value;
        return *this;
    }

    std::vector<int64_t>
    get_dim() const {
        return dim;
    }

    auto
    set_dim(std::vector<int64_t> const& value) -> Tensor_attributes& {
        dim = value;
        return *this;
    }

    std::vector<int64_t>
    get_stride() const {
        return stride;
    }

    auto
    set_stride(std::vector<int64_t> const& value) -> Tensor_attributes& {
        stride = value;
        return *this;
    }

    bool
    get_is_virtual() const {
        return is_virtual;
    }

    auto
    set_is_virtual(bool const value) -> Tensor_attributes& {
        is_virtual = value;
        return *this;
    }

    auto
    set_output(bool const value) -> Tensor_attributes& {
        return set_is_virtual(!value);
    }

    bool
    get_is_pass_by_value() const {
        return is_pass_by_value;
    }

    auto
    set_is_pass_by_value(bool const value) -> Tensor_attributes& {
        is_pass_by_value = value;
        return *this;
    }

    TensorReordering_t
    get_reordering_type() const {
        return reordering_type;
    }

    auto
    set_reordering_type(TensorReordering_t const value) -> Tensor_attributes& {
        reordering_type = value;
        return *this;
    }

    int64_t
    get_uid() const {
        return uid;
    }

    auto
    set_uid(int64_t value) -> Tensor_attributes& {
        uid = value;
        return *this;
    }

    auto
    fill_from_context(detail::Context const& context) -> Tensor_attributes& {
        if (get_data_type() == DataType_t::NOT_SET) {
            if (get_is_virtual()) {
                set_data_type(context.get_intermediate_data_type());
            } else {
                set_data_type(context.get_io_data_type());
            }
        }
        return *this;
    }
};

class Batchnorm_attributes;
class Batchnorm_backward_attributes;

template <typename DerivedT>
class Attributes {
    DerivedT&
    self() {
        return *static_cast<DerivedT*>(this);
    }
    DerivedT const&
    self() const {
        return *static_cast<DerivedT const*>(this);
    }

   protected:
    void
    fill_from_context(detail::Context const& context) {
        auto derived = static_cast<DerivedT const*>(this);
        for (auto& [name, tensor] : derived->inputs) {
            if (tensor) {
                tensor->fill_from_context(context);
            }
        }
        for (auto& [name, tensor] : derived->outputs) {
            if (tensor) {
                tensor->fill_from_context(context);
            }
        }
        // Handle special case of BN where peer_stats is also an input
        if constexpr (std::is_same_v<DerivedT, Batchnorm_attributes> ||
                      std::is_same_v<DerivedT, Batchnorm_backward_attributes>) {
            for (auto& tensor : derived->peer_stats) {
                if (tensor) {
                    tensor->fill_from_context(context);
                }
            }
        }

        if (compute_data_type == DataType_t::NOT_SET) {
            set_compute_data_type(context.get_compute_data_type());
        }
    }

   public:
    std::string name;
    DataType_t compute_data_type = DataType_t::NOT_SET;

    DerivedT&
    set_name(std::string const& value) {
        name = value;
        return self();
    }

    DerivedT&
    set_compute_data_type(DataType_t value) {
        compute_data_type = value;
        return self();
    }

    // Common input tensor validate functions
    error_t
    validate_inputs() const {
        auto derived = static_cast<DerivedT const*>(this);
        for (auto const& [enum_name, tensor] : derived->inputs) {
            if (tensor) {
                RETURN_CUDNN_FRONTEND_ERROR_IF(tensor->dim.empty(),
                                               error_code_t::ATTRIBUTE_NOT_SET,
                                               "Tensor '" + tensor->name + "' dims not set.");
                RETURN_CUDNN_FRONTEND_ERROR_IF(tensor->stride.empty(),
                                               error_code_t::ATTRIBUTE_NOT_SET,
                                               "Tensor '" + tensor->name + "' strides not set.");
            }
        }

        // Handle special case of BN where peer_stats is also an input
        if constexpr (std::is_same_v<DerivedT, Batchnorm_attributes> ||
                      std::is_same_v<DerivedT, Batchnorm_backward_attributes>) {
            for (auto const& tensor : derived->peer_stats) {
                if (tensor) {
                    RETURN_CUDNN_FRONTEND_ERROR_IF(
                        tensor->dim.empty(), error_code_t::ATTRIBUTE_NOT_SET, "peer_stats dims not set.");
                    RETURN_CUDNN_FRONTEND_ERROR_IF(
                        tensor->stride.empty(), error_code_t::ATTRIBUTE_NOT_SET, "peer_stats strides not set.");
                }
            }
        }

        return {error_code_t::OK, ""};
    }
};

class Operation {
   public:
    enum class Tag {
        BN,
        BN_inference,
        BN_finalize,
        Conv_fprop,
        Conv_dgrad,
        Conv_wgrad,
        DBN,
        DLN,
        DIN,
        DBN_weight,
        DRMSNorm,
        Genstats,
        LN,
        IN,
        Matmul,
        Pointwise,
        Reduction,
        Rng,
        RMSNorm,
        Reshape,
        Scaled_dot_product_attention,
        Scaled_dot_product_flash_attention,
        Scaled_dot_product_flash_attention_backward,
        Softmax,
    };
    Tag tag;

    std::string name;
    DataType_t compute_data_type = DataType_t::NOT_SET;

    Operation(Tag t) : tag(t) {}

    std::string const
    get_name() const {
        return name;
    }

    DataType_t
    get_compute_data_type() const {
        return compute_data_type;
    }

    virtual ~Operation() = default;
};

NLOHMANN_JSON_SERIALIZE_ENUM(
    Operation::Tag,
    {
        {Operation::Tag::BN, "BN"},
        {Operation::Tag::BN_inference, "BN_inference"},
        {Operation::Tag::BN_finalize, "BN_finalize"},
        {Operation::Tag::Conv_fprop, "Conv_fprop"},
        {Operation::Tag::Conv_dgrad, "Conv_dgrad"},
        {Operation::Tag::Conv_wgrad, "Conv_wgrad"},
        {Operation::Tag::DBN, "DBN"},
        {Operation::Tag::DBN_weight, "DBN_weight"},
        {Operation::Tag::Genstats, "Genstats"},
        {Operation::Tag::LN, "LN"},
        {Operation::Tag::Matmul, "Matmul"},
        {Operation::Tag::Pointwise, "Pointwise"},
        {Operation::Tag::Reduction, "Reduction"},
        {Operation::Tag::RMSNorm, "RMSNorm"},
        {Operation::Tag::Rng, "Rng"},
        {Operation::Tag::Reshape, "Reshape"},
        {Operation::Tag::Scaled_dot_product_attention, "Scaled_dot_product_attention"},
        {Operation::Tag::Scaled_dot_product_flash_attention, "Scaled_dot_product_flash_attention"},
        {Operation::Tag::Scaled_dot_product_flash_attention_backward, "Scaled_dot_product_flash_attention_backward"},
        {Operation::Tag::Softmax, "Softmax"},
    })

class BN_finalize_attributes : public Attributes<BN_finalize_attributes> {
    friend class Attributes<BN_finalize_attributes>;
    friend class BatchNormFinalizeNode;
    friend class Graph;

    enum class input_names {
        SUM,
        SQ_SUM,
        SCALE,
        BIAS,
        EPSILON,
        ACCUM_COUNT,
        PREV_RUNNING_MEAN,
        PREV_RUNNING_VAR,
        MOMENTUM
    };
    std::unordered_map<input_names, std::shared_ptr<Tensor_attributes>> inputs;

    enum class output_names { EQ_SCALE, EQ_BIAS, MEAN, INV_VARIANCE, NEXT_RUNNING_MEAN, NEXT_RUNNING_VAR };
    std::unordered_map<output_names, std::shared_ptr<Tensor_attributes>> outputs;

   public:
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(BN_finalize_attributes, name, inputs, outputs)

    BN_finalize_attributes&
    set_previous_running_stats(std::shared_ptr<Tensor_attributes>& mean,
                               std::shared_ptr<Tensor_attributes>& variance,
                               std::shared_ptr<Tensor_attributes>& momentum) {
        inputs[BN_finalize_attributes::input_names::PREV_RUNNING_MEAN] = mean;
        inputs[BN_finalize_attributes::input_names::PREV_RUNNING_VAR]  = variance;
        inputs[BN_finalize_attributes::input_names::MOMENTUM]          = momentum;
        return *this;
    }
};

class Genstats_attributes : public Attributes<Genstats_attributes> {
    friend class Attributes<Genstats_attributes>;
    friend class GenstatsNode;
    friend class Graph;

    enum class input_names { X };
    std::unordered_map<input_names, std::shared_ptr<Tensor_attributes>> inputs;

    enum class output_names { SUM, SQ_SUM };
    std::unordered_map<output_names, std::shared_ptr<Tensor_attributes>> outputs;

   public:
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Genstats_attributes, name, inputs, outputs)
};

class Conv_fprop_attributes : public Attributes<Conv_fprop_attributes> {
    friend class Attributes<Conv_fprop_attributes>;
    friend class ConvolutionNode;
    friend class Graph;

    enum class input_names { X, W };
    std::unordered_map<input_names, std::shared_ptr<Tensor_attributes>> inputs;

    enum class output_names { Y };
    std::unordered_map<output_names, std::shared_ptr<Tensor_attributes>> outputs;

    std::vector<int64_t> padding;
    std::vector<int64_t> stride;
    std::vector<int64_t> dilation;

   public:
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Conv_fprop_attributes, name, inputs, outputs, padding, stride, dilation)

    std::vector<int64_t>
    get_padding() const {
        return padding;
    }

    Conv_fprop_attributes&
    set_padding(std::vector<int64_t> value) {
        padding = value;
        return *this;
    }

    std::vector<int64_t>
    get_stride() const {
        return stride;
    }

    Conv_fprop_attributes&
    set_stride(std::vector<int64_t> value) {
        stride = value;
        return *this;
    }

    std::vector<int64_t>
    get_dilation() const {
        return dilation;
    }

    Conv_fprop_attributes&
    set_dilation(std::vector<int64_t> value) {
        dilation = value;
        return *this;
    }
};

class Batchnorm_backward_attributes : public Attributes<Batchnorm_backward_attributes> {
    friend class Attributes<Batchnorm_backward_attributes>;
    friend class DBNNode;
    friend class Graph;

    enum class input_names { DY, X, SCALE, MEAN, INV_VARIANCE };
    std::unordered_map<input_names, std::shared_ptr<Tensor_attributes>> inputs;
    // Only special case where one of the inputs is a vector.
    std::vector<std::shared_ptr<Tensor_attributes>> peer_stats;

    enum class output_names { DX, DSCALE, DBIAS };
    std::unordered_map<output_names, std::shared_ptr<Tensor_attributes>> outputs;

   public:
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Batchnorm_backward_attributes, name, inputs, outputs)

    Batchnorm_backward_attributes&
    set_saved_mean_and_inv_variance(std::shared_ptr<Tensor_attributes> mean,
                                    std::shared_ptr<Tensor_attributes> inv_variance) {
        inputs[Batchnorm_backward_attributes::input_names::MEAN]         = mean;
        inputs[Batchnorm_backward_attributes::input_names::INV_VARIANCE] = inv_variance;
        return *this;
    }

    Batchnorm_backward_attributes&
    set_peer_stats(std::vector<std::shared_ptr<Tensor_attributes>> const& input_peer_stats) {
        peer_stats = input_peer_stats;
        return *this;
    }
};

class DBN_weight_attributes : public Operation {
   public:
    struct Inputs {
        std::shared_ptr<Tensor_attributes> X;
        std::shared_ptr<Tensor_attributes> MEAN;
        std::shared_ptr<Tensor_attributes> INV_VARIANCE;
        std::shared_ptr<Tensor_attributes> SCALE;
        std::shared_ptr<Tensor_attributes> DY;
    } inputs;

    struct Outputs {
        std::shared_ptr<Tensor_attributes> DSCALE;
        std::shared_ptr<Tensor_attributes> DBIAS;
        std::shared_ptr<Tensor_attributes> EQ_SCALE_DY;
        std::shared_ptr<Tensor_attributes> EQ_SCALE_X;
        std::shared_ptr<Tensor_attributes> EQ_BIAS;
    } outputs;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Inputs, X, MEAN, INV_VARIANCE, SCALE, DY)

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Outputs, DSCALE, DBIAS, EQ_SCALE_DY, EQ_SCALE_X, EQ_BIAS)

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(DBN_weight_attributes, name, tag, inputs, outputs)

    DBN_weight_attributes() : Operation(Tag::DBN_weight) {}

    DBN_weight_attributes&
    set_name(std::string const& value) {
        name = value;
        return *this;
    }

    DBN_weight_attributes&
    set_compute_data_type(DataType_t value) {
        compute_data_type = value;
        return *this;
    }

    void
    make_outputs(std::function<std::shared_ptr<Tensor_attributes>(std::string const&)> output_tensor) {
        outputs.DSCALE      = output_tensor(name + "_dscale_output");
        outputs.DBIAS       = output_tensor(name + "_dbias_output");
        outputs.EQ_SCALE_DY = output_tensor(name + "_eq_scale_dy_output");
        outputs.EQ_SCALE_X  = output_tensor(name + "_eq_scale_x_output");
        outputs.EQ_BIAS     = output_tensor(name + "_eq_bias_output");
    }

    auto
    fill_from_context(detail::Context const& context) -> DBN_weight_attributes& {
        // Fill node's tensors
        inputs.X->fill_from_context(context);
        inputs.MEAN->fill_from_context(context);
        inputs.INV_VARIANCE->fill_from_context(context);
        inputs.SCALE->fill_from_context(context);
        inputs.DY->fill_from_context(context);
        outputs.DSCALE->fill_from_context(context);
        outputs.DBIAS->fill_from_context(context);
        outputs.EQ_SCALE_DY->fill_from_context(context);
        outputs.EQ_SCALE_X->fill_from_context(context);
        outputs.EQ_BIAS->fill_from_context(context);

        // Fill this node
        if (get_compute_data_type() == DataType_t::NOT_SET) {
            set_compute_data_type(context.get_compute_data_type());
        }
        return *this;
    }
};

class Conv_dgrad_attributes : public Attributes<Conv_dgrad_attributes> {
    friend class Attributes<Conv_dgrad_attributes>;
    friend class DgradNode;
    friend class Graph;

    enum class input_names { DY, W };
    std::unordered_map<input_names, std::shared_ptr<Tensor_attributes>> inputs;

    enum class output_names { DX };
    std::unordered_map<output_names, std::shared_ptr<Tensor_attributes>> outputs;

    std::vector<int64_t> padding;
    std::vector<int64_t> stride;
    std::vector<int64_t> dilation;

   public:
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Conv_dgrad_attributes, name, inputs, outputs, padding, stride, dilation)

    std::vector<int64_t>
    get_padding() const {
        return padding;
    }

    Conv_dgrad_attributes&
    set_padding(std::vector<int64_t> value) {
        padding = value;
        return *this;
    }

    std::vector<int64_t>
    get_stride() const {
        return stride;
    }

    Conv_dgrad_attributes&
    set_stride(std::vector<int64_t> value) {
        stride = value;
        return *this;
    }

    std::vector<int64_t>
    get_dilation() const {
        return dilation;
    }

    Conv_dgrad_attributes&
    set_dilation(std::vector<int64_t> value) {
        dilation = value;
        return *this;
    }
};

class Matmul_attributes : public Attributes<Matmul_attributes> {
    friend class Attributes<Matmul_attributes>;
    friend class MatmulNode;
    friend class ScaledDotProductFlashAttentionNode;
    friend class ScaledDotProductFlashAttentionBackwardNode;
    friend class Graph;

    enum class input_names { A, B, M_override, N_override, K_override };
    std::unordered_map<input_names, std::shared_ptr<Tensor_attributes>> inputs;

    enum class output_names { C };
    std::unordered_map<output_names, std::shared_ptr<Tensor_attributes>> outputs;

   public:
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Matmul_attributes, name, inputs, outputs)
};

class Pointwise_attributes : public Attributes<Pointwise_attributes> {
    friend class Attributes<Pointwise_attributes>;
    friend class PointwiseNode;
    friend class SoftmaxNode;
    friend class ScaledDotProductFlashAttentionNode;
    friend class ScaledDotProductFlashAttentionBackwardNode;
    friend class Graph;

    enum class input_names { IN_0, IN_1, IN_2 };
    std::unordered_map<input_names, std::shared_ptr<Tensor_attributes>> inputs;

    enum class output_names { OUT_0 };
    std::unordered_map<output_names, std::shared_ptr<Tensor_attributes>> outputs;

    PointwiseMode_t mode = PointwiseMode_t::NOT_SET;
    std::optional<int64_t> axis;

    std::optional<float> relu_lower_clip_slope;

   public:
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Pointwise_attributes, name, inputs, outputs, mode, axis)

    Pointwise_attributes&
    set_mode(PointwiseMode_t const value) {
        mode = value;
        return *this;
    }

    std::optional<int64_t>
    get_axis() const {
        return axis;
    }

    Pointwise_attributes&
    set_axis(int64_t const axis) {
        this->axis = axis;
        return *this;
    }

    Pointwise_attributes&
    set_relu_lower_clip_slope(float const negative_slope) {
        this->relu_lower_clip_slope = negative_slope;
        return *this;
    }
};

class Instancenorm_backward_attributes : public Attributes<Instancenorm_backward_attributes> {
    friend class Attributes<Instancenorm_backward_attributes>;
    friend class DINNode;
    friend class Graph;

    enum class input_names { DY, X, SCALE, MEAN, INV_VARIANCE };
    std::unordered_map<input_names, std::shared_ptr<Tensor_attributes>> inputs;

    enum class output_names { DX, DSCALE, DBIAS };
    std::unordered_map<output_names, std::shared_ptr<Tensor_attributes>> outputs;

   public:
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Instancenorm_backward_attributes, name, inputs, outputs)

    Instancenorm_backward_attributes&
    set_saved_mean_and_inv_variance(std::shared_ptr<Tensor_attributes> mean,
                                    std::shared_ptr<Tensor_attributes> inv_variance) {
        inputs[Instancenorm_backward_attributes::input_names::MEAN]         = mean;
        inputs[Instancenorm_backward_attributes::input_names::INV_VARIANCE] = inv_variance;
        return *this;
    }
};

class Layernorm_backward_attributes : public Attributes<Layernorm_backward_attributes> {
    friend class Attributes<Layernorm_backward_attributes>;
    friend class DLNNode;
    friend class Graph;

    enum class input_names { DY, X, SCALE, MEAN, INV_VARIANCE };
    std::unordered_map<input_names, std::shared_ptr<Tensor_attributes>> inputs;

    enum class output_names { DX, DSCALE, DBIAS };
    std::unordered_map<output_names, std::shared_ptr<Tensor_attributes>> outputs;

   public:
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Layernorm_backward_attributes, name, inputs, outputs)

    Layernorm_backward_attributes&
    set_saved_mean_and_inv_variance(std::shared_ptr<Tensor_attributes> mean,
                                    std::shared_ptr<Tensor_attributes> inv_variance) {
        inputs[Layernorm_backward_attributes::input_names::MEAN]         = mean;
        inputs[Layernorm_backward_attributes::input_names::INV_VARIANCE] = inv_variance;
        return *this;
    }
};

class Layernorm_attributes : public Attributes<Layernorm_attributes> {
    friend class Attributes<Layernorm_attributes>;
    friend class LayerNormNode;
    friend class Graph;

    enum class input_names { X, SCALE, BIAS, EPSILON };
    std::unordered_map<input_names, std::shared_ptr<Tensor_attributes>> inputs;

    enum class output_names { Y, MEAN, INV_VARIANCE };
    std::unordered_map<output_names, std::shared_ptr<Tensor_attributes>> outputs;

    NormFwdPhase_t forward_phase = NormFwdPhase_t::NOT_SET;

   public:
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Layernorm_attributes, name, inputs, outputs, forward_phase)

    Layernorm_attributes&
    set_forward_phase(NormFwdPhase_t const value) {
        forward_phase = value;
        return *this;
    }

    Layernorm_attributes&
    set_epsilon(std::shared_ptr<Tensor_attributes>& value) {
        inputs[Layernorm_attributes::input_names::EPSILON] = value;
        return *this;
    }
};

class Instancenorm_attributes : public Attributes<Instancenorm_attributes> {
    friend class Attributes<Instancenorm_attributes>;
    friend class InstanceNormNode;
    friend class Graph;

    enum class input_names { X, SCALE, BIAS, EPSILON };
    std::unordered_map<input_names, std::shared_ptr<Tensor_attributes>> inputs;

    enum class output_names { Y, MEAN, INV_VARIANCE };
    std::unordered_map<output_names, std::shared_ptr<Tensor_attributes>> outputs;

    NormFwdPhase_t forward_phase = NormFwdPhase_t::NOT_SET;

   public:
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Instancenorm_attributes, name, inputs, outputs, forward_phase)

    Instancenorm_attributes&
    set_forward_phase(NormFwdPhase_t const value) {
        forward_phase = value;
        return *this;
    }

    Instancenorm_attributes&
    set_epsilon(std::shared_ptr<Tensor_attributes>& value) {
        inputs[Instancenorm_attributes::input_names::EPSILON] = value;
        return *this;
    }
};

class Batchnorm_attributes : public Attributes<Batchnorm_attributes> {
    friend class Attributes<Batchnorm_attributes>;
    friend class BatchNormNode;
    friend class Graph;

    enum class input_names { X, SCALE, BIAS, PREV_RUNNING_MEAN, PREV_RUNNING_VAR, EPSILON, MOMENTUM };
    std::unordered_map<input_names, std::shared_ptr<Tensor_attributes>> inputs;
    // Only special case where one of the inputs is a vector.
    std::vector<std::shared_ptr<Tensor_attributes>> peer_stats;

    enum class output_names { Y, MEAN, INV_VARIANCE, NEXT_RUNNING_MEAN, NEXT_RUNNING_VAR };
    std::unordered_map<output_names, std::shared_ptr<Tensor_attributes>> outputs;

    NormFwdPhase_t forward_phase = NormFwdPhase_t::NOT_SET;

   public:
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Batchnorm_attributes, name, inputs, peer_stats, outputs, forward_phase)

    Batchnorm_attributes&
    set_forward_phase(NormFwdPhase_t const value) {
        forward_phase = value;
        return *this;
    }

    Batchnorm_attributes&
    set_previous_running_stats(std::shared_ptr<Tensor_attributes>& mean,
                               std::shared_ptr<Tensor_attributes>& variance,
                               std::shared_ptr<Tensor_attributes>& momentum) {
        inputs[input_names::PREV_RUNNING_MEAN] = mean;
        inputs[input_names::PREV_RUNNING_VAR]  = variance;
        inputs[input_names::MOMENTUM]          = momentum;
        return *this;
    }

    Batchnorm_attributes&
    set_epsilon(std::shared_ptr<Tensor_attributes>& value) {
        inputs[input_names::EPSILON] = value;
        return *this;
    }

    Batchnorm_attributes&
    set_peer_stats(std::vector<std::shared_ptr<Tensor_attributes>> const& input_peer_stats) {
        peer_stats = input_peer_stats;
        return *this;
    }
};

class Batchnorm_inference_attributes : public Operation {
   public:
    struct Inputs {
        std::shared_ptr<Tensor_attributes> X;
        std::shared_ptr<Tensor_attributes> MEAN;
        std::shared_ptr<Tensor_attributes> INV_VARIANCE;
        std::shared_ptr<Tensor_attributes> SCALE;
        std::shared_ptr<Tensor_attributes> BIAS;
    } inputs;

    struct Outputs {
        std::shared_ptr<Tensor_attributes> Y;
    } outputs;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Inputs, X, MEAN, INV_VARIANCE, SCALE, BIAS)

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Outputs, Y)

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Batchnorm_inference_attributes, name, tag, inputs, outputs)

    Batchnorm_inference_attributes() : Operation(Tag::BN_inference) {}

    Batchnorm_inference_attributes&
    set_name(std::string const& value) {
        name = value;
        return *this;
    }

    Batchnorm_inference_attributes&
    set_compute_data_type(DataType_t value) {
        compute_data_type = value;
        return *this;
    }

    auto
    fill_from_context(detail::Context const& context) -> Batchnorm_inference_attributes& {
        // Fill node's tensors
        inputs.X->fill_from_context(context);
        inputs.SCALE->fill_from_context(context);
        inputs.BIAS->fill_from_context(context);
        inputs.MEAN->fill_from_context(context);
        inputs.INV_VARIANCE->fill_from_context(context);

        outputs.Y->fill_from_context(context);

        if (get_compute_data_type() == DataType_t::NOT_SET) {
            set_compute_data_type(context.get_compute_data_type());
        }
        return *this;
    }
};

class Reduction_attributes : public Attributes<Reduction_attributes> {
    friend class Attributes<Reduction_attributes>;
    friend class ReductionNode;
    friend class SoftmaxNode;
    friend class ScaledDotProductFlashAttentionBackwardNode;
    friend class Graph;

    enum class input_names { X };
    std::unordered_map<input_names, std::shared_ptr<Tensor_attributes>> inputs;

    enum class output_names { Y };
    std::unordered_map<output_names, std::shared_ptr<Tensor_attributes>> outputs;

    std::optional<ReductionMode_t> mode;

   public:
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Reduction_attributes, name, inputs, outputs, mode)

    std::optional<ReductionMode_t>
    get_mode() const {
        return mode;
    }

    Reduction_attributes&
    set_mode(ReductionMode_t value) {
        mode = value;
        return *this;
    }
};

class Rng_attributes : public Attributes<Rng_attributes> {
    friend class Attributes<Rng_attributes>;
    friend class RngNode;
    friend class ScaledDotProductFlashAttentionNode;
    friend class ScaledDotProductFlashAttentionBackwardNode;
    friend class Graph;

    enum class input_names { Seed, Offset };
    std::unordered_map<input_names, std::shared_ptr<Tensor_attributes>> inputs;

    enum class output_names { Y };
    std::unordered_map<output_names, std::shared_ptr<Tensor_attributes>> outputs;

    RngDistribution_t distribution = RngDistribution_t::NOT_SET;
    std::vector<int64_t> dim       = {};
    std::vector<int64_t> stride    = {};
    std::optional<int64_t> seed;
    std::optional<double> bernoulli_probability;

   public:
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Rng_attributes,
                                   name,
                                   inputs,
                                   outputs,
                                   distribution,
                                   dim,
                                   stride,
                                   seed,
                                   bernoulli_probability)

    std::vector<int64_t>
    get_dim() const {
        return dim;
    }

    auto
    set_dim(std::vector<int64_t> const& value) -> Rng_attributes& {
        dim = value;
        return *this;
    }

    std::vector<int64_t>
    get_stride() const {
        return stride;
    }

    auto
    set_stride(std::vector<int64_t> const& value) -> Rng_attributes& {
        stride = value;
        return *this;
    }

    RngDistribution_t
    get_distribution() const {
        return distribution;
    }

    Rng_attributes&
    set_distribution(RngDistribution_t value) {
        distribution = value;
        return *this;
    }

    std::optional<int64_t>
    get_seed() const {
        return seed;
    }

    Rng_attributes&
    set_seed(std::optional<int64_t> value) {
        seed = value;
        return *this;
    }

    std::optional<double>
    get_bernoulli_probability() const {
        return bernoulli_probability;
    }

    Rng_attributes&
    set_bernoulli_probability(std::optional<double> value) {
        bernoulli_probability = value;
        return *this;
    }
};

class Reshape_attributes : public Operation {
   public:
    struct Inputs {
        std::shared_ptr<Tensor_attributes> X;
    } inputs;

    struct Outputs {
        std::shared_ptr<Tensor_attributes> Y;
    } outputs;

    std::vector<int64_t> dim    = {};
    std::vector<int64_t> stride = {};

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Inputs, X)

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Outputs, Y)

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Reshape_attributes, name, tag, inputs, outputs, dim, stride)

    Reshape_attributes() : Operation(Tag::Reshape) {}

    std::vector<int64_t>
    get_dim() const {
        return dim;
    }

    auto
    set_dim(std::vector<int64_t> const& value) -> Reshape_attributes& {
        dim = value;
        return *this;
    }

    std::vector<int64_t>
    get_stride() const {
        return stride;
    }

    auto
    set_stride(std::vector<int64_t> const& value) -> Reshape_attributes& {
        stride = value;
        return *this;
    }

    Reshape_attributes&
    set_name(std::string const& value) {
        name = value;
        return *this;
    }

    Reshape_attributes&
    set_compute_data_type(DataType_t value) {
        compute_data_type = value;
        return *this;
    }

    auto
    fill_from_context(detail::Context const& context) -> Reshape_attributes& {
        inputs.X->fill_from_context(context);
        outputs.Y->fill_from_context(context);

        // Fill this node
        if (get_compute_data_type() == DataType_t::NOT_SET) {
            set_compute_data_type(context.get_compute_data_type());
        }
        return *this;
    }
};

class Rmsnorm_attributes : public Attributes<Rmsnorm_attributes> {
    friend class Attributes<Rmsnorm_attributes>;
    friend class RMSNormNode;
    friend class Graph;

    enum class input_names { X, SCALE, BIAS, EPSILON };
    std::unordered_map<input_names, std::shared_ptr<Tensor_attributes>> inputs;

    enum class output_names { Y, INV_VARIANCE };
    std::unordered_map<output_names, std::shared_ptr<Tensor_attributes>> outputs;

    NormFwdPhase_t forward_phase = NormFwdPhase_t::NOT_SET;

   public:
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Rmsnorm_attributes, name, inputs, outputs, forward_phase)

    Rmsnorm_attributes&
    set_forward_phase(NormFwdPhase_t const value) {
        forward_phase = value;
        return *this;
    }

    Rmsnorm_attributes&
    set_bias(std::shared_ptr<Tensor_attributes>& value) {
        inputs[Rmsnorm_attributes::input_names::BIAS] = value;
        return *this;
    }

    Rmsnorm_attributes&
    set_epsilon(std::shared_ptr<Tensor_attributes>& value) {
        inputs[Rmsnorm_attributes::input_names::EPSILON] = value;
        return *this;
    }
};

class Rmsnorm_backward_attributes : public Attributes<Rmsnorm_backward_attributes> {
    friend class Attributes<Rmsnorm_backward_attributes>;
    friend class DRMSNormNode;
    friend class Graph;

    enum class input_names { DY, X, SCALE, INV_VARIANCE };
    std::unordered_map<input_names, std::shared_ptr<Tensor_attributes>> inputs;

    enum class output_names { DX, DSCALE, DBIAS };
    std::unordered_map<output_names, std::shared_ptr<Tensor_attributes>> outputs;
    std::optional<bool> use_dbias;

   public:
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Rmsnorm_backward_attributes, name, inputs, outputs)

    Rmsnorm_backward_attributes&
    has_dbias(bool value) {
        use_dbias = value;
        return *this;
    }
};

class Scaled_dot_product_attention_attributes : public Operation {
   public:
    struct Inputs {
        std::shared_ptr<Tensor_attributes> Q;
        std::shared_ptr<Tensor_attributes> K;
        std::shared_ptr<Tensor_attributes> Attn_scale;
        std::shared_ptr<Tensor_attributes> Bias;  // Optional bias after bmm1
        std::shared_ptr<Tensor_attributes> V;
        std::shared_ptr<Tensor_attributes> SEQ_LEN_Q;
        std::shared_ptr<Tensor_attributes> SEQ_LEN_KV;
        std::shared_ptr<Tensor_attributes> Mask;
        std::shared_ptr<Tensor_attributes> Dropout_mask;
        std::shared_ptr<Tensor_attributes> Dropout_scale;
    } inputs;

    struct Outputs {
        std::shared_ptr<Tensor_attributes> O;
        std::shared_ptr<Tensor_attributes>
            S;  // softmax output dumped when is_inference false. Users first need to check whether its nullptr.
    } outputs;

    std::optional<bool> is_inference;
    bool padding_mask = false;
    bool causal_mask  = false;
    std::optional<float> dropout_probability;
    int64_t seed;
    float dropout_scale = 1.f;

   public:
    Scaled_dot_product_attention_attributes() : Operation(Tag::Scaled_dot_product_attention), is_inference(false) {}

    Scaled_dot_product_attention_attributes&
    set_is_inference(bool const value) {
        is_inference = value;
        return *this;
    }

    Scaled_dot_product_attention_attributes&
    set_seq_len_q(std::shared_ptr<Tensor_attributes> value) {
        inputs.SEQ_LEN_Q = value;
        return *this;
    }

    Scaled_dot_product_attention_attributes&
    set_seq_len_kv(std::shared_ptr<Tensor_attributes> value) {
        inputs.SEQ_LEN_KV = value;
        return *this;
    }

    Scaled_dot_product_attention_attributes&
    set_padding_mask(bool const value) {
        padding_mask = value;
        return *this;
    }

    Scaled_dot_product_attention_attributes&
    set_causal_mask(bool const value) {
        causal_mask = value;
        return *this;
    }

    Scaled_dot_product_attention_attributes&
    set_attn_scale(std::shared_ptr<Tensor_attributes> value) {
        inputs.Attn_scale = value;
        return *this;
    }

    Scaled_dot_product_attention_attributes&
    set_bias(std::shared_ptr<Tensor_attributes> bias) {
        inputs.Bias = bias;
        return *this;
    }

    Scaled_dot_product_attention_attributes&
    set_dropout(float const probability, int64_t const seed_) {
        dropout_probability = probability;
        seed                = seed_;
        return *this;
    }

    Scaled_dot_product_attention_attributes&
    set_dropout(std::shared_ptr<Tensor_attributes> mask, std::shared_ptr<Tensor_attributes> scale) {
        inputs.Dropout_mask  = mask;
        inputs.Dropout_scale = scale;
        return *this;
    }

    Scaled_dot_product_attention_attributes&
    set_compute_data_type(DataType_t const value) {
        compute_data_type = value;
        return *this;
    }

    Scaled_dot_product_attention_attributes&
    set_name(std::string const& value) {
        name = value;
        return *this;
    }

    Scaled_dot_product_attention_attributes&
    fill_from_context(detail::Context const& context) {
        // Fill node's tensors
        inputs.Q->fill_from_context(context);
        inputs.K->fill_from_context(context);
        inputs.V->fill_from_context(context);
        inputs.SEQ_LEN_Q->fill_from_context(context);
        inputs.SEQ_LEN_KV->fill_from_context(context);
        outputs.O->fill_from_context(context);

        // Fill this node
        if (get_compute_data_type() == DataType_t::NOT_SET) {
            set_compute_data_type(context.get_compute_data_type());
        }
        return *this;
    }
};

class Scaled_dot_product_flash_attention_attributes : public Operation {
   public:
    struct Inputs {
        std::shared_ptr<Tensor_attributes> Q;
        std::shared_ptr<Tensor_attributes> K;
        std::shared_ptr<Tensor_attributes> V;
        std::shared_ptr<Tensor_attributes> Attn_scale;
        std::shared_ptr<Tensor_attributes> Bias;
        std::shared_ptr<Tensor_attributes> SEQ_LEN_Q;
        std::shared_ptr<Tensor_attributes> SEQ_LEN_KV;
        std::shared_ptr<Tensor_attributes> Seed;
        std::shared_ptr<Tensor_attributes> Offset;
        std::shared_ptr<Tensor_attributes> Dropout_mask;
        std::shared_ptr<Tensor_attributes> Dropout_scale;
    } inputs;

    struct Outputs {
        std::shared_ptr<Tensor_attributes> O;
        std::shared_ptr<Tensor_attributes> Stats;  // softmax stats dumped when in forward training mode. Users first
                                                   // need to check whether its nullptr.
    } outputs;

    std::optional<bool> is_inference;
    bool alibi_mask   = false;
    bool padding_mask = false;
    bool causal_mask  = false;
    std::optional<float> dropout_probability;
    std::optional<float> attn_scale_value;

    Scaled_dot_product_flash_attention_attributes() : Operation(Tag::Scaled_dot_product_flash_attention) {}

    Scaled_dot_product_flash_attention_attributes&
    set_is_inference(bool const value) {
        is_inference = value;
        return *this;
    }

    Scaled_dot_product_flash_attention_attributes&
    set_attn_scale(std::shared_ptr<Tensor_attributes> value) {
        inputs.Attn_scale = value;
        return *this;
    }

    Scaled_dot_product_flash_attention_attributes&
    set_attn_scale(float const value) {
        attn_scale_value = value;
        return *this;
    }

    Scaled_dot_product_flash_attention_attributes&
    set_bias(std::shared_ptr<Tensor_attributes> value) {
        inputs.Bias = value;
        return *this;
    }

    Scaled_dot_product_flash_attention_attributes&
    set_alibi_mask(bool const value) {
        alibi_mask = value;
        return *this;
    }

    Scaled_dot_product_flash_attention_attributes&
    set_padding_mask(bool const value) {
        padding_mask = value;
        return *this;
    }

    Scaled_dot_product_flash_attention_attributes&
    set_seq_len_q(std::shared_ptr<Tensor_attributes> value) {
        inputs.SEQ_LEN_Q = value;
        return *this;
    }

    Scaled_dot_product_flash_attention_attributes&
    set_seq_len_kv(std::shared_ptr<Tensor_attributes> value) {
        inputs.SEQ_LEN_KV = value;
        return *this;
    }

    Scaled_dot_product_flash_attention_attributes&
    set_causal_mask(bool const value) {
        causal_mask = value;
        return *this;
    }

    Scaled_dot_product_flash_attention_attributes&
    set_dropout(float const probability,
                std::shared_ptr<Tensor_attributes> seed,
                std::shared_ptr<Tensor_attributes> offset) {
        dropout_probability = probability;
        inputs.Seed         = seed;
        inputs.Offset       = offset;
        return *this;
    }

    Scaled_dot_product_flash_attention_attributes&
    set_dropout(std::shared_ptr<Tensor_attributes> mask, std::shared_ptr<Tensor_attributes> scale) {
        inputs.Dropout_mask  = mask;
        inputs.Dropout_scale = scale;
        return *this;
    }

    Scaled_dot_product_flash_attention_attributes&
    set_compute_data_type(DataType_t const value) {
        compute_data_type = value;
        return *this;
    }

    Scaled_dot_product_flash_attention_attributes&
    set_name(std::string const& value) {
        name = value;
        return *this;
    }

    Scaled_dot_product_flash_attention_attributes&
    fill_from_context(detail::Context const& context) {
        // Fill node's tensors
        inputs.Q->fill_from_context(context);
        inputs.K->fill_from_context(context);
        inputs.V->fill_from_context(context);
        outputs.O->fill_from_context(context);

        // Fill this node
        if (get_compute_data_type() == DataType_t::NOT_SET) {
            set_compute_data_type(context.get_compute_data_type());
        }
        return *this;
    }
};

class Scaled_dot_product_flash_attention_backward_attributes : public Operation {
   public:
    struct Inputs {
        std::shared_ptr<Tensor_attributes> Q;
        std::shared_ptr<Tensor_attributes> K;
        std::shared_ptr<Tensor_attributes> V;
        std::shared_ptr<Tensor_attributes> O;
        std::shared_ptr<Tensor_attributes> dO;
        std::shared_ptr<Tensor_attributes> Stats;
        std::shared_ptr<Tensor_attributes> Attn_scale;
        std::shared_ptr<Tensor_attributes> Bias;
        std::shared_ptr<Tensor_attributes> SEQ_LEN_Q;
        std::shared_ptr<Tensor_attributes> SEQ_LEN_KV;
        std::shared_ptr<Tensor_attributes> Seed;
        std::shared_ptr<Tensor_attributes> Offset;
        std::shared_ptr<Tensor_attributes> Dropout_mask;
        std::shared_ptr<Tensor_attributes> Dropout_scale;
        std::shared_ptr<Tensor_attributes> Dropout_scale_inv;
    } inputs;

    struct Outputs {
        std::shared_ptr<Tensor_attributes> dQ;
        std::shared_ptr<Tensor_attributes> dK;
        std::shared_ptr<Tensor_attributes> dV;
    } outputs;

    bool alibi_mask   = false;
    bool padding_mask = false;
    bool causal_mask  = false;

    std::optional<float> dropout_probability;
    std::optional<float> attn_scale_value;

   public:
    Scaled_dot_product_flash_attention_backward_attributes()
        : Operation(Tag::Scaled_dot_product_flash_attention_backward) {}

    Scaled_dot_product_flash_attention_backward_attributes&
    set_attn_scale(std::shared_ptr<Tensor_attributes> value) {
        inputs.Attn_scale = value;
        return *this;
    }

    Scaled_dot_product_flash_attention_backward_attributes&
    set_attn_scale(float const value) {
        attn_scale_value = value;
        return *this;
    }

    Scaled_dot_product_flash_attention_backward_attributes&
    set_bias(std::shared_ptr<Tensor_attributes> value) {
        inputs.Bias = value;
        return *this;
    }

    Scaled_dot_product_flash_attention_backward_attributes&
    set_alibi_mask(bool const value) {
        alibi_mask = value;
        return *this;
    }

    Scaled_dot_product_flash_attention_backward_attributes&
    set_padding_mask(bool const value) {
        padding_mask = value;
        return *this;
    }

    Scaled_dot_product_flash_attention_backward_attributes&
    set_seq_len_q(std::shared_ptr<Tensor_attributes> value) {
        inputs.SEQ_LEN_Q = value;
        return *this;
    }

    Scaled_dot_product_flash_attention_backward_attributes&
    set_seq_len_kv(std::shared_ptr<Tensor_attributes> value) {
        inputs.SEQ_LEN_KV = value;
        return *this;
    }

    Scaled_dot_product_flash_attention_backward_attributes&
    set_causal_mask(bool const value) {
        causal_mask = value;
        return *this;
    }

    Scaled_dot_product_flash_attention_backward_attributes&
    set_dropout(float const probability,
                std::shared_ptr<Tensor_attributes> seed,
                std::shared_ptr<Tensor_attributes> offset) {
        dropout_probability = probability;
        inputs.Seed         = seed;
        inputs.Offset       = offset;
        return *this;
    }

    Scaled_dot_product_flash_attention_backward_attributes&
    set_dropout(std::shared_ptr<Tensor_attributes> mask,
                std::shared_ptr<Tensor_attributes> scale,
                std::shared_ptr<Tensor_attributes> scale_inv) {
        inputs.Dropout_mask      = mask;
        inputs.Dropout_scale     = scale;
        inputs.Dropout_scale_inv = scale_inv;
        return *this;
    }

    Scaled_dot_product_flash_attention_backward_attributes&
    set_compute_data_type(DataType_t const value) {
        compute_data_type = value;
        return *this;
    }

    Scaled_dot_product_flash_attention_backward_attributes&
    set_name(std::string const& value) {
        name = value;
        return *this;
    }

    Scaled_dot_product_flash_attention_backward_attributes&
    fill_from_context(detail::Context const& context) {
        // Fill node's tensors
        inputs.Q->fill_from_context(context);
        inputs.K->fill_from_context(context);
        inputs.V->fill_from_context(context);
        inputs.O->fill_from_context(context);
        inputs.dO->fill_from_context(context);
        inputs.Stats->fill_from_context(context);
        outputs.dQ->fill_from_context(context);
        outputs.dK->fill_from_context(context);
        outputs.dV->fill_from_context(context);

        // Fill this node
        if (get_compute_data_type() == DataType_t::NOT_SET) {
            set_compute_data_type(context.get_compute_data_type());
        }
        return *this;
    }
};

class Softmax_attributes : public Attributes<Softmax_attributes> {
    friend class Attributes<Softmax_attributes>;
    friend class ScaledDotProductFlashAttentionNode;
    friend class SoftmaxNode;

    enum class input_names { P };
    std::unordered_map<input_names, std::shared_ptr<Tensor_attributes>> inputs;

    enum class output_names { S, Stats };
    std::unordered_map<output_names, std::shared_ptr<Tensor_attributes>> outputs;
    std::optional<bool> use_stats;
};

class Conv_wgrad_attributes : public Attributes<Conv_wgrad_attributes> {
    friend class Attributes<Conv_wgrad_attributes>;
    friend class WgradNode;
    friend class Graph;

    enum class input_names { DY, X };
    std::unordered_map<input_names, std::shared_ptr<Tensor_attributes>> inputs;

    enum class output_names { DW };
    std::unordered_map<output_names, std::shared_ptr<Tensor_attributes>> outputs;

    std::vector<int64_t> padding;
    std::vector<int64_t> stride;
    std::vector<int64_t> dilation;

   public:
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Conv_wgrad_attributes, name, inputs, outputs, padding, stride, dilation)

    std::vector<int64_t>
    get_padding() const {
        return padding;
    }

    Conv_wgrad_attributes&
    set_padding(std::vector<int64_t> value) {
        padding = value;
        return *this;
    }

    std::vector<int64_t>
    get_stride() const {
        return stride;
    }

    Conv_wgrad_attributes&
    set_stride(std::vector<int64_t> value) {
        stride = value;
        return *this;
    }

    std::vector<int64_t>
    get_dilation() const {
        return dilation;
    }

    Conv_wgrad_attributes&
    set_dilation(std::vector<int64_t> value) {
        dilation = value;
        return *this;
    }
};

}  // namespace graph

}  // namespace cudnn_frontend