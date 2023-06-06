#pragma once

#include <iostream>
#include <numeric>
#include <optional>
#include <unordered_map>
#include <vector>

#include "graphs/cudnn_frontend_graph_helpers.h"

namespace cudnn_frontend {

namespace graph {

// simple structure to hold all properties of a tensor.
// Each property has a getter setter.
class Tensor {
protected:
    std::string name;
    DataType_t data_type = DataType_t::NOT_SET;
    std::vector<int64_t> dim = {};
    std::vector<int64_t> stride = {};
    bool is_virtual = false;
    bool is_pass_by_value = false;
    TensorReordering_t reordering_type = TensorReordering_t::NONE;
    using uid_t = int64_t;
    uid_t uid;

public:
    bool is_dim_set = false;
    bool is_stride_set = false;
    bool is_virtual_set = false;
    bool is_pass_by_value_set = false;
    bool is_uid_set = false;

    int
    generateStrides(cudnnTensorFormat_t const filterFormat) {
        size_t const dim_count = dim.size();
        stride.resize(dim_count);
        if (filterFormat == CUDNN_TENSOR_NCHW) {
            stride[dim_count - 1] = 1;
            for (int64_t d = dim_count - 2; d >= 0; d--) {
                stride[d] = stride[d + 1] * dim[d + 1];
            }
        } else {
            // Here we assume that the format is CUDNN_TENSOR_NHWC
            stride[1]          = 1;
            stride[dim_count - 1] = stride[1] * dim[1];
            for (int64_t d = dim_count - 2; d >= 2; d--) {
                stride[d] = stride[d + 1] * dim[d + 1];
            }
            stride[0] = stride[2] * dim[2];
        }
        is_stride_set = true;
        return 0;
    }
    
    size_t 
    get_tensor_size()
    {
        size_t initialProduct = 1;
        return std::accumulate(dim.begin(), dim.end(), initialProduct, std::multiplies<int64_t>());
    }

    Tensor(const std::string &name) : name(name) {}

    std::string get_name() const {
        return name;
    }

    auto set_name(std::string const& value) -> Tensor& {
        name = value;
        return *this;
    }

    DataType_t get_data_type() const {
        return data_type;
    }

    auto set_data_type(DataType_t const value) -> Tensor& {
        data_type = value;
        return *this;
    }
    
    std::vector<int64_t> get_dim() const {
        return dim;
    }

    auto set_dim(std::vector<int64_t> const& value) -> Tensor& {
        dim = value;
        is_dim_set = true;
        return *this;
    }

    std::vector<int64_t> get_stride() const {
        return stride;
    }

    auto set_stride(std::vector<int64_t> const& value) -> Tensor& {
        stride = value;
        is_stride_set = true;
        return *this;
    }

    bool get_is_virtual() const {
        return is_virtual;
    }

    auto set_is_virtual(bool const value) -> Tensor& {
        is_virtual = value;
        is_virtual_set = true;
        return *this;
    }

    bool get_is_pass_by_value() const {
        return is_pass_by_value;
    }

    auto set_is_pass_by_value(bool const value) -> Tensor& {
        is_pass_by_value = value;
        is_pass_by_value_set = true;
        return *this;
    }
    
    TensorReordering_t get_reordering_type() const {
        return reordering_type;
    }

    auto set_reordering_type(TensorReordering_t const value) -> Tensor& {
        reordering_type = value;
        return *this;
    }

    uid_t get_uid() const {
        return uid;
    }

    auto set_uid(uid_t value) -> Tensor& {
        uid = value;
        is_uid_set = true;
        return *this;
    }

    auto fill_from_context(detail::Context const& context) -> Tensor& {
        if(get_data_type() == DataType_t::NOT_SET) {
            if(get_is_virtual()) {
                set_data_type(context.get_intermediate_data_type());
            }    
            else {
                set_data_type(context.get_io_data_type());
            }
        }
        return *this;
    }

    friend std::ostream& operator<<(std::ostream& os, const Tensor& props);
};

inline std::ostream& operator<<(std::ostream& os, const Tensor& props) {
    os << "{" 
    << " name: '" << props.get_name() << "',"
    << " dim: [";
    for(size_t i = 0; i < props.get_dim().size(); ++i) {
        os << props.get_dim()[i] << ",";
    }
    os << "],"
    << " stride: [";
    for(size_t i = 0; i < props.get_stride().size(); ++i) {
        os << props.get_stride()[i] << ",";
    }
    os << "],"
    << " is_virtual: " << props.get_is_virtual() << ","
    << " is_pass_by_value: " << props.get_is_pass_by_value() << ","
    << "}";
    return os;
}

class Operation {
public:
    enum class Tag {
        BN,
        BN_finalize,
        Conv,
        DBN_weight,
        Dgrad,
        Genstats,
        Matmul,
        Pointwise,
        Reduction,
        Scaled_dot_product_attention,
        Softmax,
        Wgrad,
    };

protected:

    std::string name;
    DataType_t compute_data_type = DataType_t::NOT_SET;

    Tag tag;
public:

    Operation(const std::string name, Tag t) : name(name), tag(t) {}

    std::string const
    get_name() const {
        return name;
    }

    Tag get_tag() const {
        return tag;
    }

    DataType_t get_compute_data_type() const {
        return compute_data_type;
    }

    virtual ~Operation() = default;
};

class BN_finalize : public Operation {
public:
    struct Inputs {
        std::shared_ptr<Tensor> SUM;
        std::shared_ptr<Tensor> SQ_SUM;
        std::shared_ptr<Tensor> MEAN;
        std::shared_ptr<Tensor> INV_VARIANCE;
        std::shared_ptr<Tensor> SCALE;
        std::shared_ptr<Tensor> BIAS;
        std::shared_ptr<Tensor> PREV_RUNNING_MEAN;
        std::shared_ptr<Tensor> PREV_RUNNING_VAR;
        std::shared_ptr<Tensor> EPSILON;
        std::shared_ptr<Tensor> EXP_AVG;
        std::shared_ptr<Tensor> ACCUM_COUNT;
    } inputs;
        
    struct Outputs {
        std::shared_ptr<Tensor> EQ_SCALE;
        std::shared_ptr<Tensor> EQ_BIAS;
        std::shared_ptr<Tensor> NEXT_RUNNING_MEAN;
        std::shared_ptr<Tensor> NEXT_RUNNING_VAR;
    } outputs;

    BN_finalize(const std::string name) : Operation(name, Tag::BN_finalize) {}

    BN_finalize& set_compute_data_type(DataType_t value) {
        compute_data_type = value;
        return *this;
    }

    void
    make_outputs(std::function<std::shared_ptr<Tensor>(std::string const &)> output_tensor) {
        outputs.EQ_SCALE = output_tensor(name + "_EQ_SCALE_output");
        outputs.EQ_BIAS = output_tensor(name + "_EQ_BIAS_output");
        outputs.NEXT_RUNNING_MEAN = output_tensor(name + "_NEXT_RUNNING_MEAN_output");
        outputs.NEXT_RUNNING_VAR = output_tensor(name + "_NEXT_RUNNING_VAR_output");
    }

    auto fill_from_context(detail::Context const& context) -> BN_finalize& {
        // Fill node's tensors
        inputs.SUM->fill_from_context(context);
        inputs.SQ_SUM->fill_from_context(context);
        inputs.MEAN->fill_from_context(context);
        inputs.INV_VARIANCE->fill_from_context(context);
        inputs.SCALE->fill_from_context(context);
        inputs.BIAS->fill_from_context(context);
        inputs.PREV_RUNNING_MEAN->fill_from_context(context);
        inputs.PREV_RUNNING_VAR->fill_from_context(context);
        inputs.EPSILON->fill_from_context(context);
        inputs.EXP_AVG->fill_from_context(context);
        inputs.ACCUM_COUNT->fill_from_context(context);
        
        outputs.EQ_SCALE->fill_from_context(context);
        outputs.EQ_BIAS->fill_from_context(context);
        outputs.NEXT_RUNNING_MEAN->fill_from_context(context);
        outputs.NEXT_RUNNING_VAR->fill_from_context(context);

        // Fill this node
        if(get_compute_data_type() == DataType_t::NOT_SET) {
            set_compute_data_type(context.get_compute_data_type());
        }
        return *this;
    }
};

class Genstats : public Operation {
public:
    struct Inputs {
        std::shared_ptr<Tensor> X;
    } inputs;

    struct Outputs {
        std::shared_ptr<Tensor> SUM;
        std::shared_ptr<Tensor> SQ_SUM;
    } outputs;

    Genstats(const std::string name) : Operation(name, Tag::Genstats) {}

    Genstats& set_compute_data_type(DataType_t value) {
        compute_data_type = value;
        return *this;
    }

    auto fill_from_context(detail::Context const& context) -> Genstats& {
        // Fill node's tensors
        inputs.X->fill_from_context(context);
        outputs.SUM->fill_from_context(context);
        outputs.SQ_SUM->fill_from_context(context);

        // Fill this node
        if(get_compute_data_type() == DataType_t::NOT_SET) {
            set_compute_data_type(context.get_compute_data_type());
        }
        return *this;
    }
};

class Convolution : public Operation {
public:
    struct Inputs {
        std::shared_ptr<Tensor> X;
        std::shared_ptr<Tensor> W;
    } inputs;

    struct Outputs {
        std::shared_ptr<Tensor> Y;
    } outputs;

private:
    std::vector<int64_t> padding  = {};
    std::vector<int64_t> stride   = {};
    std::vector<int64_t> dilation = {};

public:
    bool is_padding_set = false;
    bool is_stride_set = false;
    bool is_dilation_set = false;
    
    Convolution(const std::string name) : Operation(name, Tag::Conv) {}

    Convolution& set_compute_data_type(DataType_t const value) {
        compute_data_type = value;
        return *this;
    }

    std::vector<int64_t> get_padding() const {
        return padding;
    }

    Convolution& set_padding(std::vector<int64_t> value) {
        padding = value;
        is_padding_set = true;
        return *this;
    }

    std::vector<int64_t> get_stride() const {
        return stride;
    }

    Convolution& set_stride(std::vector<int64_t> value) {
        stride = value;
        is_stride_set = true;
        return *this;
    }

    std::vector<int64_t> get_dilation() const {
        return dilation;
    }

    Convolution& set_dilation(std::vector<int64_t> value) {
        dilation = value;
        is_dilation_set = true;
        return *this;
    }

    auto fill_from_context(detail::Context const& context) -> Convolution& {
        // Fill node's tensors
        inputs.X->fill_from_context(context);
        inputs.W->fill_from_context(context);
        outputs.Y->fill_from_context(context);

        // Fill this node
        if(get_compute_data_type() == DataType_t::NOT_SET) {
            set_compute_data_type(context.get_compute_data_type());
        }
        return *this;
    }
};

class DBN_weight : public Operation {
public:
    struct Inputs {
        std::shared_ptr<Tensor> X;
        std::shared_ptr<Tensor> MEAN;
        std::shared_ptr<Tensor> INV_VARIANCE;
        std::shared_ptr<Tensor> SCALE;
        std::shared_ptr<Tensor> DY;
    } inputs;

    struct Outputs {
        std::shared_ptr<Tensor> DSCALE;
        std::shared_ptr<Tensor> DBIAS;
        std::shared_ptr<Tensor> EQ_SCALE_DY;
        std::shared_ptr<Tensor> EQ_SCALE_X;
        std::shared_ptr<Tensor> EQ_BIAS;
    } outputs;
    
    DBN_weight(const std::string name) : Operation(name, Tag::DBN_weight) {}

    DBN_weight& set_compute_data_type(DataType_t value) {
        compute_data_type = value;
        return *this;
    }

    void
    make_outputs(std::function<std::shared_ptr<Tensor>(std::string const &)> output_tensor) {
        outputs.DSCALE = output_tensor(name + "_dscale_output");
        outputs.DBIAS = output_tensor(name + "_dbias_output");
        outputs.EQ_SCALE_DY = output_tensor(name + "_eq_scale_dy_output");
        outputs.EQ_SCALE_X = output_tensor(name + "_eq_scale_x_output");
        outputs.EQ_BIAS = output_tensor(name + "_eq_bias_output");
    }

    auto fill_from_context(detail::Context const& context) -> DBN_weight& {
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
        if(get_compute_data_type() == DataType_t::NOT_SET) {
            set_compute_data_type(context.get_compute_data_type());
        }
        return *this;
    }
};

class Dgrad : public Operation {
public:
    struct Inputs {
        std::shared_ptr<Tensor> DY;
        std::shared_ptr<Tensor> W;
    } inputs;

    struct Outputs {
        std::shared_ptr<Tensor> DX;
    } outputs;

private:
    std::vector<int64_t> padding;
    std::vector<int64_t> stride;
    std::vector<int64_t> dilation;

public:
    Dgrad(const std::string name) : Operation(name, Tag::Dgrad) {}

    Dgrad& set_compute_data_type(DataType_t value) {
        compute_data_type = value;
        return *this;
    }

    std::vector<int64_t> get_padding() const {
        return padding;
    }

    Dgrad& set_padding(std::vector<int64_t> value) {
        padding = value;
        return *this;
    }

    std::vector<int64_t> get_stride() const {
        return stride;
    }

    Dgrad& set_stride(std::vector<int64_t> value) {
        stride = value;
        return *this;
    }

    std::vector<int64_t> get_dilation() const {
        return dilation;
    }

    Dgrad& set_dilation(std::vector<int64_t> value) {
        dilation = value;
        return *this;
    }

    auto fill_from_context(detail::Context const& context) -> Dgrad& {
        // Fill node's tensors
        inputs.DY->fill_from_context(context);
        inputs.W->fill_from_context(context);
        outputs.DX->fill_from_context(context);

        // Fill this node
        if(get_compute_data_type() == DataType_t::NOT_SET) {
            set_compute_data_type(context.get_compute_data_type());
        }
        return *this;
    }
};

class Matmul : public Operation {
public:
    struct Inputs {
        std::shared_ptr<Tensor> A;
        std::shared_ptr<Tensor> B;
        std::shared_ptr<Tensor> M_override;
        std::shared_ptr<Tensor> N_override;
        std::shared_ptr<Tensor> K_override;
    } inputs;

    struct Outputs {
        std::shared_ptr<Tensor> C;
    } outputs;

    Matmul(const std::string name) : Operation(name, Tag::Matmul) {}

    Matmul& set_compute_data_type(DataType_t value) {
        compute_data_type = value;
        return *this;
    }

    auto fill_from_context(detail::Context const& context) -> Matmul& {
        // Fill node's tensors
        inputs.A->fill_from_context(context);
        inputs.B->fill_from_context(context);
        outputs.C->fill_from_context(context);

        if(inputs.M_override)inputs.M_override->fill_from_context(context);
        if(inputs.N_override)inputs.N_override->fill_from_context(context);
        if(inputs.K_override)inputs.K_override->fill_from_context(context);

        // Fill this node
        if(get_compute_data_type() == DataType_t::NOT_SET) {
            set_compute_data_type(context.get_compute_data_type());
        }
        return *this;
    }
};

class Pointwise : public Operation {
public:
    struct Inputs {
        std::shared_ptr<Tensor> IN_0;
        std::shared_ptr<Tensor> IN_1;
        std::shared_ptr<Tensor> IN_2;
    } inputs;

    struct Outputs {
        std::shared_ptr<Tensor> OUT_0;
    } outputs;

private:
    std::optional<PointwiseMode_t> mode;
    std::optional<int64_t> axis;

public:

    Pointwise(const std::string name) : Operation(name, Tag::Pointwise) {}

    Pointwise& set_compute_data_type(DataType_t const value) {
        compute_data_type = value;
        return *this;
    }

    std::optional<PointwiseMode_t> get_mode() const {
        return mode;
    }

    Pointwise& set_mode(PointwiseMode_t const value) {
        mode = value;
        return *this;
    }
    
    std::optional<int64_t> get_axis() const {
        return axis;
    }

    Pointwise& set_axis(int64_t const axis) {
        this->axis = axis;
        return *this;
    }

    auto fill_from_context(detail::Context const& context) -> Pointwise& {
        // Fill node's tensors
        inputs.IN_0->fill_from_context(context);
        if(inputs.IN_1)inputs.IN_1->fill_from_context(context);
        if(inputs.IN_2)inputs.IN_2->fill_from_context(context);
        outputs.OUT_0->fill_from_context(context);

        // Fill this node
        if(get_compute_data_type() == DataType_t::NOT_SET) {
            set_compute_data_type(context.get_compute_data_type());
        }
        return *this;
    }
};

class Batchnorm : public Operation {
public:
    struct Inputs {
        std::shared_ptr<Tensor> X;
        std::shared_ptr<Tensor> SCALE;
        std::shared_ptr<Tensor> BIAS;
        std::shared_ptr<Tensor> PREV_RUNNING_MEAN;
        std::shared_ptr<Tensor> PREV_RUNNING_VAR;
    } inputs;

    struct Outputs {
        std::shared_ptr<Tensor> Y;
        std::shared_ptr<Tensor> MEAN;
        std::shared_ptr<Tensor> INV_VARIANCE;
        std::shared_ptr<Tensor> NEXT_RUNNING_MEAN;
        std::shared_ptr<Tensor> NEXT_RUNNING_VAR;
    } outputs;
private:
    NormFwdPhase_t forward_phase;
    std::optional<float> epsilon;
    std::optional<float> momentum;

public:

    Batchnorm(const std::string name) : Operation(name, Tag::BN), forward_phase(NormFwdPhase_t::NOT_SET) {}
    
    Batchnorm& set_compute_data_type(DataType_t value) {
        compute_data_type = value;
        return *this;
    }

    NormFwdPhase_t get_forward_phase() {
        return forward_phase;
    }

    Batchnorm& set_forward_phase(NormFwdPhase_t const value) {
        forward_phase = value;
        return *this;
    }

    std::optional<float> get_epsilon() {
        return epsilon;
    }

    Batchnorm& set_epsilon(float const value) {
        epsilon = value;
        return *this;
    }

    std::optional<float> get_momentum() {
        return momentum;
    }

    Batchnorm& set_momentum(float const value) {
        momentum = value;
        return *this;
    }

    void make_outputs(std::function<std::shared_ptr<Tensor>(std::string const &)> output_tensor) {
        outputs.Y = output_tensor(name + "_Y_output");
        outputs.MEAN = output_tensor(name + "_MEAN_output");;
        outputs.INV_VARIANCE = output_tensor(name + "_INV_VARIANCE_output");;
        outputs.NEXT_RUNNING_MEAN = output_tensor(name + "_NEXT_RUNNING_MEAN_output");;
        outputs.NEXT_RUNNING_VAR = output_tensor(name + "_NEXT_RUNNING_VAR_output");;
    }

    auto fill_from_context(detail::Context const& context) -> Batchnorm& {
        // Fill node's tensors
        inputs.X->fill_from_context(context);
        inputs.SCALE->fill_from_context(context);
        inputs.BIAS->fill_from_context(context);
        inputs.PREV_RUNNING_MEAN->fill_from_context(context);
        inputs.PREV_RUNNING_VAR->fill_from_context(context);
        
        outputs.Y->fill_from_context(context);
        outputs.MEAN->fill_from_context(context);
        outputs.INV_VARIANCE->fill_from_context(context);
        outputs.NEXT_RUNNING_MEAN->fill_from_context(context);
        outputs.NEXT_RUNNING_VAR->fill_from_context(context);

        if(get_compute_data_type() == DataType_t::NOT_SET) {
            set_compute_data_type(context.get_compute_data_type());
        }
        return *this;
    }
};

class Reduction : public Operation {
public:
    
    struct Inputs {
        std::shared_ptr<Tensor> X;
    } inputs;

    struct Outputs {
        std::shared_ptr<Tensor> Y;
    } outputs;

private:
    std::optional<ReductionMode_t> mode;

public:
    Reduction(const std::string name) : Operation(name, Tag::Reduction) {}

    std::optional<ReductionMode_t> get_mode() const {
        return mode;
    }

    Reduction& set_mode(ReductionMode_t value) {
        mode = value;
        return *this;
    }
    
    Reduction& set_compute_data_type(DataType_t value) {
        compute_data_type = value;
        return *this;
    }

    auto fill_from_context(detail::Context const& context) -> Reduction& {
        // Fill node's tensors
        inputs.X->fill_from_context(context);
        outputs.Y->fill_from_context(context);

        // Fill this node
        if(get_compute_data_type() == DataType_t::NOT_SET) {
            set_compute_data_type(context.get_compute_data_type());
        }
        return *this;
    }
};

class Scaled_dot_product_attention : public Operation {
public:

    struct Inputs {
        std::shared_ptr<Tensor> Q;
        std::shared_ptr<Tensor> K;
        std::shared_ptr<Tensor> Bias;  // Optional bias after bmm1
        std::shared_ptr<Tensor> V;
        std::shared_ptr<Tensor> SEQ_LEN_Q;
        std::shared_ptr<Tensor> SEQ_LEN_K;
        std::shared_ptr<Tensor> Mask;
        std::shared_ptr<Tensor> Dropout_mask;
    } inputs;

    struct Outputs {
        std::shared_ptr<Tensor> S; // softmax output dumped when is_inference false. Users first need to check whether its nullptr.
        std::shared_ptr<Tensor> O;
    } outputs;

private:
    bool is_inference;
    bool use_causal_masking;
    double dropout_probability, dropout_scale;
    float scale_k;
    
public:
    Scaled_dot_product_attention(const std::string name) : Operation(name, Tag::Scaled_dot_product_attention), is_inference(false), scale_k(1.f) {}

    bool get_is_inference() const {
        return is_inference;
    }

    Scaled_dot_product_attention& set_is_inference(bool const value){
        is_inference = value;
        return *this;
    }

    float get_scale_k() const {
        return scale_k;
    }

    Scaled_dot_product_attention& set_scale_k(float const value){
        scale_k = value;
        return *this;
    }

    Scaled_dot_product_attention& set_masking(bool const use_causal) {
        use_causal_masking = use_causal;
        return *this;
    }

    Scaled_dot_product_attention& set_inference(bool const use_causal) {
        use_causal_masking = use_causal;
        return *this;
    }

    Scaled_dot_product_attention& set_dropout(double const probability, double const scale) {
        dropout_probability = probability;
        dropout_scale = scale;
        return *this;
    }

    Scaled_dot_product_attention& set_compute_data_type(DataType_t const value) {
        compute_data_type = value;
        return *this;
    }

    Scaled_dot_product_attention& fill_from_context(detail::Context const& context) {
        // Fill node's tensors
        inputs.Q->fill_from_context(context);
        inputs.K->fill_from_context(context);
        inputs.V->fill_from_context(context);
        inputs.SEQ_LEN_Q->fill_from_context(context);
        inputs.SEQ_LEN_K->fill_from_context(context);
        outputs.O->fill_from_context(context);

        // Fill this node
        if(get_compute_data_type() == DataType_t::NOT_SET) {
            set_compute_data_type(context.get_compute_data_type());
        }
        return *this;
    }
};

class Softmax : public Operation {
public:

    struct Inputs {
        std::shared_ptr<Tensor> P;
    } inputs;

    struct Outputs {
        std::shared_ptr<Tensor> S; // softmax output dumped when is_inference false. Users first need to check whether its nullptr.
    } outputs;

private:
    bool is_inference;

public:
    Softmax(const std::string name) : Operation(name, Tag::Softmax) {}

    bool get_is_inference() const {
        return is_inference;
    }

    Softmax& set_is_inference(bool const value){
        is_inference = value;
        return *this;
    }

    Softmax& set_compute_data_type(DataType_t const value) {
        compute_data_type = value;
        return *this;
    }

    Softmax& fill_from_context(detail::Context const& context) {
        // Fill node's tensors
        inputs.P->fill_from_context(context);
        outputs.S->fill_from_context(context);

        // Fill this node
        if(get_compute_data_type() == DataType_t::NOT_SET) {
            set_compute_data_type(context.get_compute_data_type());
        }
        return *this;
    }
};

class Wgrad : public Operation {
public:
    
    struct Inputs {
        std::shared_ptr<Tensor> DY;
        std::shared_ptr<Tensor> X;
    } inputs;

    struct Outputs {
        std::shared_ptr<Tensor> DW;
    } outputs;

private:
    std::vector<int64_t> padding;
    std::vector<int64_t> stride;
    std::vector<int64_t> dilation;

public:

    Wgrad(const std::string name) : Operation(name, Tag::Wgrad) {}

    Wgrad& set_compute_data_type(DataType_t value) {
        compute_data_type = value;
        return *this;
    }

    std::vector<int64_t> get_padding() const {
        return padding;
    }

    Wgrad& set_padding(std::vector<int64_t> value) {
        padding = value;
        return *this;
    }

    std::vector<int64_t> get_stride() const {
        return stride;
    }

    Wgrad& set_stride(std::vector<int64_t> value) {
        stride = value;
        return *this;
    }

    std::vector<int64_t> get_dilation() const {
        return dilation;
    }

    Wgrad& set_dilation(std::vector<int64_t> value) {
        dilation = value;
        return *this;
    }

    auto fill_from_context(detail::Context const& context) -> Wgrad& {
        // Fill node's tensors
        inputs.DY->fill_from_context(context);
        inputs.X->fill_from_context(context);
        outputs.DW->fill_from_context(context);

        // Fill this node
        if(get_compute_data_type() == DataType_t::NOT_SET) {
            set_compute_data_type(context.get_compute_data_type());
        }
        return *this;
    }
};

} // namespace graph

}