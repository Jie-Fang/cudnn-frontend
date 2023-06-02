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
    int64_t uid;

public:
    bool is_dim_set = false;
    bool is_stride_set = false;
    bool is_virtual_set = false;
    bool is_pass_by_value_set = false;
    bool is_uid_set = false;

    // TODO: Currently this structure takes in unrolled list of properties to set.
    // But later, it will take in the context and derive properties to set from it.
    int set_properties_from_context(cudnnTensorFormat_t const filter_format, int64_t const uid) {
        if(!is_stride_set) {
            generateStrides(filter_format);
        }
        if(!is_uid_set) {
            set_uid(uid);
        }

        return 0;
    }

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

    int64_t get_uid() const {
        return uid;
    }

    auto set_uid(int64_t value) -> Tensor& {
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
        Batchnorm,
        Batchnorm_backward_weight,
        Batchnorm_finalize,
        Convolution,
        Dgrad,
        Genstats,
        Matmul,
        Pointwise,
        Reduction,
        Scaled_dot_product_attention,
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

class Batchnorm_backward_weight : public Operation {
public:
    enum PORTS {
        X,
        MEAN,
        INV_VARIANCE,
        SCALE,
        DY,
        
        DSCALE,
        DBIAS,
        EQUIVALENT_SCALE_DY,
        EQUIVALENT_SCALE_X,
        EQUIVALENT_BIAS,
        
        COUNT
    };

    std::unordered_map<PORTS, std::string> port_to_name;
    int64_t uids[PORTS::COUNT];
    Batchnorm_backward_weight(const std::string name) : Operation(name, Tag::Batchnorm_backward_weight) {
        port_to_name[PORTS::X] = name + "::X";
        port_to_name[PORTS::MEAN] = name + "::MEAN";
        port_to_name[PORTS::INV_VARIANCE] = name + "::INV_VARIANCE";
        port_to_name[PORTS::SCALE] = name + "::SCALE";
        port_to_name[PORTS::DY] = name + "::DY";
        
        port_to_name[PORTS::DSCALE] = name + "::DSCALE";
        port_to_name[PORTS::DBIAS] = name + "::DBIAS";
        port_to_name[PORTS::EQUIVALENT_SCALE_DY] = name + "::EQUIVALENT_SCALE_DY";
        port_to_name[PORTS::EQUIVALENT_SCALE_X] = name + "::EQUIVALENT_SCALE_X";
        port_to_name[PORTS::EQUIVALENT_BIAS] = name + "::EQUIVALENT_BIAS";
    }

    Batchnorm_backward_weight& map_port_to_tensor(std::vector<std::pair<PORTS, std::string>> names) {
        for(auto const& p: names) {
            port_to_name[p.first] = p.second;
        }
        return *this;
    }

    std::string get_tensor_at_port(PORTS port) const {
        return port_to_name.at(port);
    }

    int update_uids(int64_t offset) {
        for(size_t i = 0; i < PORTS::COUNT; ++i) {
            uids[i] = i + offset;
        }
        return 0;
    }

    Batchnorm_backward_weight& set_compute_data_type(DataType_t value) {
        compute_data_type = value;
        return *this;
    }

    auto fill_from_context(detail::Context const& context) -> Batchnorm_backward_weight& {
        if(get_compute_data_type() == DataType_t::NOT_SET) {
            set_compute_data_type(context.get_compute_data_type());
        }
        return *this;
    }

    friend std::ostream& operator<<(std::ostream& os, const Batchnorm_backward_weight& props);
};

inline std::ostream& operator<<(std::ostream& os, const Batchnorm_backward_weight& props) {
    os << "{" 
    << " name: '" << props.get_name() << "',"
    << " ports: [";
    for(size_t i = 0; i < Batchnorm_backward_weight::PORTS::COUNT; ++i) {
        os << props.get_tensor_at_port(static_cast<Batchnorm_backward_weight::PORTS>(i)) << ",";
    }
    os << "],";
    return os;
}

class Batchnorm_finalize : public Operation {
public:
    enum PORTS {
        SUM,
        SQUARE_SUM,
        MEAN,
        INV_VARIANCE,
        SCALE,
        BIAS,
        Previous_running_mean,
        Previous_running_var,
        EPSILON,
        EXP_AVG,
        ACCUMULATION_COUNT,
        
        EQUIVALENT_SCALE,
        EQUIVALENT_BIAS,
        Next_running_mean,
        Next_running_var,
        
        COUNT
    };

    std::unordered_map<PORTS, std::string> port_to_name;
    int64_t uids[PORTS::COUNT];
    Batchnorm_finalize(const std::string name) : Operation(name, Tag::Batchnorm_finalize) {
        port_to_name[PORTS::SUM] = name + "::SUM";
        port_to_name[PORTS::SQUARE_SUM] = name + "::SQUARE_SUM";
        port_to_name[PORTS::MEAN] = name + "::MEAN";
        port_to_name[PORTS::INV_VARIANCE] = name + "::INV_VARIANCE";
        port_to_name[PORTS::SCALE] = name + "::SCALE";
        port_to_name[PORTS::BIAS] = name + "::BIAS";
        port_to_name[PORTS::Previous_running_mean] = name + "::Previous_running_mean";
        port_to_name[PORTS::Previous_running_var] = name + "::Previous_running_var";
        port_to_name[PORTS::EPSILON] = name + "::EPSILON";
        port_to_name[PORTS::EXP_AVG] = name + "::EXP_AVG";
        port_to_name[PORTS::ACCUMULATION_COUNT] = name + "::ACCUMULATION_COUNT";
        
        port_to_name[PORTS::EQUIVALENT_BIAS] = name + "::EQUIVALENT_BIAS";
        port_to_name[PORTS::EQUIVALENT_SCALE] = name + "::EQUIVALENT_SCALE";
        port_to_name[PORTS::Next_running_mean] = name + "::Next_running_mean";
        port_to_name[PORTS::Next_running_var] = name + "::Next_running_var";
    }

    Batchnorm_finalize& map_port_to_tensor(std::vector<std::pair<PORTS, std::string>> names) {
        for(auto const& p: names) {
            port_to_name[p.first] = p.second;
        }
        return *this;
    }

    std::string get_tensor_at_port(PORTS port) const {
        return port_to_name.at(port);
    }

    int update_uids(int64_t offset) {
        for(size_t i = 0; i < PORTS::COUNT; ++i) {
            uids[i] = i + offset;
        }
        return 0;
    }

    Batchnorm_finalize& set_compute_data_type(DataType_t value) {
        compute_data_type = value;
        return *this;
    }

    auto fill_from_context(detail::Context const& context) -> Batchnorm_finalize& {
        if(get_compute_data_type() == DataType_t::NOT_SET) {
            set_compute_data_type(context.get_compute_data_type());
        }
        return *this;
    }

    friend std::ostream& operator<<(std::ostream& os, const Batchnorm_finalize& props);
};

inline std::ostream& operator<<(std::ostream& os, const Batchnorm_finalize& props) {
    os << "{" 
    << " name: '" << props.get_name() << "',"
    << " ports: [";
    for(size_t i = 0; i < Batchnorm_finalize::PORTS::COUNT; ++i) {
        os << props.get_tensor_at_port(static_cast<Batchnorm_finalize::PORTS>(i)) << ",";
    }
    os << "],";
    return os;
}

class Genstats : public Operation {
public:
    enum PORTS {
        X,
        SUM,
        SQ_SUM,

        COUNT
    };

    std::unordered_map<PORTS, std::string> port_to_name;
    int64_t uids[PORTS::COUNT];
    Genstats(const std::string name) : Operation(name, Tag::Genstats) {
        port_to_name[PORTS::X] = name + "::X";
        port_to_name[PORTS::SUM] = name + "::SUM";
        port_to_name[PORTS::SQ_SUM] = name + "::SQ_SUM";
    }

    Genstats& map_port_to_tensor(std::vector<std::pair<PORTS, std::string>> names) {
        for(auto const& p: names) {
            port_to_name[p.first] = p.second;
        }
        return *this;
    }

    std::string get_tensor_at_port(PORTS port) const {
        return port_to_name.at(port);
    }

    int update_uids(int64_t offset) {
        for(size_t i = 0; i < PORTS::COUNT; ++i) {
            uids[i] = i + offset;
        }
        return 0;
    }

    Genstats& set_compute_data_type(DataType_t value) {
        compute_data_type = value;
        return *this;
    }

    auto fill_from_context(detail::Context const& context) -> Genstats& {
        if(get_compute_data_type() == DataType_t::NOT_SET) {
            set_compute_data_type(context.get_compute_data_type());
        }
        return *this;
    }

    friend std::ostream& operator<<(std::ostream& os, const Genstats& props);
};

inline std::ostream& operator<<(std::ostream& os, const Genstats& props) {
    os << "{" 
    << " name: '" << props.get_name() << "',"
    << " ports: [";
    for(size_t i = 0; i < Genstats::PORTS::COUNT; ++i) {
        os << props.get_tensor_at_port(static_cast<Genstats::PORTS>(i)) << ",";
    }
    os << "],";
    return os;
}

class Convolution : public Operation {
public:
    enum PORTS {
        X,
        W,
        Y,

        COUNT
    };
private:
    std::vector<int64_t> padding  = {};
    std::vector<int64_t> stride   = {};
    std::vector<int64_t> dilation = {};

public:
    bool is_padding_set = false;
    bool is_stride_set = false;
    bool is_dilation_set = false;

    std::unordered_map<PORTS, std::string> port_to_name;
    int64_t uids[PORTS::COUNT];
    Convolution(const std::string name) : Operation(name, Tag::Convolution) {
        port_to_name[PORTS::X] = name + "::X";
        port_to_name[PORTS::W] = name + "::W";
        port_to_name[PORTS::Y] = name + "::Y";
    }

    Convolution& map_port_to_tensor(std::vector<std::pair<PORTS, std::string>> names) {
        for(auto const& p: names) {
            port_to_name[p.first] = p.second;
        }
        return *this;
    }

    std::string get_tensor_at_port(PORTS port) const {
        return port_to_name.at(port);
    }

    int update_uids(int64_t offset) {
        for(size_t i = 0; i < PORTS::COUNT; ++i) {
            uids[i] = i + offset;
        }
        return 0;
    }

    Convolution& set_compute_data_type(DataType_t value) {
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
        if(get_compute_data_type() == DataType_t::NOT_SET) {
            set_compute_data_type(context.get_compute_data_type());
        }
        return *this;
    }

    Convolution& set_input(std::shared_ptr<Tensor> tensor) {
        port_to_name[PORTS::X] = tensor->get_name();
        return *this;
    }

    Convolution& set_weight(std::shared_ptr<Tensor> tensor) {
        port_to_name[PORTS::W] = tensor->get_name();
        return *this;
    }

    Convolution& set_output(std::shared_ptr<Tensor> tensor) {
        port_to_name[PORTS::Y] = tensor->get_name();
        return *this;
    }

    friend std::ostream& operator<<(std::ostream& os, const Convolution& props);
};

inline std::ostream& operator<<(std::ostream& os, const Convolution& props) {
    os << "{" 
    << " name: '" << props.get_name() << "',"
    << " dilation: [";
    for(size_t i = 0; i < props.get_dilation().size(); ++i) {
        os << props.get_dilation()[i] << ",";
    }
    os << "],"
    << " stride: [";
    for(size_t i = 0; i < props.get_stride().size(); ++i) {
        os << props.get_stride()[i] << ",";
    }
    os << "],"
    << " padding: [";
    for(size_t i = 0; i < props.get_padding().size(); ++i) {
        os << props.get_padding()[i] << ",";
    }
    os << "],"
    << " ports: [";
    for(size_t i = 0; i < Convolution::PORTS::COUNT; ++i) {
        os << props.get_tensor_at_port(static_cast<Convolution::PORTS>(i)) << ",";
    }
    os << "],";
    return os;
}

class Dgrad : public Operation {
public:
    enum PORTS {
        DY,
        W,
        DX,

        COUNT
    };
private:
    std::vector<int64_t> padding;
    std::vector<int64_t> stride;
    std::vector<int64_t> dilation;

public:

    std::unordered_map<PORTS, std::string> port_to_name;
    int64_t uids[PORTS::COUNT];
    Dgrad(const std::string name) : Operation(name, Tag::Dgrad) {
        port_to_name[PORTS::W] = name + "::W";
        port_to_name[PORTS::DY] = name + "::DY";
        port_to_name[PORTS::DX] = name + "::DX";
    }

    Dgrad& map_port_to_tensor(std::vector<std::pair<PORTS, std::string>> names) {
        for(auto const& p: names) {
            port_to_name[p.first] = p.second;
        }
        return *this;
    }

    std::string get_tensor_at_port(PORTS port) const {
        return port_to_name.at(port);
    }

    int update_uids(int64_t offset) {
        for(size_t i = 0; i < PORTS::COUNT; ++i) {
            uids[i] = i + offset;
        }
        return 0;
    }

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
        if(get_compute_data_type() == DataType_t::NOT_SET) {
            set_compute_data_type(context.get_compute_data_type());
        }
        return *this;
    }

    friend std::ostream& operator<<(std::ostream& os, const Dgrad& props);
};

inline std::ostream& operator<<(std::ostream& os, const Dgrad& props) {
    os << "{" 
    << " name: '" << props.get_name() << "',"
    << " dilation: [";
    for(size_t i = 0; i < props.get_dilation().size(); ++i) {
        os << props.get_dilation()[i] << ",";
    }
    os << "],"
    << " stride: [";
    for(size_t i = 0; i < props.get_stride().size(); ++i) {
        os << props.get_stride()[i] << ",";
    }
    os << "],"
    << " padding: [";
    for(size_t i = 0; i < props.get_padding().size(); ++i) {
        os << props.get_padding()[i] << ",";
    }
    os << "],"
    << " ports: [";
    for(size_t i = 0; i < Dgrad::PORTS::COUNT; ++i) {
        os << props.get_tensor_at_port(static_cast<Dgrad::PORTS>(i)) << ",";
    }
    os << "],";
    return os;
}

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
    enum PORTS {
        IN_0,
        IN_1,
        IN_2,
        OUT_0,

        COUNT
    };

private:
    PointwiseMode_t mode;
    std::optional<int64_t> axis;
public:
    bool is_mode_set;

    std::unordered_map<PORTS, std::string> port_to_name;
    int64_t uids[PORTS::COUNT];

    Pointwise(const std::string name) : Operation(name, Tag::Pointwise) {
        port_to_name[PORTS::IN_0] = name + "::IN_0";
        port_to_name[PORTS::IN_1] = name + "::IN_1";
        port_to_name[PORTS::IN_2] = name + "::IN_2";
        port_to_name[PORTS::OUT_0] = name + "::OUT_0";
    }

    int 
    update_uids(int64_t offset) {
        for(size_t i = 0; i < PORTS::COUNT; ++i) {
            uids[i] = i + offset;
        }
        return 0;
    }

    Pointwise& set_compute_data_type(DataType_t value) {
        compute_data_type = value;
        return *this;
    }

    PointwiseMode_t get_mode() const {
        return mode;
    }

    Pointwise& set_mode(PointwiseMode_t value) {
        mode = value;
        is_mode_set = true;
        return *this;
    }
    
    std::optional<int64_t> get_axis() const {
        return axis;
    }

    Pointwise& set_axis(int64_t const axis) {
        this->axis = axis;
        return *this;
    }

    Pointwise& map_port_to_tensor(std::vector<std::pair<PORTS, std::string>> names) {
        for(auto const& p: names) {
            port_to_name[p.first] = p.second;
        }
        return *this;
    }

    std::string get_tensor_at_port(PORTS port) const {
        return port_to_name.at(port);
    }

    auto fill_from_context(detail::Context const& context) -> Pointwise& {
        if(get_compute_data_type() == DataType_t::NOT_SET) {
            set_compute_data_type(context.get_compute_data_type());
        }
        return *this;
    }

    Pointwise& set_inputs(std::vector<std::shared_ptr<Tensor>> tensors) {
        size_t count = 0;
        for(auto const port: {PORTS::IN_0, PORTS::IN_1, PORTS::IN_2}) {
            port_to_name[port] = tensors[count]->get_name();
            count++;
            if(count >= tensors.size()) {
                break;
            }
        }

        return *this;
    }

    Pointwise& set_output(std::shared_ptr<Tensor> tensor) {
        port_to_name[PORTS::OUT_0] = tensor->get_name();
        return *this;
    }

    friend std::ostream& operator<<(std::ostream& os, const Pointwise& props);
};

inline std::ostream& operator<<(std::ostream& os, const Pointwise& props) {
    os << "{" 
    << " name: '" << props.get_name() << "',"
    << " ports: [";
    for(size_t i = 0; i < Pointwise::PORTS::COUNT; ++i) {
        os << props.get_tensor_at_port(static_cast<Pointwise::PORTS>(i)) << ",";
    }
    os << "],";
    return os;
}


class Batchnorm : public Operation {
public:
    enum PORTS {
        X = 0,
        Mean,
        Var,
        EPS,
        Scale,
        Bias,
        Previous_running_mean,
        Previous_running_var,
        Next_running_mean,
        Next_running_var,
        Y,
        EXP_AVG,

        COUNT
    };
private:
    
public:
    
    std::unordered_map<PORTS, std::string> port_to_name;
    int64_t uids[PORTS::COUNT];
    Batchnorm(const std::string name) : Operation(name, Tag::Batchnorm) {
        port_to_name[PORTS::X] = name + "::X";
        port_to_name[PORTS::Mean] = name + "::Mean";
        port_to_name[PORTS::Var] = name + "::Var";
        port_to_name[PORTS::EPS] = name + "::EPS";
        port_to_name[PORTS::EXP_AVG] = name + "::EXP_AVG";
        port_to_name[PORTS::Scale] = name + "::Scale";
        port_to_name[PORTS::Bias] = name + "::Bias";
        port_to_name[PORTS::Previous_running_mean] = name + "::Previous_running_mean";
        port_to_name[PORTS::Previous_running_var] = name + "::Previous_running_var";
        port_to_name[PORTS::Next_running_mean] = name + "::Next_running_mean";
        port_to_name[PORTS::Next_running_var] = name + "::Next_running_var";
        port_to_name[PORTS::Y] = name + "::Y";
    }

    Batchnorm& map_port_to_tensor(std::vector<std::pair<PORTS, std::string>> names) {
        for(auto const& p: names) {
            port_to_name[p.first] = p.second;
        }
        return *this;
    }

    std::string get_tensor_at_port(PORTS port) const {
        return port_to_name.at(port);
    }

    int update_uids(int64_t offset) {
        for(size_t i = 0; i < PORTS::COUNT; ++i) {
            uids[i] = i + offset;
        }
        return 0;
    }
    
    Batchnorm& set_compute_data_type(DataType_t value) {
        compute_data_type = value;
        return *this;
    }

    auto fill_from_context(detail::Context const& context) -> Batchnorm& {
        if(get_compute_data_type() == DataType_t::NOT_SET) {
            set_compute_data_type(context.get_compute_data_type());
        }
        return *this;
    }
};

class Reduction : public Operation {
public:
    enum PORTS {
        X = 0,
        Y,

        COUNT
    };

private:
    ReductionMode_t mode;

public:
    bool is_mode_set;

    std::unordered_map<PORTS, std::string> port_to_name;
    int64_t uids[PORTS::COUNT];

    Reduction(const std::string name) : Operation(name, Tag::Reduction) {
        port_to_name[PORTS::X] = name + "::X";
        port_to_name[PORTS::Y] = name + "::Y";
    }

    int update_uids(int64_t offset) {
        for(size_t i = 0; i < PORTS::COUNT; ++i) {
            uids[i] = i + offset;
        }
        return 0;
    }

    ReductionMode_t get_mode() const {
        return mode;
    }

    Reduction& set_mode(ReductionMode_t value) {
        mode = value;
        is_mode_set = true;
        return *this;
    }

    Reduction& map_port_to_tensor(std::vector<std::pair<PORTS, std::string>> const& names) {
        for(auto const& p: names) {
            port_to_name[p.first] = p.second;
        }
        return *this;
    }

    std::string get_tensor_at_port(PORTS port) const {
        return port_to_name.at(port);
    }
    
    Reduction& set_compute_data_type(DataType_t value) {
        compute_data_type = value;
        return *this;
    }

    auto fill_from_context(detail::Context const& context) -> Reduction& {
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
    Scaled_dot_product_attention(const std::string name) : Operation(name, Tag::Scaled_dot_product_attention) {}

    Scaled_dot_product_attention& set_is_inference(bool const value){
        is_inference = value;
        return *this;
    }

    Scaled_dot_product_attention& set_scale_for_K(float const value){
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

class Wgrad : public Operation {
public:
    enum PORTS {
        DY,
        X,
        DW,

        COUNT
    };
private:
    std::vector<int64_t> padding;
    std::vector<int64_t> stride;
    std::vector<int64_t> dilation;

public:

    std::unordered_map<PORTS, std::string> port_to_name;
    int64_t uids[PORTS::COUNT];
    Wgrad(const std::string name) : Operation(name, Tag::Wgrad) {
        port_to_name[PORTS::X] = name + "::X";
        port_to_name[PORTS::DW] = name + "::W";
        port_to_name[PORTS::DY] = name + "::DY";
    }

    Wgrad& map_port_to_tensor(std::vector<std::pair<PORTS, std::string>> names) {
        for(auto const& p: names) {
            port_to_name[p.first] = p.second;
        }
        return *this;
    }

    std::string get_tensor_at_port(PORTS port) const {
        return port_to_name.at(port);
    }

    int update_uids(int64_t offset) {
        for(size_t i = 0; i < PORTS::COUNT; ++i) {
            uids[i] = i + offset;
        }
        return 0;
    }

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
        if(get_compute_data_type() == DataType_t::NOT_SET) {
            set_compute_data_type(context.get_compute_data_type());
        }
        return *this;
    }

    friend std::ostream& operator<<(std::ostream& os, const Wgrad& props);
};

inline std::ostream& operator<<(std::ostream& os, const Wgrad& props) {
    os << "{" 
    << " name: '" << props.get_name() << "',"
    << " dilation: [";
    for(size_t i = 0; i < props.get_dilation().size(); ++i) {
        os << props.get_dilation()[i] << ",";
    }
    os << "],"
    << " stride: [";
    for(size_t i = 0; i < props.get_stride().size(); ++i) {
        os << props.get_stride()[i] << ",";
    }
    os << "],"
    << " padding: [";
    for(size_t i = 0; i < props.get_padding().size(); ++i) {
        os << props.get_padding()[i] << ",";
    }
    os << "],"
    << " ports: [";
    for(size_t i = 0; i < Wgrad::PORTS::COUNT; ++i) {
        os << props.get_tensor_at_port(static_cast<Wgrad::PORTS>(i)) << ",";
    }
    os << "],";
    return os;
}

} // namespace graph

}