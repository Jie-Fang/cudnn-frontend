#pragma once

#include <iostream>
#include <numeric>
#include <unordered_map>
#include <vector>

#include "graphs/cudnn_frontend_graph_helpers.h"

namespace cudnn_frontend {

// simple structure to hold all properties of a tensor.
// Each property has a getter setter.
class tensor_properties {
protected:

    std::string name;
    // TODO: use custom FE data type string/enum.
    DataType_t data_type = DataType_t::NOT_SET;
    std::vector<int64_t> dim = {};
    std::vector<int64_t> stride = {};
    bool is_virtual = false;
    bool is_pass_by_value = false;
    int64_t uid;

public:
    bool is_data_type_set = false;
    bool is_dim_set = false;
    bool is_stride_set = false;
    bool is_virtual_set = false;
    bool is_pass_by_value_set = false;
    bool is_uid_set = false;

    std::string const
    get_name() const {
        return name;
    }

    // TODO: Currently this structure takes in unrolled list of properties to set.
    // But later, it will take in the context and derive properties to set from it.
    int set_properties_from_context(cudnnTensorFormat_t const filter_format, DataType_t const data_type, int64_t const uid) {
        if(!is_stride_set) {
            generateStrides(filter_format);
        }
        if(!is_data_type_set) {
            set_data_type(data_type);
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

    tensor_properties(const std::string &name) : name(name) {}

    DataType_t const &
    get_data_type() const {
        return data_type;
    }

    error_t
    set_data_type(DataType_t value) {
        data_type = value;
        is_data_type_set = true;
        return error_t::OK;
    }

    std::vector<int64_t> const &
    get_dim() const {
        return dim;
    }
    
    std::vector<int64_t>&
    get_dim() {
        return dim;
    }

    int
    set_dim(std::vector<int64_t> const& value) {
        dim = value;
        is_dim_set = true;
        return 0;
    }

    std::vector<int64_t> const &
    get_stride() const {
        return stride;
    }

    int
    set_stride(std::vector<int64_t> const& value) {
        // empty object implies caller wants cudnn to infer strides.
        if(value.empty()) {
            return 0;
        }
        stride = value;
        is_stride_set = true;
        return 0;
    }

    bool const &
    get_is_virtual() const {
        return is_virtual;
    }

    int
    set_is_virtual(bool value) {
        is_virtual = value;
        is_virtual_set = true;
        return 0;
    }

    bool const &
    get_is_pass_by_value() const {
        return is_pass_by_value;
    }

    int
    set_is_pass_by_value(bool value) {
        is_pass_by_value = value;
        is_pass_by_value_set = true;
        return 0;
    }

    int64_t
    get_uid() const {
        return uid;
    }

    int
    set_uid(int64_t value) {
        uid = value;
        is_uid_set = true;
        return 0;
    }

    // TODO: put coorect size
    int64_t
    get_size() const {
        auto size = std::accumulate(begin(dim), end(dim), 2 /*sizeof(half)*/, std::multiplies<double>());
        return size;
    }

    friend std::ostream& operator<<(std::ostream& os, const tensor_properties& props);
};

inline std::ostream& operator<<(std::ostream& os, const tensor_properties& props) {
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

class operation_properties {
public:
    enum class Tag {
        BatchNorm,
        Convolution,
        MatMul,
        Pointwise,
        Reduction
    };

protected:

    std::string name;
    // TODO: remove setting tensor data type in operation properties.
    // The operation only has to know of the compute type. cudnn operation
    // will convert any tensor type to compute type internally.
    DataType_t tensor_data_type;
    DataType_t compute_type;

    Tag tag;
public:

    bool is_tensor_data_type_set = false;
    bool is_compute_type_set = false;

    operation_properties(const std::string name, Tag t) : name(name), tag(t) {}

    std::string const
    get_name() const {
        return name;
    }

    Tag
    get_tag() const {
        return tag;
    }

    DataType_t
    get_tensor_data_type() const {
        return tensor_data_type;
    }

    error_t
    set_tensor_data_type(DataType_t value) {
        tensor_data_type = value;
        is_tensor_data_type_set = true;
        return error_t::OK;
    }

    DataType_t
    get_compute_type() const {
        return compute_type;
    }

    error_t set_compute_type(DataType_t value) {
        compute_type = value;
        is_compute_type_set = true;
        return error_t::OK;
    }

    virtual ~operation_properties() = default;
};

class convolution_properties : public operation_properties {
public:
    enum PORTS {
        X = 0,
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
    convolution_properties(const std::string name) : operation_properties(name, Tag::Convolution) {
        port_to_name[PORTS::X] = name + "::X";
        port_to_name[PORTS::W] = name + "::W";
        port_to_name[PORTS::Y] = name + "::Y";
    }

    error_t map_port_to_tensor(std::vector<std::pair<PORTS, std::string>> const& names) {
        for(auto const& p: names) {
            port_to_name[p.first] = p.second;
        }
        return error_t::OK;
    }

    std::string
    get_port_name(PORTS port) const {
        return port_to_name.at(port);
    }

    int update_uids(int64_t offset) {
        for(size_t i = 0; i < PORTS::COUNT; ++i) {
            uids[i] = i + offset;
        }
        return 0;
    }

    std::vector<int64_t> const &
    get_padding() const {
        return padding;
    }

    int
    set_padding(std::vector<int64_t> value) {
        padding = value;
        is_padding_set = true;
        return 0;
    }

    std::vector<int64_t> const &
    get_stride() const {
        return stride;
    }

    int
    set_stride(std::vector<int64_t> value) {
        stride = value;
        is_stride_set = true;
        return 0;
    }

    std::vector<int64_t> const &
    get_dilation() const {
        return dilation;
    }

    int
    set_dilation(std::vector<int64_t> value) {
        dilation = value;
        is_dilation_set = true;
        return 0;
    }

    friend std::ostream& operator<<(std::ostream& os, const convolution_properties& props);
};

inline std::ostream& operator<<(std::ostream& os, const convolution_properties& props) {
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
    for(size_t i = 0; i < convolution_properties::PORTS::COUNT; ++i) {
        os << props.get_port_name(static_cast<convolution_properties::PORTS>(i)) << ",";
    }
    os << "],";
    return os;
}

class matmul_properties : public operation_properties {
public:
    enum PORTS {
        X = 0,
        W,
        Y,

        COUNT
    };
private:
    
public:
    
    std::unordered_map<PORTS, std::string> port_to_name;
    int64_t uids[PORTS::COUNT];
    matmul_properties(const std::string name) : operation_properties(name, Tag::MatMul) {
        port_to_name[PORTS::X] = name + "::X";
        port_to_name[PORTS::W] = name + "::W";
        port_to_name[PORTS::Y] = name + "::Y";
    }

    error_t map_port_to_tensor(std::vector<std::pair<PORTS, std::string>> const& names) {
        for(auto const& p: names) {
            port_to_name[p.first] = p.second;
        }
        return error_t::OK;
    }

    std::string
    get_port_name(PORTS port) const {
        return port_to_name.at(port);
    }

    int update_uids(int64_t offset) {
        for(size_t i = 0; i < PORTS::COUNT; ++i) {
            uids[i] = i + offset;
        }
        return 0;
    }
    friend std::ostream& operator<<(std::ostream& os, const matmul_properties& props);
};

inline std::ostream& operator<<(std::ostream& os, const matmul_properties& props) {
    os << "{" 
    << " name: '" << props.get_name() << "',"
    << " ports: [";
    for(size_t i = 0; i < matmul_properties::PORTS::COUNT; ++i) {
        os << props.get_port_name(static_cast<matmul_properties::PORTS>(i)) << ",";
    }
    os << "],";
    return os;
}

class pointwise_properties : public operation_properties {
public:
    enum PORTS {
        X = 0,
        B,
        Y,

        COUNT
    };

private:
    PointwiseMode_t mode;
public:
    bool is_mode_set;

    std::unordered_map<PORTS, std::string> port_to_name;
    int64_t uids[PORTS::COUNT];

    pointwise_properties(const std::string name) : operation_properties(name, Tag::Pointwise) {
        port_to_name[PORTS::X] = name + "::X";
        port_to_name[PORTS::B] = name + "::B";
        port_to_name[PORTS::Y] = name + "::Y";
    }

    int 
    update_uids(int64_t offset) {
        for(size_t i = 0; i < PORTS::COUNT; ++i) {
            uids[i] = i + offset;
        }
        return 0;
    }

    PointwiseMode_t const &
    get_mode() const {
        return mode;
    }

    error_t set_mode(PointwiseMode_t value) {
        mode = value;
        is_mode_set = true;
        return error_t::OK;
    }

    error_t map_port_to_tensor(std::vector<std::pair<PORTS, std::string>> const& names) {
        for(auto const& p: names) {
            port_to_name[p.first] = p.second;
        }
        return error_t::OK;
    }

    std::string
    get_port_name(PORTS port) const {
        return port_to_name.at(port);
    }

    friend std::ostream& operator<<(std::ostream& os, const pointwise_properties& props);
};

inline std::ostream& operator<<(std::ostream& os, const pointwise_properties& props) {
    os << "{" 
    << " name: '" << props.get_name() << "',"
    << " ports: [";
    for(size_t i = 0; i < pointwise_properties::PORTS::COUNT; ++i) {
        os << props.get_port_name(static_cast<pointwise_properties::PORTS>(i)) << ",";
    }
    os << "],";
    return os;
}


class batchnorm_properties : public operation_properties {
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
    batchnorm_properties(const std::string name) : operation_properties(name, Tag::BatchNorm) {
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

    error_t map_port_to_tensor(std::vector<std::pair<PORTS, std::string>> const& names) {
        for(auto const& p: names) {
            port_to_name[p.first] = p.second;
        }
        return error_t::OK;
    }

    std::string
    get_port_name(PORTS port) const {
        return port_to_name.at(port);
    }

    int update_uids(int64_t offset) {
        for(size_t i = 0; i < PORTS::COUNT; ++i) {
            uids[i] = i + offset;
        }
        return 0;
    }
};

class reduction_properties : public operation_properties {
public:
    enum PORTS {
        X = 0,
        Y,

        COUNT
    };

private:
    cudnnReduceTensorOp_t mode;

public:
    bool is_mode_set;

    std::unordered_map<PORTS, std::string> port_to_name;
    int64_t uids[PORTS::COUNT];

    reduction_properties(const std::string name) : operation_properties(name, Tag::Reduction) {
        port_to_name[PORTS::X] = name + "::X";
        port_to_name[PORTS::Y] = name + "::Y";
    }

    int update_uids(int64_t offset) {
        for(size_t i = 0; i < PORTS::COUNT; ++i) {
            uids[i] = i + offset;
        }
        return 0;
    }

    cudnnReduceTensorOp_t const &
    get_mode() const {
        return mode;
    }

    int
    set_mode(cudnnReduceTensorOp_t value) {
        mode = value;
        is_mode_set = true;
        return 0;
    }

    error_t map_port_to_tensor(std::vector<std::pair<PORTS, std::string>> const& names) {
        for(auto const& p: names) {
            port_to_name[p.first] = p.second;
        }
        return error_t::OK;
    }

    std::string
    get_port_name(PORTS port) const {
        return port_to_name.at(port);
    }
};

}