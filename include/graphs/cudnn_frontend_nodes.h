#pragma once

#include <iostream>
#include <numeric>
#include <unordered_map>
#include <vector>

#include "graphs/cudnn_frontend_graph_helpers.h"

namespace cudnn_frontend {

class graph_properties {
protected:
    std::string name;

public:
    graph_properties(const std::string &name) : name(name) {}

    std::string const
    get_name() const {
        return name;
    }
};

// simple structure to hold all properties of a tensor.
// Each property has a getter setter.
class tensor_properties : public graph_properties {
protected:
    // TODO: use custom FE data type string/enum.
    cudnnDataType_t data_type = CUDNN_DATA_FLOAT;
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

    // TODO: Currently this structure takes in unrolled list of properties to set.
    // But later, it will take in the context and derive properties to set from it.
    int set_properties_from_context(cudnnTensorFormat_t const filter_format, cudnnDataType_t const data_type, int64_t const uid) {
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

    tensor_properties(const std::string &name) : graph_properties(name) {}

    cudnnDataType_t const &
    get_data_type() const {
        return data_type;
    }

    int
    set_data_type(cudnnDataType_t value) {
        data_type = value;
        is_data_type_set = true;
        return 0;
    }

    std::vector<int64_t> const &
    get_dim() const {
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

class Node : public graph_properties {
public:
    enum class Type {
        Convolution,
        Matmul,
        Pointwise,
        Reduction
    };

    using parent_class = Node;
protected:

    // TODO: remove setting tensor data type in operation properties.
    // The operation only has to know of the compute type. cudnn operation
    // will convert any tensor type to compute type internally.
    cudnnDataType_t tensor_data_type;
    cudnnDataType_t compute_type;

    Type node_type;
public:

    bool is_tensor_data_type_set = false;
    bool is_compute_type_set = false;
    bool is_input_set = false;

    Node(const std::string name, Type t) : graph_properties(name), node_type(t) {}

    Type
    get_node_type() const {
        return node_type;
    }

    cudnnDataType_t
    get_tensor_data_type() const {
        return tensor_data_type;
    }

    int
    set_tensor_data_type(cudnnDataType_t value) {
        tensor_data_type = value;
        is_tensor_data_type_set = true;
        return 0;
    }

    cudnnDataType_t
    get_compute_type() const {
        return compute_type;
    }

    int
    set_compute_type(cudnnDataType_t value) {
        compute_type = value;
        is_compute_type_set = true;
        return 0;
    }

    virtual std::vector<std::string>
    get_inputs() const = 0;

};

class convolution_node : public Node {
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
    convolution_node(const std::string name) : Node(name, Type::Convolution) {
        port_to_name[PORTS::X] = name + "::X";
        port_to_name[PORTS::W] = name + "::W";
        port_to_name[PORTS::Y] = name + "::Y";
    }

    cudnn_frontend_error_t
    set_port_names(std::vector<std::pair<PORTS, std::string>> const& names) {
        for(auto const& p: names) {
            port_to_name[p.first] = p.second;
        }
        return cudnn_frontend_error_t::OK;
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

    std::vector<std::string>
    get_inputs() const override {
        return {port_to_name.at(PORTS::X), port_to_name.at(PORTS::W)};
    }

    friend std::ostream& operator<<(std::ostream& os, const convolution_node& props);
};

inline std::ostream& operator<<(std::ostream& os, const convolution_node& props) {
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
    for(size_t i = 0; i < convolution_node::PORTS::COUNT; ++i) {
        os << props.get_port_name(static_cast<convolution_node::PORTS>(i)) << ",";
    }
    os << "],";
    return os;
}

class matmul_node : public Node {
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
    matmul_node(const std::string name) : Node(name, Type::Matmul) {
        port_to_name[PORTS::X] = name + "::X";
        port_to_name[PORTS::W] = name + "::W";
        port_to_name[PORTS::Y] = name + "::Y";
    }

    cudnn_frontend_error_t
    set_port_names(std::vector<std::pair<PORTS, std::string>> const& names) {
        for(auto const& p: names) {
            port_to_name[p.first] = p.second;
        }
        return cudnn_frontend_error_t::OK;
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

    std::vector<std::string>
    get_inputs() const override {
        return{port_to_name.at(PORTS::X), port_to_name.at(PORTS::W)};
    }
};

class pointwise_node : public Node {
public:
    enum PORTS {
        X = 0,
        B,
        Y,

        COUNT
    };

private:
    cudnnPointwiseMode_t mode;
public:
    bool is_mode_set;

    std::unordered_map<PORTS, std::string> port_to_name;
    int64_t uids[PORTS::COUNT];

    pointwise_node(const std::string name) : Node(name, Type::Pointwise) {
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

    cudnnPointwiseMode_t const &
    get_mode() const {
        return mode;
    }


    int
    set_mode(cudnnPointwiseMode_t value) {
        mode = value;
        is_mode_set = true;
        return 0;
    }

    int
    set_mode(std::string value) {
        mode = string_to_pointwise_mode(value);
        is_mode_set = true;
        return 0;
    }

    cudnn_frontend_error_t
    set_port_names(std::vector<std::pair<PORTS, std::string>> const& names) {
        for(auto const& p: names) {
            port_to_name[p.first] = p.second;
        }
        return cudnn_frontend_error_t::OK;
    }

    std::string
    get_port_name(PORTS port) const {
        return port_to_name.at(port);
    }

    std::vector<std::string>
    get_inputs() const override {
        return{port_to_name.at(PORTS::X), port_to_name.at(PORTS::B)};
    }

    friend std::ostream& operator<<(std::ostream& os, const pointwise_node& props);
};

inline std::ostream& operator<<(std::ostream& os, const pointwise_node& props) {
    os << "{" 
    << " name: '" << props.get_name() << "',"
    << " ports: [";
    for(size_t i = 0; i < pointwise_node::PORTS::COUNT; ++i) {
        os << props.get_port_name(static_cast<pointwise_node::PORTS>(i)) << ",";
    }
    os << "],";
    return os;
}

class reduction_node : public Node {
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

    reduction_node(const std::string name) : Node(name, Type::Reduction) {
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

    std::vector<std::string>
    get_inputs() const override {
        return{port_to_name.at(PORTS::X)};
    }

    cudnn_frontend_error_t
    set_port_names(std::vector<std::pair<PORTS, std::string>> const& names) {
        for(auto const& p: names) {
            port_to_name[p.first] = p.second;
        }
        return cudnn_frontend_error_t::OK;
    }

    std::string
    get_port_name(PORTS port) const {
        return port_to_name.at(port);
    }
};

}