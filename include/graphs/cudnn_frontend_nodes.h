#pragma once

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

class tensor_properties : public graph_properties {
protected:
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

    int64_t const &
    get_uid() const {
        return uid;
    }

    int
    set_uid(int64_t value) {
        uid = value;
        is_uid_set = true;
        return 0;
    }

};

class Node : public graph_properties {
public:
    enum class Type {
        Convolution,
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

    std::vector<std::string> inputs = {};
    Type node_type;
    size_t input_port_count = 1;
public:

    bool is_tensor_data_type_set = false;
    bool is_compute_type_set = false;
    bool is_input_set = false;

    Node(const std::string name, Type t, int port_count) : graph_properties(name), node_type(t), input_port_count(port_count) {}

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

    virtual cudnn_frontend_error_t
    set_inputs(std::vector<std::string> const & value) {
        if (value.size() != input_port_count) {
            return cudnn_frontend_error_t::INPUT_PORT_COUNT_MISMATCH;
        }
        inputs = value;
        is_input_set = true;
        return cudnn_frontend_error_t::OK;
    } 

    std::vector<std::string> const &
    get_inputs() const {
        return inputs;
    } 
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
    convolution_node(const std::string name) : Node(name, Type::Convolution , 2) {
        port_to_name[PORTS::X] = name + "::X";
        port_to_name[PORTS::W] = name + "::W";
        port_to_name[PORTS::Y] = name + "::Y";
    }

    cudnn_frontend_error_t
    set_inputs(std::vector<std::string> const & value) override final {
        auto return_value = parent_class::set_inputs(value);
        if (return_value != cudnn_frontend_error_t::OK) {
            return return_value;
        }
        getLogger() << "conv node " << name << " has inputs: [";
        for (size_t i = PORTS::X; i < parent_class::input_port_count; i++) {
            port_to_name[static_cast<PORTS>(i)] = inputs[i];
            getLogger() << port_to_name[static_cast<PORTS>(i)] << ",";
        }
        getLogger() << "]" << std::endl;
        return cudnn_frontend_error_t::OK;

    }

    cudnn_frontend_error_t
    set_port_names(std::vector<std::pair<PORTS, std::string>> const& names) {
        for(auto const& p: names) {
            port_to_name[static_cast<PORTS>(p.first)] = p.second;
        }
        return cudnn_frontend_error_t::OK;
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

    pointwise_node(const std::string name) : Node(name, Type::Pointwise, 2) {
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
    set_inputs(std::vector<std::string> const & value) override final {
        auto return_value = parent_class::set_inputs(value);
        if (return_value != cudnn_frontend_error_t::OK) {
            return return_value;
        }
        for (size_t i = PORTS::X; i < parent_class::input_port_count; i++) {
            port_to_name[static_cast<PORTS>(i)] = inputs[i];
        }
        return cudnn_frontend_error_t::OK;
    }

    cudnn_frontend_error_t
    set_port_names(std::vector<std::pair<PORTS, std::string>> const& names) {
        for(auto const& p: names) {
            port_to_name[static_cast<PORTS>(p.first)] = p.second;
        }
        return cudnn_frontend_error_t::OK;
    }
};

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

    reduction_node(const std::string name) : Node(name, Type::Reduction, 1) {
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

    cudnn_frontend_error_t
    set_inputs(std::vector<std::string> const & value) override final {
        auto return_value = parent_class::set_inputs(value);
        if (return_value != cudnn_frontend_error_t::OK) {
            return return_value;
        }
        for (size_t i = PORTS::X; i < parent_class::input_port_count; i++) {
            port_to_name[static_cast<PORTS>(i)] = inputs[i];
        }
        return cudnn_frontend_error_t::OK;
    }

    cudnn_frontend_error_t
    set_port_names(std::vector<std::pair<PORTS, std::string>> const& names) {
        for(auto const& p: names) {
            port_to_name[static_cast<PORTS>(p.first)] = p.second;
        }
        return cudnn_frontend_error_t::OK;
    }
};

}