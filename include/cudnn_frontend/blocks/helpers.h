#pragma once

#include <unordered_map>
#include <vector>

#include <cudnn_ops_infer.h>

namespace cudnn_frontend {

static void
generateStrides(std::vector<int64_t>& dim, std::vector<int64_t>& stride, cudnnTensorFormat_t const filterFormat) {
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
}

class tensor_properties {
public:
    std::string name;
    int64_t uid;

    std::vector<int64_t> dim;
    std::vector<int64_t> stride;

    cudnnDataType_t data_type;

    bool is_virtual;

    bool is_pass_by_value;
};

class convolution_properties {
public:
    convolution_properties() {
        port_to_name[PORTS::X] = "X";
        port_to_name[PORTS::W] = "W";
        port_to_name[PORTS::Y] = "Y";
    }

    std::vector<int64_t> padding;
    std::vector<int64_t> stride;
    std::vector<int64_t> dilation;

    cudnnDataType_t tensor_data_type;
    cudnnDataType_t compute_data_type;

    enum PORTS {
        X = 0,
        W,
        Y,

        COUNT
    };
    std::unordered_map<PORTS, std::string> port_to_name;
    int64_t uids[PORTS::COUNT];

    void update_uids(int64_t offset) {
        for(size_t i = 0; i < PORTS::COUNT; ++i) {
            uids[i] = i + offset;
        }
    }
};

class pointwise_properties {
public:
    pointwise_properties() {
        port_to_name[PORTS::X] = "X";
        port_to_name[PORTS::B] = "B";
        port_to_name[PORTS::Y] = "Y";
    }

    cudnnPointwiseMode_t mode;

    cudnnDataType_t tensor_data_type;
    cudnnDataType_t compute_data_type;

    enum PORTS {
        X = 0,
        B,
        Y,

        COUNT
    };
    std::unordered_map<PORTS, std::string> port_to_name;
    int64_t uids[PORTS::COUNT];

    void update_uids(int64_t offset) {
        for(size_t i = 0; i < PORTS::COUNT; ++i) {
            uids[i] = i + offset;
        }
    }
};

static bool
allowAllConfig(cudnnBackendDescriptor_t engine_config) {
    (void)engine_config;
    return false;
}

} // namespace cudnn_frontend