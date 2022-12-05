#pragma once

#include <cudnn_ops_infer.h>

namespace cudnn_frontend {

static void
generateStrides(int64_t const* const dimA, int64_t* const strideA, int const nbDims, cudnnTensorFormat_t const filterFormat) {
    if (filterFormat == CUDNN_TENSOR_NCHW) {
        strideA[nbDims - 1] = 1;
        for (int64_t d = nbDims - 2; d >= 0; d--) {
            strideA[d] = strideA[d + 1] * dimA[d + 1];
        }
    } else {
        // Here we assume that the format is CUDNN_TENSOR_NHWC
        strideA[1]          = 1;
        strideA[nbDims - 1] = strideA[1] * dimA[1];
        for (int64_t d = nbDims - 2; d >= 2; d--) {
            strideA[d] = strideA[d + 1] * dimA[d + 1];
        }
        strideA[0] = strideA[2] * dimA[2];
    }
}

class tensor_properties {
public:
    int64_t uid;

    int64_t dim_count;
    int64_t dim[CUDNN_DIM_MAX];
    int64_t stride[CUDNN_DIM_MAX];

    cudnnDataType_t data_type;

    bool is_virtual;

    bool is_pass_by_value;
};

class convolution_properties {
public:
    int64_t dim_count;

    int64_t padding[CUDNN_DIM_MAX];
    int64_t stride[CUDNN_DIM_MAX];
    int64_t dilation[CUDNN_DIM_MAX];

    cudnnDataType_t tensor_data_type;
    cudnnDataType_t compute_data_type;

    enum UIDs {
        X_UID = 0,
        W_UID,
        Y_UID,

        UID_COUNT
    };
    int64_t uids[static_cast<size_t>(UIDs::UID_COUNT)];

    void update_uids(int64_t offset) {
        for(size_t i = 0; i < static_cast<size_t>(UIDs::UID_COUNT); ++i) {
            uids[i] = i + offset;
        }
    }
};

class pointwise_properties {
public:
    int64_t dim_count;

    cudnnPointwiseMode_t mode;

    cudnnDataType_t tensor_data_type;
    cudnnDataType_t compute_data_type;

    enum UIDs {
        X_UID = 0,
        B_UID,
        Y_UID,

        UID_COUNT
    };
    int64_t uids[static_cast<size_t>(UIDs::UID_COUNT)];

    void update_uids(int64_t offset) {
        for(size_t i = 0; i < static_cast<size_t>(UIDs::UID_COUNT); ++i) {
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