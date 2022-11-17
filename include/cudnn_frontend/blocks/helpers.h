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
    int64_t dim_count;
    int64_t dim[CUDNN_DIM_MAX];
    int64_t stride[CUDNN_DIM_MAX];

    cudnnDataType_t data_type;

    enum UIDs {
        TENSOR_UID = 0,

        UID_COUNT
    };
    int64_t uids_with_offset[static_cast<size_t>(UIDs::UID_COUNT)];
};

class convolution_properties {
public:
    int64_t dim_count;
    int64_t input_dim[CUDNN_DIM_MAX];
    int64_t weight_dim[CUDNN_DIM_MAX];
    int64_t output_dim[CUDNN_DIM_MAX];

    int64_t input_stride[CUDNN_DIM_MAX];
    int64_t weight_stride[CUDNN_DIM_MAX];
    int64_t output_stride[CUDNN_DIM_MAX];

    int64_t padding[CUDNN_DIM_MAX];
    int64_t stride[CUDNN_DIM_MAX];
    int64_t dilation[CUDNN_DIM_MAX];

    cudnnDataType_t tensor_data_type;
    cudnnDataType_t compute_type;

    enum UIDs {
        INPUT_UID = 0,
        WEIGHT_UID,
        OUTPUT_UID,

        UID_COUNT
    };
    int64_t uids_with_offset[static_cast<size_t>(UIDs::UID_COUNT)];

    void update_uids(int64_t offset) {
        for(size_t i = 0; i < static_cast<size_t>(UIDs::UID_COUNT); ++i) {
            uids_with_offset[i] = i + offset;
        }
    }
};

static bool
allowAllConfig(cudnnBackendDescriptor_t engine_config) {
    (void)engine_config;
    return false;
}

} // namespace cudnn_frontend