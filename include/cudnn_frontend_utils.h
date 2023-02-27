/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#pragma once
#include <exception>
#include <string>
#include <vector>

#include "cudnn_backend_base.h"
#include "cudnn_frontend_Logging.h"

#ifndef NV_CUDNN_DISABLE_EXCEPTION
#ifdef _MSC_VER
#pragma warning(disable:4702) // if exceptions are enabled there are unreachable return statements
#endif
#endif

#define CUDNN_FRONTEND_UNUSED(X) ((void)X)
namespace cudnn_frontend {

/// Detailed feature_vector. Generally the Tensor and Operation properties
using feature_vector_t = std::vector<int64_t>;

#ifndef NV_CUDNN_DISABLE_EXCEPTION
class cudnnException : public std::runtime_error {
   public:
    cudnnException(const char *message, cudnnStatus_t status) throw() : std::runtime_error(message) {
        error_status = status;
    }
    virtual const char *
    what() const throw() {
        return std::runtime_error::what();
    }
    cudnnStatus_t getCudnnStatus() {
        return error_status;
    }

    cudnnStatus_t error_status;
};
#endif

static inline bool
AllowAll(cudnnBackendDescriptor_t engine_config) {
    (void)engine_config;
    return false;
}

static inline void
throw_if(std::function<bool()> expr, const char *message, cudnnStatus_t status) {
    if (expr()) {
#ifndef NV_CUDNN_DISABLE_EXCEPTION
        throw cudnnException(message, status);
#endif
    }
}
static inline void
throw_if(bool expr, const char *message, cudnnStatus_t status) {
    if (expr) {
#ifndef NV_CUDNN_DISABLE_EXCEPTION
        throw cudnnException(message, status);
#endif
    }
}

static inline std::string
to_string(cudnnStatus_t status) {
    switch(status) {
        case CUDNN_STATUS_SUCCESS:
            return std::string("CUDNN_STATUS_SUCCESS");
        case CUDNN_STATUS_NOT_INITIALIZED:
            return std::string("CUDNN_STATUS_NOT_INITIALIZED");
        case CUDNN_STATUS_ALLOC_FAILED:
            return std::string("CUDNN_STATUS_ALLOC_FAILED");
        case CUDNN_STATUS_BAD_PARAM:
            return std::string("CUDNN_STATUS_BAD_PARAM");
        case CUDNN_STATUS_INTERNAL_ERROR:
            return std::string("CUDNN_STATUS_INTERNAL_ERROR");
        case CUDNN_STATUS_INVALID_VALUE:
            return std::string("CUDNN_STATUS_INVALID_VALUE");
        case CUDNN_STATUS_ARCH_MISMATCH:
            return std::string("CUDNN_STATUS_ARCH_MISMATCH");
        case CUDNN_STATUS_MAPPING_ERROR:
            return std::string("CUDNN_STATUS_MAPPING_ERROR");
        case CUDNN_STATUS_EXECUTION_FAILED:
            return std::string("CUDNN_STATUS_EXECUTION_FAILED");
        case CUDNN_STATUS_NOT_SUPPORTED:
            return std::string("CUDNN_STATUS_NOT_SUPPORTED");
        case CUDNN_STATUS_LICENSE_ERROR:
            return std::string("CUDNN_STATUS_LICENSE_ERROR");
        case CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING:
            return std::string("CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING");
        case CUDNN_STATUS_RUNTIME_IN_PROGRESS:
            return std::string("CUDNN_STATUS_RUNTIME_IN_PROGRESS");
        case CUDNN_STATUS_RUNTIME_FP_OVERFLOW:
            return std::string("CUDNN_STATUS_RUNTIME_FP_OVERFLOW");
        case CUDNN_STATUS_VERSION_MISMATCH:
            return std::string("CUDNN_STATUS_VERSION_MISMATCH");
#ifndef NO_DEFAULT_IN_SWITCH
        default:
            return std::string("UNKNOWN_CUDNN_STATUS");
#endif
    }
    return std::string("");
}

static inline void
set_error_and_throw_exception(BackendDescriptor const *desc, cudnnStatus_t status, const char *message) {
    if (desc != nullptr) {
        desc->set_status(status);
        desc->set_error(message);
    }
#ifndef NV_CUDNN_DISABLE_EXCEPTION
    throw cudnnException(
        std::string(std::string(message) + std::string(" cudnn_status: ") + to_string(status)).c_str(), status);
#endif
}

static inline std::string
to_string(cudnnDataType_t type) {
    switch(type) {
        case CUDNN_DATA_FLOAT:
            return std::string("CUDNN_DATA_FLOAT");
        case CUDNN_DATA_DOUBLE:
            return std::string("CUDNN_DATA_DOUBLE");
        case CUDNN_DATA_HALF:
            return std::string("CUDNN_DATA_HALF");
        case CUDNN_DATA_INT8:
            return std::string("CUDNN_DATA_INT8");
        case CUDNN_DATA_INT32:
            return std::string("CUDNN_DATA_INT32");
        case CUDNN_DATA_INT8x4: // x4 and x32 are replaced by vectorized dimension in the v8 API 
            return std::string("CUDNN_DATA_INT8x4");
        case CUDNN_DATA_UINT8:
            return std::string("CUDNN_DATA_UINT8");
        case CUDNN_DATA_UINT8x4: // x4 and x32 are replaced by vectorized dimension in the v8 API 
            return std::string("CUDNN_DATA_UINT8x4");
        case CUDNN_DATA_INT8x32: // x4 and x32 are replaced by vectorized dimension in the v8 API 
            return std::string("CUDNN_DATA_INT8x32");
        case CUDNN_DATA_INT64:
            return std::string("CUDNN_DATA_INT64");
        case CUDNN_DATA_BFLOAT16:
            return std::string("CUDNN_DATA_BFLOAT16");
#if (CUDNN_VERSION >= 8300)
        case CUDNN_DATA_BOOLEAN:
            return std::string("CUDNN_DATA_BOOLEAN");
#endif
#if (CUDNN_VERSION >= 8600)
        case CUDNN_DATA_FP8_E5M2:
            return std::string("CUDNN_DATA_FP8_E5M2");
        case CUDNN_DATA_FP8_E4M3:
            return std::string("CUDNN_DATA_FP8_E4M3");
#endif
#ifndef NO_DEFAULT_IN_SWITCH
        default:
            return std::string("UNKNOWN DATA_TYPE");
#endif
    }
    return std::string("");
}

#if (CUDNN_VERSION >= 8200)  
static inline std::string
to_string(cudnnBackendBehaviorNote_t note) {
    switch(note) {
        case CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION:
            return std::string("CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION");
#if (CUDNN_VERSION >= 8300)
        case CUDNN_BEHAVIOR_NOTE_REQUIRES_FILTER_INT8x32_REORDER:
            return std::string("CUDNN_BEHAVIOR_NOTE_REQUIRES_FILTER_INT8x32_REORDER");
        case CUDNN_BEHAVIOR_NOTE_REQUIRES_BIAS_INT8x32_REORDER:
            return std::string("CUDNN_BEHAVIOR_NOTE_REQUIRES_BIAS_INT8x32_REORDER");
#endif
        case CUDNN_BEHAVIOR_NOTE_TYPE_COUNT:
            return std::string("CUDNN_BEHAVIOR_NOTE_TYPE_COUNT");
#ifndef NO_DEFAULT_IN_SWITCH
        default:
            return std::string("UNKNOWN_BEHAVIOR_NOTE");
#endif
    }
    return std::string("INVALID_BEHAVIOR_NOTE");
}
#endif

static inline std::string
to_string(cudnnBackendNumericalNote_t note) {
    switch(note) {
        case CUDNN_NUMERICAL_NOTE_TENSOR_CORE:
            return std::string("CUDNN_NUMERICAL_NOTE_TENSOR_CORE");
        case CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS:
            return std::string("CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS");
        case CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION:
            return std::string("CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION");
        case CUDNN_NUMERICAL_NOTE_FFT:
            return std::string("CUDNN_NUMERICAL_NOTE_FFT");
        case CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC:
            return std::string("CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC");
        case CUDNN_NUMERICAL_NOTE_WINOGRAD:
            return std::string("CUDNN_NUMERICAL_NOTE_WINOGRAD");
#if (CUDNN_VERSION >= 8300)
        case CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_4x4:
            return std::string("CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_4x4");
        case CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_6x6:
            return std::string("CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_6x6");
        case CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_13x13:
            return std::string("CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_13x13");
#endif
        case CUDNN_NUMERICAL_NOTE_TYPE_COUNT:
            return std::string("CUDNN_NUMERICAL_NOTE_TYPE_COUNT");
#ifndef NO_DEFAULT_IN_SWITCH
        default:
            return std::string("UNKNOWN_NUMERICAL_NOTE");
#endif
    }
    return std::string("INVALID_NUMERICAL_NOTE");
}

static inline std::string
to_string(cudnnPointwiseMode_t mode) {
    switch(mode) {
        case CUDNN_POINTWISE_ADD:
            return std::string("CUDNN_POINTWISE_ADD");
        case CUDNN_POINTWISE_MUL:
            return std::string("CUDNN_POINTWISE_MUL");
#if (CUDNN_VERSION >= 8300)
        case CUDNN_POINTWISE_DIV:
            return std::string("CUDNN_POINTWISE_DIV");
        case CUDNN_POINTWISE_ADD_SQUARE:
            return std::string("CUDNN_POINTWISE_ADD_SQUARE");
        case CUDNN_POINTWISE_SUB:
            return std::string("CUDNN_POINTWISE_SUB");
        case CUDNN_POINTWISE_CMP_EQ:
            return std::string("CUDNN_POINTWISE_CMP_EQ");
        case CUDNN_POINTWISE_CMP_NEQ:
            return std::string("CUDNN_POINTWISE_CMP_NEQ");
        case CUDNN_POINTWISE_CMP_GT:
            return std::string("CUDNN_POINTWISE_CMP_GT");
        case CUDNN_POINTWISE_CMP_GE:
            return std::string("CUDNN_POINTWISE_CMP_GE");
        case CUDNN_POINTWISE_CMP_LT:
            return std::string("CUDNN_POINTWISE_CMP_LT");
        case CUDNN_POINTWISE_CMP_LE:
            return std::string("CUDNN_POINTWISE_CMP_LE");
        case CUDNN_POINTWISE_LOGICAL_AND:
            return std::string("CUDNN_POINTWISE_LOGICAL_AND");
        case CUDNN_POINTWISE_LOGICAL_OR:
            return std::string("CUDNN_POINTWISE_LOGICAL_OR");
#endif
        case CUDNN_POINTWISE_MIN:
            return std::string("CUDNN_POINTWISE_MIN");
        case CUDNN_POINTWISE_MAX:
            return std::string("CUDNN_POINTWISE_MAX");
        case CUDNN_POINTWISE_RELU_BWD:
            return std::string("CUDNN_POINTWISE_RELU_BWD");
        case CUDNN_POINTWISE_TANH_BWD:
            return std::string("CUDNN_POINTWISE_TANH_BWD");
        case CUDNN_POINTWISE_SIGMOID_BWD:
            return std::string("CUDNN_POINTWISE_SIGMOID_BWD");
        case CUDNN_POINTWISE_ELU_BWD:
            return std::string("CUDNN_POINTWISE_ELU_BWD");
        case CUDNN_POINTWISE_GELU_BWD:
            return std::string("CUDNN_POINTWISE_GELU_BWD");
        case CUDNN_POINTWISE_SOFTPLUS_BWD:
            return std::string("CUDNN_POINTWISE_SOFTPLUS_BWD");
        case CUDNN_POINTWISE_SWISH_BWD:
            return std::string("CUDNN_POINTWISE_SWISH_BWD");
#if (CUDNN_VERSION >= 8500)
        case CUDNN_POINTWISE_GELU_APPROX_TANH_BWD:
            return std::string("CUDNN_POINTWISE_GELU_APPROX_TANH_BWD");
#endif
        case CUDNN_POINTWISE_SQRT:
            return std::string("CUDNN_POINTWISE_SQRT");
        case CUDNN_POINTWISE_RELU_FWD:
            return std::string("CUDNN_POINTWISE_RELU_FWD");
        case CUDNN_POINTWISE_TANH_FWD:
            return std::string("CUDNN_POINTWISE_TANH_FWD");
        case CUDNN_POINTWISE_SIGMOID_FWD:
            return std::string("CUDNN_POINTWISE_SIGMOID_FWD");
        case CUDNN_POINTWISE_ELU_FWD:
            return std::string("CUDNN_POINTWISE_ELU_FWD");
        case CUDNN_POINTWISE_GELU_FWD:
            return std::string("CUDNN_POINTWISE_GELU_FWD");
        case CUDNN_POINTWISE_SOFTPLUS_FWD:
            return std::string("CUDNN_POINTWISE_SOFTPLUS_FWD");
        case CUDNN_POINTWISE_SWISH_FWD:
            return std::string("CUDNN_POINTWISE_SWISH_FWD");
#if (CUDNN_VERSION >= 8300)
        case CUDNN_POINTWISE_EXP:
            return std::string("CUDNN_POINTWISE_EXP");
        case CUDNN_POINTWISE_LOG:
            return std::string("CUDNN_POINTWISE_LOG");
        case CUDNN_POINTWISE_NEG:
            return std::string("CUDNN_POINTWISE_NEG");
        case CUDNN_POINTWISE_MOD:
            return std::string("CUDNN_POINTWISE_MOD");
        case CUDNN_POINTWISE_POW:
            return std::string("CUDNN_POINTWISE_POW");
        case CUDNN_POINTWISE_ABS:
            return std::string("CUDNN_POINTWISE_ABS");
        case CUDNN_POINTWISE_CEIL:
            return std::string("CUDNN_POINTWISE_CEIL");
        case CUDNN_POINTWISE_FLOOR:
            return std::string("CUDNN_POINTWISE_FLOOR");
        case CUDNN_POINTWISE_COS:
            return std::string("CUDNN_POINTWISE_COS");
        case CUDNN_POINTWISE_TAN:
            return std::string("CUDNN_POINTWISE_TAN");
        case CUDNN_POINTWISE_SIN:
            return std::string("CUDNN_POINTWISE_SIN");
        case CUDNN_POINTWISE_RSQRT:
            return std::string("CUDNN_POINTWISE_RSQRT");
        case CUDNN_POINTWISE_LOGICAL_NOT:
            return std::string("CUDNN_POINTWISE_LOGICAL_NOT");
#endif
#if (CUDNN_VERSION >= 8400)
        case CUDNN_POINTWISE_GEN_INDEX:
            return std::string("CUDNN_POINTWISE_GEN_INDEX");
        case CUDNN_POINTWISE_BINARY_SELECT:
            return std::string("CUDNN_POINTWISE_BINARY_SELECT");
#endif
#if (CUDNN_VERSION >= 8500)
        case CUDNN_POINTWISE_ERF:
            return std::string("CUDNN_POINTWISE_ERF");
        case CUDNN_POINTWISE_GELU_APPROX_TANH_FWD:
            return std::string("CUDNN_POINTWISE_GELU_APPROX_TANH_FWD");
        case CUDNN_POINTWISE_IDENTITY:
            return std::string("CUDNN_POINTWISE_IDENTITY");
#endif
#ifndef NO_DEFAULT_IN_SWITCH
        default:
            return std::string("UNKNOWN_CUDNN_POINTWISE_MODE");
#endif
    }
    return std::string("");
}

#if (CUDNN_VERSION >= 8700)
static inline std::string
to_string(cudnnRngDistribution_t distribution) {
    switch(distribution) {
        case CUDNN_RNG_DISTRIBUTION_BERNOULLI:
            return std::string("CUDNN_RNG_DISTRIBUTION_BERNOULLI");
        case CUDNN_RNG_DISTRIBUTION_UNIFORM:
            return std::string("CUDNN_RNG_DISTRIBUTION_UNIFORM");
        case CUDNN_RNG_DISTRIBUTION_NORMAL:
            return std::string("CUDNN_RNG_DISTRIBUTION_NORMAL");
#ifndef NO_DEFAULT_IN_SWITCH
        default:
            return std::string("UNKNOWN_CUDNN_DISTRIBUTION");
#endif
    }
    return std::string("");
}
#endif


enum class TensorReordering_t {
    NONE,
    INT8x32,
    F16x16,
};

enum class ResampleMode_t{
    NOT_SET,

    AVGPOOL_EXCLUDE_PADDING,
    AVGPOOL_INCLUDE_PADDING,
    BILINEAR,
    NEAREST,
    MAXPOOL,
};

enum class PaddingMode_t{
    NOT_SET,

    EDGE_VAL_PAD,
    NEG_INF_PAD,
    ZERO_PAD
};

enum class NormFwdPhase_t {
    NOT_SET,

    INFERENCE,
    TRAINING
};

enum class DescriptorType_t {
    NOT_SET,

    POINTWISE_DESCRIPTOR,
    CONVOLUTION_DESCRIPTOR,
    ENGINE_DESCRIPTOR,
    ENGINECFG_DESCRIPTOR,
    ENGINEHEUR_DESCRIPTOR,
    EXECUTION_PLAN_DESCRIPTOR,
    INTERMEDIATE_INFO_DESCRIPTOR,
    KNOB_CHOICE_DESCRIPTOR,
    KNOB_INFO_DESCRIPTOR,
    LAYOUT_INFO_DESCRIPTOR,
    OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR,
    OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR,
    OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR,
    OPERATION_POINTWISE_DESCRIPTOR,
    OPERATION_GEN_STATS_DESCRIPTOR,
    OPERATIONGRAPH_DESCRIPTOR,
    VARIANT_PACK_DESCRIPTOR,
    TENSOR_DESCRIPTOR,
    MATMUL_DESCRIPTOR,
    OPERATION_MATMUL_DESCRIPTOR,
    OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR,
    REDUCTION_DESCRIPTOR,
    OPERATION_REDUCTION_DESCRIPTOR,
    OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR,
    RESAMPLE_DESCRIPTOR,
    OPERATION_RESAMPLE_FWD_DESCRIPTOR,
    OPERATION_RESAMPLE_BWD_DESCRIPTOR,
    OPERATION_CONCAT_DESCRIPTOR,
    OPERATION_SIGNAL_DESCRIPTOR,
    OPERATION_NORM_FORWARD_DESCRIPTOR,
    OPERATION_NORM_BACKWARD_DESCRIPTOR,
    OPERATION_RESHAPE_DESCRIPTOR,
    RNG_DESCRIPTOR,
    OPERATION_RNG_DESCRIPTOR
};

enum class NormMode_t {
    NOT_SET,

    LAYER_NORM,
    INSTANCE_NORM,
    BATCH_NORM,
    GROUP_NORM,
};

static inline std::ostream& operator<<(std::ostream& os, const NormMode_t& mode) {
    switch (mode) {
        case NormMode_t::LAYER_NORM:
            os << "LAYER_NORM";
            break;
        case NormMode_t::INSTANCE_NORM:
            os << "INSTANCE_NORM";
            break;
        case NormMode_t::BATCH_NORM:
            os << "BATCH_NORM";
            break;
        case NormMode_t::GROUP_NORM:
            os << "GROUP_NORM";
            break;
        case NormMode_t::NOT_SET:
            os << "NOT_SET";
            break;
    }
    return os;
} 

static inline std::ostream& operator<<(std::ostream& os, const DescriptorType_t& mode) {
    switch (mode) {
        case DescriptorType_t::POINTWISE_DESCRIPTOR:
            os << "POINTWISE_DESCRIPTOR";
            break;
        case DescriptorType_t::CONVOLUTION_DESCRIPTOR:
            os << "CONVOLUTION_DESCRIPTOR";
            break;
        case DescriptorType_t::ENGINE_DESCRIPTOR:
            os << "ENGINE_DESCRIPTOR";
            break;
        case DescriptorType_t::ENGINECFG_DESCRIPTOR:
            os << "ENGINECFG_DESCRIPTOR";
            break;
        case DescriptorType_t::ENGINEHEUR_DESCRIPTOR:
            os << "ENGINEHEUR_DESCRIPTOR";
            break;
        case DescriptorType_t::EXECUTION_PLAN_DESCRIPTOR:
            os << "EXECUTION_PLAN_DESCRIPTOR";
            break;
        case DescriptorType_t::INTERMEDIATE_INFO_DESCRIPTOR:
            os << "INTERMEDIATE_INFO_DESCRIPTOR";
            break;
        case DescriptorType_t::KNOB_CHOICE_DESCRIPTOR:
            os << "KNOB_CHOICE_DESCRIPTOR";
            break;
        case DescriptorType_t::KNOB_INFO_DESCRIPTOR:
            os << "KNOB_INFO_DESCRIPTOR";
            break;
        case DescriptorType_t::LAYOUT_INFO_DESCRIPTOR:
            os << "LAYOUT_INFO_DESCRIPTOR";
            break;
        case DescriptorType_t::OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR:
            os << "OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR";
            break;
        case DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR:
            os << "OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR";
            break;
        case DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR:
            os << "OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR";
            break;
        case DescriptorType_t::OPERATION_POINTWISE_DESCRIPTOR:
            os << "OPERATION_POINTWISE_DESCRIPTOR";
            break;
        case DescriptorType_t::OPERATION_GEN_STATS_DESCRIPTOR:
            os << "OPERATION_GEN_STATS_DESCRIPTOR";
            break;
        case DescriptorType_t::OPERATIONGRAPH_DESCRIPTOR:
            os << "OPERATIONGRAPH_DESCRIPTOR";
            break;
        case DescriptorType_t::VARIANT_PACK_DESCRIPTOR:
            os << "VARIANT_PACK_DESCRIPTOR";
            break;
        case DescriptorType_t::TENSOR_DESCRIPTOR:
            os << "TENSOR_DESCRIPTOR";
            break;
        case DescriptorType_t::MATMUL_DESCRIPTOR:
            os << "MATMUL_DESCRIPTOR";
            break;
        case DescriptorType_t::OPERATION_MATMUL_DESCRIPTOR:
            os << "OPERATION_MATMUL_DESCRIPTOR";
            break;
        case DescriptorType_t::OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR:
            os << "OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR";
            break;
        case DescriptorType_t::REDUCTION_DESCRIPTOR:
            os << "REDUCTION_DESCRIPTOR";
            break;
        case DescriptorType_t::OPERATION_REDUCTION_DESCRIPTOR:
            os << "OPERATION_REDUCTION_DESCRIPTOR";
            break;
        case DescriptorType_t::OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR:
            os << "OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR";
            break;
        case DescriptorType_t::RESAMPLE_DESCRIPTOR:
            os << "RESAMPLE_DESCRIPTOR";
            break;
        case DescriptorType_t::OPERATION_RESAMPLE_FWD_DESCRIPTOR:
            os << "OPERATION_RESAMPLE_FWD_DESCRIPTOR";
            break;
        case DescriptorType_t::OPERATION_RESAMPLE_BWD_DESCRIPTOR:
            os << "OPERATION_RESAMPLE_BWD_DESCRIPTOR";
            break;
        case DescriptorType_t::OPERATION_CONCAT_DESCRIPTOR:
            os << "OPERATION_CONCAT_DESCRIPTOR";
            break;
        case DescriptorType_t::OPERATION_SIGNAL_DESCRIPTOR:
            os << "OPERATION_SIGNAL_DESCRIPTOR";
            break;
        case DescriptorType_t::OPERATION_NORM_FORWARD_DESCRIPTOR:
            os << "OPERATION_NORM_FORWARD_DESCRIPTOR";
            break;
        case DescriptorType_t::OPERATION_NORM_BACKWARD_DESCRIPTOR:
            os << "OPERATION_NORM_BACKWARD_DESCRIPTOR";
            break;
        case DescriptorType_t::OPERATION_RESHAPE_DESCRIPTOR:
            os << "OPERATION_RESHAPE_DESCRIPTOR";
            break;
        case DescriptorType_t::RNG_DESCRIPTOR:
            os << "RNG_DESCRIPTOR";
            break;
        case DescriptorType_t::OPERATION_RNG_DESCRIPTOR:
            os << "OPERATION_RNG_DESCRIPTOR";
            break;
        case DescriptorType_t::NOT_SET:
            os << "NOT_SET";
            break;
    }
    return os;
} 

static inline std::ostream& operator<<(std::ostream& os, const NormFwdPhase_t& mode) {
    switch (mode)
    {
        case NormFwdPhase_t::INFERENCE:
            os << "INFERENCE";
            break;
        case NormFwdPhase_t::TRAINING:
            os << "TRAINING";
            break;
        case NormFwdPhase_t::NOT_SET:
            os << "NOT_SET";
            break;
    }
    return os;
} 

static inline std::ostream& operator<<(std::ostream& os, const TensorReordering_t& mode) {
    switch (mode)
    {
        case TensorReordering_t::INT8x32:
            os << "INT8x32";
            break;
        case TensorReordering_t::F16x16:
            os << "F16x16";
            break;
        case TensorReordering_t::NONE:
            os << "NONE";
            break;
    }
    return os;
} 

static inline std::ostream& operator<<(std::ostream& os, const ResampleMode_t& mode) {
    switch (mode)
    {
        case ResampleMode_t::AVGPOOL_EXCLUDE_PADDING:
            os << "AVGPOOL_EXCLUDE_PADDING";
            break;
        case ResampleMode_t::AVGPOOL_INCLUDE_PADDING:
            os << "AVGPOOL_INCLUDE_PADDING";
            break;
        case ResampleMode_t::BILINEAR:
            os << "BILINEAR";
            break; 
        case ResampleMode_t::NEAREST:
            os << "NEAREST";
            break; 
        case ResampleMode_t::MAXPOOL:
            os << "MAXPOOL";
            break; 
        case ResampleMode_t::NOT_SET:
            os << "NOT_SET";
            break;
    }
    return os;
} 

static inline std::ostream& operator<<(std::ostream& os, const PaddingMode_t& mode) {
    switch (mode)
    {
        case PaddingMode_t::ZERO_PAD:
            os << "ZERO_PAD";
            break;
        case PaddingMode_t::NEG_INF_PAD:
            os << "NEG_INF_PAD";
            break;
        case PaddingMode_t::EDGE_VAL_PAD:
            os << "EDGE_VAL_PAD";
            break; 
        case PaddingMode_t::NOT_SET:
            os << "NOT_SET";
            break;
    }
    return os;
} 

namespace detail {
         
    static inline cudnnStatus_t convert_to_cudnn_type(cudnn_frontend::DescriptorType_t const mode, cudnnBackendDescriptorType_t& cudnn_mode) {
        switch (mode) {
            case DescriptorType_t::POINTWISE_DESCRIPTOR:
                cudnn_mode = CUDNN_BACKEND_POINTWISE_DESCRIPTOR;
                return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
            case DescriptorType_t::CONVOLUTION_DESCRIPTOR:
                cudnn_mode = CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR;
                return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
            case DescriptorType_t::ENGINE_DESCRIPTOR:
                cudnn_mode = CUDNN_BACKEND_ENGINE_DESCRIPTOR;
                return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
            case DescriptorType_t::ENGINECFG_DESCRIPTOR:
                cudnn_mode = CUDNN_BACKEND_ENGINECFG_DESCRIPTOR;
                return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
            case DescriptorType_t::ENGINEHEUR_DESCRIPTOR:
                cudnn_mode = CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR;
                return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
            case DescriptorType_t::EXECUTION_PLAN_DESCRIPTOR:
                cudnn_mode = CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR;
                return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
            case DescriptorType_t::INTERMEDIATE_INFO_DESCRIPTOR:
                cudnn_mode = CUDNN_BACKEND_INTERMEDIATE_INFO_DESCRIPTOR;
                return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
            case DescriptorType_t::KNOB_CHOICE_DESCRIPTOR:
                cudnn_mode = CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR;
                return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
            case DescriptorType_t::KNOB_INFO_DESCRIPTOR:
                cudnn_mode = CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR;
                return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
            case DescriptorType_t::LAYOUT_INFO_DESCRIPTOR:
                cudnn_mode = CUDNN_BACKEND_LAYOUT_INFO_DESCRIPTOR;
                return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
            case DescriptorType_t::OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR:
                cudnn_mode = CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR;
                return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
            case DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR:
                cudnn_mode = CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR;
                return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
            case DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR:
                cudnn_mode = CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR;
                return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
            case DescriptorType_t::OPERATION_POINTWISE_DESCRIPTOR:
                cudnn_mode = CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR;
                return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
            case DescriptorType_t::OPERATION_GEN_STATS_DESCRIPTOR:
                cudnn_mode = CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR;
                return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
            case DescriptorType_t::OPERATIONGRAPH_DESCRIPTOR:
                cudnn_mode = CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR;
                return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
            case DescriptorType_t::VARIANT_PACK_DESCRIPTOR:
                cudnn_mode = CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR;
                return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
            case DescriptorType_t::TENSOR_DESCRIPTOR:
                cudnn_mode = CUDNN_BACKEND_TENSOR_DESCRIPTOR;
                return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
            case DescriptorType_t::MATMUL_DESCRIPTOR:
                cudnn_mode = CUDNN_BACKEND_MATMUL_DESCRIPTOR;
                return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
            case DescriptorType_t::OPERATION_MATMUL_DESCRIPTOR:
                cudnn_mode = CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR;
                return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
            case DescriptorType_t::OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR:
                cudnn_mode = CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR;
                return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
            case DescriptorType_t::REDUCTION_DESCRIPTOR:
                cudnn_mode = CUDNN_BACKEND_REDUCTION_DESCRIPTOR;
                return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
            case DescriptorType_t::OPERATION_REDUCTION_DESCRIPTOR:
                cudnn_mode = CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR;
                return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
            
            case DescriptorType_t::OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR:
        #if (CUDNN_VERSION >= 8400)
                cudnn_mode = CUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR;
                return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        #else
                return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
        #endif
            case DescriptorType_t::RESAMPLE_DESCRIPTOR:
        #if (CUDNN_VERSION >= 8500)
                cudnn_mode = CUDNN_BACKEND_RESAMPLE_DESCRIPTOR;
                return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        #else
                return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
        #endif

            case DescriptorType_t::OPERATION_RESAMPLE_FWD_DESCRIPTOR:
        #if (CUDNN_VERSION >= 8500)
                cudnn_mode = CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR;
                return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        #else
                return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
        #endif

            case DescriptorType_t::OPERATION_RESAMPLE_BWD_DESCRIPTOR:
        #if (CUDNN_VERSION >= 8600)
                cudnn_mode = CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR;
                return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        #else
                return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
        #endif

            case DescriptorType_t::OPERATION_CONCAT_DESCRIPTOR:
        #if (CUDNN_VERSION >= 8500)
                cudnn_mode = CUDNN_BACKEND_OPERATION_CONCAT_DESCRIPTOR;
                return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        #else
                return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
        #endif
            
            case DescriptorType_t::OPERATION_SIGNAL_DESCRIPTOR:
        #if (CUDNN_VERSION >= 8500)
                cudnn_mode = CUDNN_BACKEND_OPERATION_SIGNAL_DESCRIPTOR;
                return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        #else
                return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
        #endif

            case DescriptorType_t::OPERATION_NORM_FORWARD_DESCRIPTOR:
        #if (CUDNN_VERSION >= 8500)
                cudnn_mode = CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR;
                return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        #else
                return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
        #endif

            case DescriptorType_t::OPERATION_NORM_BACKWARD_DESCRIPTOR:
        #if (CUDNN_VERSION >= 8500)
                cudnn_mode = CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR;
                return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        #else
                return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
        #endif

            case DescriptorType_t::OPERATION_RESHAPE_DESCRIPTOR:
        #if (CUDNN_VERSION >= 8700)
                cudnn_mode = CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR;
                return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        #else
                return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
        #endif
        
            case DescriptorType_t::RNG_DESCRIPTOR:
        #if (CUDNN_VERSION >= 8700)
                cudnn_mode = CUDNN_BACKEND_RNG_DESCRIPTOR;
                return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        #else
                return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
        #endif
        
            case DescriptorType_t::OPERATION_RNG_DESCRIPTOR:
        #if (CUDNN_VERSION >= 8700)
                cudnn_mode = CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR;
                return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
        #else
                return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
        #endif
        
            #ifndef NO_DEFAULT_IN_SWITCH
            default:
                return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
            #endif
        }
        return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
    }

#if (CUDNN_VERSION >= 8500)
    static inline cudnnStatus_t convert_to_cudnn_type(cudnn_frontend::ResampleMode_t const mode, cudnnResampleMode_t& cudnn_mode) {
        switch (mode)
        {
#if (CUDNN_VERSION >= 8600)
            case cudnn_frontend::ResampleMode_t::AVGPOOL_EXCLUDE_PADDING:
                cudnn_mode = CUDNN_RESAMPLE_AVGPOOL_EXCLUDE_PADDING;
                return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
            case cudnn_frontend::ResampleMode_t::AVGPOOL_INCLUDE_PADDING:
                cudnn_mode = CUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING;
                return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
#else
            case cudnn_frontend::ResampleMode_t::AVGPOOL_INCLUDE_PADDING:
                cudnn_mode = CUDNN_RESAMPLE_AVGPOOL;
                return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
#endif
            case cudnn_frontend::ResampleMode_t::BILINEAR:
                cudnn_mode = CUDNN_RESAMPLE_BILINEAR;
                return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
            case cudnn_frontend::ResampleMode_t::NEAREST:
                cudnn_mode = CUDNN_RESAMPLE_NEAREST;
                return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
            case cudnn_frontend::ResampleMode_t::MAXPOOL:
                cudnn_mode = CUDNN_RESAMPLE_MAXPOOL;
                return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
#ifndef NO_DEFAULT_IN_SWITCH
            default:
                return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
#endif
        }
        return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
    }
     
    static inline cudnnStatus_t convert_to_cudnn_type(cudnn_frontend::PaddingMode_t const mode, cudnnPaddingMode_t& cudnn_mode) {
        switch (mode)
        {
            case cudnn_frontend::PaddingMode_t::ZERO_PAD:
                cudnn_mode = CUDNN_ZERO_PAD;
                return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
            case cudnn_frontend::PaddingMode_t::NEG_INF_PAD:
                cudnn_mode = CUDNN_NEG_INF_PAD;
                return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
            case cudnn_frontend::PaddingMode_t::EDGE_VAL_PAD:
                cudnn_mode = CUDNN_EDGE_VAL_PAD;
                return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
            #ifndef NO_DEFAULT_IN_SWITCH
            default:
                return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
            #endif
        }
        return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
    }

    static inline cudnnStatus_t convert_to_cudnn_type(cudnn_frontend::NormMode_t const mode, cudnnBackendNormMode_t& cudnn_mode) {
        switch (mode)
        {
            #if (CUDNN_VERSION >= 8500)
            case NormMode_t::LAYER_NORM:
                cudnn_mode = CUDNN_LAYER_NORM;
                return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
            case NormMode_t::INSTANCE_NORM:
                cudnn_mode = CUDNN_INSTANCE_NORM;
                return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
            case NormMode_t::BATCH_NORM:
                cudnn_mode = CUDNN_BATCH_NORM;
                return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
            case NormMode_t::GROUP_NORM:
                cudnn_mode = CUDNN_GROUP_NORM;
                return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
            #endif

            #ifndef NO_DEFAULT_IN_SWITCH
            default:
                return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
            #endif
        }
        return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
    }

    static inline cudnnStatus_t convert_to_cudnn_type(cudnn_frontend::NormFwdPhase_t const mode, cudnnBackendNormFwdPhase_t& cudnn_mode) {
        switch (mode)
        {
            #if (CUDNN_VERSION >= 8500)
            case NormFwdPhase_t::INFERENCE:
                cudnn_mode = CUDNN_NORM_FWD_INFERENCE;
                return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
            case NormFwdPhase_t::TRAINING:
                cudnn_mode = CUDNN_NORM_FWD_TRAINING;
                return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
            #endif

            #ifndef NO_DEFAULT_IN_SWITCH
            default:
                return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
            #endif
        }
        return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
    }

    // To be deprecated. Only exists as setResampleMode(cudnnPaddingMode_t) requires it.
    static inline void convert_from_cudnn_type(cudnnPaddingMode_t const cudnn_mode, cudnn_frontend::PaddingMode_t& mode) {
        mode = cudnn_frontend::PaddingMode_t::NOT_SET;
        switch (cudnn_mode)
        {
            case CUDNN_EDGE_VAL_PAD:
                mode = cudnn_frontend::PaddingMode_t::EDGE_VAL_PAD;
                break; 
            case CUDNN_NEG_INF_PAD:
                mode = cudnn_frontend::PaddingMode_t::NEG_INF_PAD;
                break; 
            case CUDNN_ZERO_PAD:
                mode = cudnn_frontend::PaddingMode_t::ZERO_PAD;
                break;
    #ifndef NO_DEFAULT_IN_SWITCH
            default:
                break;
    #endif
        }
    }

    // To be deprecated. Only exists as setResampleMode(cudnnResampleMode_t) requires it.
    static inline void convert_from_cudnn_type(cudnnResampleMode_t const cudnn_mode, cudnn_frontend::ResampleMode_t& mode) {
        mode = cudnn_frontend::ResampleMode_t::NOT_SET;
        switch (cudnn_mode)
        {
#if (CUDNN_VERSION >= 8600)
            case CUDNN_RESAMPLE_AVGPOOL_EXCLUDE_PADDING:
                mode = cudnn_frontend::ResampleMode_t::AVGPOOL_EXCLUDE_PADDING;
                break;
            case CUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING:
                mode = cudnn_frontend::ResampleMode_t::AVGPOOL_INCLUDE_PADDING;
                break;
#else
            case CUDNN_RESAMPLE_AVGPOOL:
                mode = cudnn_frontend::ResampleMode_t::AVGPOOL_INCLUDE_PADDING;
                break;
#endif
            case CUDNN_RESAMPLE_BILINEAR:
                mode = cudnn_frontend::ResampleMode_t::BILINEAR;
                break; 
            case CUDNN_RESAMPLE_NEAREST:
                mode = cudnn_frontend::ResampleMode_t::NEAREST;
                break; 
            case CUDNN_RESAMPLE_MAXPOOL:
                mode = cudnn_frontend::ResampleMode_t::MAXPOOL;
                break;
    #ifndef NO_DEFAULT_IN_SWITCH
            default:
                break;
    #endif
        }
    }

    // To be deprecated. Only exists as setNormalizationMode(cudnnBackendNormMode_t) requires it.
    static inline void convert_from_cudnn_type(cudnnBackendNormMode_t const cudnn_mode, cudnn_frontend::NormMode_t& mode) {
        mode = NormMode_t::NOT_SET;
        switch (cudnn_mode)
        {
        #if (CUDNN_VERSION >= 8500)
            case CUDNN_LAYER_NORM:
                mode = NormMode_t::LAYER_NORM;
                break; 
            case CUDNN_INSTANCE_NORM:
                mode = NormMode_t::INSTANCE_NORM;
                break; 
            case CUDNN_BATCH_NORM:
                mode = NormMode_t::BATCH_NORM;
                break; 
            case CUDNN_GROUP_NORM:
                mode = NormMode_t::GROUP_NORM;
                break;
        #endif
    
        #ifndef NO_DEFAULT_IN_SWITCH
            default:
                break;
        #endif
        }
    }

    // To be deprecated. Only exists as setNormFwdPhase(cudnnBackendNormFwdPhase_t) requires it.
    static inline void convert_from_cudnn_type(cudnnBackendNormFwdPhase_t const cudnn_mode, cudnn_frontend::NormFwdPhase_t& mode) {
        mode = NormFwdPhase_t::NOT_SET;
        switch (cudnn_mode)
        {
        #if (CUDNN_VERSION >= 8500)
            case CUDNN_NORM_FWD_INFERENCE:
                mode = NormFwdPhase_t::INFERENCE;
                break; 
            case CUDNN_NORM_FWD_TRAINING:
                mode = NormFwdPhase_t::TRAINING;
                break;
        #endif
    
        #ifndef NO_DEFAULT_IN_SWITCH
            default:
                break;
        #endif
        }
    }

#endif

#if (CUDNN_VERSION >= 8300)
static inline cudnnStatus_t convert_to_cudnn_type(cudnn_frontend::TensorReordering_t const mode, cudnnBackendTensorReordering_t& cudnn_mode) {
        switch (mode)
        {
            case cudnn_frontend::TensorReordering_t::NONE:
                cudnn_mode = CUDNN_TENSOR_REORDERING_NONE;
                return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
            case cudnn_frontend::TensorReordering_t::INT8x32:
                cudnn_mode = CUDNN_TENSOR_REORDERING_INT8x32;
                return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
            case cudnn_frontend::TensorReordering_t::F16x16:
    #if CUDNN_VERSION >= 8800
                cudnn_mode = CUDNN_TENSOR_REORDERING_F16x16;
                return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
    #elif CUDNN_VERSION >= 8700
                cudnn_mode = CUDNN_TENSOR_REORDERING_NONE;
                return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
    #else
                return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
    #endif
    #ifndef NO_DEFAULT_IN_SWITCH
            default:
                return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
    #endif
        }
        return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
    }

    // To be deprecated. Only exists as setReorderType(cudnnBackendTensorReordering_t) requires it.
    static inline void convert_from_cudnn_type(cudnnBackendTensorReordering_t const cudnn_mode, cudnn_frontend::TensorReordering_t& mode) {
        mode = cudnn_frontend::TensorReordering_t::NONE;
        switch (cudnn_mode)
        {
            case CUDNN_TENSOR_REORDERING_INT8x32:
                mode = cudnn_frontend::TensorReordering_t::INT8x32;
                break;
    #if CUDNN_VERSION >= 8800
            case CUDNN_TENSOR_REORDERING_F16x16:
                mode = cudnn_frontend::TensorReordering_t::F16x16;
                break;
    #endif
    #ifndef NO_DEFAULT_IN_SWITCH
            default:
                break;
    #endif
        }
    }

#endif

    // To be deprecated. Only exists as OperationBuilder_v8(::cudnnBackendDescriptorType_t mode) requires it.
    static inline cudnn_frontend::DescriptorType_t convert_from_cudnn_type(cudnnBackendDescriptorType_t const cudnn_mode) {
        switch (cudnn_mode)
        {
            case CUDNN_BACKEND_POINTWISE_DESCRIPTOR:
                return DescriptorType_t::POINTWISE_DESCRIPTOR;
            case CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR:
                return DescriptorType_t::CONVOLUTION_DESCRIPTOR;
            case CUDNN_BACKEND_ENGINE_DESCRIPTOR:
                return DescriptorType_t::ENGINE_DESCRIPTOR;
            case CUDNN_BACKEND_ENGINECFG_DESCRIPTOR:
                return DescriptorType_t::ENGINECFG_DESCRIPTOR;
            case CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR:
                return DescriptorType_t::ENGINEHEUR_DESCRIPTOR;
            case CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR:
                return DescriptorType_t::EXECUTION_PLAN_DESCRIPTOR;
            case CUDNN_BACKEND_INTERMEDIATE_INFO_DESCRIPTOR:
                return DescriptorType_t::INTERMEDIATE_INFO_DESCRIPTOR;
            case CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR:
                return DescriptorType_t::KNOB_CHOICE_DESCRIPTOR;
            case CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR:
                return DescriptorType_t::KNOB_INFO_DESCRIPTOR;
            case CUDNN_BACKEND_LAYOUT_INFO_DESCRIPTOR:
                return DescriptorType_t::LAYOUT_INFO_DESCRIPTOR;
            case CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR:
                return DescriptorType_t::OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR;
            case CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR:
                return DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR;
            case CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR:
                return DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR;
            case CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR:
                return DescriptorType_t::OPERATION_POINTWISE_DESCRIPTOR;
            case CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR:
                return DescriptorType_t::OPERATION_GEN_STATS_DESCRIPTOR;
            case CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR:
                return DescriptorType_t::OPERATIONGRAPH_DESCRIPTOR;
            case CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR:
                return DescriptorType_t::VARIANT_PACK_DESCRIPTOR;
            case CUDNN_BACKEND_TENSOR_DESCRIPTOR:
                return DescriptorType_t::TENSOR_DESCRIPTOR;
            case CUDNN_BACKEND_MATMUL_DESCRIPTOR:
                return DescriptorType_t::MATMUL_DESCRIPTOR;
            case CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR:
                return DescriptorType_t::OPERATION_MATMUL_DESCRIPTOR;
            case CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR:
                return DescriptorType_t::OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR;
            case CUDNN_BACKEND_REDUCTION_DESCRIPTOR:
                return DescriptorType_t::REDUCTION_DESCRIPTOR;
            case CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR:
                return DescriptorType_t::OPERATION_REDUCTION_DESCRIPTOR;

        #if (CUDNN_VERSION >= 8400)
            case CUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR:
                return DescriptorType_t::OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR;
        #endif
        
        #if (CUDNN_VERSION >= 8500)
            case CUDNN_BACKEND_RESAMPLE_DESCRIPTOR:
                return DescriptorType_t::RESAMPLE_DESCRIPTOR;
            case CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR:
                return DescriptorType_t::OPERATION_RESAMPLE_FWD_DESCRIPTOR;
            case CUDNN_BACKEND_OPERATION_CONCAT_DESCRIPTOR:
                return DescriptorType_t::OPERATION_CONCAT_DESCRIPTOR;
            case CUDNN_BACKEND_OPERATION_SIGNAL_DESCRIPTOR:
                return DescriptorType_t::OPERATION_SIGNAL_DESCRIPTOR;
            case CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR:
                return DescriptorType_t::OPERATION_NORM_FORWARD_DESCRIPTOR;
            case CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR:
                return DescriptorType_t::OPERATION_NORM_BACKWARD_DESCRIPTOR;
        #endif

        #if (CUDNN_VERSION >= 8600)
            case CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR:
                return DescriptorType_t::OPERATION_RESAMPLE_BWD_DESCRIPTOR;
        #endif

        #if (CUDNN_VERSION >= 8700)
            case CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR:
                return DescriptorType_t::OPERATION_RESHAPE_DESCRIPTOR;
            case CUDNN_BACKEND_RNG_DESCRIPTOR:
                return DescriptorType_t::RNG_DESCRIPTOR;        
            case CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR:
                return DescriptorType_t::OPERATION_RNG_DESCRIPTOR;
        #endif
        
    #ifndef NO_DEFAULT_IN_SWITCH
            default:
                return DescriptorType_t::NOT_SET;
                break;
    #endif
        }
        return DescriptorType_t::NOT_SET;
    }

} // namespace detail

}

