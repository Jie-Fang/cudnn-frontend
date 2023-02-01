#pragma once

#include <unordered_map>
#include <vector>

#include<bits/stdc++.h>
#include<algorithm>
#include <string>


namespace cudnn_frontend {

enum class cudnn_frontend_error_t {
    OK = 0,
    TENSOR_DIMENSIONS_NOT_SET,
    POINTWISE_MODE_NOT_SET,
    SHAPE_DEDUCTION_FAILED,
    OUTPUT_TENSOR_NODE_NOT_FOUND,
    UNKNOWN_TENSOR_NAME,
    INPUT_PORT_COUNT_MISMATCH,
    GRAPH_PARTITION_EXECUTION_PLAN_CREATION_FAILED
};

static cudnnDataType_t
string_to_data_type(std::string data_type) {
    if (data_type == "float") {
        return CUDNN_DATA_FLOAT;
    }
    else if (data_type == "double") {
        return CUDNN_DATA_DOUBLE;
    }
    else if (data_type == "half") {
        return CUDNN_DATA_HALF;
    }
    
    // default :
    return CUDNN_DATA_FLOAT;
}

static cudnnPointwiseMode_t
string_to_pointwise_mode(std::string mode) {

    std::transform(std::begin(mode), std::end(mode), std::begin(mode), [](auto c) { return std::toupper(c); });

    if (mode.find("ADD") != std::string::npos) {return CUDNN_POINTWISE_ADD;}
    if (mode.find("MUL")    != std::string::npos) {return CUDNN_POINTWISE_MUL;}
    if (mode.find("SQRT")   != std::string::npos) {return CUDNN_POINTWISE_SQRT;}
    if (mode.find("MAX")    != std::string::npos) {return CUDNN_POINTWISE_MAX;}
    if (mode.find("MIN")    != std::string::npos) {return CUDNN_POINTWISE_MIN;}
    if (mode.find("RELU_FWD") != std::string::npos) {return CUDNN_POINTWISE_RELU_FWD;}
    if (mode.find("TANH_FWD") != std::string::npos) {return CUDNN_POINTWISE_TANH_FWD;}
    if (mode.find("SIGMOID_FWD") != std::string::npos) {return CUDNN_POINTWISE_SIGMOID_FWD;}
    if (mode.find("ELU_FWD") != std::string::npos) {return CUDNN_POINTWISE_ELU_FWD;}
    if (mode.find("GELU_FWD") != std::string::npos) {return CUDNN_POINTWISE_GELU_FWD;}
    if (mode.find("SOFTPLUS_FWD") != std::string::npos) {return CUDNN_POINTWISE_SOFTPLUS_FWD;}
    if (mode.find("SWISH_FWD") != std::string::npos) {return CUDNN_POINTWISE_SWISH_FWD;}
    if (mode.find("RELU_BWD") != std::string::npos) {return CUDNN_POINTWISE_RELU_BWD;}
    if (mode.find("TANH_BWD") != std::string::npos) {return CUDNN_POINTWISE_TANH_BWD;}
    if (mode.find("SIGMOID_BWD") != std::string::npos) {return CUDNN_POINTWISE_SIGMOID_BWD;}
    if (mode.find("ELU_BWD") != std::string::npos) {return CUDNN_POINTWISE_ELU_BWD;}
    if (mode.find("GELU_BWD") != std::string::npos) {return CUDNN_POINTWISE_GELU_BWD;}
    if (mode.find("SOFTPLUS_BWD") != std::string::npos) {return CUDNN_POINTWISE_SOFTPLUS_BWD;}
    if (mode.find("SWISH_BWD") != std::string::npos) {return CUDNN_POINTWISE_SWISH_BWD;}

#if (CUDNN_VERSION >= 8500)
    if (mode.find("ERF")    != std::string::npos) {return CUDNN_POINTWISE_ERF;}
    if (mode.find("IDENTITY") != std::string::npos) {return CUDNN_POINTWISE_IDENTITY;}
    if (mode.find("GELU_APPROX_TANH_BWD") != std::string::npos) {return CUDNN_POINTWISE_GELU_APPROX_TANH_BWD;}
    if (mode.find("GELU_APPROX_TANH_FWD") != std::string::npos) {return CUDNN_POINTWISE_GELU_APPROX_TANH_FWD;}
#endif
#if (CUDNN_VERSION >= 8400)
    if (mode.find("GEN_INDEX") != std::string::npos) {return CUDNN_POINTWISE_GEN_INDEX;}
    if (mode.find("BINARY_SELECT") != std::string::npos) {return CUDNN_POINTWISE_BINARY_SELECT;}
#endif
#if (CUDNN_VERSION >= 8300)
    if (mode.find("EXP")    != std::string::npos) {return CUDNN_POINTWISE_EXP;}
    if (mode.find("LOG")    != std::string::npos) {return CUDNN_POINTWISE_LOG;}
    if (mode.find("NEG")    != std::string::npos) {return CUDNN_POINTWISE_NEG;}
    if (mode.find("MOD")    != std::string::npos) {return CUDNN_POINTWISE_MOD;}
    if (mode.find("POW")    != std::string::npos) {return CUDNN_POINTWISE_POW;}
    if (mode.find("ABS")    != std::string::npos) {return CUDNN_POINTWISE_ABS;}
    if (mode.find("CEIL")   != std::string::npos) {return CUDNN_POINTWISE_CEIL;}
    if (mode.find("COS")    != std::string::npos) {return CUDNN_POINTWISE_COS;}
    if (mode.find("FLOOR")  != std::string::npos) {return CUDNN_POINTWISE_FLOOR;}
    if (mode.find("RSQRT")  != std::string::npos) {return CUDNN_POINTWISE_RSQRT;}
    if (mode.find("SIN")    != std::string::npos) {return CUDNN_POINTWISE_SIN;}
    if (mode.find("LOGICAL_NOT") != std::string::npos) {return CUDNN_POINTWISE_LOGICAL_NOT;}
    if (mode.find("TAN")    != std::string::npos) {return CUDNN_POINTWISE_TAN;}
    if (mode.find("SUB")    != std::string::npos) {return CUDNN_POINTWISE_SUB;}
    if (mode.find("ADD_SQUARE") != std::string::npos) {return CUDNN_POINTWISE_ADD_SQUARE;}
    if (mode.find("DIV")    != std::string::npos) {return CUDNN_POINTWISE_DIV;}
    if (mode.find("CMP_EQ") != std::string::npos) {return CUDNN_POINTWISE_CMP_EQ;}
    if (mode.find("CMP_NEQ") != std::string::npos) {return CUDNN_POINTWISE_CMP_NEQ;}
    if (mode.find("CMP_GT") != std::string::npos) {return CUDNN_POINTWISE_CMP_GT;}
    if (mode.find("CMP_GE") != std::string::npos) {return CUDNN_POINTWISE_CMP_GE;}
    if (mode.find("CMP_LT") != std::string::npos) {return CUDNN_POINTWISE_CMP_LT;}
    if (mode.find("CMP_LE") != std::string::npos) {return CUDNN_POINTWISE_CMP_LE;}
    if (mode.find("LOGICAL_AND") != std::string::npos) {return CUDNN_POINTWISE_LOGICAL_AND;}
    if (mode.find("LOGICAL_OR") != std::string::npos) {return CUDNN_POINTWISE_LOGICAL_OR;}
#endif
    return CUDNN_POINTWISE_ADD;
}

// TODO: add full data type support
static bool
is_valid_type(std::string data_type) {
    auto is_valid = 
        data_type == "float"  ||
        data_type == "double" ||
        data_type == "half";
    
    return is_valid;
}

static bool
allowAllConfig(cudnnBackendDescriptor_t engine_config) {
    (void)engine_config;
    return false;
}

} // namespace cudnn_frontend