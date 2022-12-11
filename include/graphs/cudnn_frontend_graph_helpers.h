#pragma once

#include <unordered_map>
#include <vector>

#include "cudnn_frontend_nodes.h"

namespace cudnn_frontend {

enum class cudnn_frontend_error_t {
    OK,
    TENSOR_DIMENSIONS_NOT_SET,
    POINTWISE_MODE_NOT_SET,
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