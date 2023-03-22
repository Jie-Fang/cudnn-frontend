#pragma once

#include <unordered_map>
#include <vector>

#include<bits/stdc++.h>
#include<algorithm>
#include <string>


namespace cudnn_frontend {

enum class error_t {
    OK
    , ATTRIBUTE_NOT_SET
    , SHAPE_DEDUCTION_FAILED
    , INVALID_TENSOR_NAME
    , INVALID_VARIANT_PACK
    , GRAPH_EXECUTION_PLAN_CREATION_FAILED
    , GRAPH_EXECUTION_FAILED
};

static inline std::ostream& operator<<(std::ostream& os, const error_t& mode) {
    switch (mode)
    {
        case error_t::OK:
            os << "OK";
            break;
        case error_t::ATTRIBUTE_NOT_SET:
            os << "ATTRIBUTE_NOT_SET";
            break;
        case error_t::SHAPE_DEDUCTION_FAILED:
            os << "SHAPE_DEDUCTION_FAILED";
            break;
        case error_t::INVALID_TENSOR_NAME:
            os << "INVALID_TENSOR_NAME";
            break;
        case error_t::INVALID_VARIANT_PACK:
            os << "INVALID_VARIANT_PACK";
            break;
        case error_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED:
            os << "GRAPH_EXECUTION_PLAN_CREATION_FAILED";
            break;
        case error_t::GRAPH_EXECUTION_FAILED:
            os << "GRAPH_EXECUTION_FAILED";
            break;
    }
    return os;
}

static bool
allowAllConfig(cudnnBackendDescriptor_t engine_config) {
    (void)engine_config;
    return false;
}

} // namespace cudnn_frontend