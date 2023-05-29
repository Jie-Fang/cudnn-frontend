#pragma once

#include <unordered_map>
#include <vector>

#include<bits/stdc++.h>
#include<algorithm>
#include <string>


namespace cudnn_frontend {

using op_graph_to_engine_configs = std::unordered_map<std::string, EngineConfigList>;

enum class [[nodiscard]] error_t {
    OK
    , ATTRIBUTE_NOT_SET
    , SHAPE_DEDUCTION_FAILED
    , INVALID_TENSOR_NAME
    , INVALID_VARIANT_PACK
    , GRAPH_EXECUTION_PLAN_CREATION_FAILED
    , GRAPH_EXECUTION_FAILED
    , HEURISTIC_QUERY_FAILED
    , UNSUPPORTED_GRAPH_FORMAT
    , INVALID_CUDA_DEVICE
};

#define CHECK_CUDNN_FRONTEND_ERROR(x) do { \
  error_t retval = (x); \
  if (retval != error_t::OK) { \
    getLogger() << "[cudnn_frontend] ERROR: " << #x << " returned " << retval << " at " << __FILE__ << ":" <<  __LINE__; \
    return retval; \
  } \
} while (0)

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
        case error_t::HEURISTIC_QUERY_FAILED:
            os << "HEURISTIC_QUERY_FAILED";
            break;
        case error_t::INVALID_CUDA_DEVICE:
            os << "INVALID_CUDA_DEVICE";
            break;
        case error_t::UNSUPPORTED_GRAPH_FORMAT:
            os << "UNSUPPORTED_GRAPH_FORMAT";
            break;
    }
    return os;
}

static bool
allowAllConfig(cudnnBackendDescriptor_t engine_config) {
    (void)engine_config;
    return false;
}

namespace detail {

    class Context {
        DataType_t compute_data_type = DataType_t::NOT_SET;
        DataType_t intermediate_data_type = DataType_t::NOT_SET;
        DataType_t io_data_type       = DataType_t::NOT_SET;

    public:
        Context& set_intermediate_data_type(DataType_t const type) {
            intermediate_data_type = type;
            return *this;
        }

        Context& set_io_data_type(DataType_t const type) {
            io_data_type = type;
            return *this;
        }

        Context& set_compute_data_type(DataType_t const type) {
            compute_data_type = type;
            return *this;
        }

        DataType_t get_io_data_type() const {
            return io_data_type;
        }

        DataType_t get_intermediate_data_type() const {
            return intermediate_data_type;
        }

        DataType_t get_compute_data_type() const {
            return compute_data_type;
        }

        Context& fill_missing_properties(Context const& global_context) {
            if(get_compute_data_type() == DataType_t::NOT_SET) {
                set_compute_data_type(global_context.get_compute_data_type());
            }
            if(get_intermediate_data_type() == DataType_t::NOT_SET) {
                set_intermediate_data_type(global_context.get_intermediate_data_type());
            }
            if(get_io_data_type() == DataType_t::NOT_SET) {
                set_io_data_type(global_context.get_io_data_type());
            }
            return *this;
        }
    };

    static inline std::ostream& operator<<(std::ostream& os, const Context& context) {
        os << "compute_data_type: " << context.get_compute_data_type() << ", intermediate_data_type: " << context.get_intermediate_data_type() << ", io_data_type: " << context.get_io_data_type() << std::endl;
        return os;
    }

} // namespace detail

} // namespace cudnn_frontend