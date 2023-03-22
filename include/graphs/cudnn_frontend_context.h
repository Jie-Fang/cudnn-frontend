#pragma once

#include <cudnn.h>

#include "cudnn_frontend_utils.h"
#include "cudnn_frontend_graph_helpers.h"

namespace cudnn_frontend {

class cuDNNFEContext {

public:
    enum class Layout {
        ChannelFirst, // NCHW
        ChannelLast  // NHWC
    };

protected:
    int64_t tensor_dims  = 4;
    int64_t spatial_dims = 2;
    DataType_t compute_type           = DataType_t::FLOAT;
    DataType_t intermediate_data_type = DataType_t::FLOAT;
    DataType_t tensor_data_type       = DataType_t::HALF;
    Layout layout                          = Layout::ChannelLast;

public:    

    cuDNNFEContext() {}

    error_t
    set_intermediate_data_type(DataType_t type) {
        intermediate_data_type = type;
        return error_t::OK;
    }

    error_t
    set_tensor_data_type(DataType_t type) {
        tensor_data_type = type;
        return error_t::OK;
    }

    error_t
    set_compute_type(DataType_t type) {
        compute_type = type;
        return error_t::OK;
    }
    
    int
    set_tensor_dims(int64_t x) {
        tensor_dims = x;
        return true;
    }

    int
    set_spatial_dims(int64_t x) {
        spatial_dims = x;
        return true;
    }

    int
    set_layout(std::string layout_) {
        if (layout_ == "ChannelFirst" || layout_ == "NCHW") {
            layout = Layout::ChannelFirst;
            return true;
        }
        else if (layout_ == "ChannelLast" || layout_ == "NHWC") {
            layout = Layout::ChannelLast;
            return true;
        }

        return false;
    }

    DataType_t
    get_tensor_data_type() const {
        return tensor_data_type;
    }

    DataType_t
    get_intermediate_data_type() const {
        return intermediate_data_type;
    }

    DataType_t
    get_compute_type() const {
        return compute_type;
    }
        
    int64_t
    get_tensor_dims() {
        return tensor_dims;
    }

    int64_t
    get_spatial_dims() {
        return spatial_dims;
    }

    std::string
    get_layout_string() const {
        switch (layout) {
            case Layout::ChannelFirst:
                return "ChannelFirst";
            case Layout::ChannelLast:
                return "ChannelLast";
        }
        return "ChannelLast";
    }
    
    Layout
    get_layout() const {
        return layout;
    }
};

}