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
    cudnnDataType_t compute_type           = CUDNN_DATA_FLOAT;
    cudnnDataType_t intermediate_data_type = CUDNN_DATA_FLOAT;
    cudnnDataType_t tensor_data_type       = CUDNN_DATA_HALF;
    Layout layout                          = Layout::ChannelLast;

public:    

    cuDNNFEContext() {}

    int
    set_intermediate_data_type(std::string type) {
        auto ret_val = is_valid_type(type);
        intermediate_data_type = string_to_data_type(type);
        return ret_val;
    }

    int
    set_tensor_data_type(std::string type) {
        auto ret_val = is_valid_type(type);
        tensor_data_type = string_to_data_type(type);
        return ret_val;
    }

    int
    set_compute_type(std::string type) {
        auto ret_val = is_valid_type(type);
        compute_type = string_to_data_type(type);
        return ret_val;
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

    std::string
    get_tensor_data_type_string() const {
        return to_string(tensor_data_type);
    }

    std::string
    get_intermediate_data_type_string() const {
        return to_string(intermediate_data_type);
    }

    std::string
    get_compute_type_string() const {
        return to_string(compute_type);
    }

    cudnnDataType_t
    get_tensor_data_type() const {
        return (tensor_data_type);
    }

    cudnnDataType_t
    get_intermediate_data_type() const {
        return (intermediate_data_type);
    }

    cudnnDataType_t
    get_compute_type() const {
        return (compute_type);
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

    std::string
    describe() const {
        return 
        "cuDNNFEContext Tensor Dims: "  + std::to_string(tensor_dims)
        + " Spatial Dims: "  + std::to_string(spatial_dims)
        + " Layout: " + get_layout_string()
        + " Compute precision: " + get_compute_type_string()
        + " Tensor type: " + get_tensor_data_type_string()
        + " Intermediate type: " + get_intermediate_data_type_string();
    }
};

}