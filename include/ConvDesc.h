#pragma once

#include <array>
#include <memory>
#include <sstream>
#include <algorithm>
#include <functional>
#include <utility>

#include <cudnn.h>
#include <cudnn_backend.h>

#include "cudnn_backend_wrap_utils.h"

namespace cudnn_api_wrapper {

///
/// Convolution Descriptor Class
/// This class tells the properties of the Convolution operation
/// Properties:
///    - padLower
///    - padUpper
///    - Dilation
///    - Stride
///    - Math Operation Data Type
///    - Convolution Mode
///    - Convolution spatial dimensions
///
/// Use ConvDescBuilder to build this class.
/// Describe returns a string describing the convolution operation
///
class ConvDesc : public BackendDescriptor {
   public:
    friend class ConvDescBuilder;
    std::string
    describe() const override {
        std::stringstream ss;
        char sep = ' ';
        ss << "CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR :"
           << " Datatype: " << std::to_string(data_type) << " Mode: " << std::to_string(mode)
           << " Num Dimensions: " << std::to_string(nDims);
        ss << " PadLower [";
        std::for_each(std::begin(padLower), std::end(padLower), [&ss, sep](int x) mutable {
            ss << sep << x;
            sep = ',';
        });
        ss << " ] PadUpper [";
        std::for_each(std::begin(padUpper), std::end(padUpper), [&ss, sep](int x) mutable {
            ss << sep << x;
            sep = ',';
        });
        ss << " ] Dilation [";
        std::for_each(std::begin(dilation), std::end(dilation), [&ss, sep](int x) mutable {
            ss << sep << x;
            sep = ',';
        });
        ss << " ] Stride [";
        std::for_each(std::begin(stride), std::end(stride), [&ss, sep](int x) mutable {
            ss << sep << x;
            sep = ',';
        });
        ss << "]";
        return ss.str();
    }

    ConvDesc(ConvDesc &&from)
        : BackendDescriptor(from.desc, from.get_status(), from.get_error()), data_type(from.data_type), mode(from.mode), nDims(from.nDims) {
        std::copy(std::begin(from.padLower), std::end(from.padLower), padLower);
        std::copy(std::begin(from.padUpper), std::end(from.padUpper), padUpper);
        std::copy(std::begin(from.dilation), std::end(from.dilation), dilation);
        std::copy(std::begin(from.stride), std::end(from.stride), stride);
        from.desc = nullptr;
    }

    ~ConvDesc() {
        if (desc != nullptr) {
            cudnnBackendDestroyDescriptor(desc);
        }
    }

   private:
    ConvDesc()                 = default;
    ConvDesc(ConvDesc const &) = delete;
    ConvDesc &
    operator=(ConvDesc const &) = delete;

    cudnnDataType_t data_type           = CUDNN_DATA_FLOAT;
    cudnnConvolutionMode_t mode         = CUDNN_CONVOLUTION;
    int64_t nDims                       = -1;
    int64_t padLower[CUDNN_DIM_MAX + 1] = {0};   // n, g, c, d, h, w
    int64_t padUpper[CUDNN_DIM_MAX + 1] = {0};   // n, g, c, d, h, w
    int64_t dilation[CUDNN_DIM_MAX + 1] = {0};   // n, g, c, d, h, w
    int64_t stride[CUDNN_DIM_MAX + 1]   = {-1};  // n, g, c, d, h, w
};

///
/// ConvDescBuilder Class
/// Helper class used to build ConvDesc class
class ConvDescBuilder {
   public:
    /** @defgroup ConvDescBuilder
     *  Set individual property of ConvDesc class
     *  @{
     */
    //! Set Datatype for the Convolution Operation
    auto
    setDataType(cudnnDataType_t data_type_) -> ConvDescBuilder & {
        m_convDesc.data_type = data_type_;
        return *this;
    }
    //! Set Padding Lower of the convDesc
    auto
    setPrePadding(int64_t ndims, int64_t *padding) -> ConvDescBuilder & {
        std::copy(padding, padding + ndims, m_convDesc.padLower);
        return *this;
    }
    //! Set Padding Upper of the convDesc
    auto
    setPostPadding(int64_t ndims, int64_t *padding) -> ConvDescBuilder & {
        std::copy(padding, padding + ndims, m_convDesc.padUpper);
        return *this;
    }
    //! Set Dilation of the convDesc
    auto
    setDilation(int64_t ndims, int64_t *dilation) -> ConvDescBuilder & {
        std::copy(dilation, dilation + ndims, m_convDesc.dilation);
        return *this;
    }
    //! Set Strides of the convDesc
    auto
    setStrides(int64_t ndims, int64_t *strides) -> ConvDescBuilder & {
        std::copy(strides, strides + ndims, m_convDesc.stride);
        return *this;
    }
    //! Set Num Spatial Dimensions of the convolution Operation
    auto
    setNDims(int64_t nDims_) -> ConvDescBuilder & {
        m_convDesc.nDims = nDims_;
        return *this;
    }
    //! Set Convolution Mode of the convolution Operation
    auto
    setMathMode(cudnnConvolutionMode_t mode_) -> ConvDescBuilder & {
        m_convDesc.mode = mode_;
        return *this;
    }
    /** @} */

    //! constructs the ConvDesc by calling the cudnn API
    //! Throws the appropriate error message
    ConvDesc &&
    build() {
        // Sanity check if non-default fields have been set correctly.
        if (m_convDesc.nDims <= 0) {
            set_error_and_throw_exception(&m_convDesc, CUDNN_STATUS_BAD_PARAM, "CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR: Check and Set the CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS field");
            return std::move(m_convDesc);
        };
        if (m_convDesc.stride[0] <= 0) {
            set_error_and_throw_exception(&m_convDesc, CUDNN_STATUS_BAD_PARAM, "CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR: Check and Set the CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES field");
            return std::move(m_convDesc);
        }
        if (m_convDesc.desc != nullptr) {
            set_error_and_throw_exception(&m_convDesc, CUDNN_STATUS_BAD_PARAM, "CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR: Bad descriptor created");
            return std::move(m_convDesc);
        }

        // Create a descriptor. Memory allocation happens here.
        auto status = CUDNN_STATUS_SUCCESS;
        status      = cudnnBackendCreateDescriptor(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR, &m_convDesc.desc);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_convDesc, status, "CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR: cudnnCreate Failed");
            return std::move(m_convDesc);
        }

        // Once Created lets set the descriptor parameters.
        status = cudnnBackendSetAttribute(
            m_convDesc.desc, CUDNN_ATTR_CONVOLUTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &m_convDesc.data_type);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_convDesc, status, "CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR: SetAttribute CUDNN_ATTR_CONVOLUTION_COMP_TYPE Failed");
            return std::move(m_convDesc);
        }

        status = cudnnBackendSetAttribute(
            m_convDesc.desc, CUDNN_ATTR_CONVOLUTION_MODE, CUDNN_TYPE_CONVOLUTION_MODE, 1, &m_convDesc.mode);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_convDesc, status, "CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR: SetAttribute CUDNN_ATTR_CONVOLUTION_MODE Failed");
            return std::move(m_convDesc);
        }

        status = cudnnBackendSetAttribute(
            m_convDesc.desc, CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS, CUDNN_TYPE_INT64, 1, &m_convDesc.nDims);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_convDesc, status, "CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR: SetAttribute CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS Failed");
            return std::move(m_convDesc);
        }

        status = cudnnBackendSetAttribute(m_convDesc.desc,
                                          CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS,
                                          CUDNN_TYPE_INT64,
                                          m_convDesc.nDims,
                                          m_convDesc.padLower);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_convDesc, status, "CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR: SetAttribute CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS Failed");
            return std::move(m_convDesc);
        }

        status = cudnnBackendSetAttribute(m_convDesc.desc,
                                          CUDNN_ATTR_CONVOLUTION_POST_PADDINGS,
                                          CUDNN_TYPE_INT64,
                                          m_convDesc.nDims,
                                          m_convDesc.padUpper);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_convDesc, status, "CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR: SetAttribute CUDNN_ATTR_CONVOLUTION_POST_PADDINGS Failed");
            return std::move(m_convDesc);
        }

        status = cudnnBackendSetAttribute(m_convDesc.desc,
                                          CUDNN_ATTR_CONVOLUTION_DILATIONS,
                                          CUDNN_TYPE_INT64,
                                          m_convDesc.nDims,
                                          m_convDesc.dilation);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_convDesc, status, "CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR: SetAttribute CUDNN_ATTR_CONVOLUTION_DILATIONS Failed");
            return std::move(m_convDesc);
        }

        status = cudnnBackendSetAttribute(m_convDesc.desc,
                                          CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES,
                                          CUDNN_TYPE_INT64,
                                          m_convDesc.nDims,
                                          m_convDesc.stride);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_convDesc, status, "CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR: SetAttribute CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES Failed");
            return std::move(m_convDesc);
        }

        // Finalizing the descriptor
        status = cudnnBackendFinalize(m_convDesc.desc);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_convDesc, status, "CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR: cudnnFinalize Failed");
            return std::move(m_convDesc);
        }

        return std::move(m_convDesc);
    }

    explicit ConvDescBuilder()               = default;
    ~ConvDescBuilder()                       = default;
    ConvDescBuilder(ConvDescBuilder &&)      = delete;
    ConvDescBuilder(ConvDescBuilder const &) = delete;
    ConvDescBuilder &
    operator=(ConvDescBuilder const &) = delete;

   private:
    ConvDesc m_convDesc;
};
}
