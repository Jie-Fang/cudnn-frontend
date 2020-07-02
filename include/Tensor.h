#pragma once

#include <array>
#include <memory>
#include <sstream>
#include <algorithm>
#include <functional>
#include <utility>

#include "cudnn_frontend_utils.h"

namespace cudnn_frontend {

///
/// Tensor Class
/// This class tells the properties of the Tensor on which the operation will be performed
/// Properties:
///    - dataType
///    - alignment
///    - unique identifier
///    - tensor dimensions
///    - tensor strides
///
/// Use TensorBuilder to build this class.
/// Describe returns a string describing the tensor class
///
class Tensor : public BackendDescriptor {
   public:
    friend class TensorBuilder;
    std::string
    describe() const override {
        std::stringstream ss;
        char sep = ' ';
        ss << "CUDNN_BACKEND_TENSOR_DESCRIPTOR :"
           << " Datatype: " << std::to_string(data_type) << " Id: " << std::to_string(id)
           << " Alignment: " << std::to_string(alignment) << " nDims " << nDims;
        ss << " Dim [";
        std::for_each(std::begin(btensor_dimA), std::end(btensor_dimA), [&ss, sep](int x) mutable {
            ss << sep << x;
            sep = ',';
        });
        ss << " ] Str [";
        std::for_each(std::begin(btensor_strA), std::end(btensor_strA), [&ss, sep](int x) mutable {
            ss << sep << x;
            sep = ',';
        });
        ss << "]";
        return ss.str();
    }

    Tensor(Tensor &&from)
        : BackendDescriptor(from.desc, from.get_status(), from.get_error()),
          data_type(from.data_type),
          id(from.id),
          alignment(from.alignment),
          nDims(from.nDims) {
        std::copy(std::begin(from.btensor_dimA), std::end(from.btensor_dimA), btensor_dimA);
        std::copy(std::begin(from.btensor_strA), std::end(from.btensor_strA), btensor_strA);
        from.desc = nullptr;
    }

    ~Tensor() {
        if (desc != nullptr) {
            cudnnBackendDestroyDescriptor(desc);
        }
    }

   private:
    Tensor()               = default;
    Tensor(Tensor const &) = delete;
    Tensor &
    operator=(Tensor const &) = delete;

    cudnnDataType_t data_type               = CUDNN_DATA_FLOAT;
    int64_t btensor_dimA[CUDNN_DIM_MAX + 1] = {-1};  // n, g, c, d, h, w
    int64_t btensor_strA[CUDNN_DIM_MAX + 1] = {-1};  // n, g, c, d, h, w
    int64_t id                              = -1;
    int64_t alignment                       = -1;
    int64_t nDims                           = -1;
};

///
/// TensorBuilder Class
/// Helper class used to build Tensor class
class TensorBuilder {
   public:
    /** @defgroup TensorBuilder
     *  Set individual property of Tensor class
     *  @{
     */
    //! Set Datatype for the Tensor
    auto
    setDataType(cudnnDataType_t data_type_) -> TensorBuilder & {
        m_tensor.data_type = data_type_;
        return *this;
    }
    //! Set Dimensions of the tensor
    auto
    setDim(int64_t ndim, int64_t *dim) -> TensorBuilder & {
        std::copy((dim), dim + ndim, m_tensor.btensor_dimA);
        m_tensor.nDims = ndim;
        return *this;
    }
    //! Set Strides of the tensor
    auto
    setStrides(int64_t ndim, int64_t *strides) -> TensorBuilder & {
        std::copy(strides, strides + ndim, m_tensor.btensor_strA);
        return *this;
    }
    //! Set Unique Id  of the tensor
    auto
    setId(int64_t id_) -> TensorBuilder & {
        m_tensor.id = id_;
        return *this;
    }
    //! Set Alignment of the tensor
    auto
    setAlignment(int64_t alignment_) -> TensorBuilder & {
        m_tensor.alignment = alignment_;
        return *this;
    }
    /** @} */

    //! constructs the Tensor by calling the cudnn API
    //! Throws the appropriate error message
    Tensor &&
    build() {
        // Sanity check if non-default fields have been set correctly.
        if(m_tensor.alignment <= 0) {
            set_error_and_throw_exception(&m_tensor, CUDNN_STATUS_BAD_PARAM, "CUDNN_BACKEND_TENSOR_DESCRIPTOR: Check and Set the CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT field");
            return std::move(m_tensor);
        }
        if(m_tensor.id <= 0) {
            set_error_and_throw_exception(&m_tensor, CUDNN_STATUS_BAD_PARAM, "CUDNN_BACKEND_TENSOR_DESCRIPTOR: Check and Set the CUDNN_ATTR_TENSOR_UNIQUE_ID as a valid value");
            return std::move(m_tensor);
        }
        if(m_tensor.btensor_strA[0] <= 0) {
            set_error_and_throw_exception(&m_tensor, CUDNN_STATUS_BAD_PARAM, "CUDNN_BACKEND_TENSOR_DESCRIPTOR: Check and Set the CUDNN_ATTR_TENSOR_STRIDES Correctly");
            return std::move(m_tensor);
        }
        if(m_tensor.btensor_dimA[0] <= 0) {
            set_error_and_throw_exception(&m_tensor, CUDNN_STATUS_BAD_PARAM, "CUDNN_BACKEND_TENSOR_DESCRIPTOR: Check and Set the CUDNN_ATTR_TENSOR_DIMENSIONS Correctly");
            return std::move(m_tensor);
        }
        if(m_tensor.desc != nullptr) {
            set_error_and_throw_exception(&m_tensor, CUDNN_STATUS_BAD_PARAM, "CUDNN_BACKEND_TENSOR_DESCRIPTOR: Bad tensor created. The tensor already seems to be pointing to something");
            return std::move(m_tensor);
        }
    
        // Create a descriptor. Memory allocation happens here.
        auto status = CUDNN_STATUS_SUCCESS;
        status      = cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &m_tensor.desc);

        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_tensor, status, "CUDNN_BACKEND_TENSOR_DESCRIPTOR: cudnnCreate Descriptor Failed");
            return std::move(m_tensor);
        }

        // Once Created lets set the descriptor parameters.
        status = cudnnBackendSetAttribute(
            m_tensor.desc, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &m_tensor.data_type);
        if(status != CUDNN_STATUS_SUCCESS){
            set_error_and_throw_exception(&m_tensor, status, "CUDNN_BACKEND_TENSOR_DESCRIPTOR: SetAttribute CUDNN_ATTR_TENSOR_DATA_TYPE Failed");
            return std::move(m_tensor);
        }
        status = cudnnBackendSetAttribute(
            m_tensor.desc, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, m_tensor.nDims, m_tensor.btensor_dimA);
        if(status != CUDNN_STATUS_SUCCESS){
            set_error_and_throw_exception(&m_tensor, status, "CUDNN_BACKEND_TENSOR_DESCRIPTOR: SetAttribute CUDNN_ATTR_TENSOR_DIMENSIONS Failed");
            return std::move(m_tensor);
        }
        status = cudnnBackendSetAttribute(
            m_tensor.desc, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, m_tensor.nDims, m_tensor.btensor_strA);
        if(status != CUDNN_STATUS_SUCCESS){
            set_error_and_throw_exception(&m_tensor, status, "CUDNN_BACKEND_TENSOR_DESCRIPTOR: SetAttribute CUDNN_ATTR_TENSOR_STRIDES Failed");
            return std::move(m_tensor);
        }
        status =
            cudnnBackendSetAttribute(m_tensor.desc, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &m_tensor.id);
        if(status != CUDNN_STATUS_SUCCESS){
            set_error_and_throw_exception(&m_tensor, status, "CUDNN_BACKEND_TENSOR_DESCRIPTOR: SetAttribute CUDNN_ATTR_TENSOR_UNIQUE_ID Failed");
            return std::move(m_tensor);
        }
        cudnnBackendSetAttribute(
            m_tensor.desc, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &m_tensor.alignment);
        if(status != CUDNN_STATUS_SUCCESS){
            set_error_and_throw_exception(&m_tensor, status, "CUDNN_BACKEND_TENSOR_DESCRIPTOR: SetAttribute CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT Failed");
            return std::move(m_tensor);
        }

        // Finalizing the descriptor
        status = cudnnBackendFinalize(m_tensor.desc);
        if(status != CUDNN_STATUS_SUCCESS){
            set_error_and_throw_exception(&m_tensor, status, "CUDNN_BACKEND_TENSOR_DESCRIPTOR cudnnFinalize failed");
            return std::move(m_tensor);
        }

        return std::move(m_tensor);
    }

    explicit TensorBuilder()             = default;
    ~TensorBuilder()                     = default;
    TensorBuilder(TensorBuilder &&)      = delete;
    TensorBuilder(TensorBuilder const &) = delete;
    TensorBuilder &
    operator=(TensorBuilder const &) = delete;

   private:
    Tensor m_tensor;
};
}
