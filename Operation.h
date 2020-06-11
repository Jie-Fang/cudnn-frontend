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
#include "Tensor.h"
#include "ConvDesc.h"

namespace cudnn_api_wrapper {

///
/// Operation Class
/// This class has the properties of the operation
/// Properties:
///    - xDesc
///    - yDesc
///    - wdesc
///    - cdesc
///    - alpha
///    - beta
///
/// Use OperationBuilder to build this class.
/// Describe returns a string describing the convolution operation
///
class Operation : public BackendDescriptor {
   public:
    friend class OperationBuilder;
    std::string
    describe() const override {
        std::stringstream ss;
        ss << "CUDNN_BACKEND_OPERATION :"
           << " OpMode: " << std::to_string(op_mode);
        ss << std::hex << " X " << xdesc;
        ss << std::hex << " Y " << wdesc;
        ss << std::hex << " W " << ydesc;
        ss << std::hex << " C " << cdesc;
        ss << std::dec << " alphabetaType " << alphabetaType;
        ss << " Alpha: " << alpha_s << " " << alpha_d;
        ss << " Beta: " << beta_s << " " << beta_d;
        return ss.str();
    }

    Operation(Operation &&from)
        : BackendDescriptor(from.desc),
          op_mode(from.op_mode),
          xdesc(from.xdesc),
          ydesc(from.ydesc),
          wdesc(from.wdesc),
          cdesc(from.cdesc),
          alphabetaType(from.alphabetaType),
          alpha_s(from.alpha_s),
          alpha_d(from.alpha_d),
          beta_s(from.beta_s),
          beta_d(from.beta_d) {
        from.xdesc = nullptr;
        from.ydesc = nullptr;
        from.wdesc = nullptr;
        from.cdesc = nullptr;
    }

    ~Operation() {
        if (desc != nullptr) {
            cudnnBackendDestroyDescriptor(desc);
        }
        if (xdesc != nullptr) {
            cudnnBackendDestroyDescriptor(xdesc);
        }
        if (ydesc != nullptr) {
            cudnnBackendDestroyDescriptor(ydesc);
        }
        if (wdesc != nullptr) {
            cudnnBackendDestroyDescriptor(wdesc);
        }
        if (cdesc != nullptr) {
            cudnnBackendDestroyDescriptor(cdesc);
        }
    }

   private:
    Operation()                  = default;
    Operation(Operation const &) = delete;
    Operation &
    operator=(Operation const &) = delete;

    cudnnBackendDescriptorType_t op_mode = CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR;

    manager<cudnnBackendDescriptor_t> xdesc = nullptr;
    manager<cudnnBackendDescriptor_t> ydesc = nullptr;
    manager<cudnnBackendDescriptor_t> wdesc = nullptr;
    manager<cudnnBackendDescriptor_t> cdesc = nullptr;

    cudnnBackendAttributeType_t alphabetaType = CUDNN_TYPE_FLOAT;
    float alpha_s = .0f, beta_s = .0f;
    double alpha_d = 0.0, beta_d = 0.0;
};

///
/// OperationBuilder Class
/// Helper class used to build Tensor class

class OperationBuilder {
   public:
    /** @defgroup OperationBuilder
     *  Set individual property of Operation class
     *  @{
     */
    auto
    setxDesc(Tensor const &tensor) -> OperationBuilder & {
        m_operation.xdesc = tensor.get_desc();
        return *this;
    }
    auto
    setyDesc(Tensor const &tensor) -> OperationBuilder & {
        m_operation.ydesc = tensor.get_desc();
        return *this;
    }
    auto
    setwDesc(Tensor const &tensor) -> OperationBuilder & {
        m_operation.wdesc = tensor.get_desc();
        return *this;
    }
    auto
    setcDesc(ConvDesc const &conv) -> OperationBuilder & {
        m_operation.cdesc = conv.get_desc();
        return *this;
    }
    auto
    setOpMode(cudnnBackendDescriptorType_t mode) -> OperationBuilder & {
        throw_if(
            [mode]() {
                return !((mode == CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR) ||
                         (mode == CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR) ||
                         (mode == CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR));
            },
            "Check the operation mode you have set for Operation");
        m_operation.op_mode = mode;
        return *this;
    }
    auto
    setAlpha(float alpha) -> OperationBuilder & {
        m_operation.alphabetaType = CUDNN_TYPE_FLOAT;
        m_operation.alpha_s = alpha;
        return *this;
    }
    auto
    setAlpha(double alpha) -> OperationBuilder & {
        m_operation.alphabetaType = CUDNN_TYPE_DOUBLE;
        m_operation.alpha_d = alpha;
        return *this;
    }
    auto
    setBeta(float beta) -> OperationBuilder & {
        m_operation.alphabetaType = CUDNN_TYPE_FLOAT;
        m_operation.beta_s = beta;
        return *this;
    }
    auto
    setBeta(double beta) -> OperationBuilder & {
        m_operation.alphabetaType = CUDNN_TYPE_DOUBLE;
        m_operation.beta_d = beta;
        return *this;
    }
    /** @} */

    //! constructs the backend Operation by calling the cudnn API
    //! Throws the appropriate error message
    Operation &&
    build() {
        throw_if([this]() { return m_operation.xdesc == nullptr; }, "cudnn xDesc is not set correctly");
        throw_if([this]() { return m_operation.wdesc == nullptr; }, "cudnn wDesc is not set correctly");
        throw_if([this]() { return m_operation.ydesc == nullptr; }, "cudnn yDesc is not set correctly");
        throw_if([this]() { return m_operation.cdesc == nullptr; }, "cudnn cDesc is not set correctly");

        // Create the descriptor.
        auto status = CUDNN_STATUS_SUCCESS;
        switch (m_operation.op_mode) {
            case CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR:
                status = cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR,
                                                      &m_operation.desc);
                break;
            case CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR:
                status = cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR,
                                                      &m_operation.desc);
                break;
            case CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR:
                status = cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR,
                                                      &m_operation.desc);
                break;
            default:
                throw(cudnnException("Check the operation op_mode you have set for Operation"));
                break;
        }
        throw_if([status]() { return status != CUDNN_STATUS_SUCCESS; }, "Bad Allocation.");
        if (m_operation.op_mode == CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR) {
            status = cudnnBackendSetAttribute(m_operation.desc,
                                              CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                                              CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                              1,
                                              &m_operation.xdesc);
            throw_if([status]() { return status != CUDNN_STATUS_SUCCESS; }, "xdesc not set for forward op.");
            status = cudnnBackendSetAttribute(m_operation.desc,
                                              CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W,
                                              CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                              1,
                                              &m_operation.wdesc);
            throw_if([status]() { return status != CUDNN_STATUS_SUCCESS; }, "ydesc not set for forward op.");
            status = cudnnBackendSetAttribute(m_operation.desc,
                                              CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y,
                                              CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                              1,
                                              &m_operation.ydesc);
            throw_if([status]() { return status != CUDNN_STATUS_SUCCESS; }, "wdesc not set for forward op.");
            status = cudnnBackendSetAttribute(m_operation.desc,
                                              CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC,
                                              CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                              1,
                                              &m_operation.cdesc);
            throw_if([status]() { return status != CUDNN_STATUS_SUCCESS; }, "cdesc not set for forward op.");
            void *alpha = (m_operation.alphabetaType == CUDNN_TYPE_FLOAT ? static_cast<void *>(&m_operation.alpha_s)
                                                                         : static_cast<void *>(&m_operation.alpha_d));
            void *beta = (m_operation.alphabetaType == CUDNN_TYPE_FLOAT ? static_cast<void *>(&m_operation.beta_s)
                                                                        : static_cast<void *>(&m_operation.beta_d));
            status = cudnnBackendSetAttribute(
                m_operation.desc, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA, m_operation.alphabetaType, 1, alpha);
            throw_if([status]() { return status != CUDNN_STATUS_SUCCESS; }, "cdesc not set for forward alpha.");
            status = cudnnBackendSetAttribute(
                m_operation.desc, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA, m_operation.alphabetaType, 1, beta);
            throw_if([status]() { return status != CUDNN_STATUS_SUCCESS; }, "cdesc not set for forward beta.");
        } else if (m_operation.op_mode == CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR) {
            status = cudnnBackendSetAttribute(m_operation.desc,
                                              CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X,
                                              CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                              1,
                                              &m_operation.xdesc);
            throw_if([status]() { return status != CUDNN_STATUS_SUCCESS; }, "xdesc not set for bwd_filter op.");
            status = cudnnBackendSetAttribute(m_operation.desc,
                                              CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW,
                                              CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                              1,
                                              &m_operation.wdesc);
            throw_if([status]() { return status != CUDNN_STATUS_SUCCESS; }, "wdesc not set for bwd_filter op.");
            status = cudnnBackendSetAttribute(m_operation.desc,
                                              CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY,
                                              CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                              1,
                                              &m_operation.ydesc);
            throw_if([status]() { return status != CUDNN_STATUS_SUCCESS; }, "ydesc not set for bwd_filter op.");
            status = cudnnBackendSetAttribute(m_operation.desc,
                                              CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC,
                                              CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                              1,
                                              &m_operation.cdesc);
            throw_if([status]() { return status != CUDNN_STATUS_SUCCESS; }, "cdesc not set for bwd_filter op.");
            void *alpha = (m_operation.alphabetaType == CUDNN_TYPE_FLOAT ? static_cast<void *>(&m_operation.alpha_s)
                                                                         : static_cast<void *>(&m_operation.alpha_d));
            void *beta = (m_operation.alphabetaType == CUDNN_TYPE_FLOAT ? static_cast<void *>(&m_operation.beta_s)
                                                                        : static_cast<void *>(&m_operation.beta_d));
            status = cudnnBackendSetAttribute(m_operation.desc,
                                              CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA,
                                              m_operation.alphabetaType,
                                              1,
                                              alpha);
            throw_if([status]() { return status != CUDNN_STATUS_SUCCESS; }, "cdesc not set for bwd_filter alpha.");
            status = cudnnBackendSetAttribute(
                m_operation.desc, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA, m_operation.alphabetaType, 1, beta);
            throw_if([status]() { return status != CUDNN_STATUS_SUCCESS; }, "cdesc not set for bwd_filter beta.");
        } else if (m_operation.op_mode == CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR) {
            status = cudnnBackendSetAttribute(m_operation.desc,
                                              CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX,
                                              CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                              1,
                                              &m_operation.xdesc);
            throw_if([status]() { return status != CUDNN_STATUS_SUCCESS; }, "xdesc not set for bwd_data op.");
            status = cudnnBackendSetAttribute(m_operation.desc,
                                              CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W,
                                              CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                              1,
                                              &m_operation.wdesc);
            throw_if([status]() { return status != CUDNN_STATUS_SUCCESS; }, "wdesc not set for bwd_data op.");
            status = cudnnBackendSetAttribute(m_operation.desc,
                                              CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY,
                                              CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                              1,
                                              &m_operation.ydesc);
            throw_if([status]() { return status != CUDNN_STATUS_SUCCESS; }, "ydesc not set for bwd_data op.");
            status = cudnnBackendSetAttribute(m_operation.desc,
                                              CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC,
                                              CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                              1,
                                              &m_operation.cdesc);
            throw_if([status]() { return status != CUDNN_STATUS_SUCCESS; }, "cdesc not set for bwd_data op.");
            void *alpha = (m_operation.alphabetaType == CUDNN_TYPE_FLOAT ? static_cast<void *>(&m_operation.alpha_s)
                                                                         : static_cast<void *>(&m_operation.alpha_d));
            void *beta = (m_operation.alphabetaType == CUDNN_TYPE_FLOAT ? static_cast<void *>(&m_operation.beta_s)
                                                                        : static_cast<void *>(&m_operation.beta_d));
            status = cudnnBackendSetAttribute(
                m_operation.desc, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA, m_operation.alphabetaType, 1, alpha);
            throw_if([status]() { return status != CUDNN_STATUS_SUCCESS; }, "cdesc not set for bwd_data alpha.");
            status = cudnnBackendSetAttribute(
                m_operation.desc, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA, m_operation.alphabetaType, 1, beta);
            throw_if([status]() { return status != CUDNN_STATUS_SUCCESS; }, "cdesc not set for bwd_data beta.");
        }
        status = cudnnBackendFinalize(m_operation.desc);
        throw_if([this, status]() { return (status != CUDNN_STATUS_SUCCESS); }, "cudnnFinalize for Operation failed");
        return std::move(m_operation);
    }

   private:
    Operation m_operation;
};
}
