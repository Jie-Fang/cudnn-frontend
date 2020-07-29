#pragma once

#include <algorithm>
#include <array>
#include <functional>
#include <memory>
#include <sstream>
#include <utility>

#include <cudnn.h>
#include <cudnn_backend.h>

#include "Operation.h"
#include "cudnn_frontend_utils.h"

namespace cudnn_frontend {

///
/// OperationGraph Class
/// This class tells the properties of the Tensor on which the operation will be
/// performed
/// Properties:
///    - handle
///    - operation
///
/// Use OperationGraphBuilder to build this class.
/// Describe returns a string describing the tensor class
///
class OperationGraph : public BackendDescriptor {
   public:
    friend class OperationGraphBuilder;
    std::string
    describe() const override {
        std::stringstream ss;
        ss << "CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR :";
        return ss.str();
    }

    OperationGraph(OperationGraph &&from)
        : BackendDescriptor(from.desc, from.get_status(), from.get_error()),
          handle(from.handle),
          ops(from.ops),
          numOps(from.numOps) {
        for (auto i = 0; i < ops.size(); i++) {  // TODO: Use std::fill
            from.ops[i] = nullptr;
        }
    }

    ~OperationGraph() {
        if (desc != nullptr) {
            cudnnBackendDestroyDescriptor(desc);
        }

        for (auto i = 0; i < ops.size(); i++) {
            if (ops[i] != nullptr) {
                cudnnBackendDestroyDescriptor(ops[i]);
                ops[i] = nullptr;
            }
        }
    }

    /** @defgroup OperationGraphQuery
     *  Query individual property of OperationGraph class
     *  @{
     */
    //! Query the total count of the engines for the Operation Set
    auto
    getEngineCount(void) -> const int64_t {
        int64_t global_count = -1;
        auto status          = cudnnBackendGetAttribute(
            desc, CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT, CUDNN_TYPE_INT64, 1, NULL, &global_count);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(this,
                                          status,
                                          "CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR: GetAttribute "
                                          "CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT Failed");
        }
        return global_count;
    }
    /** @} */

   private:
    OperationGraph()                       = default;
    OperationGraph(OperationGraph const &) = delete;
    OperationGraph &
    operator=(OperationGraph const &) = delete;

    cudnnHandle_t handle = nullptr;
    std::array<manager<cudnnBackendDescriptor_t>, 10> ops{};
    int64_t numOps = -1;
};

///
/// OperationGraphBuilder Class
/// Helper class used to build OperationGraph class
class OperationGraphBuilder {
   public:
    /** @defgroup OperationGraphBuilder
     *  Set individual property of OperationGraph class
     *  @{
     */
    //! Set cudnnHandle for the operations
    auto
    setHandle(cudnnHandle_t handle_) -> OperationGraphBuilder & {
        m_operationGraph.handle = handle_;
        return *this;
    }
    //! Set numoperations and the operations
    auto
    setOperationGraph(int64_t numOps_, Operation const **ops_) -> OperationGraphBuilder & {
        m_operationGraph.numOps = numOps_;
        for (auto i = 0u; i < numOps_; i++) {
            m_operationGraph.ops[i] = ops_[i]->get_desc();
        }
        return *this;
    }
    /** @} */

    //! constructs the OperationGraph by calling the cudnn API
    //! Throws the appropriate error message
    OperationGraph &&
    build() {
        if (m_operationGraph.numOps <= 0) {
            set_error_and_throw_exception(
                &m_operationGraph,
                CUDNN_STATUS_BAD_PARAM,
                "CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR: Check and Set the CUDNN_ATTR_OPERATIONGRAPH_OPS Count field");
            return std::move(m_operationGraph);
        }
        if (m_operationGraph.ops[0] == nullptr) {
            set_error_and_throw_exception(
                &m_operationGraph,
                CUDNN_STATUS_BAD_PARAM,
                "CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR: Check and set CUDNN_ATTR_OPERATIONGRAPH_OPS field");
            return std::move(m_operationGraph);
        }
        if (m_operationGraph.handle == nullptr) {
            set_error_and_throw_exception(
                &m_operationGraph,
                CUDNN_STATUS_BAD_PARAM,
                "CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR: Check and Set CUDNN_ATTR_OPERATIONGRAPH_HANDLE");
            return std::move(m_operationGraph);
        }

        // Create a descriptor. Memory allocation happens here.
        auto status = CUDNN_STATUS_SUCCESS;
        status      = cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &m_operationGraph.desc);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_operationGraph, status, "CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR: cudnnCreate Failed");
            return std::move(m_operationGraph);
        }

        status = cudnnBackendSetAttribute(m_operationGraph.desc,
                                          CUDNN_ATTR_OPERATIONGRAPH_OPS,
                                          CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                          m_operationGraph.numOps,
                                          m_operationGraph.ops.data());
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_operationGraph,
                status,
                "CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR: SetAttribute CUDNN_ATTR_OPERATIONGRAPH_OPS Failed");
            return std::move(m_operationGraph);
        }
        status = cudnnBackendSetAttribute(
            m_operationGraph.desc, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, 1, &m_operationGraph.handle);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_operationGraph,
                status,
                "CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR: SetAttribute CUDNN_ATTR_OPERATIONGRAPH_HANDLE Failed");
            return std::move(m_operationGraph);
        }

        // Finalizing the descriptor
        status = cudnnBackendFinalize(m_operationGraph.desc);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_operationGraph, status, "CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR: cudnnFinalize Failed");
            return std::move(m_operationGraph);
        }

        return std::move(m_operationGraph);
    }

    explicit OperationGraphBuilder()                     = default;
    ~OperationGraphBuilder()                             = default;
    OperationGraphBuilder(OperationGraphBuilder &&)      = delete;
    OperationGraphBuilder(OperationGraphBuilder const &) = delete;
    OperationGraphBuilder &
    operator=(OperationGraphBuilder const &) = delete;

   private:
    OperationGraph m_operationGraph;
};
}
