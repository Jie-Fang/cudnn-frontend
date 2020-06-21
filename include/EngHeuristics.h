#pragma once

#include <cudnn.h>
#include <cudnn_backend.h>

#include "cudnn_backend_wrap_utils.h"
#include "OperationGraph.h"

namespace cudnn_api_wrapper {
///
/// Engine Heuristic Class
/// This class helps determine the engine from the operation graph
/// based on the heuristics
/// Properties:
///    - heuristic mode
///    - operation graph
///
/// Use EngineHeuristicsBuilder to build this class.
/// Describe returns a string describing the EngineHeuristics class
///
class EngineHeuristics : public BackendDescriptor {
   public:
    friend class EngineHeuristicsBuilder;
    std::string
    describe() const override {
        std::stringstream ss;
        ss << "CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR :";
        return ss.str();
    }

    EngineHeuristics(EngineHeuristics &&from) : BackendDescriptor(from.desc, from.get_status(), from.get_error()), mode(from.mode), opGraph(from.opGraph) {
        from.opGraph = nullptr;
    }

    ~EngineHeuristics() {
        if (desc != nullptr) {
            cudnnBackendDestroyDescriptor(desc);
        }
        for (auto i = 0u; i < m_heuristic_results.size(); ++i) {
            if (m_heuristic_results[i] != nullptr) {
                cudnnBackendDestroyDescriptor(m_heuristic_results[i]);
            }
        }
        m_heuristic_results.clear();
        if (opGraph != nullptr) {
            cudnnBackendDestroyDescriptor(opGraph);
        }
    }

    /** @defgroup EngineHeuristicsQuery
     *  Query individual property of EngineHeuristics class
     *  @{
     */
    //! Query the total count of the engines for the Operation Set
    auto
    getEngineConfig(int64_t count = 1) -> std::vector<cudnnBackendDescriptor_t> & {
        cudnnStatus_t status;
        for (auto i = 0u; i < count; ++i) {
            cudnnBackendDescriptor_t engConfig = nullptr;
            status = cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, &engConfig);
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(this, status, "Engine Heuristics: Engine Config cudnnCreate failed");
                return m_heuristic_results;
            };
            m_heuristic_results.emplace_back(engConfig);
        }
        int64_t result = -1;
        status = cudnnBackendGetAttribute(desc,
                CUDNN_ATTR_ENGINEHEUR_RESULTS,
                CUDNN_TYPE_BACKEND_DESCRIPTOR,
                count,
                &result,
                m_heuristic_results.data());
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(this, status, "Engine Heuristics: Get Attribute Heuristic failed");
        };
        return m_heuristic_results;
    }

    //! Query the total count of the engine config for the Operation Set
    auto
    getEngineConfigCount(void) -> int64_t {
        cudnnStatus_t status;
        int64_t count = -1;
        status = cudnnBackendGetAttribute(desc,
                CUDNN_ATTR_ENGINEHEUR_RESULTS,
                CUDNN_TYPE_BACKEND_DESCRIPTOR,
                0,
                &count,
                nullptr);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(this, status, "Engine Heuristics: Get Attribute Heuristic Count failed");
        };
        return count;
    }
    /** @} */

   private:
    EngineHeuristics()                     = default;
    EngineHeuristics(EngineHeuristics const &) = delete;
    EngineHeuristics &
    operator=(EngineHeuristics const &) = delete;

    cudnnBackendHeurMode_t mode             = CUDNN_HEUR_MODE_INSTANT;
    manager<cudnnBackendDescriptor_t> opGraph = nullptr;
    std::vector<cudnnBackendDescriptor_t> m_heuristic_results; //! storage of heuristic results
};

///
/// EngineHeuristicsBuilder Class
/// Helper class used to build EngineHeuristics class
class EngineHeuristicsBuilder {
   public:
    /** @defgroup EngineHeuristicsBuilder
     *  Set individual property of EngineHeuristics class
     *  @{
     */
    //! Set operationGraph for the engine
    auto
    setOperationGraph(OperationGraph &opGraph_) -> EngineHeuristicsBuilder & {
        m_heuristics.opGraph = opGraph_.get_desc();
        return *this;
    }
    //! Set cudnnHandle for the operations
    auto
    setHeurMode(cudnnBackendHeurMode_t mode_) -> EngineHeuristicsBuilder & {
        m_heuristics.mode = mode_;
        return *this;
    }
    /** @} */

    //! constructs the EngineHeuristics by calling the cudnn API
    //! Throws the appropriate error message
    EngineHeuristics &&
    build() {
        if (m_heuristics.opGraph == nullptr) {
            set_error_and_throw_exception(&m_heuristics, CUDNN_STATUS_BAD_PARAM, "Check and set the opset for heuristic field");
            return std::move(m_heuristics);
        };

        // Create a descriptor. Memory allocation happens here.
        auto status = CUDNN_STATUS_SUCCESS;
        status      = cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, &m_heuristics.desc);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_heuristics, status, "Engine Heuristics: cudnnCreate failed");
            return std::move(m_heuristics);
        };

        status = cudnnBackendSetAttribute(m_heuristics.desc,
                                          CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                          CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                          1,
                                          &m_heuristics.opGraph);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_heuristics, status, "Engine Heuristics: Set Attribute  opset failed");
            return std::move(m_heuristics);
        };
        status = cudnnBackendSetAttribute(
            m_heuristics.desc, CUDNN_ATTR_ENGINEHEUR_MODE, CUDNN_TYPE_HEUR_MODE, 1, &m_heuristics.mode);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_heuristics, status, "Engine Heuristics: Set Attribute Heuristic Mode failed");
            return std::move(m_heuristics);
        };

        // Finalizing the descriptor
        status = cudnnBackendFinalize(m_heuristics.desc);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_heuristics, status, "Engine Heuristics: cudnn Finalize failed");
            return std::move(m_heuristics);
        };

        return std::move(m_heuristics);
    }

    explicit EngineHeuristicsBuilder()                   = default;
    ~EngineHeuristicsBuilder()                           = default;
    EngineHeuristicsBuilder(EngineHeuristicsBuilder &&)      = delete;
    EngineHeuristicsBuilder(EngineHeuristicsBuilder const &) = delete;
    EngineHeuristicsBuilder &
    operator=(EngineHeuristicsBuilder const &) = delete;

   private:
    EngineHeuristics m_heuristics;
};
}
