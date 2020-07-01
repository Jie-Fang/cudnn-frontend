#pragma once

#include <array>
#include <memory>
#include <sstream>
#include <algorithm>
#include <functional>
#include <utility>

#include <cudnn.h>
#include <cudnn_backend.h>

#include "cudnn_frontend_utils.h"
#include "Engine.h"

namespace cudnn_frontend {
///
/// ExecutionPlan Class
/// This class tells the Configuration of the Engine in terms of the knob choices
/// Properties:
///    - num knobs
///    - Choice
///    - Engine
///
/// Use ExecutionPlanBuilder to build this class.
/// Describe returns a string describing the tensor class
///
class ExecutionPlan : public BackendDescriptor {
   public:
    friend class ExecutionPlanBuilder;
    std::string
    describe() const override {
        std::stringstream ss;
        ss << "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR :";
        return ss.str();
    }
    ExecutionPlan(ExecutionPlan &&from)
        : BackendDescriptor(from.desc, from.get_status(), from.get_error()), handle(from.handle), engine_config(from.engine_config) {
        from.engine_config = nullptr;
    }
    ~ExecutionPlan() {
        if (desc != nullptr) {
            cudnnBackendDestroyDescriptor(desc);
        }
        if (engine_config != nullptr) {
            cudnnBackendDestroyDescriptor(engine_config);
        }
    }
    /** @defgroup ExecutionPlanQuery
     *  Query individual property of ExecutionPlan class
     *  @{
     */
    //! Query the workspace requirement for the given plan
    auto
    getWorkspaceSize(void) -> const int64_t {
        uint64_t workSpaceSize = 0;
        auto status            = cudnnBackendGetAttribute(
            desc, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, CUDNN_TYPE_INT64, 1, NULL, &workSpaceSize);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(this, status, "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: GetAttribute CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE Failed");
            return workSpaceSize;
        }
        if (workSpaceSize < 0) {
            set_error_and_throw_exception(this, status, "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: GetAttribute Workspace Size Invalid");
            return workSpaceSize;
        }
        return workSpaceSize;
    }

   private:
    ExecutionPlan()                      = default;
    ExecutionPlan(ExecutionPlan const &) = delete;
    ExecutionPlan &
    operator=(ExecutionPlan const &) = delete;

    manager<cudnnBackendDescriptor_t> engine_config = nullptr;
    cudnnHandle_t handle                   = nullptr;
};

///
/// ExecutionPlanBuilder Class
/// Helper class used to build ExecutionPlan class
class ExecutionPlanBuilder {
   public:
    /** @defgroup ExecutionPlanBuilder
     *  Set individual property of ExecutionPlan class
     *  @{
     */
    //! Set engine for the ExecutionPlan
    auto
    setHandle(cudnnHandle_t handle_) -> ExecutionPlanBuilder & {
        m_execution_plan.handle = handle_;
        return *this;
    }
    //! Set engine Config for the Plan
    auto
    setEngineConfig(EngineConfig const &engine_config_) -> ExecutionPlanBuilder & {
        m_execution_plan.engine_config = engine_config_.get_desc();
        return *this;
    }
    //! Set engine Config for the Plan
    auto
    setEngineConfig(cudnnBackendDescriptor_t & desc) -> ExecutionPlanBuilder & {
        m_execution_plan.engine_config = desc;
        desc = nullptr;
        return *this;
    }
    /** @} */

    //! constructs the Engine Config by calling the cudnn API
    //! Throws the appropriate error message
    ExecutionPlan &&
    build() {
        if(m_execution_plan.handle == nullptr) {
            set_error_and_throw_exception(&m_execution_plan, CUDNN_STATUS_BAD_PARAM, "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: Check and Set the CUDNN_ATTR_EXECUTION_PLAN_HANDLE");
            return std::move(m_execution_plan);
        };
        if(m_execution_plan.engine_config == nullptr) {
            set_error_and_throw_exception(&m_execution_plan, CUDNN_STATUS_BAD_PARAM, "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: Check and Set the CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG");
            return std::move(m_execution_plan);
        };

        // Create a descriptor. Memory allocation happens here.
        auto status = CUDNN_STATUS_SUCCESS;
        status      = cudnnBackendCreateDescriptor(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &m_execution_plan.desc);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_execution_plan, status, "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: cudnnCreate Failed");
            return std::move(m_execution_plan);
        }

        status = cudnnBackendSetAttribute(m_execution_plan.desc,
                                          CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                          CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                          1,
                                          &m_execution_plan.engine_config);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_execution_plan, status, "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: SetAttribute CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG Failed");
            return std::move(m_execution_plan);
        }
        status = cudnnBackendSetAttribute(
            m_execution_plan.desc, CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, 1, &m_execution_plan.handle);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_execution_plan, status, "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: SetAttribute CUDNN_ATTR_EXECUTION_PLAN_HANDLE Failed");
            return std::move(m_execution_plan);
        }

        // Finalizing the descriptor
        status = cudnnBackendFinalize(m_execution_plan.desc);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_execution_plan, status, "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: cudnnFinalize Descriptor Failed");
            return std::move(m_execution_plan);
        }
        return std::move(m_execution_plan);
    }

    explicit ExecutionPlanBuilder()                    = default;
    ~ExecutionPlanBuilder()                            = default;
    ExecutionPlanBuilder(ExecutionPlanBuilder &&)      = delete;
    ExecutionPlanBuilder(ExecutionPlanBuilder const &) = delete;
    ExecutionPlanBuilder &
    operator=(ExecutionPlanBuilder const &) = delete;

   private:
    ExecutionPlan m_execution_plan;
};
}
