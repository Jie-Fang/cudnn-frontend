/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include <cudnn_frontend.h>
#include <map>

struct executionOption {
    cudnn_frontend::ExecutionPlan plan;  // One can get the underlying EngineConfig from the ExecutionPlan
    float time_ms;
};

using executionOptions = std::vector<struct executionOption>;
using executionPlans   = std::vector<cudnn_frontend::ExecutionPlan>;
using Predicate        = std::function<bool(cudnn_frontend::ExecutionPlan &plan)>;

enum class CudnnFindSamplingTechnique {
    CUDNN_FIND_SAMPLE_ONCE,             // Sample once quick but may have unstable values
    CUDNN_FIND_SAMPLE_MEDIAN_OF_THREE,  // Sample 3 times and take median.
    CUDNN_FIND_SAMPLE_TILL_STABLE       // Sample multiple times till stable.
};

using engine_config_generator = std::function<cudnn_frontend::EngineConfigList(cudnn_frontend::OperationGraph &)>;
class EngineConfigGenerator {
   private:
    std::vector<engine_config_generator> engine_config_generators;
    EngineConfigGenerator() = default;

   public:
    void
    register_engine_config_generator(engine_config_generator fn_ptr) {
        engine_config_generators.push_back(fn_ptr);
    };
    auto
    generate_engine_config(cudnn_frontend::OperationGraph& opGraph) -> cudnn_frontend::EngineConfigList {
        cudnn_frontend::EngineConfigList engine_configs;
        for (auto fn : engine_config_generators) {
            cudnn_frontend::EngineConfigList new_engine_config = fn(opGraph);
            std::copy(new_engine_config.begin(),
                      new_engine_config.end(),
                      std::back_inserter(engine_configs));
            new_engine_config.clear();
        }
        return engine_configs;
    }
    static EngineConfigGenerator &
    getInstance() {
        static EngineConfigGenerator instance;
        return instance;
    }
};

// Filter out the execution plan based on the prerequisite conditions.
auto
filter(Predicate pred, executionPlans &plans) -> executionPlans {
    executionPlans filtered_plans;
    for (auto &plan : plans) {
        if (pred(plan)) {
            filtered_plans.emplace_back(std::move(plan));
        }
    }
    return filtered_plans;
}

template <CudnnFindSamplingTechnique samplingTechnique>
auto
time_sorted_plan(cudnnHandle_t handle, executionPlans plans, cudnn_frontend::VariantPack &&variantPack)
    -> executionOptions {
    executionOptions time_sorted_plans;
    std::map<float, cudnn_frontend::ExecutionPlan &> timed_execution_plans;

    const int maxIterCount =
        (samplingTechnique == CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_ONCE)
            ? 1
            : (samplingTechnique == CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_MEDIAN_OF_THREE) ? 3 : 100;
    const float threshhold = 0.95f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaDeviceSynchronize();
    float time_ms;

    for (auto &plan : plans) {
        float time_ms       = 0.0f;
        float total_time_ms = 0.0f;
        float min_time_ms   = std::numeric_limits<float>::max();

        // Warm-up run
        ::cudnnBackendExecute(handle, plan.get_raw_desc(), variantPack.get_raw_desc());
        cudaDeviceSynchronize();

        for (int i = 0; i < maxIterCount; i++) {
            cudaEventRecord(start);

            ::cudnnBackendExecute(handle, plan.get_raw_desc(), variantPack.get_raw_desc());

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&time_ms, start, stop);

            if (samplingTechnique == CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_TILL_STABLE) {
                if (time_ms / min_time_ms < threshhold) {
                    min_time_ms = std::min<float>(min_time_ms, time_ms);
                } else {
                    time_ms = std::min(min_time_ms, time_ms);
                    break;
                }
            } else {
                total_time_ms += time_ms;
            }
        }
        if (samplingTechnique == CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_TILL_STABLE) {
            timed_execution_plans.insert({time_ms, plan});
        } else {
            timed_execution_plans.insert({total_time_ms / maxIterCount, plan});
        }
    }
    std::transform(
        timed_execution_plans.begin(),
        timed_execution_plans.end(),
        std::back_inserter(time_sorted_plans),
        [](const std::map<float, cudnn_frontend::ExecutionPlan &>::value_type &pair) -> struct executionOption {
            return {std::move(pair.second), pair.first};
        });

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return time_sorted_plans;
}

template <CudnnFindSamplingTechnique samplingTechnique>
auto
cudnnFind(cudnnHandle_t handle,
          cudnn_frontend::OperationGraph &&opGraph,
          cudnn_frontend::VariantPack &&variantPack,
          Predicate pred) -> executionOptions {
    // Creating a set of execution plans that are supported.
    executionPlans plans;
    for (auto& engine_config : EngineConfigGenerator::getInstance().generate_engine_config(opGraph)) {
        try {
            plans.push_back(
                cudnn_frontend::ExecutionPlanBuilder().setHandle(handle).setEngineConfig(engine_config).build());
        } catch (cudnn_frontend::cudnnException e) {
        }
    }
    return time_sorted_plan<samplingTechnique>(handle, filter(pred, plans), std::move(variantPack));
}
