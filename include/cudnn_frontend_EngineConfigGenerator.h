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

struct executionOption {
    cudnn_frontend::ExecutionPlan plan;  // One can get the underlying EngineConfig from the ExecutionPlan
    float time_ms;
};

using executionOptions = std::vector<struct executionOption>;
using executionPlans   = std::vector<cudnn_frontend::ExecutionPlan>;
using Predicate        = std::function<bool(cudnn_frontend::ExecutionPlan const &plan)>;
using generatorSource  = std::function<cudnn_frontend::EngineConfigList(cudnn_frontend::OperationGraph &)>;

enum class CudnnFindSamplingTechnique {
    CUDNN_FIND_SAMPLE_ONCE,             // Sample once quick but may have unstable values
    CUDNN_FIND_SAMPLE_MEDIAN_OF_THREE,  // Sample 3 times and take median.
    CUDNN_FIND_SAMPLE_TILL_STABLE       // Sample multiple times till stable.
};

class EngineConfigGenerator {
   private:
    std::vector<generatorSource> engine_config_generators;

   public:
    EngineConfigGenerator(int const sourceSize, generatorSource const *sources) {
        for (int i = 0; i < sourceSize; i++) {
            engine_config_generators.push_back(sources[i]);
        }
    };

    auto
    generate_engine_config(cudnn_frontend::OperationGraph &opGraph) -> cudnn_frontend::EngineConfigList {
        cudnn_frontend::EngineConfigList engine_configs;
        for (auto fn : engine_config_generators) {
            cudnn_frontend::EngineConfigList new_engine_config = fn(opGraph);
            std::copy(new_engine_config.begin(), new_engine_config.end(), std::back_inserter(engine_configs));
            new_engine_config.clear();
        }
        return engine_configs;
    }

    auto
    cudnnGetPlan(cudnnHandle_t handle, cudnn_frontend::OperationGraph &&opGraph, Predicate pred) -> executionPlans;

    template <CudnnFindSamplingTechnique samplingTechnique>
    auto
    cudnnFindPlan(cudnnHandle_t handle,
                  cudnn_frontend::OperationGraph &&opGraph,
                  cudnn_frontend::VariantPack &variantPack,
                  Predicate pred) -> executionOptions;
};

// Filter out the execution plan based on the prerequisite conditions.
auto
filter(Predicate pred, executionPlans &plans) -> executionPlans {
    executionPlans filtered_plans;
    for (auto &plan : plans) {
        if (!pred(plan)) {
            filtered_plans.emplace_back(std::move(plan));
        }
    }
    return filtered_plans;
}
