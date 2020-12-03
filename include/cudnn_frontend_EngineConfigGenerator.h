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

using executionPlans          = std::vector<cudnn_frontend::ExecutionPlan>;
using Predicate               = std::function<bool(cudnn_frontend::ExecutionPlan const& plan)>;
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
        if (!pred(plan)) {
            filtered_plans.emplace_back(std::move(plan));
        }
    }
    return filtered_plans;
}
