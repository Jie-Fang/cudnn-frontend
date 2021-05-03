
/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include "contrib/nlohmann/json/json.hpp"

#include <cstdlib>
#include <fstream>
#pragma once

using json = nlohmann::json;

namespace cudnn_frontend {

// Loads the json handle from the json file 
// json file is defined by environment variable
// CUDNN_ERRATA_JSON_FILE
static void
load_from_config(json &json_handle) {
    const char * errata_json = std::getenv("CUDNN_ERRATA_JSON_FILE");
    if (errata_json == nullptr) {return;}
    std::ifstream ifs("./errata.json" , std::ifstream::in);
    if (!ifs.is_open() || !ifs.good()) {return;}
    ifs >> json_handle;
    return;
}

static bool 
check_rule(const json &json_handle, const std::string & executionPlanTag,
    cudnnHandle_t handle) {
    (void) json_handle;
    (void) executionPlanTag;
    (void) handle;
    return true;
}

// Takes in an initialzed json handle and checks if it satisfies the 
// condition for running it. Returns true if the given executionPlanTag
// is faulty.
static bool
check_errata(const json &json_handle, const std::string & executionPlanTag,
    cudnnHandle_t handle) {

    for (auto const &rule : json_handle["rules"]) {
        if (check_rule(rule, executionPlanTag, handle)) {
            return true;
        }
    }

    return false;
}

}
