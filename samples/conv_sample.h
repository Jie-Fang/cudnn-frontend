
#pragma once

#include <iostream>
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <tuple>
#include <functional>

#include <cudnn_frontend.h>
#include "fp16_dev.h"
#include "fp16_emu.h"
#include "helpers.h"


void run_from_global_index(
    int64_t* dimA_padded,
    int64_t* padA,
    int64_t* convstrideA,
    int64_t* dilationA,
    int64_t* filterdimA_padded,
    int64_t* outdimA_padded,
    cudnnDataType_t dataType,
    cudnnConvolutionMode_t mode,
    float * devPtrI,
    float * devPtrF,
    float * devPtrO);

void run_from_heuristics(
    int64_t* dimA_padded,
    int64_t* padA,
    int64_t* convstrideA,
    int64_t* dilationA,
    int64_t* filterdimA_padded,
    int64_t* outdimA_padded,
    cudnnDataType_t dataType,
    cudnnConvolutionMode_t mode,
    float * devPtrI,
    float * devPtrF,
    float * devPtrO);

void run_with_external_config(
    int64_t* dimA_padded,
    int64_t* padA,
    int64_t* convstrideA,
    int64_t* dilationA,
    int64_t* filterdimA_padded,
    int64_t* outdimA_padded,
    cudnnDataType_t dataType,
    cudnnConvolutionMode_t mode,
    float * devPtrI,
    float * devPtrF,
    float * devPtrO);
