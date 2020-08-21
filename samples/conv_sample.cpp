// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

//
// This example demonstrates how to use CUDNN library calls cudnnConvolutionForward,
// cudnnConvolutionBackwardData, and cudnnConvolutionBackwardFilter with the option
// to enable Tensor Cores on Volta with cudnnSetConvolutionMathType.
//
// 1. Make sure cuda and cudnn are installed in the same directory.
//
// 2. Run make from the directory of the sample specifying the cuda installation path:
//        make CUDA_PATH=<cuda installation path>
//
// 3. Use the following arguments to run sample with different convolution parameters:
//        -c2048 -h7 -w7 -k512 -r1 -s1 -pad_h0 -pad_w0 -u1 -v1
//        -c512 -h28 -w28 -k128 -r1 -s1 -pad_h0 -pad_w0 -u1 -v1
//        -c512 -h28 -w28 -k1024 -r1 -s1 -pad_h0 -pad_w0 -u2 -v2
//        -c512 -h28 -w28 -k256 -r1 -s1 -pad_h0 -pad_w0 -u2 -v2
//        -c256 -h14 -w14 -k256 -r3 -s3 -pad_h1 -pad_w1 -u1 -v1
//        -c256 -h14 -w14 -k1024 -r1 -s1 -pad_h0 -pad_w0 -u1 -v1
//        -c1024 -h14 -w14 -k256 -r1 -s1 -pad_h0 -pad_w0 -u1 -v1
//        -c1024 -h14 -w14 -k2048 -r1 -s1 -pad_h0 -pad_w0 -u2 -v2
//        -c1024 -h14 -w14 -k512 -r1 -s1 -pad_h0 -pad_w0 -u2 -v2
//        -c512 -h7 -w7 -k512 -r3 -s3 -pad_h1 -pad_w1 -u1 -v1
//        -c512 -h7 -w7 -k2048 -r1 -s1 -pad_h0 -pad_w0 -u1 -v1
//        -c2048 -h7 -w7 -k512 -r1 -s1 -pad_h0 -pad_w0 -u1 -v1
//
// 4. Use the following arguments to run sample with int8x4 and int8x32 benchmarks:
//        -filterFormat2 -n1 -c512 -h100 -w100 -k512 -r8 -s8 -pad_h0 -pad_w0 -u1 -v1 -b
//        -filterFormat2 -n1 -c4096 -h64 -w64 -k512 -r4 -s4 -pad_h1 -pad_w1 -u1 -v1 -b
//        -filterFormat2 -n1 -c512 -h100 -w100 -k512 -r8 -s8 -pad_h1 -pad_w1 -u1 -v1 -b
//        -filterFormat2 -n1 -c512 -h128 -w128 -k512 -r13 -s13 -pad_h1 -pad_w1 -u1 -v1 -b
//
// 5. Use the following additional arguments to run the layer with a different setup:
//        -mathType1     : enable Tensor Cores on Volta.
//        -dgrad         : run cudnnConvolutionBackwardData() instead of cudnnConvolutionForward().
//        -wgrad         : run cudnnConvolutionBackwardFilter() instead of cudnnConvolutionForward().
//        -n<int>        : mini batch size. (use -b with large n)
//        -b             : benchmark mode. Bypass the CPU correctness check.
//        -filterFormat0 : Use tensor format CUDNN_TENSOR_NCHW (Default).
//        -filterFormat1 : Use tensor format CUDNN_TENSOR_NHWC.
//        -filterFormat2 : Use tensor format CUDNN_TENSOR_NCHW_VECT_C. Using this
//                         format switches to int8x4 and int8x32 testing
//
// 6. Note that changing the "-filterFormat" flag will automatically switch to valid data types for
//    that format. CUDNN_TENSOR_NCHW and CUDNN_TENSOR_NHWC support single and half precision
//    tests, while CUDNN_TENSOR_NCHW_VECT_C supports int8x4 and int8x32 tests.
//
// 7. The "-fold" flag is useful for strided cases, FFT algorithm is chosen for demo purposes, 
//    but it can be applied to other algorithms as well

#include <iostream>
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <cuda_runtime.h>
#include <assert.h>

#include <cudnn_frontend.h>
#include "fp16_dev.h"
#include "fp16_emu.h"
#include "helpers.h"

#define SWITCH_CHAR '-'

#if defined(__linux__)
#include <stddef.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/sysinfo.h>
static double second(void) 
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}
#elif defined(__QNX__)
#include <time.h>
static double second(void) 
{
    struct timespec tp;
    clock_gettime(CLOCK_REALTIME, &tp);
    return ((double)tp.tv_sec + (double)tp.tv_nsec / 1000000000.0);
}
#else
#error unsupported platform
#endif

// Generate uniform numbers [0,1)
void initImage(float* image, int64_t imageSize) {
    static unsigned seed = 123456789;
    for (int index = 0; index < imageSize; index++) {
        seed         = (1103515245 * seed + 12345) & 0xffffffff;
        image[index] = float(seed) * 2.3283064e-10;  // 2^-32
    }
}

void initImage(half1* image, int64_t imageSize) {
    static unsigned seed = 123456789;
    for (int index = 0; index < imageSize; index++) {
        seed         = (1103515245 * seed + 12345) & 0xffffffff;
        image[index] = cpu_float2half_rn(float(seed) * 2.3283064e-10);  // 2^-32
    }
}

// Currently set to generate uniform integers [-2, 2] to avoid int8 overflow
void initImage(int8_t* image, int imageSize) {
    static unsigned seed = 123456789;
    for (int64_t index = 0; index < imageSize; index++) {
        seed = (1103515245 * seed + 12345) & 0xffffffff;
        // Takes floats from [0, 1), scales and casts to ints from [0, 4], then subtracts from 2
        image[index] = 2 - (int8_t)(5 * float(seed) * 2.3283064e-10);  // 2^-32
    }
}

void initImagePadded(int8_t* image, int64_t dimA[], int64_t dimPadded[], int64_t stridePadded[], cudnnDataType_t dataType) {
    static unsigned seed = 123456789;
    int resizeFactor     = (dataType == CUDNN_DATA_INT8x4) ? 4 : 32;
    int totalSize        = dimPadded[0] * dimPadded[1] * dimPadded[2] * dimPadded[3];

    // #pragma omp parallel for
    for (int i = 0; i < totalSize; i++) {
        int n  = (i / stridePadded[0]) % dimPadded[0];
        int c1 = (i / (stridePadded[1] * resizeFactor)) % (dimPadded[1] / resizeFactor);
        int c2 = i % resizeFactor;
        int c  = c1 * resizeFactor + c2;
        if (n < dimA[0] && c < dimA[1]) {
            image[i] = 2 - (int8_t)(5 * float(seed) * 2.3283064e-10);  // 2^-32
        } else {
            image[i] = 0;
        }
    }
}

int checkCudaError(cudaError_t code, const char* expr, const char* file, int line) {
    if (code) {
        printf("CUDA error at %s:%d, code=%d (%s) in '%s'", file, line, (int)code, cudaGetErrorString(code), expr);
        return 1;
    }
    return 0;
}

int checkCudnnError(cudnnStatus_t code, const char* expr, const char* file, int line) {
    if (code) {
        printf("CUDNN error at %s:%d, code=%d (%s) in '%s'\n", file, line, (int)code, cudnnGetErrorString(code), expr);
        return 1;
    }
    return 0;
}

static void printPerf(
    double cudaTime,
    double cudaGflops,
    double cudaBandwithGb,
    const char* cpuLib,
    double cpuTime,
    double cpuGflops,
    double cpuBandwithGb) 
{
    printf("^^^^ CUDA : elapsed = %g sec,  ", cudaTime);
    if (cudaGflops > 0) printf("Gflops = %.3f ", cudaGflops);
    if (cudaBandwithGb > 0) printf("Bandwidth = %.3f ", cudaBandwithGb);
    printf("\n");
    if (cpuLib) {
        printf("^^^^%s : elapsed = %g sec, ", cpuLib, cpuTime);
        if (cpuGflops > 0) printf("Gflops = %.3f ", cpuGflops);
        if (cpuBandwithGb > 0) printf("Bandwidth = %.3f, ", cpuBandwithGb);
        printf("Speedup %.2f\n", cpuTime / cudaTime);
    }
}

void generateStrides(const int64_t* dimA, int64_t* strideA, int nbDims, cudnnTensorFormat_t filterFormat) {
    // For INT8x4 and INT8x32 we still compute standard strides here to input
    // into the cuDNN functions. We will manually scale by resizeFactor in the cpu ref.
    if (filterFormat == CUDNN_TENSOR_NCHW) {
        strideA[nbDims - 1] = 1;
        for (int64_t d = nbDims - 2; d >= 0; d--) {
            strideA[d] = strideA[d + 1] * dimA[d + 1];
        }
    } else {
        // Here we assume that the format is CUDNN_TENSOR_NHWC
        strideA[1]          = 1;
        strideA[nbDims - 1] = strideA[1] * dimA[1];
        for (int64_t d = nbDims - 2; d >= 2; d--) {
            strideA[d] = strideA[d + 1] * dimA[d + 1];
        }
        strideA[0] = strideA[2] * dimA[2];
    }
}

// Convert a linear index
// i = d_1 s_1 ... s_n + d_2 s_2 ... s_n + d_n-1 s_n + d_n
// into a multidimensional index
// (d_1, d_2, ..., d_n)
void lin2dim(int id, int64_t* ids, const int64_t* dims, int length) {
    int idrem = id;
    int prod  = 1;  // accumulates the product of the dimensions
    for (int i = length - 1; i >= 0; i--) {
        ids[i] = (idrem / prod) % dims[i];
        idrem  = id - ids[i] * prod;
        prod *= dims[i];
    }
}

// Convert a multidimensional index
// (d_1, d_2, ..., d_n)
// into a linear index
// i = d_1 s_1 + ... + d_n s_n
static int dim2lin(const int64_t* ids, const int64_t* strides, int length) {
    int res = 0;
    for (int i = 0; i < length; i++) {
        res += ids[i] * strides[i];
    }
    return res;
}
void doEpilog(float* out, int idx, float alphaAcc, float beta) {
    if (beta == 0.f) {
        out[idx] = alphaAcc;
    } else {
        out[idx] = alphaAcc + out[idx] * beta;
    }
}

void doEpilog(half1* out, int idx, float alphaAcc, float beta) {
    if (beta == 0.f) {
        out[idx] = cpu_float2half_rn(alphaAcc);
    } else {
        out[idx] = cpu_float2half_rn(alphaAcc + cpu_half2float(out[idx]) * beta);
    }
}

void doEpilog(int8_t* out, int idx, int32_t alphaAcc, float beta) {
    int32_t val;
    if (beta == 0.f) {
        val = alphaAcc;
    } else {
        val = alphaAcc + out[idx] * beta;
    }
    // Properly handle overflow errors in the same way cuDNN does
    if (val > 127) {
        val = 127;
    } else if (val < -128) {
        val = -128;
    }
    out[idx] = val;
}

// T_ELEM is the type the data is stored in, T_MATH is the type the calculations are done in.
template <typename T_ELEM, typename T_MATH> 
static void conv_cpu_ref(
    const T_ELEM* inputData,
    const T_ELEM* filterData,
    T_ELEM* outputData,
    float alpha,
    float beta,
    int resizeFactor,
    cudnnTensorFormat_t filterFormat,
    const int64_t* inDims,
    const int64_t* filDims,
    const int64_t* outDims,
    const int64_t* inStride,
    const int64_t* outStride,
    const int64_t* stride,
    const int64_t* pad,
    const int64_t* dilation,
    int64_t nbDims) 
{
    int imDims = nbDims - 2;

    int64_t filStride[8] = {0};
    generateStrides(filDims, filStride, nbDims, filterFormat);

    bool isConv = true;  //(CUDNN_CONVOLUTION == mode) ;

    // Number of pixels in output
    int nPixelsOut = 1;
    for (int i = 2; i < nbDims; i++) {
        nPixelsOut *= outDims[i];
    }

    // Number of pixels in filter
    int nPixelsFil = 1;
    for (int i = 2; i < nbDims; i++) {
        nPixelsFil *= filDims[i];
    }

    // Used to store coordinates
    int64_t filIds[8] = {0};
    int64_t outIds[8] = {0};
    int64_t inIds[8]  = {0};
    int64_t tmpIds[8] = {0};

    // For each image in the output
    for (int64_t ni = 0; ni < outDims[0]; ni++) {
        // For each outer feature layer of the output image
        for (int ki_outer = 0; ki_outer < outDims[1] / resizeFactor; ki_outer++) {
            int outputOffset = ni * outStride[0] / resizeFactor + ki_outer * outStride[1];
            // For every pixel in this output image's feature layer
            for (int outId = 0; outId < nPixelsOut; outId++) {
                // Get output pixel ids
                lin2dim(outId, outIds, outDims + 2, imDims);  // Skip n and k dimensions
                // Now we get the coordinates in input space of the "top left" corner 
                // of the filter: multiply by stride and remove pad
                for (int d = 0; d < imDims; d++) {
                    inIds[d] = outIds[d] * stride[d] - pad[d];
                }
                // For each inner feature layer of the output image
                for (int ki_inner = 0; ki_inner < resizeFactor; ki_inner++) {
                    // We prepare to accumulate
                    T_MATH tmp = 0;
                    // For each outer feature layer of the input image and filter
                    for (int ci = 0; ci < inDims[1] / resizeFactor; ci++) {
                        int inputOffset = ni * inStride[0] / resizeFactor + ci * inStride[1];
                        int filterOffset = (ki_outer * resizeFactor + ki_inner) * filStride[0] / resizeFactor + ci * filStride[1];
                        // Now for every pixel in the filter
                        for (int filId = 0; filId < nPixelsFil; filId++) {
                            // Get the position of the pixel
                            lin2dim(filId, filIds, filDims + 2, imDims);
                            // Compute the corresponding output pixel
                            // and check whether we are in the padding area on the fly too
                            // (not that for convolution, we flip the image patch;
                            // equivalent to flipping the filter patch).
                            bool inside = true;
                            for (int d = 0; d < imDims && inside; d++) {
                                if (isConv) {
                                    tmpIds[d] = inIds[d] + dilation[d] * (filDims[2 + d] - 1 - filIds[d]);
                                } else {
                                    tmpIds[d] = inIds[d] + dilation[d] * filIds[d];
                                }
                                // If we are in the padding area: stop and skip computations
                                inside &= (tmpIds[d] >= 0 && tmpIds[d] < inDims[2 + d]);
                            }
                            if (inside) {
                                int actualTmpId = inputOffset + dim2lin(tmpIds, (inStride) + 2, imDims);
                                // int actualFilId = filterOffset + filId ;
                                int actualFilId = filterOffset + dim2lin(filIds, (filStride) + 2, imDims);

                                // For each inner feature layer of the input image and filter
                                for (int i = 0; i < resizeFactor; i++) {
                                    T_ELEM fval = filterData[actualFilId * resizeFactor + i];
                                    T_ELEM ival = inputData[actualTmpId * resizeFactor + i];
                                    tmp         = doFma(fval, ival, tmp);
                                }
                            }
                        }
                    }

                    // Store final result in proper position in output image
                    int actualOutId = outputOffset + dim2lin(outIds, (outStride) + 2, imDims);
                    doEpilog(outputData, actualOutId * resizeFactor + ki_inner, alpha * tmp, beta);
                }
            }
        }
    }
}

template <typename T_ELEM> 
static void dataGrad_cpu_ref(
    const T_ELEM* weight,
    const T_ELEM* top_diff,
    T_ELEM* output,
    float alpha,
    float beta,
    cudnnTensorFormat_t filterFormat,
    const int64_t* inDims,
    const int64_t* filDims,
    const int64_t* outDims,
    const int64_t* inStride,
    const int64_t* outStride,
    const int64_t* stride,
    const int64_t* pad,
    const int64_t* dilation,
    int nbDims,
    cudnnConvolutionMode_t mode) 
{
    // Sanity checks
    // output is n x c x h x w
    // diff   is n x k x p x q
    // filter is k x c x r x s
    assert(inDims[0] == outDims[0]);   // n
    assert(inDims[1] == filDims[0]);   // k
    assert(outDims[1] == filDims[1]);  // cactualOutId

    int64_t strideA_padded[8];
    int64_t outstrideA_padded[8];

    generateStrides(inDims, strideA_padded, nbDims, filterFormat);
    generateStrides(outDims, outstrideA_padded, nbDims, filterFormat);

    int64_t filStride[8] = {0};
    generateStrides(filDims, filStride, nbDims, filterFormat);

    // true for convolution and false for cross-correlation
    bool isConv = (mode == CUDNN_CONVOLUTION) ? true : false;

    // For every output pixel (n x c x h x w)
    for (int ni = 0; ni < outDims[0]; ni++) {
        for (int ci = 0; ci < outDims[1]; ci++) {
            for (int hi = 0; hi < outDims[2]; hi++) {
                for (int wi = 0; wi < outDims[3]; wi++) {
                    int outIdx = ni * outStride[0] + ci * outStride[1] + hi * outStride[2] + wi * outStride[3];
                    float val  = 0.0;

                    // For every diff channel (k)
                    for (int ki = 0; ki < inDims[1]; ki++) {  // Sum over k channels
                        int offset_filter = ki * filStride[0] + ci * filStride[1];
                        int offset_diff   = ni * inStride[0] + ki * inStride[1];
                        // For every pixel if filter (r x s)
                        for (int ri = 0; ri < filDims[2]; ri++) {
                            int p = hi + pad[0];

                            if (isConv) {
                                p -= (filDims[2] - 1 - ri) * dilation[0];
                            } else {
                                p -= ri * dilation[0];
                            }

                            if (p % stride[0]) {
                                continue;
                            }

                            p /= stride[0];

                            for (int si = 0; si < filDims[3]; si++) {
                                int q = wi + pad[1];

                                // Fetch the value in filter and diff, product and accumulate
                                // So basically, for the convolution, we replace r by dim-1-r 
                                // and s by dim-1-s to "flip" the filter
                                // We can then just reason in term of correlation
                                if (isConv) {
                                    q -= (filDims[3] - 1 - si) * dilation[1];
                                } else {
                                    q -= si * dilation[1];
                                }

                                // Skip if q or p isn't multiple of strides
                                if (q % stride[1]) {
                                    continue;
                                }

                                q /= stride[1];

                                int inBounds = ((p >= 0) && (p < inDims[2]) && (q >= 0) && (q < inDims[3]));
                                if (inBounds) {
                                    int filterIdx = offset_filter + ri * filStride[2] + si * filStride[3];
                                    int diffIdx   = offset_diff + p * inStride[2] + q * inStride[3];
                                    T_ELEM imTmp  = top_diff[diffIdx];
                                    T_ELEM filTmp = weight[filterIdx];
                                    val           = doFma(filTmp, imTmp, val);
                                }
                            }
                        }
                    }
                    doEpilog(output, outIdx, alpha * val, beta);
                }
            }
        }
    }
}

float getError(float dev, float ref) {
    if (ref > 1.0 || ref < -1.0)
        return (dev - ref) / ref;
    else
        return dev - ref;
}

float getError(half1 dev, half1 ref) {
    if (cpu_half2float(ref) > 1.0 || cpu_half2float(ref) < -1.0)
        return (cpu_half2float(dev) - cpu_half2float(ref)) / cpu_half2float(ref);
    else
        return cpu_half2float(dev) - cpu_half2float(ref);
}

int8_t getError(int8_t dev, int8_t ref) {
    return dev - ref;
}

int getFwdConvDilatedFilterDim(int filterDim, int dilation) {
    return ((filterDim - 1) * dilation) + 1;
}

int getFwdConvPaddedImageDim(int tensorDim, int pad) {
    return tensorDim + (2 * pad);
}

int getFwdConvOutputDim(
    int tensorDim, 
    int pad, 
    int filterDim, 
    int stride, 
    int dilation) 
{
    int p = (getFwdConvPaddedImageDim(tensorDim, pad) - getFwdConvDilatedFilterDim(filterDim, dilation)) / stride + 1;
    return (p);
}

int doTest(
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
    float * devPtrO)
{
    cudnnHandle_t handle_;
    const int convDim = 2;

    float alpha     = 1.0f;
    float beta      = 0.0;

    int64_t strideA_padded[4];
    int64_t outstrideA_padded[4];
    int64_t filterstrideA_padded[4];

    generateStrides(filterdimA_padded, filterstrideA_padded, 4, CUDNN_TENSOR_NCHW);
    generateStrides(dimA_padded, strideA_padded, 4, CUDNN_TENSOR_NCHW);
    generateStrides(outdimA_padded, outstrideA_padded, 4, CUDNN_TENSOR_NCHW);

    try {
        auto tensor_x = cudnn_frontend::TensorBuilder()
            .setDim(4, dimA_padded)
            .setStrides(4,strideA_padded)
            .setId('x')
            .setAlignment(4)
            .setDataType(dataType)
            .build();
        std::cout << tensor_x.describe() << std::endl;
        auto tensor_y = cudnn_frontend::TensorBuilder()
            .setDim(4, outdimA_padded)
            .setStrides(4,outstrideA_padded)
            .setId('y')
            .setAlignment(4)
            .setDataType(dataType)
            .build();
        std::cout << tensor_y.describe() << std::endl;
        auto tensor_w = cudnn_frontend::TensorBuilder()
            .setDim(4, filterdimA_padded)
            .setStrides(4,filterstrideA_padded)
            .setId('w')
            .setAlignment(4)
            .setDataType(dataType)
            .build();
        std::cout << tensor_w.describe() << std::endl;
        auto conv_desc = cudnn_frontend::ConvDescBuilder()
            .setDataType(dataType)
            .setMathMode(mode)
            .setNDims(convDim)
            .setStrides(convDim, convstrideA)
            .setPrePadding(convDim, padA)
            .setPostPadding(convDim, padA)
            .setDilation(convDim, dilationA)
            .build();
        std::cout << conv_desc.describe() << std::endl;

        checkCudnnErr(cudnnCreate(&handle_));

        auto op = cudnn_frontend::OperationBuilder()
            .setxDesc(tensor_x)
            .setyDesc(tensor_y)
            .setwDesc(tensor_w)
            .setcDesc(conv_desc)
            .setAlpha(alpha)
            .setBeta(beta)
            .setOpMode(CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR)
            .build();
        std::cout << op.describe() << std::endl;
        std::array<cudnn_frontend::Operation const *, 1> ops = {&op};
        auto opGraph = cudnn_frontend::OperationGraphBuilder()
            .setHandle(handle_)
            .setOperationGraph(ops.size(), ops.data())
            .build();
        std::cout << opGraph.describe() << std::endl;
#if 0
        auto total_engines = opGraph.getEngineCount();
        // We have to randomly pick one engine from [0, total_engines)
        // Selecting "0" by default
        auto engine = cudnn_frontend::EngineBuilder()
            .setGlobalEngineIdx(0)
            .setOperationGraph(opGraph)
            .build();
        std::cout << engine.describe() << std::endl;
        auto knobs = engine.getKnobs();
        for (auto it = begin(knobs); it != end(knobs); ++it) {
            std::cout << it->describe() << std::endl;
        }
        if (knobs.begin() != knobs.end()) {
            std::cout << "Updated knob choice" << std::endl;
            knobs.begin()->setChoice(knobs.begin()->getMinValue() + 1);
            std::cout << knobs.begin()->describe() << std::endl;
        }
        auto engine_config = cudnn_frontend::EngineConfigBuilder()
            .setEngine(engine)
            .build();
        std::cout << engine_config.describe() << std::endl;
        auto plan = cudnn_frontend::ExecutionPlanBuilder()
            .setHandle(handle_)
            .setEngineConfig(engine_config)
            .build();
#else 
        auto heuristics = cudnn_frontend::EngineHeuristicsBuilder()
            .setOperationGraph(opGraph)
            .setHeurMode(CUDNN_HEUR_MODE_INSTANT)
            .build();
        std::cout << "Heuristic has " << heuristics.getEngineConfigCount() << " configurations " << std::endl;
        auto &engine_config = heuristics.getEngineConfig();

        auto fallback = cudnn_frontend::EngineFallbackListBuilder()
            .setOperationGraph(opGraph)
            .setOperation(CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR)
            .build();
        auto &fallback_list = fallback.getFallbackList();
        std::cout << "Fallback List has " << fallback_list.size() << " configurations " << std::endl;

        std::vector <cudnnBackendDescriptor_t> filtered_configs;
        cudnn_frontend::filter(engine_config, filtered_configs, cudnn_frontend::isNonDeterministic);
        cudnn_frontend::filter(fallback_list, filtered_configs, cudnn_frontend::isNonDeterministic);

        std::cout << "Heuristic has " << heuristics.getEngineConfigCount() << " configurations " << std::endl;
        std::cout << "Fallback List has " << fallback_list.size() << " configurations " << std::endl;
        std::cout << "Filter config list has " << filtered_configs.size() << " configurations " << std::endl;

        auto plan = cudnn_frontend::ExecutionPlanBuilder()
            .setHandle(handle_)
            .setEngineConfig(filtered_configs[0])
            .build();

        std::cout << "Filter config list has " << filtered_configs.size() << " configurations " << std::endl;
        for (auto i = 0; i < filtered_configs.size(); i++) {
            if (filtered_configs[i] != nullptr) {
                cudnnBackendDestroyDescriptor(filtered_configs[i]);
            }
        }
#endif
        std::cout << plan.describe() << std::endl;
        auto workspace_size = plan.getWorkspaceSize(); 
        void * data_ptrs[] = {devPtrI, devPtrO, devPtrF};
        int64_t uids[] = {'x', 'y', 'w'};
        auto variantPack = cudnn_frontend::VariantPackBuilder()
            .setWorkspacePointer(nullptr)
            .setDataPointers(3, data_ptrs)
            .setUids(3, uids)
            .build();
        std::cout << "variantPack " << variantPack.describe() << std::endl;
        cudnnStatus_t status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
        cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error");

    } catch (cudnn_frontend::cudnnException e) {
        std::cout << "Exception " << e.what() << std::endl;
        return 1;
    }

clean:
    if (handle_) cudnnDestroy(handle_);

    return 0;
}


// int main(int argc, char** argv) 
// {
//     int64_t dimA[]       = {1, 32, 4, 4};
//     int64_t filterdimA[] = {32, 32, 1, 1};
// 
//     int64_t padA[]        = {0, 0};
//     int64_t convstrideA[] = {1, 1};
// 
//     // batch size and feature layers must be multiples of 4 or 32 when using int8x4 or int8x32 respectively
// 
//     cudnnConvolutionMode_t mode      = CUDNN_CONVOLUTION;
// 
//     auto ret = doTest<float>(dimA, padA, convstrideA, filterdimA, CUDNN_DATA_FLOAT, mode);
// 
//     return (ret ? -1 : 0);
// }

