#pragma once
#include <iostream>
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <cuda_runtime.h>
#include <assert.h>

#include "fp16_dev.h"
#include "fp16_emu.h"

#define THRESHOLD 2.0e-2

int getFwdConvDilatedFilterDim(int filterDim, int dilation);

int getFwdConvPaddedImageDim(int tensorDim, int pad);

int getFwdConvOutputDim( int tensorDim, int pad, int filterDim, int stride, int dilation);

void generateStrides(const int64_t* dimA, int64_t* strideA, int nbDims, cudnnTensorFormat_t filterFormat);

int checkCudaError(cudaError_t code, const char* expr, const char* file, int line);

int checkCudnnError(cudnnStatus_t code, const char* expr, const char* file, int line);

#define checkCudaErr(...)                                                        \
    do {                                                                         \
        int err = checkCudaError(__VA_ARGS__, #__VA_ARGS__, __FILE__, __LINE__); \
        if (err) {                                                               \
            goto clean;                                                          \
        }                                                                        \
    } while (0)

#define checkCudnnErr(...)                                                        \
    do {                                                                          \
        int err = checkCudnnError(__VA_ARGS__, #__VA_ARGS__, __FILE__, __LINE__); \
        if (err) {                                                                \
            goto clean;                                                           \
        }                                                                         \
    } while (0)

void initImage(float* image, int64_t imageSize);
void initImage(half1* image, int64_t imageSize);
void initImage(int8_t* image, int imageSize);
void initImagePadded(int8_t* image, int64_t dimA[], int64_t dimPadded[], int64_t stridePadded[], cudnnDataType_t dataType);

void doEpilog(float* out, int idx, float alphaAcc, float beta);
void doEpilog(half1* out, int idx, float alphaAcc, float beta);
void doEpilog(int8_t* out, int idx, int32_t alphaAcc, float beta);


static float doFma(float fval, float ival, float tmp) {
    return fval * ival + tmp;
}

static float doFma(half1 fval, half1 ival, float tmp) {
    return cpu_half2float(fval) * cpu_half2float(ival) + tmp;
}

static int32_t doFma(int8_t fval, int8_t ival, int32_t tmp) {
    return int32_t(fval) * int32_t(ival) + tmp;
}

// Garbage function, resolves overloaded function ambiguity for an invalid type combination
static int32_t doFma(float fval, float ival, int32_t tmp) {
    return 0;
}

// Garbage function, resolves overloaded function ambiguity for an invalid type combination
static int32_t doFma(half1 fval, half1 ival, int32_t tmp) {
    return 0;
}

// Garbage function, resolves overloaded function ambiguity for an invalid type combination
static float doFma(int8_t fval, int8_t ival, float tmp) {
    return 0;
}

float getError(float dev, float ref);
float getError(half1 dev, half1 ref);
int8_t getError(int8_t dev, int8_t ref);

template <typename T_ELEM>
void weightGrad_cpu_ref(
    const T_ELEM* image,
    const T_ELEM* diffData,
    T_ELEM* output,
    cudnnTensorFormat_t filterFormat,
    const int64_t* inDims,
    const int64_t* filDims,
    const int64_t* diffDims,
    const int64_t* stride,
    const int64_t* pad,
    const int64_t* dilation,
    int nbDims) 
{
    float alpha     = 1.0f;
    float beta      = 0.0;
    // Some sanity checks
    // image   is n x c x h x w
    // diff    is n x k x p x q
    // filter  is k x c x r x s
    assert(inDims[0] == diffDims[0]);
    assert(inDims[1] == filDims[1]);
    assert(diffDims[1] == filDims[0]);

    // Filter stride
    int64_t filterStride[8];
    int64_t inStride[8];
    int64_t diffStride[8];

    generateStrides(inDims, inStride, nbDims, filterFormat);
    generateStrides(diffDims, diffStride, nbDims, filterFormat);
    generateStrides(filDims, filterStride, nbDims, filterFormat);

    bool isConv = true;  //(CUDNN_CONVOLUTION == mode) ;

    // For every filter pixel (k x c x r x s)
    for (int ci = 0; ci < inDims[1]; ci++) {               // Loop over filter output pixels
        for (int ri = 0; ri < filDims[2]; ri++) {          //        ^
            for (int si = 0; si < filDims[3]; si++) {      //    ^
                for (int ki = 0; ki < filDims[0]; ki++) {  // ^
                    int filIdx = ki * filterStride[0] + ci * filterStride[1] + ri * filterStride[2] + si * filterStride[3];
                    float val = 0.f;
                    // For every image (n)
                    for (int ni = 0; ni < inDims[0]; ni++) {  // Sum over the batch
                        int offset_image = ni * inStride[0] + ci * inStride[1];
                        int offset_diff  = ni * diffStride[0] + ki * diffStride[1];
                        // For every pixel in diff (p x q)
                        for (int pi = 0; pi < diffDims[2]; pi++) {      // Sum over the pixels of diff
                            for (int qi = 0; qi < diffDims[3]; qi++) {  //  ^
                                // Fetch the value in image and diff, product and accumulate
                                int y = pi * stride[0] - pad[0];
                                int x = qi * stride[1] - pad[1];
                                // Convolution = Correlation with a flipped filter
                                // So basically, for the convolution, we replace r by dim-1-r 
                                // and s by dim-1-s to "flip" the filter.
                                // We can then just reason in term of correlation
                                if (isConv) {
                                    y += (filDims[2] - 1 - ri) * dilation[0];
                                    x += (filDims[3] - 1 - si) * dilation[1];
                                } else {
                                    // The effect of dilation on the gradient is to start
                                    // the "zone of influence" of a given pixel further
                                    // into the image, so dilation
                                    // only produces a shift in x and y
                                    y += ri * dilation[0];
                                    x += si * dilation[1];
                                }
                                // Image value
                                int inBounds = ((x >= 0) && (x < inDims[3]) && (y >= 0) && (y < inDims[2]));
                                if (inBounds) {
                                    int imIdx = offset_image + y * inStride[2] + x * inStride[3];
                                    // Diff value
                                    int diffIdx = offset_diff + pi * diffStride[2] + qi * diffStride[3];
                                    // Prod and accumulate
                                    T_ELEM imTmp   = image[imIdx];
                                    T_ELEM diffTmp = diffData[diffIdx];
                                    val            = doFma(diffTmp, imTmp, val);
                                }
                            }
                        }
                    }
                    doEpilog(output, filIdx, alpha * val, beta);
                }
            }
        }
    }
}
