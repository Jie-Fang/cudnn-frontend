#include <inttypes.h>
#include "catch.hpp"
#include <cudnn.h>
#include "helpers.h"

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
    float * devPtrO);

TEST_CASE( "Use heuristics for execution", "[frontend][heuristics]" ) {
    INFO("TEST_CASE :: Use heuristics for engine generation");
    int64_t dimA[]        = {1, 32, 4, 4};
    int64_t filterdimA[]  = {32, 32, 1, 1};
    int64_t outdimA[]     = {0, 0, 0, 0}; // Conputed Below
    int64_t padA[]        = {0, 0};
    int64_t dilationA[] = {1, 1};
    int64_t convstrideA[] = {1, 1};

    int64_t dimA_padded[4];
    int64_t outdimA_padded[4];
    int64_t filterdimA_padded[4];

    int numErrors = 0;

    outdimA[0] = dimA[0];
    outdimA[1] = filterdimA[0];
    for (int dim = 0; dim < 2; dim++) {
        outdimA[dim + 2] = getFwdConvOutputDim(dimA[dim + 2], padA[dim], filterdimA[dim + 2], convstrideA[dim], dilationA[dim]);
    }

    cudnnConvolutionMode_t mode      = CUDNN_CONVOLUTION;

    printf("====USER DIMENSIONS====\n");
    printf("input dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", dimA[0], dimA[1], dimA[2], dimA[3]);
    printf("filter dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", filterdimA[0], filterdimA[1], filterdimA[2], filterdimA[3]);
    printf("output dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", outdimA[0], outdimA[1], outdimA[2], outdimA[3]);

    for (int i = 0; i < 4; i++) {
        dimA_padded[i]       = dimA[i];
        outdimA_padded[i]    = outdimA[i];
        filterdimA_padded[i] = filterdimA[i];
    }

    printf("====PADDING DIMENSIONS====\n");
    printf("padded input dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", dimA_padded[0], dimA_padded[1], dimA_padded[2], dimA_padded[3]);
    printf("padded filter dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", filterdimA_padded[0], filterdimA_padded[1], filterdimA_padded[2], filterdimA_padded[3]);
    printf("padded output dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", outdimA_padded[0], outdimA_padded[1], outdimA_padded[2], outdimA_padded[3]);

    int insize = dimA_padded[0] * dimA_padded[1] * dimA_padded[2] * dimA_padded[3];
    int filtersize = filterdimA_padded[0] * filterdimA_padded[1] * filterdimA_padded[2] * filterdimA_padded[3];
    int outsize = outdimA_padded[0] * outdimA_padded[1] * outdimA_padded[2] * outdimA_padded[3];


    float* devPtrX = NULL;
    float* devPtrW = NULL;
    float* devPtrY = NULL;
    float* hostX     = NULL;
    float* hostW     = NULL;
    float* hostW_ref = NULL;
    float* hostY     = NULL;

    checkCudaErr(cudaMalloc((void**)&(devPtrX), (insize) * sizeof(devPtrX[0])));
    checkCudaErr(cudaMalloc((void**)&(devPtrW), (filtersize) * sizeof(devPtrW[0])));
    checkCudaErr(cudaMalloc((void**)&(devPtrY), (outsize) * sizeof(devPtrY[0])));

    hostX     = (float*) calloc(insize, sizeof(hostX[0]));
    hostW     = (float*) calloc(filtersize, sizeof(hostW[0]));
    hostW_ref = (float*) calloc(filtersize, sizeof(hostW_ref[0]));
    hostY     = (float*) calloc(outsize, sizeof(hostY[0]));

    initImage(hostX, insize);
    initImage(hostW, filtersize);
    initImage(hostY, outsize);

    checkCudaErr(cudaMemcpy(devPtrX, hostX, sizeof(hostX[0]) * insize, cudaMemcpyHostToDevice));
    checkCudaErr(cudaMemcpy(devPtrW, hostW, sizeof(hostW[0]) * filtersize, cudaMemcpyHostToDevice));
    checkCudaErr(cudaMemcpy(devPtrY, hostY, sizeof(hostY[0]) * outsize, cudaMemcpyHostToDevice));
    checkCudaErr(cudaDeviceSynchronize());

    doTest(dimA, padA, convstrideA, dilationA, filterdimA_padded, outdimA, CUDNN_DATA_FLOAT, mode, devPtrX, devPtrW, devPtrY);

    checkCudaErr(cudaDeviceSynchronize());
    checkCudaErr(cudaMemcpy(hostW, devPtrW, sizeof(hostW[0]) * filtersize, cudaMemcpyDeviceToHost));
    checkCudaErr(cudaDeviceSynchronize());

    weightGrad_cpu_ref<float>(hostX, hostY, hostW_ref, CUDNN_TENSOR_NCHW, dimA, filterdimA, outdimA, convstrideA, padA, dilationA, 4/*Dims*/);

    for (int index = 0; index < filtersize; index++) {  // assuming in data is packed
        float diff         = getError(hostW[index], hostW_ref[index]);
        if (diff < 0) diff = -diff;
        if (diff > THRESHOLD) { numErrors++; }
    }
    REQUIRE(numErrors == 0);

clean:
    if (devPtrX) cudaFree(devPtrX);
    if (devPtrW) cudaFree(devPtrW);
    if (devPtrY) cudaFree(devPtrY);
    if (hostX) free(hostX);
    if (hostW) free(hostW);
    if (hostW_ref) free(hostW_ref);
    if (hostY) free(hostY);

}
