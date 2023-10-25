LOG_RUNTIME = False

if LOG_RUNTIME:
    import time 
    g_clk_id = time.CLOCK_MONOTONIC_RAW

class ImplementationError(Exception):
    def __init__(self, reason):
        self.reason = reason

def getFwdConvInputDims(outputTensorDim, pad, filterDim, stride, dilation):
    inputTensorDim = [0] * len(outputTensorDim)
    inputTensorDim[0] = outputTensorDim[0]
    inputTensorDim[1] = filterDim[1]
    for dim in range(2, len(inputTensorDim)):
        inputTensorDim[dim] = getSingleFwdConvInputDim(outputTensorDim[dim], pad[dim-2], filterDim[dim], stride[dim - 2], dilation[dim -2])

    return inputTensorDim

def getSingleFwdConvInputDim(outputTensorDim, pad, filterDim, stride, dilation):
    paddedTensorDim = (outputTensorDim - 1) * stride + getSingleFwdConvDilatedFilterDim(filterDim, dilation)
    inputTensorDim  = getSingleFwdConvImageDimFromPadded(paddedTensorDim, pad)
    return int(inputTensorDim)

def getSingleFwdConvImageDimFromPadded(paddedTensorDim, pad):
    tensorDim = paddedTensorDim - (2 * pad)
    return tensorDim

def getSingleFwdConvDilatedFilterDim(filterDim, dilation):
    return ((filterDim - 1) * dilation) + 1

def getFwdConvDilatedFilterDim(filterDim, dilation):
    return ((filterDim - 1) * dilation) + 1

def getFwdConvPaddedImageDim(tensorDim, pad):
    return tensorDim + (2 * pad)

def getFwdConvOutputDim(tensorDim, pad, filterDim, stride, dilation):
    p = (getFwdConvPaddedImageDim(tensorDim, pad) - getFwdConvDilatedFilterDim(filterDim, dilation)) / stride + 1
    return int(p)

def computeStrideNdTransposedPacked(nbDims, dims, axesOrder):
    inverseTranspose = dict()
    for i in range(nbDims):
        inverseTranspose[axesOrder[i]] = i

    print (dims)
    
    strides = [1] * nbDims
    strides[inverseTranspose[nbDims - 1]] = 1
    for dim in range(nbDims - 2, -1, -1):
        strides[inverseTranspose[dim]] = dims[inverseTranspose[dim + 1]] * strides[inverseTranspose[dim + 1]]
    
    return strides

def reportCurrentTime(msg):
    if LOG_RUNTIME:
        cur_time = time.clock_gettime_ns(g_clk_id)
        print("[MB_PROFILE] {} {}".format(msg, cur_time))