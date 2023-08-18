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