# cuDNN Frontend API

## Introduction
The cuDNN frontend API is a C++ header-only library that wraps the [cuDNN C backend API](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnn-backend-api). Both the frontend and backend APIs are entry points to the same set of functionality that we commonly refer to as the "[graph API](https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#op-fusion)".

While there are two entry points to the graph API (i.e. backend and frontend), we expect that most users will use the frontend entry point because:

- It is less verbose without loss of control. All functionality accessible through the backend API is also accessible through the frontend API.
- It adds functionality on top of the backend API, like errata filters and autotuning.

Also, for those who want to use the backend API, the frontend source can serve as a reference implementation.

cudnn Frontend v1.0 API extends the groundwork in the earlier revisions. In FE 1.0, the user can describe multiple operation that form subgraph through a persistent Graph class object. Unlike the v0.x API, the user need not worry about specifying the shapes and sizes of the intermediate virtual tensors. For more information, see README.v1.0.md. 

cuDNN Frontend v1.0, also provides python bindings to its new API through pybind11. Please, look at our python documentation  

## Usage
In order to include the entire library, include the cudnn_frontend header file `cudnn_frontend.h` into your compilation unit.

## How to build samples:
    - Provide CUDA according to: https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html
    - CUDNN_PATH has the cudnn installation.
        - Headers are in CUDNN_PATH/include
        - Libraries are in CUDNN_PATH/lib or CUDNN_PATH/lib64 or CUDNN_PATH/lib/x64

    From Project Root,

    mkdir build; cd build
    cmake -DCUDNN_PATH=/path/to/cudnn -DCUDAToolkit_ROOT=/path/to/cuda  ../
    cmake --build . -j16
    bin/samples

    - You can skip building samples by providing CUDNN_FRONTEND_BUILD_SAMPLES=0 to the cmake.
    - You can skip building python bindings by providing CUDNN_FRONTEND_BUILD_PYTHON_BINDINGS=0 to the cmake.

## Logging
cuDNN Frontend API logging records execution flow through cuDNN frontend API. This functionality is disabled by default, and can be enabled through methods described in this section.

### Method 1: Using Environment Variables:
| Environment variables                             | CUDNN_FRONTEND_LOG_INFO=0 | CUDNN_FRONTEND_LOG_INFO=1 |
| ------------------------------------------------- | ------------------------- | -----------               |
| CUDNN_FRONTEND_LOG_FILE not set                   | No Logging                | No Logging                |
| CUDNN_FRONTEND_LOG_FILE set to stdout or stderr   | No Logging                | Logging to cout or cerr   |
| CUDNN_FRONTEND_LOG_FILE set to filename.txt       | No Logging                | Logging to the filename   |

### Method 2: Using API calls:
Calling `cudnn_frontend::isLoggingEnabled() = true|false` has same effect of setting the environment variable.
Calling `cudnn_frontend::getStream() = stream_name` can be used to assign the output stream directly. 

## Documentation
Documentation can be found at https://nvidia.github.io/cudnn-frontend/

## Contributing:
At this point we are not accepting any external PRs. Please create an issue in github and we will get to it.

## Feedback
Support, resources, and information about cuDNN can be found online at https://developer.nvidia.com/cudnn. 

For questions or to provide feedback, please contact cuDNN@nvidia.com.
