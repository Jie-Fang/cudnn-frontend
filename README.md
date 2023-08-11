# cuDNN FrontEnd(FE) API

## Introduction
The cuDNN FrontEnd(FE) API is a C++ header-only library that wraps the [cuDNN C backend API](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnn-backend-api). Both the FE and backend APIs are entry points to the same set of functionality that is commonly referred to as the "[graph API](https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#op-fusion)".

While there are two entry points to the graph API (i.e. backend and frontend), it is expected that most users will use the FE API. Reasons being:

- FE API is less verbose without loss of control. All functionality accessible through the backend API is also accessible through the FE API.
- FE API adds functionality on top of the backend API, like errata filters and autotuning.

Also, for those using backend API, FE API source and samples can serve as reference implementation.

FE v1.0 API extends the groundwork of earlier versions. In FE v1.0 API, users can describe multiple operations that form subgraph through a persistent cudnn_frontend::graph::Graph object. Unlike the FE v0.x API, users dont need to worry about specifying shapes and sizes of the intermediate virtual tensors. For detailed information of FE v1.0 API, see README.FE.v1.0.md. 

Additionally, FE v1.0 API provides python bindings to all API through pybind11. It is recommended that new users of cuDNN start with the frontend v1.0 API. See `samples/cpp` and `samples/python` for more details on its usage.

## Usage
In order to include the entire library, include the cudnn_frontend header file `include/cudnn_frontend.h` into your compilation unit.

## Build:

### Dependencies
Minimum cudnn version required is 8.5.0
Minimum python version needed 3.6
The python installation requires development package which can be installed by running `apt-get install python-dev`.

### C++ API
Provide CUDA according to: https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html  

CUDNN_PATH has the cudnn installation:
- Headers are in CUDNN_PATH/include.
- Libraries are in CUDNN_PATH/lib or CUDNN_PATH/lib64 or CUDNN_PATH/lib/x64.

From Project Root,

```
mkdir build; cd build
cmake -DCUDNN_PATH=/path/to/cudnn -DCUDA_PATH=/path/to/cuda  ../
cmake --build . -j16
bin/samples
```

Skip building samples by providing `CUDNN_FRONTEND_BUILD_SAMPLES=OFF` as cmake parameter.  
Skip building python bindings by providing `CUDNN_FRONTEND_BUILD_PYTHON_BINDINGS=OFF` as cmake parameter.

### Python API
Install FE python API by running: `python setup.py build`.  
To provide a custom CUDA, export environment variable: `CUDA_PATH`.  
To provide a custom CUDNN, export environment variable: `CUDNN_PATH`.  

NOTE: Only v1.0 API is exposed via python bindings.

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
No external contribution to this repository is accepted. Please create an issue in github for bugs or feature requests.

## Feedback
Support, resources, and information about cuDNN can be found online at https://developer.nvidia.com/cudnn. 

For questions or to provide feedback, please contact cuDNN@nvidia.com.
