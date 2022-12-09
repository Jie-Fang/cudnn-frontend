cmake_minimum_required(VERSION 3.18)

Include(FetchContent)

# Find the cuda compiler
find_package(CUDAToolkit)

# Find cudnn
include(${CMAKE_SOURCE_DIR}/cmake/cuDNN.cmake)

pybind11_add_module(cudnn_frontend_blocks cudnn_frontend_blocks_py.cpp)

target_link_libraries(
    cudnn_frontend_blocks
    PRIVATE cudnn_frontend

    PRIVATE CUDA::cudart
    PRIVATE CUDA::cublas
    PRIVATE CUDA::nvrtc
    PRIVATE CUDNN::cudnn_all
)

set_target_properties(
    cudnn_frontend_blocks
    PROPERTIES
    LINK_WHAT_YOU_USE TRUE
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib64
)
