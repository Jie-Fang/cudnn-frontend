cmake_minimum_required(VERSION 3.18)

Include(FetchContent)

# Find the cuda compiler
find_package(CUDAToolkit)

# Find cudnn
include(${CMAKE_SOURCE_DIR}/cmake/cuDNN.cmake)

# Find Python
find_package(Python COMPONENTS Interpreter REQUIRED)

# detect virtualenv and set Pip args accordingly
if(DEFINED ENV{VIRTUAL_ENV} OR DEFINED ENV{CONDA_PREFIX})
set(_pip_args)
else()
set(_pip_args "--user")
endif()

# install cuda-python PyPI Python package using pip
execute_process(COMMAND ${Python_EXECUTABLE} -m pip install ${_pip_args} cuda-python)

pybind11_add_module(
    cudnn_frontend_blocks
    
    cudnn_frontend_blocks_py.cpp
    cudnn_frontend_pygraph.cpp
    cudnn_frontend_properties.cpp
)

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
