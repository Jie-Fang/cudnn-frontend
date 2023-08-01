cmake_minimum_required(VERSION 3.18)

# Find the cuda compiler
find_package(CUDAToolkit)

Include(FetchContent)

# Fetch and build catch2
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.3.2
)
FetchContent_MakeAvailable(Catch2)

# Find cudnn
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/cuDNN.cmake)

add_executable(
    unittests

    serialize.cpp
)

if (MSVC)
    target_compile_options(
        unittests PRIVATE
        /W4 /WX # warning level 3 and all warnings as errors
        /wd4100 # allow unused parameters
        /wd4458 # local hides class member (currently a problem for all inline setters)
        /wd4505 # unreferenced function with internal linkage has been removed
        /wd4101 /wd4189 # unreferenced local
    )
else()
    target_compile_options(
        unittests PRIVATE
        -Wall
        -Wextra
        -Werror
        -Wno-unused-function
    )
endif()

target_link_libraries(
    unittests
    cudnn_frontend
    Catch2::Catch2WithMain
    CUDA::cudart
    CUDNN::cudnn_all
)

# cuDNN dlopen's its libraries
# Add all libraries in link line as NEEDED
set_target_properties(
    unittests
    PROPERTIES
    LINK_WHAT_YOU_USE TRUE
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
