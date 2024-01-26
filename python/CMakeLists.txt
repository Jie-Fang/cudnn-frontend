cmake_minimum_required(VERSION 3.18)

Include(FetchContent)

# Fetch and build dlpack
FetchContent_Declare(
  dlpack
  GIT_REPOSITORY https://github.com/dmlc/dlpack
  GIT_TAG        v0.8
)
FetchContent_MakeAvailable(dlpack)

find_package(Python 3.7 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Add a library using FindPython's tooling (pybind11 also provides a helper like
# this)
python_add_library(
    compiled_module
    
    MODULE
    pycudnn.cpp
    properties.cpp

    pygraph/pygraph.cpp
    pygraph/norm.cpp
    pygraph/sdpa.cpp
    pygraph/pointwise.cpp

    WITH_SOABI
)
target_link_libraries(compiled_module PRIVATE pybind11::headers)

target_link_libraries(
    compiled_module
    
    PRIVATE cudnn_frontend
    PRIVATE dlpack
)

set_target_properties(
    compiled_module

    PROPERTIES
    LINK_FLAGS "-Wl,--no-as-needed"
    LINK_WHAT_YOU_USE TRUE
)

# using python bindings directly with cmake build system is not supported
# Temparorily use below parameter
if(CUDNN_FRONTEND_KEEP_PYBINDS_IN_BINARY_DIR)
    set_target_properties(
        compiled_module

        PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/cudnn
    )
    file(COPY ${CMAKE_SOURCE_DIR}/python/cudnn DESTINATION ${CMAKE_BINARY_DIR})
endif()