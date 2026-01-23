# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

cuDNN Frontend (FE) is a modern, open-source entry point to the NVIDIA cuDNN library. It provides both a C++ header-only library and Python bindings to access the cuDNN Graph API and open-source kernels. The project supports building complex deep learning operations as computational graphs with automated optimization and execution.

## Build Commands

### C++ Build
```bash
# Standard build with samples and tests
mkdir build && cd build
cmake -DCUDNN_PATH=/path/to/cudnn -DCUDAToolkit_ROOT=/path/to/cuda ../
cmake --build . -j16

# Run C++ tests
./bin/tests

# Run C++ samples
./bin/samples
```

CMake options:
- `CUDNN_FRONTEND_BUILD_SAMPLES=ON/OFF` - Build C++ samples (default: ON)
- `CUDNN_FRONTEND_BUILD_TESTS=ON/OFF` - Build C++ tests (default: ON)
- `CUDNN_FRONTEND_BUILD_PYTHON_BINDINGS=ON/OFF` - Build Python bindings (default: OFF)
- `CUDNN_FRONTEND_SKIP_JSON_LIB=ON/OFF` - Skip nlohmann/json.hpp (default: OFF)

### Python Build
```bash
# Install from PyPI
pip install nvidia_cudnn_frontend

# Build from source
pip install -v .
# or with editable install
pip install -e .

# Environment variables for custom paths
export CUDAToolkit_ROOT=/path/to/cuda
export CUDNN_PATH=/path/to/cudnn
```

## Testing

### Python Tests
Python tests use pytest with hierarchical test levels (L0-L4). By default, only L0 tests run.

```bash
# Run all L0 tests (default, quick tests)
cd test/python
pytest

# Run specific test levels
pytest -m L1  # Medium complexity tests
pytest -m L2  # More comprehensive tests
pytest -m L3  # Extended tests
pytest -m L4  # Stress tests

# Run specific test file
pytest test_mhas.py

# Run with verbose output
pytest -v

# Run specific test within file
pytest test_matmul_bias_relu.py::test_function_name
```

Test levels are defined in `test/python/pytest.ini`. Use the appropriate marker when adding new tests.

### C++ Tests
```bash
cd build
./bin/tests
```

C++ tests use Catch2 framework. Tests are in `test/cpp/`.

## Code Formatting

The project uses automated code formatting:

- **C++**: clang-format (config in `.clang-format`)
- **Python**: black with line length 160 (config in `.pre-commit-config.yaml`)

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run formatters manually
pre-commit run --all-files
```

## Architecture

### Repository Structure

```
include/cudnn_frontend/        # C++ header-only library
├── backend/                   # Backend descriptor and execution helpers
├── node/                      # Operation node implementations (conv, matmul, attention, etc.)
├── graph_interface.h          # Main Graph API
├── graph_properties.h         # Graph property definitions
├── graph_helpers.h            # Helper functions for graph operations
├── node_interface.h           # Node base interface
├── plans.h                    # Execution plan management
└── context.h                  # cuDNN context/handle management

python/cudnn/                  # Python bindings
├── _compiled_module.*.so      # Compiled pybind11 module
├── wrapper.py                 # High-level Python wrapper with Graph class
├── api_base.py                # Base API classes
├── gemm_amax/                 # Open-source GEMM+Amax kernel
├── gemm_swiglu/               # Open-source GEMM+SwiGLU kernel
└── native_sparse_attention/   # Open-source Native Sparse Attention

samples/
├── cpp/                       # C++ usage examples (convolution, matmul, sdpa, etc.)
└── python/                    # Python notebooks and examples

test/
├── cpp/                       # C++ unit tests (Catch2)
└── python/                    # Python tests (pytest)
```

### Core Concepts

#### Graph API (v1.0)
The v1.0 API uses a functional, composable style where operations take input tensors and return output tensors. The workflow:

1. **Create Graph**: Instantiate `cudnn_frontend::graph::Graph` and set global properties (data types, compute precision)
2. **Define Tensors**: Create input/output tensors with dimensions, strides, data types
3. **Add Operations**: Chain operations (conv, matmul, pointwise, attention, etc.) - outputs become inputs to next ops
4. **Validate**: Check graph soundness, infer unspecified properties
5. **Build**: Lower graph to cuDNN backend descriptors
6. **Create Plans**: Generate execution plans based on heuristics
7. **Execute**: Run graph with device pointers

Unlike v0.x API, v1.0 automatically infers intermediate tensor properties and provides simpler composition.

#### Node Types
Operations are implemented as node classes in `include/cudnn_frontend/node/`:
- **Convolution**: `conv_fprop.h`, `conv_dgrad.h`, `conv_wgrad.h`
- **Matrix Multiplication**: `matmul.h`, `matmul_fp8.h`
- **Attention**: `scaled_dot_product_flash_attention.h`, `sdpa_fp8_bwd.h`
- **Normalization**: `batchnorm.h`, `layernorm.h`, `instancenorm.h`, `rmsnorm.h`
- **Pointwise**: `pointwise.h` (relu, gelu, add, mul, etc.)
- **Advanced**: `moe_grouped_matmul.h`, `block_scale_quantize.h`, `slice.h`, `resample.h`

Each node defines attributes (via `*_attributes` classes) and operation semantics.

#### Backend Integration
The `backend/` directory handles:
- **backend_descriptor.h**: Wraps cuDNN backend descriptors
- **execution_helpers.h**: Device memory management, execution orchestration
- **kernel_cache.h**: Caches compiled kernels across executions
- **plan_helpers.h**: Execution plan filtering and selection

#### Python Wrapper
`python/cudnn/wrapper.py` provides the `Graph` context manager that:
- Simplifies graph creation with automatic validation/compilation
- Integrates with PyTorch via DLPack for zero-copy tensor sharing
- Manages cuDNN handles and workspace allocation
- Maps named tensors to operation inputs/outputs

### Open-Source Kernels (OSS)
The project now ships reference kernel implementations:
- **GEMM + Amax** (`python/cudnn/gemm_amax/`): FP8 matrix multiplication with absolute maximum
- **GEMM + SwiGLU** (`python/cudnn/gemm_swiglu/`): Fused SwiGLU activation
- **Native Sparse Attention** (`python/cudnn/native_sparse_attention/`): Hardware-aligned sparse attention

These are pure Python/C++ implementations meant for education and customization.

## Common Development Patterns

### Adding a New Operation Node
1. Create header in `include/cudnn_frontend/node/` (e.g., `my_operation.h`)
2. Define `*_attributes` class with setters for operation parameters
3. Implement node creation logic and backend descriptor mapping
4. Include in `graph_interface.h` and expose as graph method
5. Add Python binding in `python/pygraph/pygraph.cpp` (pybind11)
6. Write tests in `test/cpp/` and `test/python/`

### Working with Tests
- Python tests often use `torch` for tensor creation and reference implementations
- Tests validate numerical accuracy against reference (typically PyTorch ops)
- Use appropriate test markers (L0 for quick smoke tests, L1+ for comprehensive coverage)
- C++ tests focus on API correctness and descriptor validation

### Debugging
Enable logging via environment variables:
```bash
export CUDNN_FRONTEND_LOG_INFO=1
export CUDNN_FRONTEND_LOG_FILE=stdout  # or path to log file
```

Or programmatically via `cudnn_frontend::isLoggingEnabled()`.

## Version and Release
- Current version: 1.18.0 (see `CMakeLists.txt` project version)
- Minimum cuDNN backend: 8.5.0 (v1.0 API requirement)
- Recommended cuDNN backend: 9.17.0+ for latest features
- Release notes in root directory: `1.15.0.md`, `1.16.0.md`, `1.17.0.md`, etc.
- Main development branch: `develop`
- Branch naming convention: `<name>-issue-<issue_number>`

## Contributing
- File issues at https://github.com/NVIDIA/cudnn-frontend/issues
- PRs should target `develop` branch
- Include unit tests for new functionality
- Run formatters before committing (pre-commit hooks handle this)
- Follow existing code patterns in node implementations
