
# cuDNN FrontEnd(FE)

**cuDNN FE** is the modern, open-source entry point to the NVIDIA cuDNN library and high performance open-source kernels. It provides a C++ header-only library and a Python interface to access the powerful cuDNN Graph API and open-source kernels.

## 🚀 Embracing Open Source

We are making a major transition to embrace open source. FE is no longer just a wrapper; it is becoming a hub for shipping high-performance, open-source kernels directly to users.

We are now shipping **OSS kernels** written in Python, allowing you to inspect, modify, and contribute to the core logic. Check out our latest implementations:

*   **[GEMM + Amax](https://github.com/NVIDIA/cudnn-frontend/tree/main/python/cudnn/gemm_amax):** Optimized FP8 matrix multiplication with absolute maximum calculation.
*   **[GEMM + SwiGLU](https://github.com/NVIDIA/cudnn-frontend/tree/main/python/cudnn/gemm_swiglu):** High-performance implementation of the SwiGLU activation fused with GEMM.

**This is just the beginning.** We are committed to releasing more open-source kernels and features in the future. We believe the future of deep learning optimization is collaborative, and we want you to be a part of it.

## Key Features

*   **Unified Graph API:** Create reusable, persistent `cudnn_frontend::graph::Graph` objects to describe complex subgraphs.
*   **Ease of Use:** Simplified C++ and Python bindings (via `pybind11`) that abstract away the boilerplate of the backend API.
*   **Performance:** Built-in autotuning and support for the latest NVIDIA GPU architectures.

## Installation

### 🐍 Python

The easiest way to get started is via pip:

```bash
pip install nvidia_cudnn_frontend
```

**Requirements:**
*   Python 3.8+
*   NVIDIA driver and CUDA Toolkit

### ⚙️ C++ (Header Only)

Since the C++ API is header-only, integration is seamless. Simply include the header in your compilation unit:

```cpp
#include <cudnn_frontend.h>
```

Ensure your include path points to the `include/` directory of this repository.

## Building from Source

If you want to build the Python bindings from source or run the C++ samples:

**1. Dependencies**
*   `python-dev` (e.g., `apt-get install python-dev`)
*   Dependencies listed in `requirements.txt` (`pip install -r requirements.txt`)

**2. Python Source Build**
```bash
pip install -v git+https://github.com/NVIDIA/cudnn-frontend.git
```
*Environment variables `CUDAToolkit_ROOT` and `CUDNN_PATH` can be used to override default paths.*

**3. C++ Samples Build**
```bash
mkdir build && cd build
cmake -DCUDNN_PATH=/path/to/cudnn -DCUDAToolkit_ROOT=/path/to/cuda ../
cmake --build . -j16
./bin/samples
```

## Documentation & Examples

*   **Developer Guide:** [Official NVIDIA Documentation](https://docs.nvidia.com/deeplearning/cudnn/frontend/v1.9.0/developer/overview.html)
*   **C++ Samples:** See `samples/cpp` for comprehensive usage examples.
*   **Python Samples:** See `samples/python` for pythonic implementations.

## 🤝 Contributing

We strictly welcome contributions! Whether you are fixing a bug, improving documentation, or optimizing one of our new OSS kernels, your help makes cuDNN better for everyone.

1.  Check the [Contribution Guide](CONTRIBUTING.md) for details.
2.  Fork the repo and create your branch.
3.  Submit a Pull Request.

## Debugging

To view the execution flow and debug issues, you can enable logging via environment variables:

```bash
# Log to stdout
export CUDNN_FRONTEND_LOG_INFO=1
export CUDNN_FRONTEND_LOG_FILE=stdout

# Log to a file
export CUDNN_FRONTEND_LOG_INFO=1
export CUDNN_FRONTEND_LOG_FILE=execution_log.txt
```

Alternatively, you can control logging programmatically via `cudnn_frontend::isLoggingEnabled()`.

## License

This project is licensed under the [MIT License](LICENSE).