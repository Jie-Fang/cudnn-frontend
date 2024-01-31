# FE - Programming Samples

## Python Interface Samples
Samples leveraging FE's Python interface are located in [samples/python](/samples/python/).
* [00_basic_gemm](/samples/python/00_basic_gemm.ipynb)
    Walks through pycudnn installation and then defining, building, executing a GEMM graph.

* [01_epilogue](/samples/python/01_epilogue.ipynb)
    Shows how to easily fuse elementwise functions to a GEMM graph.

* [02_caching](/samples/python/02_caching.ipynb)
    Shows how to cache already built cudnn graphs for faster execution in the future.

* [03_flash_attention](/samples/python/03_flash_attention.ipynb)
    Shows how to run causal self attention with dropout in forward and backward pass.

## C++ Interface Samples
Samples leveraging FE's C++ interface are located in [samples/cpp](/samples/cpp/).

## [Deprecated] C++ v0.x Interface Samples
Samples leveraging FE's C++ 0.x interface are located in [samples/legacy_samples](/samples/legacy_samples/).
