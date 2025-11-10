# GEMM + SwiGLU (SM100)

**This is an experimental API and subject to change.**

## Overview

**GEMM + SwiGLU fusion**: A persistent, batched dense GEMM fused with a SwiGLU epilogue on NVIDIA Blackwell GPUs (SM100+), implemented with CUTLASS/CUTE. It produces both the full GEMM output `C` and a SwiGLU-projected tensor `Glu` in a single pass.

### Shapes

- Inputs:
  - `A`: shape `(M, K, L)`
  - `B`: shape `(N, K, L)`
- Outputs:
  - `C`: shape `(M, N, L)` — full GEMM result
  - `Glu`: shape `(M, N/2, L)` — SwiGLU-projected result

    `L` is the batch dimension.

### Equations

- GEMM (per batch l):

$$
C[m, n, l] = \alpha \sum_{k} A[m, k, l] \, B[n, k, l]
$$

- SwiGLU epilogue (performed by pairing 32-column blocks along `N`):

  Let block size `G = 32`. For each pair of consecutive 32-wide column blocks in `C`:
  - Input block:  `X_b = C[:, 2*b*G : 2*b*G + G, :]`
  - Gate block:   `G_b = C[:, 2*b*G + G : 2*b*G + 2*G, :]`

$$
\mathrm{Glu}[:, \, bG:(b+1)G, \, :] = X_b \cdot \operatorname{swish}(G_b), \quad
\operatorname{swish}(x) = x \cdot \sigma(x)
$$

Notes:
- The `alpha` scaling is applied before the SwiGLU; both `X_b` and `G_b` are from the scaled GEMM results.
- `C` stores the entire scaled GEMM output (both input and gate blocks), while `Glu` stores the fused SwiGLU-projected result with half the columns.

### Diagram

```
 A (MxKxL)     B (NxKxL)
      |              |
      \__ GEMM (per L): C = alpha * A @ B  _________________________
                            C (MxNxL)                               \
                            |                                        \
                            |  Pair 32-col blocks along N:           |
                            |   [X0 | G0 | X1 | G1 | ...]           |
                            |    |     |    |     |                  |
                            |    \_swish(G_b)<____/                  |
                            |           |                            |
                            \___ Glu[:, b*32:(b+1)*32, :] = X_b * swish(G_b)
                                              Glu (MxN/2xL)
```

---

## API Usage

### High-level wrapper
```python
c, glu = gemm_swiglu_wrapper_sm100(
    a_tensor,
    b_tensor,
    alpha=1.0,
    c_major="n",
    c_dtype=torch.float32,
    glu_dtype=torch.float16,
    acc_dtype=torch.float32,
    use_2cta_instrs=False,
    mma_tiler_mn=(128, 128),
    cluster_shape_mn=(1, 1),
    stream=None,
)
```

### Class API
```python
gemm = GemmSwigluSm100(
    sample_a,
    sample_b,
    sample_c,
    sample_glu,
    alpha=1.0,
    acc_dtype=torch.float32,
    use_2cta_instrs=False,
    mma_tiler_mn=(128, 128),
    cluster_shape_mn=None,
)
assert gemm.check_support()
gemm.compile(
    current_stream=None
)
gemm.execute(
    a_tensor,
    b_tensor,
    c_tensor,
    glu_tensor,
    alpha=1.0,
    current_stream=None
    skip_compile=False
)
```
---

## Parameters

### Input/Output tensors
- Input tensor  **A**: `a_tensor` (wrapper) or `sample_a`, `a_tensor` (class)
  - Shape: `(M, K, L)`
  - Stride: `(1, M, M·K)` for `m`-major or `(K, 1, M·K)` for `k`-major
  - Dtype: `{float16, bfloat16, float32, float8_e4m3fn, float8_e5m2}`
- Input tensor **B**: `b_tensor` (wrapper) or `sample_b`, `b_tensor` (class)
  - Shape: `(N, K, L)`
  - Stride: `(1, N, N·K)` for `n`-major or `(K, 1, N·K)` for `k`-major
  - Dtype: Must match `A`
- Output tensor **C**: return value (wrapper) or `sample_c`, `c_tensor` (class)
  - Shape: `(M, N, L)`
  - Stride: `(1, M, M·N)` for `m`-major or `(N, 1, M·N)` for `n`-major. Provided as `c_major` argument for wrapper
  - Dtype: `{float32, float16, bfloat16}` if `acc_dtype == float32`, `{float16, bfloat16}` if `acc_dtype == float16`. Provided as `c_dtype` argument for wrapper
- Output tensor **Glu**: return value (wrapper) or `sample_glu`, `glu_tensor` (class)
  - Shape: `(M, N/2, L)`
  - Stride: `(1, M, M·N/2)` for `m`-major or `(N/2, 1, M·N/2)` for `n`-major. Must match with `C`
  - Dtype: `{float16, bfloat16}`. Provided as `glu_dtype` argument for wrapper

### Common parameters
- `alpha: float`
  - Scalar multiplier applied to the GEMM result before SwiGLU.
  - Default: `1.0`
- `acc_dtype: torch.dtype`
  - Accumulator dtype. Allowed: `{float32, float16}`.
  - Default: `torch.float32`
- `use_2cta_instrs: bool`
  - Enables 2-CTA MMA instructions. Required for `mma_tiler_mn[0] == 256`.
  - Default: `False`
- `mma_tiler_mn: Tuple[int, int]`
  - Kernel tile size `(TILE_M, TILE_N)`. Default: `(128, 128)`
  - `TILE_M ∈ {128, 256}`
  - `TILE_N ∈ {32, 64, ..., 224, 256}`
- `cluster_shape_mn: Tuple[int, int] | None`
  - Thread Block cluster shape `(CLUSTER_M, CLUSTER_N)`
  - Constraints: positive powers of 2, `CLUSTER_M*CLUSTER_N ≤ 16`.
  - Default: `(1,1)` if `use_2cta_instrs==False` else `(2,2)`.
- CUDA stream (`current_stream` in class API, `stream` in wrapper)


### Wrapper-specific parameters: `gemm_swiglu_wrapper_sm100`

- `a_tensor`, `b_tensor`, `c_tensor`, `glu_tensor`: see Input/Output tensors
- `c_major: str`: see Input/Output tensors. Default: `"n"`
- `c_dtype: torch.dtype`: see Input/Output tensors. Default: `torch.float32`
- `glu_dtype: torch.dtype`: see Input/Output tensors. Default: `torch.float16`

### Class-specific parameters

#### `GemmSwigluSm100` (constructor)

- `sample_a`, `sample_b`, `sample_c`, `sample_glu` — see Input/Output tensors

#### `GemmSwigluSm100.execute`
- `a_tensor`, `b_tensor`, `c_tensor`, `glu_tensor` — see Input/Output tensors. Must have same layout as sample tensors provided in constructor.
- `skip_compile: bool` — Default: `False`
 
---

## Support surface and constraints

### Layouts and strides

- `C` and `Glu` must have the same major order.
- `A`, `B`, `C` must be 16-byte aligned along the contiguous dimension.

### Dtypes

- `A`/`B` must have the same dtype.
- `C ∈ {float8_e4m3fn, float8_e5m2}` is currently disabled
- `acc_dtype == float16` is only supported with `A/B ∈ {float16, float8_e4m3fn, float8_e5m2}`
- `C ∈ {float32}` requires `acc_dtype == float32` and `use_2cta_instrs == True`

### Tiling and cluster

- using `TILE_M == 256` requires `use_2cta_instrs=True`.
- If `use_2cta_instrs == False`, `cluster_shape_mn` must be `(1, 1)`.
- If `mma_tiler_mn == (128, 128)` and `cluster_shape_mn == (1, 1)`, `c_major` must be `"m"`.
- If `mma_tiler_mn != (128, 128)`, `c_major` must be `"m"`.
- If `TILE_M == 128` and `cluster_shape_mn != (1, 1)`, `mma_tiler_mn` must be exactly `(128, 128)`.
- `TILE_M == 256` and `C ∈ {float32}` is currently disabled.
- If `use_2cta_instrs == True`, `CLUSTER_M` must be divisible by 2

### Environment

- Requires CUDA with SM100+ compute capability

---

## Usage examples

For usage examples, see test cases in `test/python/fe_api/test_gemm_swiglu.py`
