# Plan: Add THD (Variable Sequence Length) Support to FP8 SDPA Test

## Background

THD (ragged/packed) layout packs variable-length sequences contiguously in memory as `[total_tokens, H, D]`
instead of padded `[B, H, S_max, D]`. This avoids wasting compute on padding tokens. The FP16 test (`fp16.py`)
already supports THD; we need to add equivalent support to the FP8 test (`fp8.py`).

## Key Findings from Code Exploration

1. **FP8 Forward**: The `sdpa_fp8()` Python API goes through `sdpa_internal()` which uses `SDPA_attributes` —
   the same attributes class as FP16 SDPA. This class **already supports ragged offsets** via
   `tensor.set_ragged_offset()`. So THD forward should work with no C++ changes.

2. **FP8 Backward**: The `sdpa_fp8_backward()` Python API uses `SDPA_fp8_backward_attributes` which currently
   **lacks** `max_total_seq_len_q`/`max_total_seq_len_kv` fields. However, ragged offsets are set on the tensors
   themselves (not on the attributes), so the backward may still work if the backend kernel supports it.
   We'll test forward-only first, then attempt backward.

3. **UID allocation**: The FP8 forward `GraphFwdUid` already has ragged offset UIDs (17-21) reserved but unused.
   The backward `GraphBwdUid` does **not** have ragged offset UIDs yet.

4. **Reference implementation** (`fp8_ref.py`): Currently assumes BSHD layout `(b, s, h, d)`. For THD, we'll
   convert packed→uniform before calling the reference, then convert results back — same pattern as `fp16.py`.

## Changes

### 1. Add backward ragged offset UIDs to `GraphBwdUid` (fp8.py)

Add ragged offset UIDs to `GraphBwdUid`:
```python
q_ragged_offset = 125
k_ragged_offset = 126
v_ragged_offset = 127
o_ragged_offset = 128
stats_ragged_offset = 129
dO_ragged_offset = 130
dQ_ragged_offset = 131
dK_ragged_offset = 132
dV_ragged_offset = 133
```

### 2. Modify `generate_graph_fwd()` to support ragged tensors (fp8.py)

Add an `is_ragged` parameter. When `is_ragged=True`:
- Use BSHD strides: `(s*h*d, d, h*d, 1)` for Q, K, V, O
- Create ragged offset graph tensors with dim `(b+1, 1, 1, 1)`, dtype INT64
- Call `q.set_ragged_offset(q_ragged_offset)` etc. on Q, K, V
- Set `o.set_ragged_offset(o_ragged_offset)` on output
- Set ragged stats stride: `(s_qo * h_q, 1, h_q, 1)` instead of `(s_qo * h_q, s_qo, 1, 1)`
- If training, set `stats.set_ragged_offset(stats_ragged_offset)`
- Set `use_padding_mask=True` (required for ragged)

### 3. Modify `generate_graph_bwd()` to support ragged tensors (fp8.py)

Add `is_ragged`, `max_t_q`, `max_t_kv` parameters. When `is_ragged=True`:
- Use BSHD strides for Q, K, V, O, dO, dQ, dK, dV
- Create ragged offset graph tensors
- Call `set_ragged_offset()` on Q, K, V, O, dO, stats, dQ, dK, dV
- Set `use_padding_mask=True`
- Pass `seq_len_q` and `seq_len_kv` tensors

### 4. Modify `exec_sdpa_fp8()` tensor allocation for ragged mode (fp8.py)

When `cfg.is_ragged=True`:
- Compute `max_t_q = max(64, ((sum(seq_len_q) + 63) // 64) * 64)`, similarly for `max_t_kv`
- Allocate Q as `(max_t_q, h_q, d_qk)`, K as `(max_t_kv, h_k, d_qk)`, etc.
- Compute ragged offsets using `prefix_sum(seq_len_q_gpu) * h * d`
- Pack data into THD format using `convert_uniform_to_packed()`
- Add ragged offset tensors to variant pack

### 5. Modify reference comparison for ragged mode (fp8.py)

When `cfg.is_ragged=True`:
- Before calling `compute_ref()` / `compute_ref_backward()`: convert packed→uniform using
  `convert_packed_to_uniform()`
- After getting reference results: convert uniform→packed using `convert_uniform_to_packed()`
- Compare packed tensors directly

### 6. Test parametrization

Add THD test cases to the existing FP8 test parametrization (likely in `test_sdpa_thd.py` or through
random config). Start with a focused set:
- Forward-only THD with FP8 E4M3
- Backward THD with FP8 (if backend supports it)
- GQA + THD with FP8

## Execution Order

1. Modify `fp8.py`: Add UIDs, modify graph generation functions, modify `exec_sdpa_fp8()`
2. Add test cases
3. Run forward-only THD test to validate
4. Attempt backward THD test
5. If backward fails due to missing `max_total_seq_len` in FP8 backward attributes, we may need to:
   - Add those fields to `SDPA_fp8_backward_attributes` in `graph_properties.h`
   - Wire them through `sdpa_fp8_backward()` in `sdpa.cpp`
   - Or skip backward THD for now with a TODO
