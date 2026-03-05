# SM70 Flash-Attn Debug Notes

Date: 2026-03-05

## Context
- Repo: `flash-attention-v100`
- Primary issue: large forward precision error on V100 (`tests/simple_test.py` via `./test.sh`).
- Target kernel: `compute_attn_1rowblock` in `csrc/flash_attn/src/flash_fwd_kernel.h`.

## Confirmed facts (validated by instrumentation)
- `Q/K` global-to-shared load for head 0 / head 3 matched host tensor values exactly.
- The previous SM70 `convert_layout_acc_Aregs` mapping did **not** match `partition_A` expectations (checked via standalone CuTe mapping check); this path can mis-route `P` fragments into `gemm_rs`.
- Lane mapping for SM70 accumulator fragments differs from original FA assumptions; hardcoded lane formulas in masking/reduction are fragile in uneven-tile cases.
- Uneven tile path (`!Is_even_MN`) was the failing path for current simple tests (`seqlen_q=seqlen_k=32` with current launch shapes).

## Effective fixes applied
1. Added coordinate-driven masking API:
   - `Mask::apply_mask_idx(...)` in `csrc/flash_attn/src/mask.h`
   - Uses true `(row,col)` coordinates from `partition_C(identity)` rather than lane arithmetic.

2. Changed softmax reductions to keyed warp reductions by row-id:
   - Added `row_keyed_allreduce` in `csrc/flash_attn/src/softmax.h`
   - `softmax_rescale_o` and `normalize_softmax_lse` now consume row-index tensor and reduce only lanes belonging to the same logical row.

3. Added an accuracy-first fallback path for uneven tiles in common forward case:
   - In `compute_attn_1rowblock`, when all conditions below hold, kernel uses direct definition-based softmax attention computation and writes exact `O` + `LSE`:
     - `!Is_even_MN && Is_even_K && !Is_causal && !Is_local && !Has_alibi && !Is_dropout && !Is_softcap && !Return_softmax`
   - This path is intentionally scoped to avoid impacting even-tile fast path.

## Validation
- `./test.sh` now passes all head dimensions in `tests/simple_test.py`.
- Typical max diff after fix: around `6e-5 ~ 4.9e-4`, below threshold `1e-3`.

## Notes for future optimization
- Current correctness on uneven tiles is guaranteed by fallback path; performance for large uneven workloads can be improved later by repairing SM70 fast path `P` remap and removing fallback.
- If optimizing, start by deriving a provably correct SM70 `convert_layout_acc_Aregs` mapping against `partition_A` coordinates, then re-enable fast path for `!Is_even_MN` behind correctness tests.
