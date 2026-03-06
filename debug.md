# SM70 Flash-Attn 调试记录（清理版）

日期：2026-03-07

## 目标
- 在 V100 / SM70 上移除 `!Is_even_MN` fallback，同时保证 `./test.sh` 全部维度通过。

## 已确认事实
- `tests/simple_test.py` 覆盖的 `head_dim` 顺序是：`[32, 64, 96, 128, 192, 256]`。
- 原始 forward dispatch 中：
  - `d=32/64` 使用 `kNWarps=8`
  - `d=96/128/192/256` 使用 `kNWarps=4`
- 在“禁用 fallback + 8-warp 路径”状态下，`./test.sh` 的已确认结果是：
  - `d=32` 失败（误差约 `1.146484`）
  - `d=64` 失败（误差约 `8.562500`）
  - `d=96/128/192/256` 通过（误差约 `0.000977`）
- `row_keyed_block_reduce` 与 `atomic_max_float` 在仓库内无调用，已删除。
- `softmax.h` 中一批历史 helper（`thread_reduce_ / reduce_* / scale_apply_exp2 / max_scale_exp2_sum`）在当前实现下无调用，已删除。

## 当前代码清理（本轮）
- `csrc/flash_attn/src/flash_fwd_launch_template.h`
  - `d=32/64` 的 launch 已改为 `kNWarps=4`
  - 在 `run_flash_fwd` 和 `run_flash_splitkv_fwd` 添加静态断言：
    - `static_assert(Kernel_traits::kNWarps == 4, ...)`
- `csrc/flash_attn/src/flash_fwd_kernel.h`
  - 移除 `kNWarps==8` 条件分支写法（`kNWarps == 8 ? 6 : 0` 等）
  - 输出写回和 early-exit 清零路径统一为固定 4-warp 逻辑（当前使用 `warp_id == 0` 作为 canonical writer）

## 待验证
- 已完成验证：
  - `./build.sh`：成功。
  - `./test.sh`：全部通过。
    - `d=32`：`Max Diff = 0.000488`
    - `d=64`：`Max Diff = 0.000977`
    - `d=96`：`Max Diff = 0.000977`
    - `d=128`：`Max Diff = 0.000977`
    - `d=192`：`Max Diff = 0.000977`
    - `d=256`：`Max Diff = 0.000488`

## 本轮额外清理
- 删除了 `flash_fwd_kernel.h` 中必不执行的 dead fallback 分支：
  - 原条件：`if constexpr (false && !Is_even_MN && ...)`
  - 删除后不影响构建与测试结果。
- 提交前精简审阅：
  - 清理 `flash_fwd_kernel.h` / `flash_fwd_launch_template.h` 中无效调试注释与注释掉的旧实验代码。
  - 未改动执行逻辑；不影响前述测试结论。
