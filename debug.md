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

## 2026-03-07：`sP` 重排可行性补充验证（`__shfl_sync` 路线）
- 背景：
  - 目标是评估能否把 forward 中 `rP -> sP -> tOrP` 的 shared memory 重排替换为纯寄存器重排（`__shfl_sync`）。
- 验证方法：
  - 写最小化 CuTe 探针，统计 `partition_C`（写 `sP`）与 `partition_A`（读 `sP`）之间的数据来源关系。
  - 使用当前 SM70 前向配置：
    - `TiledMma = TiledMMA<MMA_Atom<SM70_8x8x4...>, Layout<Shape<_4,_1,_1>>, Tile<_32,_16,_4>>`
    - `kNWarps = 4`
- 已确认事实（实测）：
  - 对 `BM=64, BN=64`：`self=2048, cross=14336, other_warp=12288`
  - 对 `BM=128, BN=64`：`self=4096, cross=28672, other_warp=24576`
  - 对 `BM=128, BN=128`：`self=8192, cross=57344, other_warp=49152`
  - 逐行 warp 归属统计显示：
    - `C_rows_multiwarp=BM/BM`
    - `A_rows_multiwarp=BM/BM`
    - 每行 mask 均为 `0xf`（4 个 warp 全覆盖），即“每行只在一个 warp 内”假设不成立。
- 结论：
  - 在当前映射定义下，`sP` 这一步存在大量跨 warp 数据依赖。
  - 仅靠 `__shfl_sync`（warp 内通信）无法完整替代该重排过程。
  - 若坚持去掉 `smem`，需要更大范围重构（例如更改线程/原子映射或改变第二个 GEMM 的消费布局语义），不属于简单替换。

## 2026-03-07：`simple_test.py` 全场景矩阵验证（seqlen/causal/dropout/local/alibi/varlen）
- 执行环境：`Tesla V100-SXM2-32GB`
- 执行命令：`./build.sh` -> `./test.sh`
- 新版 `tests/simple_test.py` 使用笛卡尔积矩阵：
  - `seqlen_q/seqlen_k`：`(16,16) / (32,48) / (64,64) / (96,80)`
  - `causal ∈ {False, True}`
  - `dropout ∈ {0.0, 0.17}`
  - `local ∈ {False, True}`
  - `alibi ∈ {False, True}`
  - `varlen ∈ {False, True}`
  - 合计 `4 * 2^5 = 128` 个 case

### 已确认事实
- 构建期 `nvcc` 参数包含 `-DFLASHATTENTION_DISABLE_DROPOUT`。
- 因此所有 `dropout=0.17` case 都抛出同一异常：
  - `This flash attention build does not support dropout.`
- 全量结果：
  - 总数：`128`
  - 通过：`54`
  - 失败：`74`
- 非 dropout（`dropout=0`）结果：
  - 通过：`54`
  - 失败：`10`
- 非 dropout 且通过的 case，数值误差范围（相对 FP32 参考实现）：
  - `max_diff <= 0.001953`
  - `mean_diff <= 0.000044`

### 非 dropout 下的 NaN 失败模式（已复现）
- `Sq=64, Sk=64, causal=False, local=False, alibi=False, varlen={False,True}`
  - case_id: `65, 66`
  - 现象：`kernel_output_has_nan_or_inf`
- `Sq=96, Sk=80, causal=False, local=False, alibi={False,True}, varlen={False,True}`
  - case_id: `97, 98, 99, 100`
  - 现象：`kernel_output_has_nan_or_inf`
- `Sq=96, Sk=80, causal=False, local=True, alibi=True, varlen={False,True}`
  - case_id: `103, 104`
  - 现象：`kernel_output_has_nan_or_inf`
- `Sq=96, Sk=80, causal=True, local=True, alibi=True, varlen={False,True}`
  - case_id: `119, 120`
  - 现象：`kernel_output_has_nan_or_inf`

### 结论
- 当前构建配置下，`dropout` 场景不支持（属于编译开关限制，不是测试脚本问题）。
- 除 dropout 外，SM70 前向在多数组合可通过；但存在稳定可复现的 NaN 角落场景，主要集中在较长序列配置。

## 2026-03-07：按要求固定 `dropout=0` 后重测
- 修改：`tests/simple_test.py` 的 case 矩阵不再枚举 `dropout=True`，统一 `dropout=0`。
- 覆盖维度更新为：
  - `seqlen_q/seqlen_k`：`(16,16) / (32,48) / (64,64) / (96,80)`
  - `causal × local × alibi × varlen`
  - 合计 `4 * 2^4 = 64` 个 case
- 执行：`./test.sh`

### 最新结果
- 总数：`64`
- 通过：`54`
- 失败：`10`
- 失败原因全部一致：`kernel_output_has_nan_or_inf`

### 失败模式（dropout 已固定为 0）
- `Sq=64, Sk=64, causal=False, local=False, alibi=False, varlen={False,True}`
  - case_id: `33, 34`
- `Sq=96, Sk=80, causal=False, local=False, alibi={False,True}, varlen={False,True}`
  - case_id: `49, 50, 51, 52`
- `Sq=96, Sk=80, causal=False, local=True, alibi=True, varlen={False,True}`
  - case_id: `55, 56`
- `Sq=96, Sk=80, causal=True, local=True, alibi=True, varlen={False,True}`
  - case_id: `63, 64`

### 结论
- 去掉 dropout 后，先前 `dropout` 编译开关导致的异常已消失。
- 仍存在 10 个稳定可复现的 NaN 前向角落场景，定位范围已收敛到上述组合。
