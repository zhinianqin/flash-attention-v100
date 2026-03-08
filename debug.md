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

## 2026-03-07：SM70 NaN 修复完成（dropout=0 全场景 64/64）
- 修改文件：
  - `csrc/flash_attn/src/flash_fwd_kernel.h`
  - `csrc/flash_attn/src/mask.h`
  - `csrc/flash_attn/src/flash_fwd_launch_template.h`

### 修复点（均通过 `./build.sh` + `./test.sh` 验证）
1. `softmax_rescale_o` 的 `Check_inf` 条件补齐 `!Is_even_MN`：
   - 原先仅在 `Is_causal || Is_local` 时启用，导致非整块边界场景可能把无效分数带入归一化。
   - 修复后在非整块 (`!Is_even_MN`) 也执行无穷值检查，避免后续 NaN 传播。
2. `apply_mask_idx` 在 `!Causal_mask && !Is_local && !Is_even_MN` 路径同时屏蔽越界行/列：
   - 从仅判断 `col_idx >= max_seqlen_k`，
   - 改为 `col_idx >= max_seqlen_k || row_idx >= max_seqlen_q`。
   - 目的：防止 varlen/尾块中越界 query 行参与 softmax 统计。
3. `hdim=64` 的前向 launch tile 从 `BlockM=128` 调整为 `BlockM=64`（`64x64`）：
   - 在 SM70 上该配置更稳，避免先前特定 varlen 组合出现 NaN。

### 验证结果
- 执行顺序：`./build.sh` -> `./test.sh`
- 设备：`Tesla V100-SXM2-32GB`
- 用例总数：`64`（`dropout=0`，覆盖 `seqlen_q/seqlen_k × causal × local × alibi × varlen`）
- 结果：
  - 通过：`64`
  - 失败：`0`
- 数值误差（相对 FP32 参考实现）：
  - `max_diff <= 0.001953`
  - `mean_diff <= 0.000047`（量级稳定在 `1e-5 ~ 1e-4`）

### 结论
- 该轮修复已消除先前 10 个稳定复现的 NaN 角落场景。
- 在当前编译开关（`FLASHATTENTION_DISABLE_DROPOUT`）下，前向 kernel 的 `dropout=0` 全矩阵验证通过。

## 2026-03-07：dropout 随机状态注释逻辑与原版对照
### 对照范围
- 当前仓库：`csrc/flash_attn/flash_api.cpp`、`csrc/flash_attn/flash_api_sparse.cpp`
- 原版仓库：`/root/flash-attention/csrc/flash_attn/flash_api.cpp`、`/root/flash-attention/csrc/flash_attn/flash_api_sparse.cpp`

### 已确认事实
- 在两边仓库中，推理前向路径都存在相同注释块：
  - 注释中明确写 `Commented out because they are not used in inference.`
  - 被注释掉的内容包括 `counter_offset`、`rng_state` 分配与 `params.philox_args` 初始化。
- 同时，两边仓库在支持 dropout 的接口路径（带 `rng_state` 参数的函数）仍保留有效 RNG 赋值逻辑（非注释）：
  - `params.rng_state = ...`
  - `params.philox_args = gen->philox_cuda_state(counter_offset)`
  - `params.rng_state[0/1]` 回写 seed/offset

### 结论
- 你看到“dropout 随机状态代码被注释掉”这一点属实，但这是推理路径上的既有设计，且与原版一致，不是本仓库独有改动。

## 2026-03-07：`compute_attn_1rowblock` warp-stationary 重构调试（本轮新增）

### 背景
- 目标：`compute_attn_1rowblock` 改为 warp-stationary（每个 warp 负责一组 Q 行，K 共享，P 仅 warp 持有并做寄存器重排）。
- 初始状态（本轮起点）现象：
  - `Sq=16,Sk=16` 可通过；
  - 但更长序列大量失败，且 `Sq=32,Sk=48` 呈现明显“warp0 对、warp1 错”。

### 关键定位事实（已通过脚本复现）
1. 在 `Sq=32,Sk=48`（case 17）下，逐行误差分组为：
   - 行 `0~15`（warp0）误差约 `2.9e-05`；
   - 行 `16~31`（warp1）误差约 `2.5e-01`；
   - 说明问题集中在 `warp1+` 路径，而不是整体 softmax/mask 公式。
2. 先前在 `P` 重排处加 `*4` 后，`Sq=16` 小 case 可以恢复，但不是最终解；长序列仍失败。
3. `local_tile` 在 swizzle layout 上虽然 `layout()` 看起来相同，但 `data()` 指针实际有偏移（warp1 对应 +1024 元素），因此 `sQ_warp/sP_warp` 不是主因。
4. 真正有效的修复方向是 `PV` 链路：
   - 原 `gemm_rs` 依赖 `smem_thr_copy_V.get_thread_slice(tidx)`；
   - warp-stationary 分支中 `tOrVt` 来自 `lane_id` 视角，warp1+ 会与 `tidx` 视角失配。

### 最终修复（已验证）
1. `utils.h`：给 `gemm_rs` 增加模板参数 `B_in_regs`（默认 false）
   - `B_in_regs=false`：保持原有 shared->reg copy + prefetch 行为不变；
   - `B_in_regs=true`：跳过 shared->reg copy，直接消费寄存器里的 `B` 片段。
2. `flash_fwd_kernel.h`：在 warp-specialized 分支（`kBlockN==64 && (kWarpRows==16 || kWarpRows==32)`）
   - 新增 `tOsVtWarp = thr_mma_pv.partition_B(sVt)`；
   - 先 `cute::copy(tOsVtWarp, tOrVt)`，再调用 `gemm_rs<B_in_regs=true>`；
   - 非 warp-specialized 分支仍走原 `gemm_rs` 路径。
3. 保留已验证正确的 `P` 寄存器重排公式（双 `__shfl_sync` + lane bit1 选择），并保留 `*4` 补偿。

### 构建提速改动（便于高频迭代）
1. `CMakeLists.txt`
   - 从 `FA2_GEN_SRCS` 里排除 `flash_fwd_split_*.cu`。
2. `flash_api.cpp`
   - `run_mha_fwd` 临时强制 `params.num_splits = 1`，仅走非 split 路径。
3. 效果：`_vllm_fa2_C` 增量构建从原先 27 个对象降到 7 个对象（明显提速）。

### 本轮最终验证结果
- 命令：`./build.sh` 后执行 `./test.sh`。
- 设备：`Tesla V100-SXM2-32GB`。
- 结果：
  - 总用例 `64`，通过 `64`，失败 `0`。
  - 典型数值误差：`max_diff <= 0.001953`，`mean_diff` 维持在 `1e-5 ~ 1e-4`。
- 结论：
  - warp-stationary 改造在当前测试矩阵下已数值正确；
  - `warp1+` 错位问题已通过 `gemm_rs<B_in_regs>` + warp-local `V` 装载修复。

### 备注（2026-03-08）
- 为了保持主线行为一致，调试期的两项“构建提速改动”已在提交前回退：
  - `CMakeLists.txt` 中排除 `flash_fwd_split_*.cu` 的过滤；
  - `flash_api.cpp` 中 `run_mha_fwd` 强制 `num_splits=1`。
- 因此当前代码为：保留功能修复（`gemm_rs<B_in_regs>` 与 warp-local `V` 装载），不保留上述临时提速开关。

## 2026-03-08：`kNWarps==8` 回归失败再次定位与修复（本轮）

### 复现与失败分布
- 在当前 `head_dim=64` 使用 `Flash_fwd_kernel_traits<64,128,64,8>` 的配置下，`./test.sh` 初始结果为 `48/64 PASS`、`16/64 FAIL`。
- 失败 case 固定为：
  - `1,2,9,10,17,18,25,26,33,34,41,42,49,50,57,58`
  - 共同条件：`local=False && alibi=False`（`causal` 两态都可能失败）。
- 典型现象：
  - 小尺寸（`Sq=16,Sk=16`）为大误差（`max_abs` 可到几千）；
  - 大尺寸直接出现 `NaN/Inf`。

### 关键排查事实
1. 仅在 `CMakeLists.txt` 排除 `flash_fwd_split_*.cu` 会导致运行时导入失败：
   - `undefined symbol: run_mha_fwd_splitkv_dispatch<...>`
   - 原因是 `flash_api.cpp` 仍保留 split-kv 调度符号引用。
2. 将 `run_mha_fwd` 临时固定为非 split 路径后可正常导入与测试：
   - `params.num_splits = 1`
   - `TORCH_CHECK(!force_split_kernel, "Split-KV forward path is disabled in this build configuration.")`
3. 对 warp-local `TiledMMA` 做布局探针确认：
   - `QK` 的 `(row,col)` 逻辑点在 warp 内存在固定 `4x` 片段重复（SM70 累加器布局特性）。
   - `LSE` 可出现 `+ln(4)` 常量偏移，这本身不是最终输出错误的直接根因。
4. 真正决定正确性的点在 `PV`：
   - `P` 的 warp 寄存器重排（`__shfl_sync` + `*4`）必须保留；
   - 但 `V` 侧若走 `B_in_regs=true` 的 warp-copy 路径，在 `kNWarps=8` 下会触发错配/不稳定。

### 最终有效修复
- 文件：`csrc/flash_attn/src/flash_fwd_kernel.h`
- 在两个主循环的 `kBlockN==64 && (kWarpRows==16 || kWarpRows==32)` 分支中：
  - 保留 `tOrP` 的 warp-specialized 重排（含 `*4`）；
  - 将 `PV` 计算改为通用 `gemm_rs(..., tOsVt, ...)`，不再使用 `gemm_rs<B_in_regs=true>` 的 `tOsVtWarp -> tOrVt` 路径。
- 即本轮最终稳定组合是：
  - `P`：specialized（寄存器重排）
  - `V`：generic（shared->reg 由 `gemm_rs` 常规路径负责）

### 本轮验证结果
- 快速回归（16 个历史失败组合）：`16/16 PASS`。
- 全量回归：`./test.sh` 结果 `64/64 PASS`，所有 case `status=PASS`。
- 数值误差回到正常量级：
  - `max_diff <= 0.001953`
  - `mean_diff` 约 `2e-5 ~ 5e-5`

### 当前状态说明
- 为了提速，本轮保留了两项调试开关：
  - `CMakeLists.txt`：排除 `flash_fwd_split_*.cu` 编译；
  - `flash_api.cpp`：禁用 split-kv forward 调度入口（仅走非 split 路径）。
- 在该前提下，当前 `kNWarps==8` 路径已通过现有 `simple_test.py` 全矩阵验证。

## 2026-03-08：SM70 前向 QK GEMM 切换到 `FLASH_NAMESPACE::gemm` 的编译问题排查（本轮）

### 复现到的编译错误
- 命令：`./build.sh`
- 失败点：`csrc/flash_attn/src/flash_fwd_kernel.h:323`
- 报错：
  - `utils.h(147): static assertion failed: size<1>(tCsA) == size<1>(tCrA_copy_view)`
  - `utils.h(149): static assertion failed: size<1>(tCsB) == size<1>(tCrB_copy_view)`
- 结论：直接把手写 QK 循环替换为原版 `FLASH_NAMESPACE::gemm` 后，SM70 warp-stationary 路径里的 source 视图分解方式与 `retile_D(...)` 目标视图不一致，导致静态断言失败。

### 本轮确认的关键事实
1. `tSrK` 需要继续基于 `sK`（整块 K tile）做 `partition_fragment_B`，不能切到 `sK_warp`。
2. `sK_warp = local_tile(sK, Shape<kBlockN,kHeadDim>, make_coord(warp_id,0))` 在语义上不安全：`sK` 的 M 维本来就是一个完整 tile，`warp_id>0` 会把基址移到下一个 tile（越过当前 K tile）。
3. 对于当前 SM70 warp-stationary 代码，`FLASH_NAMESPACE::gemm` 所需的 `tCsA/tCsB` 应使用 copy atom 的 `retile_S(...)` 视图（`tOsQ/tOsK`），才能和 `retile_D(tSrQ/tSrK)` 的 mode 分解对齐。

### 最终修改（已实测可编译）
- 文件：`csrc/flash_attn/src/flash_fwd_kernel.h`
- 修改点：
  - 新增 copy atom retiling 变量：
    - `smem_tiled_copy_Q / smem_thr_copy_Q / tSsQ`
    - `smem_tiled_copy_K / smem_thr_copy_K / tSsK`
  - 其中：
    - `tSsQ = smem_thr_copy_Q.retile_S(tOsQ)`
    - `tSsK = smem_thr_copy_K.retile_S(tOsK)`
  - 将 masking loop 内原手写循环替换为：
    - `FLASH_NAMESPACE::gemm<false>(acc_s, tSrQ, tSrK, tSsQ, tSsK, ...)`

### 验证结果
1. `./build.sh`：通过（本轮构建成功安装 `vllm-flash-attn`）。
2. `./test.sh`：`64` 个用例里 `60` 个通过，`4` 个失败。
   - 失败用例：`55, 56, 63, 64`
   - 共同条件：`Sq=96, Sk=80, local=True, alibi=True`
   - 失败类型：`kernel_output_has_nan_or_inf`

### 当前判断
- 本轮目标“修复 `FLASH_NAMESPACE::gemm` 编译失败”已完成。
- 仍有 4 个数值失败 case，是否由本轮改动引入尚未做 A/B 回归（需与改动前二进制做同样 case 对比）。

## 2026-03-08：QK `gemm` 回归（local+alibi NaN）根因确认与最终修复

### 现象确认
- 将 masking loop 的手写 QK 主循环改为 `FLASH_NAMESPACE::gemm` 后：
  - `./build.sh` 可通过；
  - `./test.sh` 出现 `4/64` 失败：`55,56,63,64`；
  - 失败条件固定为 `Sq=96, Sk=80, local=True, alibi=True`，错误为 `kernel_output_has_nan_or_inf`。

### A/B 结论（用于定位根因）
1. 仅将 masking loop 临时改回手写 `copy+gemm` 后，4 个失败 case 全部恢复 PASS。
2. 这说明回归确实来自“QK 在 masking loop 的新 `FLASH_NAMESPACE::gemm` 调用路径”。

### 根因（不是 case 条件绕过）
- 当前 SM70 warp-stationary 前向路径中，Q/K 的可用 source 视图是 `tOsQ / tOsK`（fragment 对齐视图）。
- 原版 `gemm` 默认路径内部依赖 `smem_tiled_copy + retile_D`，它要求 source tensor 的 copy-mode 与 tiled-copy 映射一致。
- 在本分支的 warp-stationary 组织下：
  - 若强行走默认 tiled-copy 路径，存在两类问题：
    1) 某些 kernel traits 直接触发静态断言（`size<1>(tCsA/B) == size<1>(tCrA/B_copy_view)`）；
    2) 对能编译的 traits（比如 hdim64 测试覆盖），会出现 source/dest mode 对齐偏差，导致寄存器片段装载错误，进而在 `local+alibi` 组合下放大为 NaN。

### 最终修复
1. 在 `csrc/flash_attn/src/utils.h` 的 `FLASH_NAMESPACE::gemm` 增加第三个模板参数：
   - `Use_smem_tiled_copy`（默认 `true`，保持原行为不变）。
2. 新增 `Use_smem_tiled_copy=false` 路径：
   - 对 `tCsA/tCsB -> tCrA/tCrB` 使用 fragment 维度一一对应的 `cute::copy`（带同样的预取顺序）；
   - 不走 `retile_D + smem_tiled_copy`。
3. 在 `csrc/flash_attn/src/flash_fwd_kernel.h` 的 masking loop 中，统一调用：
   - `FLASH_NAMESPACE::gemm<false, false, false>(acc_s, tSrQ, tSrK, tOsQ, tOsK, ...)`
- 这是对该 kernel 数据布局的结构性适配，不依赖 `local/alibi` 等运行时条件，不是“加条件绕过失败 case”。

### 验证结果
- `./build.sh`：通过。
- 单测失败集复测（55/56/63/64）：全部 PASS。
- `./test.sh` 全量：`64/64 PASS`。

## 2026-03-08：按要求恢复 `gemm` 原样后的最终方案

### 用户要求
- `csrc/flash_attn/src/utils.h` 里的 `gemm` 必须保持原样，不增加额外分支。
- 通过外部（`flash_fwd_kernel.h`）适配来解决回归。

### 最终实现
1. `utils.h`：已恢复到原始 `gemm` 实现（无 `Use_smem_tiled_copy` 额外模板参数）。
2. `flash_fwd_kernel.h`（masking loop）改为“外部预装载 + 原版 gemm 纯计算”模式：
   - 使用 lane-local 的 copy 视图：
     - `tSsQ = smem_thr_copy_Q.retile_S(tOsQ)`
     - `tSsK = smem_thr_copy_K.retile_S(tOsK)`
   - 在调用 `gemm` 前，先用已验证正确的 warp-stationary 映射把 `Q/K` 片段装入寄存器：
     - `cute::copy(tOsQ(_,_,i), tSrQ(_,_,i))`
     - `cute::copy(tOsK(_,_,i), tSrK(_,_,i))`
   - 然后调用原版：
     - `FLASH_NAMESPACE::gemm<true, true>(...)`
     - 即让 `gemm` 只执行 MMA 循环，不再由内部 copy 路径装载 A/B。

### 结论（根因对应）
- 回归点在于：SM70 warp-stationary 路径下，`gemm` 内部默认 copy 路径（`retile_D + smem_tiled_copy`）与本分支 `Q/K` source 视图不完全同构，导致部分 traits 编译断言失败，或在可编译配置下触发数值异常。
- 外部先按已验证映射把 `A/B` 放入寄存器，再让原版 `gemm<true,true>` 只做计算，可同时满足：
  - `gemm` 不改动；
  - 数值正确；
  - 代码调用仍走 `FLASH_NAMESPACE::gemm`。

### 验证结果
- `./build.sh`：通过。
- 失败集复测（55/56/63/64）：全部 PASS。
- `./test.sh` 全量：`64/64 PASS`。

## 2026-03-08：继续验证“传入正确 copy 契约”后的最终稳定点（不改 `utils.h::gemm`）

### 目标
- 保持 `csrc/flash_attn/src/utils.h` 的 `FLASH_NAMESPACE::gemm` 原样。
- 优先尝试通过外部传入 `tCsA/tCsB`（copy 契约）让 `gemm` 内部 copy 路径可用，并且通过全量测试。

### 实验与结论（按实际顺序）
1. `gemm<false, true>` + `tSsQ = partition_S(sQ)`：
   - 结果：`./build.sh` 编译失败。
   - 错误：`utils.h(147)` 静态断言 `size<1>(tCsA) == size<1>(tCrA_copy_view)`。
   - 说明：Q 的 source 视图 mode 分解与 `retile_D(tSrQ)` 不匹配。

2. `gemm<false, true>` + `tSsQ = retile_S(tOsQ)`，K 外部预装载（`copy tOsK -> tSrK`）：
   - 结果：可编译、可运行。
   - 快速回归（55/56/63/64）：全 PASS。
   - 全量 `./test.sh`：`62/64 PASS`，失败仅 `59/60`：
     - `Sq=96, Sk=80, causal=True, local=False, alibi=True`
     - 失败类型：`kernel_output_has_nan_or_inf`
   - 补充：将 Q thread slice 从 `tidx` 改为 `lane_id` 后，`59/60` 仍失败，问题不在该索引切换。

3. `gemm<false, true>` + `tSsQ = partition_S(sQ_warp)`：
   - 结果：编译失败（同类静态断言）。
   - 说明：warp-local tile 直接 `partition_S` 在当前 traits 下不能满足 `gemm` 的 A-copy 契约。

4. `gemm<true, false>` + Q 外部预装载（`copy tOsQ -> tSrQ`）+ K 走 `gemm` 内部 copy：
   - 关键参数：
     - `tSsQ = smem_thr_copy_Q.retile_S(tOsQ)`（A_in_regs=true 时仅保留形状一致性，不参与实际 A copy）
     - `tSsK = smem_thr_copy_K.retile_S(tOsK)`（B_in_regs=false，实际用于 K copy）
   - 结果：
     - `./build.sh`：通过
     - 关键失败集（55/56/59/60/63/64）：全 PASS
     - `./test.sh`：`64/64 PASS`

### 本轮最终确认的事实
- 在当前 SM70 warp-stationary 实现中，Q 由 `gemm` 内部 copy（`A_in_regs=false`）仍存在不可稳定契约：
  - 要么编译期静态断言失败；
  - 要么在 `causal=True, local=False, alibi=True` 组合触发 NaN。
- K 的内部 copy 契约是可稳定成立的（`B_in_regs=false` 可全量通过）。
- 因此当前“既不改 `gemm`、又保持高性能 copy”的稳定方案是：
  - **Q 外部预装载到寄存器 + K 使用 `gemm` 内部 copy**（`gemm<true,false>`）。
