# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

Flash Attention 移植到 V100 (SM70) GPU，移除了 SM80 支持。

- **原版项目**: `/root/flash-attention`
- **目标架构**: SM70 (V100)
- **包名**: `vllm_flash_attn`

## 分支状态

- `develop`: 可构建/测试通过，但有大量寄存器 spill
- `debug/sm70-linear16-spill-only` (当前): 重构中，spill 已优化，但精度尚未对齐

## 构建和测试

```bash
./build.sh   # 构建 (CUDA 12.8, ccache 100G)
./test.sh    # 测试
```

**重要**: 构建非常耗时，绝对不要终止 build 任务。

**工具路径**: `.venv/bin/{python,pytest,cmake,ninja}`

## 关键文件

- `csrc/flash_attn/src/flash_fwd_kernel.h` - 前向 kernel 主实现
- `csrc/flash_attn/src/kernel_traits.h` - TiledMMA 定义
- `flash_attn/flash_attn_interface.py` - Python 接口
