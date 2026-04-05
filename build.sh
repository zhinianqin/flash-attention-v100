#!/bin/bash

source .venv/bin/activate

export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH
export D_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH

export CMAKE_CXX_COMPILER_LAUNCHER=ccache
export CMAKE_CUDA_COMPILER_LAUNCHER=ccache

ccache -M 100G

export MAX_JOBS=$(nproc)
export NVCC_THREADS=1

# 定位 site-packages
SITE_PACKAGES=$(.venv/bin/python -c "import sysconfig; print(sysconfig.get_path('purelib'))")

FLASH_DIR="$SITE_PACKAGES/vllm_flash_attn"
VLLM_DIR="$SITE_PACKAGES/vllm"

rm -rf "$VLLM_DIR/vllm_flash_attn"

uv pip uninstall vllm-flash-attn
uv pip install --no-build-isolation . -v 2>&1 | \
sed -E 's/.*([1-9][0-9]* bytes (stack frame|spill stores|spill loads)).*/\x1b[31m&\x1b[0m/g'

cp -r "$FLASH_DIR" "$VLLM_DIR/"
rm -rf "$SITE_PACKAGES/flash_attn"
