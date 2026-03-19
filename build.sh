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

uv pip uninstall vllm-flash-attn
uv pip install --no-build-isolation . -v 2>&1 | \
sed -E 's/.*[1-9][0-9]* bytes spill stores.*/\x1b[31m&\x1b[0m/g'
