#!/bin/bash

source .venv/bin/activate

export CUDA_HOME=/usr/local/cuda-12.4
export PATH=/usr/local/cuda-12.4/bin:$PATH
export D_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH

export CMAKE_BUILD_TYPE=Debug
export CMAKE_CXX_COMPILER_LAUNCHER=ccache
export CMAKE_CUDA_COMPILER_LAUNCHER=ccache

ccache -M 100G

export MAX_JOBS=$(nproc)
export NVCC_THREADS=1

uv pip uninstall vllm-flash-attn
uv pip install --no-build-isolation . -v
