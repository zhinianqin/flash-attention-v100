#!/bin/bash
set -euo pipefail

source .venv/bin/activate

TEST_TARGET="${1:-all}"
CUDA_DEV="${CUDA_VISIBLE_DEVICES:-1}"

case "$TEST_TARGET" in
  dense)
    CUDA_VISIBLE_DEVICES="$CUDA_DEV" python tests/test_dense.py
    ;;
  splitkv)
    # onlysplitkv 分支：只测试 splitkv 路径
    CUDA_VISIBLE_DEVICES="$CUDA_DEV" DENSE_SUITE=splitkv python tests/test_dense.py
    ;;
  sparse)
    CUDA_VISIBLE_DEVICES="$CUDA_DEV" python tests/test_sparse.py
    ;;
  all)
    CUDA_VISIBLE_DEVICES="$CUDA_DEV" python tests/test_dense.py
    CUDA_VISIBLE_DEVICES="$CUDA_DEV" python tests/test_sparse.py
    ;;
  *)
    echo "Usage: ./test.sh [all|dense|splitkv|sparse]"
    exit 1
    ;;
esac
