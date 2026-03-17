#!/bin/bash
set -euo pipefail

source .venv/bin/activate

TEST_TARGET="${1:-sparse}"
CUDA_DEV="${CUDA_VISIBLE_DEVICES:-1}"

case "$TEST_TARGET" in
  dense)
    CUDA_VISIBLE_DEVICES="$CUDA_DEV" python tests/test_dense.py
    ;;
  sparse)
    CUDA_VISIBLE_DEVICES="$CUDA_DEV" python tests/test_sparse.py
    ;;
  all)
    CUDA_VISIBLE_DEVICES="$CUDA_DEV" python tests/test_dense.py
    CUDA_VISIBLE_DEVICES="$CUDA_DEV" python tests/test_sparse.py
    ;;
  *)
    echo "Usage: ./test.sh [all|dense|sparse]"
    exit 1
    ;;
esac
