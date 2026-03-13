#!/bin/bash
set -euo pipefail

source .venv/bin/activate

TEST_TARGET="${1:-simple}"
CUDA_DEV="${CUDA_VISIBLE_DEVICES:-1}"

case "$TEST_TARGET" in
  simple)
    CUDA_VISIBLE_DEVICES="$CUDA_DEV" python tests/simple_test.py
    ;;
  sparse)
    CUDA_VISIBLE_DEVICES="$CUDA_DEV" python tests/test_sparse.py
    ;;
  all)
    CUDA_VISIBLE_DEVICES="$CUDA_DEV" python tests/simple_test.py
    CUDA_VISIBLE_DEVICES="$CUDA_DEV" python tests/test_sparse.py
    ;;
  *)
    echo "Usage: ./test.sh [simple|sparse|all]"
    exit 1
    ;;
esac
