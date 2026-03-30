#!/bin/bash

.venv/bin/python benchmarks/attention_benchmarks/benchmark.py --backends FLASH_ATTN TRITON_ATTN --batch-specs "q1k" "q2k" "q4k" "q8k" "q128s1k" "q128s2k" "q128s4k" "q128s8k"