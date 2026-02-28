#!/bin/bash

source .venv/bin/activate

CUDA_VISIBLE_DEVICES=1 python tests/simple_test.py
