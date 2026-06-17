#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

INPUT_ROOT="${INPUT_ROOT:-dataset/query_nonvideo_large}"
OUTPUT_ROOT="${OUTPUT_ROOT:-dataset/query_nonvideo_large_strict}"
DEV_RATIO="${DEV_RATIO:-0.2}"
SEED="${SEED:-42}"

python tools/build_large_train_dev_split.py \
  --input_root "$INPUT_ROOT" \
  --output_root "$OUTPUT_ROOT" \
  --dev_ratio "$DEV_RATIO" \
  --seed "$SEED"
