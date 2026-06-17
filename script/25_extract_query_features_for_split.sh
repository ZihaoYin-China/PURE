#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ -n "${PYTHON_BIN:-}" ]; then
  :
elif [ -x "/opt/conda/envs/universalrag/bin/python" ]; then
  PYTHON_BIN="/opt/conda/envs/universalrag/bin/python"
else
  PYTHON_BIN="python"
fi
INPUT_DIR="${INPUT_DIR:-dataset/query_nonvideo_large_strict/dev}"
QUERY_BGE_DIR="${QUERY_BGE_DIR:-eval/features/query_dev/bge-large}"
QUERY_INTERNVIDEO_DIR="${QUERY_INTERNVIDEO_DIR:-eval/features/query_dev/internvideo}"
RUN_BGE="${RUN_BGE:-1}"
RUN_INTERNVIDEO="${RUN_INTERNVIDEO:-1}"
BGE_BATCH_SIZE="${BGE_BATCH_SIZE:-64}"
BGE_DEVICE="${BGE_DEVICE:-auto}"

if [ ! -d "$INPUT_DIR" ]; then
  echo "[ERROR] INPUT_DIR does not exist: $INPUT_DIR"
  exit 1
fi

if [ "$RUN_BGE" = "1" ]; then
  echo "[INFO] Extracting BGE query features: $INPUT_DIR -> $QUERY_BGE_DIR"
  "$PYTHON_BIN" preprocess/extract_query_feats_bge.py \
    --input_path "$INPUT_DIR" \
    --output_path "$QUERY_BGE_DIR" \
    --batch_size "$BGE_BATCH_SIZE" \
    --device "$BGE_DEVICE"
fi

if [ "$RUN_INTERNVIDEO" = "1" ]; then
  if [ -n "${INTERNVIDEO_PATH:-}" ]; then
    if [ -d "${INTERNVIDEO_PATH}/InternVideo2/multi_modality" ]; then
      export PYTHONPATH="${INTERNVIDEO_PATH}/InternVideo2:${INTERNVIDEO_PATH}/InternVideo2/multi_modality${PYTHONPATH:+:${PYTHONPATH}}"
    elif [ -d "${INTERNVIDEO_PATH}/multi_modality" ]; then
      export PYTHONPATH="${INTERNVIDEO_PATH}:${INTERNVIDEO_PATH}/multi_modality${PYTHONPATH:+:${PYTHONPATH}}"
    fi
  fi

  echo "[INFO] Extracting InternVideo query features: $INPUT_DIR -> $QUERY_INTERNVIDEO_DIR"
  echo "[INFO] Python: $PYTHON_BIN"
  "$PYTHON_BIN" preprocess/extract_query_feats_internvideo.py \
    --input_path "$INPUT_DIR" \
    --output_path "$QUERY_INTERNVIDEO_DIR"
fi

echo "[DONE] Query feature extraction finished."
