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

"$PYTHON_BIN" route/train/route_distilbert_vib.py \
  --checkpoint_dir "${CHECKPOINT_DIR:-route/train/checkpoints/distilbert_vib}" \
  --input_dir "${INPUT_DIR:-dataset/query}" \
  --output_dir "${OUTPUT_DIR:-route/results_vib}" \
  --router_name "${ROUTER_NAME:-distilbert}" \
  --batch_size "${BATCH_SIZE:-256}" \
  --max_input_length "${MAX_INPUT_LENGTH:-512}" \
  --device "${DEVICE:-auto}" \
  --include_targets "${INCLUDE_TARGETS:-}" \
  --exclude_targets "${EXCLUDE_TARGETS:-lvbench,videorag_synth,videorag_wikihow}"
