#!/usr/bin/env bash
set -euo pipefail

# CPU-8c32G-friendly sampled latency/cost benchmark.
# Requires the same API key environment used by eval/utils/models/qwen_api.py,
# e.g. DASHSCOPE_API_KEY or QWEN_API_KEY for qwen-api:qwen3.6-plus.

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Match the refreshed HotpotQA retrieval setting used in the paper tables.
export HOTPOTQA_TEXT_FEATS="${HOTPOTQA_TEXT_FEATS:-eval/features/text/hotpotqa_raw_context.pkl}"

MODEL_PATH="${MODEL_PATH:-qwen-api:qwen3.6-plus}"
OUT_DIR="${OUT_DIR:-analysis/results/cpu8c32g_latency_cost_500x5x6}"
WORKERS="${WORKERS:-8}"
SAMPLE_PER_TARGET="${SAMPLE_PER_TARGET:-500}"
SEED="${SEED:-20260531}"

python analysis/run_sampled_latency_cost_benchmark.py \
  --mode all \
  --output-dir "${OUT_DIR}" \
  --model-path "${MODEL_PATH}" \
  --sample-per-target "${SAMPLE_PER_TARGET}" \
  --seed "${SEED}" \
  --workers "${WORKERS}" \
  --verifier-policy observed \
  --continue-on-error \
  "$@"
