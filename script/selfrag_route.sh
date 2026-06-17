#!/usr/bin/env bash
#
# Self-RAG Routing Adapter
# ========================
# Runs Self-RAG's Llama-2 model to make binary retrieval decisions,
# maps them to COVER's 4-class action space, and outputs route files
# compatible with COVER's eval.py.
#
# The eval.py pipeline then uses:
#   - SAME generator (Qwen3.6-plus) as all other baselines
#   - SAME retrievers (BGE / InternVideo) as all other baselines
#
# Only the ROUTING LOGIC differs.
#
# Usage:
#   bash script/selfrag_route.sh
#   SELFRAG_TARGETS=mmlu,squad,webqa bash script/selfrag_route.sh
#   SELFRAG_THRESHOLD=0.5 SELFRAG_BATCH_SIZE=16 bash script/selfrag_route.sh
#
# Prerequisites:
#   - CUDA GPU(s) with sufficient VRAM (7B model needs ~14GB)
#   - vllm installed (pip install vllm)
#   - Self-RAG model downloaded (auto-downloaded from HuggingFace)
#

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

# vLLM spawns subprocesses that may import torch/numpy after libgomp is loaded.
# Conda MKL can conflict with libgomp unless the GNU threading layer is used.
if [ "${MKL_THREADING_LAYER:-}" = "INTEL" ]; then
    export MKL_THREADING_LAYER=GNU
else
    export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"
fi

# --- Defaults ---
MODEL_NAME="${SELFRAG_MODEL_NAME:-selfrag/selfrag_llama2_7b}"
SOURCE_ROUTE_DIR="${SELFRAG_SOURCE_DIR:-route/results/distilbert}"
OUTPUT_DIR="${SELFRAG_OUTPUT_DIR:-route/results/selfrag}"
TARGETS="${SELFRAG_TARGETS:-mmlu,squad,natural_questions,hotpotqa,webqa}"
THRESHOLD="${SELFRAG_THRESHOLD:-0.2}"
RETRIEVAL_MAP="${SELFRAG_RETRIEVAL_MAP:-}"
BATCH_SIZE="${SELFRAG_BATCH_SIZE:-32}"
DTYPE="${SELFRAG_DTYPE:-half}"
WORLD_SIZE="${SELFRAG_WORLD_SIZE:-1}"
DOWNLOAD_DIR="${SELFRAG_DOWNLOAD_DIR:-$ROOT_DIR/.cache}"
if [[ "$DOWNLOAD_DIR" != /* ]]; then
    DOWNLOAD_DIR="$ROOT_DIR/$DOWNLOAD_DIR"
fi
SKIP_IMAGE_QUERIES="${SELFRAG_SKIP_IMAGE_QUERIES:-1}"

# --- Help ---
if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    echo "Self-RAG Routing Adapter"
    echo ""
    echo "Environment variables:"
    echo "  SELFRAG_MODEL_NAME     Model path (default: selfrag/selfrag_llama2_7b)"
    echo "  SELFRAG_SOURCE_DIR     Source route dir for queries (default: route/results/distilbert)"
    echo "  SELFRAG_OUTPUT_DIR     Output dir for Self-RAG routes (default: route/results/selfrag)"
    echo "  SELFRAG_TARGETS        Comma-separated targets (default: mmlu,squad,natural_questions,hotpotqa,webqa)"
    echo "  SELFRAG_THRESHOLD      Retrieval threshold, 0.2=aggressive, 0.5=balanced (default: 0.2)"
    echo "  SELFRAG_BATCH_SIZE     vllm batch size (default: 32)"
    echo "  SELFRAG_DTYPE          Model dtype (default: half)"
    echo "  SELFRAG_WORLD_SIZE     Number of GPUs (default: 1)"
    echo "  SELFRAG_DOWNLOAD_DIR   Model cache dir (default: .cache)"
    echo "  SELFRAG_SKIP_IMAGE_QUERIES"
    echo "                         1: fallback image GT rows to source router; 0: pure Self-RAG text routing (default: 1)"
    exit 0
fi

echo "============================================"
echo " Self-RAG Routing Adapter"
echo "============================================"
echo " Model:      ${MODEL_NAME}"
echo " Targets:    ${TARGETS}"
echo " Threshold:  ${THRESHOLD}"
echo " Retrieval map: ${RETRIEVAL_MAP:-<default>}"
echo " Output:     ${OUTPUT_DIR}"
echo " Batch size: ${BATCH_SIZE}"
echo " GPUs:       ${WORLD_SIZE}"
echo " Image fallback: ${SKIP_IMAGE_QUERIES}"
echo "============================================"
echo ""

EXTRA_ARGS=()
if [ -n "$RETRIEVAL_MAP" ]; then
    EXTRA_ARGS+=(--retrieval_map "$RETRIEVAL_MAP")
fi
if [ "$SKIP_IMAGE_QUERIES" = "1" ]; then
    EXTRA_ARGS+=(--skip_image_queries)
fi

python route/selfrag_route_adapter.py \
    --model_name "${MODEL_NAME}" \
    --source_route_dir "${SOURCE_ROUTE_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --targets "${TARGETS}" \
    --threshold "${THRESHOLD}" \
    --batch_size "${BATCH_SIZE}" \
    --dtype "${DTYPE}" \
    --world_size "${WORLD_SIZE}" \
    --download_dir "${DOWNLOAD_DIR}" \
    "${EXTRA_ARGS[@]}"
