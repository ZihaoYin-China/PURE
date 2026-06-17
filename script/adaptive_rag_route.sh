#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

ADAPTIVE_RAG_CLASSIFIER="${ADAPTIVE_RAG_CLASSIFIER:-}"
ADAPTIVE_RAG_PREDICTION_DIR="${ADAPTIVE_RAG_PREDICTION_DIR:-}"
ADAPTIVE_RAG_PREDICTION_FILE="${ADAPTIVE_RAG_PREDICTION_FILE:-}"
ADAPTIVE_RAG_TARGETS="${ADAPTIVE_RAG_TARGETS:-${TARGETS:-mmlu,squad,natural_questions,hotpotqa,webqa}}"
ADAPTIVE_RAG_SOURCE_DIR="${ADAPTIVE_RAG_SOURCE_DIR:-${ROUTE_DIR:-route/results}/distilbert}"
ADAPTIVE_RAG_OUTPUT_DIR="${ADAPTIVE_RAG_OUTPUT_DIR:-${ROUTE_DIR:-route/results}/adaptive_rag}"
ADAPTIVE_RAG_BATCH_SIZE="${ADAPTIVE_RAG_BATCH_SIZE:-64}"
ADAPTIVE_RAG_SINGLE_MAP="${ADAPTIVE_RAG_SINGLE_MAP:-}"
ADAPTIVE_RAG_MULTI_MAP="${ADAPTIVE_RAG_MULTI_MAP:-}"
ADAPTIVE_RAG_DEVICE="${ADAPTIVE_RAG_DEVICE:-auto}"
ADAPTIVE_RAG_MAX_SEQ_LENGTH="${ADAPTIVE_RAG_MAX_SEQ_LENGTH:-384}"
ADAPTIVE_RAG_SKIP_IMAGE_QUERIES="${ADAPTIVE_RAG_SKIP_IMAGE_QUERIES:-1}"

echo "============================================"
echo " Adaptive-RAG Routing Adapter"
echo "============================================"
echo " Classifier:      ${ADAPTIVE_RAG_CLASSIFIER:-<none>}"
echo " Prediction dir:  ${ADAPTIVE_RAG_PREDICTION_DIR:-<none>}"
echo " Prediction file: ${ADAPTIVE_RAG_PREDICTION_FILE:-<none>}"
echo " Targets:         ${ADAPTIVE_RAG_TARGETS}"
echo " Source:          ${ADAPTIVE_RAG_SOURCE_DIR}"
echo " Output:          ${ADAPTIVE_RAG_OUTPUT_DIR}"
echo " Batch size:      ${ADAPTIVE_RAG_BATCH_SIZE}"
echo " Single map:      ${ADAPTIVE_RAG_SINGLE_MAP:-<default>}"
echo " Multi map:       ${ADAPTIVE_RAG_MULTI_MAP:-<default>}"
echo " Skip image:      ${ADAPTIVE_RAG_SKIP_IMAGE_QUERIES}"
echo " Device:          ${ADAPTIVE_RAG_DEVICE}"
echo "============================================"

ARGS=(
    --source_route_dir "$ADAPTIVE_RAG_SOURCE_DIR"
    --output_dir "$ADAPTIVE_RAG_OUTPUT_DIR"
    --targets "$ADAPTIVE_RAG_TARGETS"
    --batch_size "$ADAPTIVE_RAG_BATCH_SIZE"
    --device "$ADAPTIVE_RAG_DEVICE"
    --max_seq_length "$ADAPTIVE_RAG_MAX_SEQ_LENGTH"
)

if [[ -n "$ADAPTIVE_RAG_CLASSIFIER" ]]; then
    ARGS+=(--classifier_model_path "$ADAPTIVE_RAG_CLASSIFIER")
fi
if [[ -n "$ADAPTIVE_RAG_PREDICTION_DIR" ]]; then
    ARGS+=(--prediction_dir "$ADAPTIVE_RAG_PREDICTION_DIR")
fi
if [[ -n "$ADAPTIVE_RAG_PREDICTION_FILE" ]]; then
    ARGS+=(--prediction_file "$ADAPTIVE_RAG_PREDICTION_FILE")
fi
if [[ -n "$ADAPTIVE_RAG_SINGLE_MAP" ]]; then
    ARGS+=(--single_retrieval_map "$ADAPTIVE_RAG_SINGLE_MAP")
fi
if [[ -n "$ADAPTIVE_RAG_MULTI_MAP" ]]; then
    ARGS+=(--multi_retrieval_map "$ADAPTIVE_RAG_MULTI_MAP")
fi
if [[ "$ADAPTIVE_RAG_SKIP_IMAGE_QUERIES" == "1" ]]; then
    ARGS+=(--skip_image_queries)
fi

python route/adaptive_rag_route_adapter.py "${ARGS[@]}"
