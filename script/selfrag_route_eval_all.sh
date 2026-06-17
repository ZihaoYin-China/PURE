#!/usr/bin/env bash
#
# Self-RAG + COVER Full Evaluation Pipeline
# ========================================
# 1. Runs Self-RAG routing on selected targets
# 2. Runs COVER eval.py with the same generator/retrievers as baselines
# 3. Scores results with eval/score.py
#
# This isolates routing quality: same generator, same retrievers, only routing
# logic differs (Self-RAG reflection tokens vs PURE routers).
#
# Usage:
#   bash script/selfrag_route_eval_all.sh
#   bash script/selfrag_route_eval_all.sh --targets mmlu,squad,webqa
#   RESULTS_ROOT=eval/results_qwen36plus_api_selfrag bash script/selfrag_route_eval_all.sh
#

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

# --- Configurable defaults ---
MODEL_PATH="${SELFRAG_EVAL_MODEL_PATH:-${MODEL_PATH:-qwen-api:qwen3.6-plus}}"
MODEL_NAME="${MODEL_PATH##*/}"
ROUTER_MODEL="selfrag"
TARGETS_ENV="${SELFRAG_EVAL_TARGETS:-${TARGETS:-mmlu,squad,natural_questions,hotpotqa,webqa}}"
ROUTE_DIR="${ROUTE_DIR:-route/results}"
RESULTS_ROOT="${RESULTS_ROOT:-eval/results}"
QUERY_BGE_DIR="${QUERY_BGE_DIR:-eval/features/query/bge-large}"
QUERY_INTERNVIDEO_DIR="${QUERY_INTERNVIDEO_DIR:-eval/features/query/internvideo}"
TOP_K="${TOP_K:-1}"
ALPHA="${ALPHA:-0.2}"
NFRAMES="${SELFRAG_EVAL_NFRAMES:-${NFRAMES:-1}}"
FORCE_EVAL="${FORCE_EVAL:-0}"
if [ -n "${SCORE_PYTHON_BIN:-}" ]; then
    :
elif [ -x "/opt/conda/envs/universalrag/bin/python" ]; then
    SCORE_PYTHON_BIN="/opt/conda/envs/universalrag/bin/python"
else
    SCORE_PYTHON_BIN="python"
fi

# Self-RAG routing defaults
SELFRAG_MODEL="${SELFRAG_MODEL_NAME:-selfrag/selfrag_llama2_7b}"
SELFRAG_THRESHOLD="${SELFRAG_THRESHOLD:-0.2}"
SELFRAG_BATCH_SIZE="${SELFRAG_BATCH_SIZE:-32}"

show_help() {
    echo "Self-RAG + COVER Full Evaluation Pipeline"
    echo ""
    echo "Environment variables:"
    echo "  SELFRAG_EVAL_MODEL_PATH   Generator model (default: qwen-api:qwen3.6-plus)"
    echo "  SELFRAG_EVAL_TARGETS      Targets (default: mmlu,squad,natural_questions,hotpotqa,webqa)"
    echo "  SELFRAG_MODEL_NAME        Self-RAG model (default: selfrag/selfrag_llama2_7b)"
    echo "  SELFRAG_SOURCE_DIR        Source route dir for query rows (default: ${ROUTE_DIR}/distilbert)"
    echo "  SELFRAG_THRESHOLD         Retrieval threshold (default: 0.2)"
    echo "  SELFRAG_BATCH_SIZE        vLLM batch size (default: 32)"
    echo "  RESULTS_ROOT              Evaluation output root (default: eval/results)"
    echo "  ROUTE_DIR                 Route root used by eval.py (default: route/results)"
    echo "  TOP_K, ALPHA, NFRAMES     Eval settings (defaults: 1, 0.2, 1)"
    echo "  FORCE_EVAL                1 to rerun existing generation results"
    echo ""
    echo "Options:"
    echo "  --targets LIST            Comma- or space-separated targets"
    echo "  --model_path PATH         Generator model"
    echo "  --results_root DIR        Evaluation output root"
    echo "  --route_dir DIR           Route root"
    echo "  --threshold FLOAT         Self-RAG retrieval threshold"
    echo "  --batch_size INT          Self-RAG vLLM batch size"
    echo "  --top_k INT               Number of retrieved items"
    echo "  --alpha FLOAT             Image caption feature weight"
    echo "  --nframes STR             Frame tag passed to eval.py"
    echo ""
    echo "Steps:"
    echo "  1. bash script/selfrag_route.sh"
    echo "  2. bash script/4_eval.sh --model_path ... --router_model selfrag --target ..."
    echo "  3. python eval/score.py --result_file ..."
}

while [ $# -gt 0 ]; do
    case "$1" in
        --targets)
            TARGETS_ENV="$(echo "$2" | tr " " ",")"
            shift 2
            ;;
        --model_path)
            MODEL_PATH="$2"
            MODEL_NAME="${MODEL_PATH##*/}"
            shift 2
            ;;
        --results_root)
            RESULTS_ROOT="$2"
            shift 2
            ;;
        --route_dir)
            ROUTE_DIR="$2"
            shift 2
            ;;
        --threshold)
            SELFRAG_THRESHOLD="$2"
            shift 2
            ;;
        --batch_size)
            SELFRAG_BATCH_SIZE="$2"
            shift 2
            ;;
        --top_k)
            TOP_K="$2"
            shift 2
            ;;
        --alpha)
            ALPHA="$2"
            shift 2
            ;;
        --nframes)
            NFRAMES="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage."
            exit 1
            ;;
    esac
done

TARGETS_ENV="$(echo "$TARGETS_ENV" | tr " " ",")"
read -r -a TARGETS <<< "$(echo "$TARGETS_ENV" | tr "," " ")"
SELFRAG_OUTPUT_DIR="${SELFRAG_OUTPUT_DIR:-${ROUTE_DIR}/${ROUTER_MODEL}}"
SELFRAG_SOURCE_DIR="${SELFRAG_SOURCE_DIR:-${ROUTE_DIR}/distilbert}"

# =========================================================================
# Step 1: Self-RAG Routing
# =========================================================================
echo "============================================"
echo " Step 1/3: Self-RAG Routing"
echo "============================================"
echo "MODEL_PATH=${MODEL_PATH}"
echo "ROUTE_DIR=${ROUTE_DIR}"
echo "RESULTS_ROOT=${RESULTS_ROOT}"
echo "TARGETS=${TARGETS[*]}"
echo "SELFRAG_MODEL=${SELFRAG_MODEL}"
echo "SELFRAG_THRESHOLD=${SELFRAG_THRESHOLD}"
echo "SELFRAG_OUTPUT_DIR=${SELFRAG_OUTPUT_DIR}"
echo "SELFRAG_SOURCE_DIR=${SELFRAG_SOURCE_DIR}"
echo "SCORE_PYTHON_BIN=${SCORE_PYTHON_BIN}"

export SELFRAG_TARGETS="$TARGETS_ENV"
export SELFRAG_OUTPUT_DIR
export SELFRAG_SOURCE_DIR
export SELFRAG_THRESHOLD
export SELFRAG_BATCH_SIZE
bash script/selfrag_route.sh

# =========================================================================
# Step 2: COVER Evaluation
# =========================================================================
echo ""
echo "============================================"
echo " Step 2/3: Evaluation"
echo "============================================"

for target in "${TARGETS[@]}"; do
    result_file="${RESULTS_ROOT}/${MODEL_NAME}/${ROUTER_MODEL}/${target}_top${TOP_K}_${ALPHA}_${NFRAMES}.json"
    if [ "$FORCE_EVAL" != "1" ] && [ -f "$result_file" ]; then
        echo "[SKIP] Existing result: ${result_file}"
        continue
    fi

    echo ""
    echo "--- Evaluating ${target} ---"
    bash script/4_eval.sh \
        --model_path "${MODEL_PATH}" \
        --router_model "${ROUTER_MODEL}" \
        --target "${target}" \
        --top_k "${TOP_K}" \
        --alpha "${ALPHA}" \
        --nframes "${NFRAMES}" \
        --query_bge_dir "${QUERY_BGE_DIR}" \
        --query_internvideo_dir "${QUERY_INTERNVIDEO_DIR}" \
        --route_dir "${ROUTE_DIR}" \
        --output_root "${RESULTS_ROOT}"
done

# =========================================================================
# Step 3: Scoring
# =========================================================================
echo ""
echo "============================================"
echo " Step 3/3: Scoring"
echo "============================================"

RESULTS_DIR="${RESULTS_ROOT}/${MODEL_NAME}/${ROUTER_MODEL}"

for target in "${TARGETS[@]}"; do
    echo ""
    echo "--- Scoring ${target} ---"
    RESULT_FILE="${RESULTS_DIR}/${target}_top${TOP_K}_${ALPHA}_${NFRAMES}.json"
    if [ ! -f "$RESULT_FILE" ]; then
        echo "[MISS] ${RESULT_FILE}" | tee -a "eval_results_all.log"
        continue
    fi
    "${SCORE_PYTHON_BIN}" eval/score.py --result_file "${RESULT_FILE}" --target "${target}" \
        2>&1 | tee -a "eval_results_all.log"
done

echo ""
echo "============================================"
echo " Pipeline complete!"
echo " Results in: ${RESULTS_DIR}/"
echo "============================================"
