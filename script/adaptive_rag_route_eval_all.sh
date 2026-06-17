#!/usr/bin/env bash
#
# Adaptive-RAG-style router + PURE evaluation pipeline.
# Keeps the generator fixed (default qwen-api:qwen3.6-plus) and only swaps the
# routing policy to Adaptive-RAG's A/B/C complexity classifier.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

MODEL_PATH="${ADAPTIVE_RAG_EVAL_MODEL_PATH:-${MODEL_PATH:-qwen-api:qwen3.6-plus}}"
MODEL_NAME="${MODEL_PATH##*/}"
ROUTER_MODEL="adaptive_rag"
TARGETS_ENV="${ADAPTIVE_RAG_EVAL_TARGETS:-${TARGETS:-mmlu,squad,natural_questions,hotpotqa,webqa}}"
ROUTE_DIR="${ROUTE_DIR:-route/results}"
RESULTS_ROOT="${RESULTS_ROOT:-eval/results}"
QUERY_BGE_DIR="${QUERY_BGE_DIR:-eval/features/query/bge-large}"
QUERY_INTERNVIDEO_DIR="${QUERY_INTERNVIDEO_DIR:-eval/features/query/internvideo}"
TOP_K="${TOP_K:-1}"
ALPHA="${ALPHA:-0.2}"
NFRAMES="${ADAPTIVE_RAG_EVAL_NFRAMES:-${NFRAMES:-1}}"
FORCE_EVAL="${FORCE_EVAL:-0}"

show_help() {
    echo "Adaptive-RAG-style router + PURE evaluation"
    echo ""
    echo "Required unless using precomputed predictions:"
    echo "  ADAPTIVE_RAG_CLASSIFIER   Trained T5 classifier checkpoint"
    echo ""
    echo "Optional:"
    echo "  ADAPTIVE_RAG_PREDICTION_DIR / ADAPTIVE_RAG_PREDICTION_FILE"
    echo "  ADAPTIVE_RAG_EVAL_TARGETS  Targets"
    echo "  ROUTE_DIR, RESULTS_ROOT, QUERY_BGE_DIR, QUERY_INTERNVIDEO_DIR"
    echo "  FORCE_EVAL=1 to rerun existing generation outputs"
    echo ""
    echo "Options:"
    echo "  --targets LIST"
    echo "  --model_path PATH"
    echo "  --results_root DIR"
    echo "  --route_dir DIR"
    echo "  --top_k INT"
    echo "  --alpha FLOAT"
    echo "  --nframes STR"
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

export ADAPTIVE_RAG_TARGETS="$TARGETS_ENV"
export ADAPTIVE_RAG_SOURCE_DIR="${ADAPTIVE_RAG_SOURCE_DIR:-${ROUTE_DIR}/distilbert}"
export ADAPTIVE_RAG_OUTPUT_DIR="${ADAPTIVE_RAG_OUTPUT_DIR:-${ROUTE_DIR}/${ROUTER_MODEL}}"

echo "============================================"
echo " Step 1/3: Adaptive-RAG Routing"
echo "============================================"
echo "MODEL_PATH=${MODEL_PATH}"
echo "ROUTE_DIR=${ROUTE_DIR}"
echo "RESULTS_ROOT=${RESULTS_ROOT}"
echo "TARGETS=${TARGETS[*]}"
echo "ADAPTIVE_RAG_SOURCE_DIR=${ADAPTIVE_RAG_SOURCE_DIR}"
echo "ADAPTIVE_RAG_OUTPUT_DIR=${ADAPTIVE_RAG_OUTPUT_DIR}"

bash script/adaptive_rag_route.sh

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

echo ""
echo "============================================"
echo " Step 3/3: Scoring"
echo "============================================"

RESULTS_DIR="${RESULTS_ROOT}/${MODEL_NAME}/${ROUTER_MODEL}"

for target in "${TARGETS[@]}"; do
    echo ""
    echo "--- Scoring ${target} ---"
    result_file="${RESULTS_DIR}/${target}_top${TOP_K}_${ALPHA}_${NFRAMES}.json"
    if [ ! -f "$result_file" ]; then
        echo "[MISS] ${result_file}" | tee -a "eval_results_all.log"
        continue
    fi
    python eval/score.py --result_file "$result_file" --target "$target" \
        2>&1 | tee -a "eval_results_all.log"
done

echo ""
echo "============================================"
echo " Pipeline complete!"
echo " Results in: ${RESULTS_DIR}/"
echo "============================================"
