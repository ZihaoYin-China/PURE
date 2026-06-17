#!/usr/bin/env bash
#
# CRAG-style evidence routing + PURE evaluation pipeline.
# Keeps the generator, retrievers, prompts, and scoring fixed; only swaps the
# routing policy to CRAG's evidence evaluator.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

MODEL_PATH="${CRAG_EVAL_MODEL_PATH:-${MODEL_PATH:-qwen-api:qwen3.6-plus}}"
MODEL_NAME="${MODEL_PATH##*/}"
ROUTER_MODEL="crag"
TARGETS_ENV="${CRAG_EVAL_TARGETS:-${TARGETS:-mmlu,squad,natural_questions,hotpotqa,webqa}}"
ROUTE_DIR="${ROUTE_DIR:-route/results}"
RESULTS_ROOT="${RESULTS_ROOT:-eval/results}"
QUERY_BGE_DIR="${QUERY_BGE_DIR:-eval/features/query/bge-large}"
QUERY_INTERNVIDEO_DIR="${QUERY_INTERNVIDEO_DIR:-eval/features/query/internvideo}"
TOP_K="${TOP_K:-1}"
ALPHA="${ALPHA:-0.2}"
NFRAMES="${CRAG_EVAL_NFRAMES:-${NFRAMES:-1}}"
FORCE_EVAL="${FORCE_EVAL:-0}"
if [ -n "${CRAG_PYTHON_BIN:-}" ]; then
    :
elif python -c "import numpy, torch, transformers" >/dev/null 2>&1; then
    CRAG_PYTHON_BIN="python"
elif [ -x "/opt/conda/bin/python" ] && /opt/conda/bin/python -c "import numpy, torch, transformers" >/dev/null 2>&1; then
    CRAG_PYTHON_BIN="/opt/conda/bin/python"
elif [ -x "/opt/conda/envs/universalrag/bin/python" ] && /opt/conda/envs/universalrag/bin/python -c "import numpy, torch, transformers" >/dev/null 2>&1; then
    CRAG_PYTHON_BIN="/opt/conda/envs/universalrag/bin/python"
else
    CRAG_PYTHON_BIN="python"
fi

if [ -n "${SCORE_PYTHON_BIN:-}" ]; then
    :
else
    SCORE_PYTHON_BIN="$CRAG_PYTHON_BIN"
fi
export CRAG_PYTHON_BIN
export EVAL_PYTHON_BIN="${EVAL_PYTHON_BIN:-$CRAG_PYTHON_BIN}"

show_help() {
    echo "CRAG-style evidence routing + PURE evaluation"
    echo ""
    echo "Required:"
    echo "  CRAG_EVALUATOR_PATH      CRAG evaluator checkpoint; defaults to similarity fallback"
    echo ""
    echo "Options:"
    echo "  --evaluator_path PATH    CRAG evaluator checkpoint, or similarity"
    echo "  --python_bin PATH        Python executable for CRAG route/scoring"
    echo "  --targets LIST           Comma- or space-separated targets"
    echo "  --model_path PATH        Generator model"
    echo "  --results_root DIR       Evaluation output root"
    echo "  --route_dir DIR          Route root"
    echo "  --query_bge_dir DIR      BGE query feature dir; must match route split"
    echo "  --top_k INT              Number of retrieved items for final generation"
    echo "  --alpha FLOAT            Image caption feature weight"
    echo "  --nframes STR            Frame tag passed to eval.py"
    echo "  --ndocs INT              Evidence docs CRAG scores per action"
    echo "  --upper_threshold FLOAT  CRAG correct threshold"
    echo "  --lower_threshold FLOAT  CRAG ambiguous threshold"
    echo "  --candidate_actions STR  paragraph,document or target"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --evaluator_path)
            export CRAG_EVALUATOR_PATH="$2"
            shift 2
            ;;
        --python_bin)
            export CRAG_PYTHON_BIN="$2"
            export EVAL_PYTHON_BIN="$2"
            export SCORE_PYTHON_BIN="$2"
            shift 2
            ;;
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
        --query_bge_dir)
            QUERY_BGE_DIR="$2"
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
        --ndocs)
            export CRAG_NDOCS="$2"
            shift 2
            ;;
        --upper_threshold)
            export CRAG_UPPER_THRESHOLD="$2"
            shift 2
            ;;
        --lower_threshold)
            export CRAG_LOWER_THRESHOLD="$2"
            shift 2
            ;;
        --candidate_actions)
            export CRAG_CANDIDATE_ACTIONS="$2"
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

export CRAG_TARGETS="$TARGETS_ENV"
export CRAG_SOURCE_DIR="${CRAG_SOURCE_DIR:-${ROUTE_DIR}/distilbert}"
export CRAG_OUTPUT_DIR="${CRAG_OUTPUT_DIR:-${ROUTE_DIR}/${ROUTER_MODEL}}"
export CRAG_QUERY_BGE_DIR="${CRAG_QUERY_BGE_DIR:-${QUERY_BGE_DIR}}"
export CRAG_NDOCS="${CRAG_NDOCS:-${TOP_K}}"

echo "============================================"
echo " Step 1/3: CRAG Routing"
echo "============================================"
echo "MODEL_PATH=${MODEL_PATH}"
echo "ROUTE_DIR=${ROUTE_DIR}"
echo "RESULTS_ROOT=${RESULTS_ROOT}"
echo "TARGETS=${TARGETS[*]}"
echo "CRAG_SOURCE_DIR=${CRAG_SOURCE_DIR}"
echo "CRAG_OUTPUT_DIR=${CRAG_OUTPUT_DIR}"
echo "QUERY_BGE_DIR=${QUERY_BGE_DIR}"
echo "CRAG_NDOCS=${CRAG_NDOCS}"
echo "CRAG_PYTHON_BIN=${CRAG_PYTHON_BIN}"
echo "EVAL_PYTHON_BIN=${EVAL_PYTHON_BIN}"
echo "SCORE_PYTHON_BIN=${SCORE_PYTHON_BIN}"

bash script/crag_route.sh

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
    "${SCORE_PYTHON_BIN}" eval/score.py --result_file "$result_file" --target "$target" \
        2>&1 | tee -a "eval_results_all.log"
done

echo ""
echo "============================================"
echo " Pipeline complete!"
echo " Results in: ${RESULTS_DIR}/"
echo "============================================"
