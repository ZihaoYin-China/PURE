#!/usr/bin/env bash
#
# CRAG Routing Adapter
# ====================
# Scores retrieved paragraph/document evidence with a CRAG evaluator, maps the
# scores to PURE's route action space, and writes route files compatible
# with eval/eval.py.
#
# The downstream evaluation then uses:
#   - SAME generator as all other baselines
#   - SAME retrievers as all other baselines
#   - SAME scoring script as all other baselines
#
# Only the routing / evidence-judging logic differs.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

CRAG_EVALUATOR_PATH="${CRAG_EVALUATOR_PATH:-${CRAG_EVALUATOR:-similarity}}"
CRAG_TARGETS="${CRAG_TARGETS:-${TARGETS:-mmlu,squad,natural_questions,hotpotqa,webqa}}"
CRAG_SOURCE_DIR="${CRAG_SOURCE_DIR:-${ROUTE_DIR:-route/results}/distilbert}"
CRAG_OUTPUT_DIR="${CRAG_OUTPUT_DIR:-${ROUTE_DIR:-route/results}/crag}"
CRAG_QUERY_BGE_DIR="${CRAG_QUERY_BGE_DIR:-${QUERY_BGE_DIR:-eval/features/query/bge-large}}"
CRAG_ALLOW_MISSING_QUERY_FEATURES="${CRAG_ALLOW_MISSING_QUERY_FEATURES:-0}"
CRAG_CANDIDATE_ACTIONS="${CRAG_CANDIDATE_ACTIONS:-paragraph,document}"
CRAG_NDOCS="${CRAG_NDOCS:-${TOP_K:-1}}"
CRAG_BATCH_SIZE="${CRAG_BATCH_SIZE:-32}"
CRAG_DEVICE="${CRAG_DEVICE:-auto}"
CRAG_MAX_SEQ_LENGTH="${CRAG_MAX_SEQ_LENGTH:-512}"
CRAG_MAX_CHARS="${CRAG_MAX_CHARS:-3000}"
CRAG_UPPER_THRESHOLD="${CRAG_UPPER_THRESHOLD:-0.592}"
CRAG_LOWER_THRESHOLD="${CRAG_LOWER_THRESHOLD:--0.995}"
CRAG_AMBIGUOUS_ACTION_MAP="${CRAG_AMBIGUOUS_ACTION_MAP:-best}"
CRAG_INCORRECT_ACTION_MAP="${CRAG_INCORRECT_ACTION_MAP:-no}"
CRAG_SKIP_IMAGE_QUERIES="${CRAG_SKIP_IMAGE_QUERIES:-1}"

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

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    echo "CRAG Routing Adapter"
    echo ""
    echo "Required:"
    echo "  CRAG_EVALUATOR_PATH     CRAG evaluator checkpoint, or similarity fallback"
    echo ""
    echo "Optional:"
    echo "  CRAG_TARGETS            Targets (default: mmlu,squad,natural_questions,hotpotqa,webqa)"
    echo "  CRAG_SOURCE_DIR         Source route dir (default: route/results/distilbert)"
    echo "  CRAG_OUTPUT_DIR         Output route dir (default: route/results/crag)"
    echo "  CRAG_CANDIDATE_ACTIONS  paragraph,document or target (default: paragraph,document)"
    echo "  CRAG_NDOCS              Evidence docs scored per action (default: TOP_K or 1)"
    echo "  CRAG_UPPER_THRESHOLD    Correct threshold (default: 0.592)"
    echo "  CRAG_LOWER_THRESHOLD    Ambiguous threshold (default: -0.995)"
    echo "  CRAG_AMBIGUOUS_ACTION_MAP  best/action/target:action (default: best)"
    echo "  CRAG_INCORRECT_ACTION_MAP  action/target:action (default: no)"
    echo "  CRAG_SKIP_IMAGE_QUERIES 1 keeps source image routes for image rows (default: 1)"
    echo "  CRAG_PYTHON_BIN       Python executable for CRAG routing"
    echo "  CRAG_ALLOW_MISSING_QUERY_FEATURES 1 skips missing query-feature rows"
    exit 0
fi

echo "============================================"
echo " CRAG Routing Adapter"
echo "============================================"
echo " Evaluator:        ${CRAG_EVALUATOR_PATH}"
echo " Targets:          ${CRAG_TARGETS}"
echo " Source:           ${CRAG_SOURCE_DIR}"
echo " Output:           ${CRAG_OUTPUT_DIR}"
echo " Query BGE dir:    ${CRAG_QUERY_BGE_DIR}"
echo " Candidate actions:${CRAG_CANDIDATE_ACTIONS}"
echo " N docs:           ${CRAG_NDOCS}"
echo " Batch size:       ${CRAG_BATCH_SIZE}"
echo " Thresholds:       upper=${CRAG_UPPER_THRESHOLD}, lower=${CRAG_LOWER_THRESHOLD}"
echo " Ambiguous map:    ${CRAG_AMBIGUOUS_ACTION_MAP}"
echo " Incorrect map:    ${CRAG_INCORRECT_ACTION_MAP}"
echo " Image fallback:   ${CRAG_SKIP_IMAGE_QUERIES}"
echo " Device:           ${CRAG_DEVICE}"
echo " Python:           ${CRAG_PYTHON_BIN}"
echo "============================================"

ARGS=(
    --evaluator_path "$CRAG_EVALUATOR_PATH"
    --source_route_dir "$CRAG_SOURCE_DIR"
    --output_dir "$CRAG_OUTPUT_DIR"
    --targets "$CRAG_TARGETS"
    --query_bge_dir "$CRAG_QUERY_BGE_DIR"
    --candidate_actions "$CRAG_CANDIDATE_ACTIONS"
    --ndocs "$CRAG_NDOCS"
    --batch_size "$CRAG_BATCH_SIZE"
    --device "$CRAG_DEVICE"
    --max_seq_length "$CRAG_MAX_SEQ_LENGTH"
    --max_chars "$CRAG_MAX_CHARS"
    --upper_threshold "$CRAG_UPPER_THRESHOLD"
    --lower_threshold "$CRAG_LOWER_THRESHOLD"
    --ambiguous_action_map "$CRAG_AMBIGUOUS_ACTION_MAP"
    --incorrect_action_map "$CRAG_INCORRECT_ACTION_MAP"
)

if [[ "$CRAG_SKIP_IMAGE_QUERIES" == "1" ]]; then
    ARGS+=(--skip_image_queries)
fi
if [[ "$CRAG_ALLOW_MISSING_QUERY_FEATURES" == "1" ]]; then
    ARGS+=(--allow_missing_query_features)
fi

"$CRAG_PYTHON_BIN" route/crag_route_adapter.py "${ARGS[@]}"
