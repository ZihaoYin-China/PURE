#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

MODEL_PATH="${MODEL_PATH:-qwen2.5vl:7b}"
MODEL_NAME="${MODEL_PATH##*/}"
TOP_K="${TOP_K:-1}"
ALPHA="${ALPHA:-0.2}"
NFRAMES="${NFRAMES:-1}"
NFRAMES_TAG="${NFRAMES//,/_}"
NFRAMES_TAG="${NFRAMES_TAG//:/}"

ROUTE_DIR="${ROUTE_DIR:-route/results_vib}"
OUTPUT_ROOT="${OUTPUT_ROOT:-eval/results_bayes_vib}"
QUERY_BGE_DIR="${QUERY_BGE_DIR:-eval/features/query/bge-large}"
QUERY_INTERNVIDEO_DIR="${QUERY_INTERNVIDEO_DIR:-eval/features/query/internvideo}"

TARGETS_ENV="${TARGETS:-}"
if [ -n "$TARGETS_ENV" ]; then
  read -r -a TARGETS <<< "$(echo "$TARGETS_ENV" | tr ',' ' ')"
else
  TARGETS=("mmlu" "squad" "natural_questions" "hotpotqa" "webqa")
fi

FORCE_EVAL="${FORCE_EVAL:-0}"

BAYES_ALPHA_PRIOR="${BAYES_ALPHA_PRIOR:-1,1,1,1}"
BAYES_TAU="${BAYES_TAU:-8.0}"
BAYES_BETA_COST="${BAYES_BETA_COST:-0.1}"
BAYES_MODALITY_COSTS="${BAYES_MODALITY_COSTS:-0.0,0.25,0.45,0.60}"
BAYES_DEFAULT_CONFIDENCE="${BAYES_DEFAULT_CONFIDENCE:-0.72}"
BAYES_UNCERTAINTY_THRESHOLD="${BAYES_UNCERTAINTY_THRESHOLD:-0.35}"
BAYES_DECISION_MODE="${BAYES_DECISION_MODE:-mean}"
BAYES_FALLBACK_WHEN_UNCERTAIN="${BAYES_FALLBACK_WHEN_UNCERTAIN:-1}"
BAYES_ONLINE_UPDATE="${BAYES_ONLINE_UPDATE:-0}"
BAYES_ETA="${BAYES_ETA:-1.0}"
BAYES_RHO="${BAYES_RHO:-0.0}"
BAYES_SEED="${BAYES_SEED:-42}"
BAYES_TAG="${BAYES_TAG:-vib_tau${BAYES_TAU}_beta${BAYES_BETA_COST}}"
BAYES_TAG="${BAYES_TAG//./p}"
BAYES_SOFT_TOP_N="${BAYES_SOFT_TOP_N:-1}"
BAYES_SOFT_WEIGHT_MODE="${BAYES_SOFT_WEIGHT_MODE:-theta}"
BAYES_SOFT_FUSION_MODE="${BAYES_SOFT_FUSION_MODE:-auto}"
BAYES_SOFT_FUSION_MAX_NEW_TOKENS="${BAYES_SOFT_FUSION_MAX_NEW_TOKENS:-64}"
BAYES_SOFT_STORE_CANDIDATES="${BAYES_SOFT_STORE_CANDIDATES:-1}"
BAYES_TAG_EFFECTIVE="$BAYES_TAG"
if [ "${BAYES_SOFT_TOP_N}" -gt 1 ]; then
  SOFT_TAG="softtop${BAYES_SOFT_TOP_N}_${BAYES_SOFT_WEIGHT_MODE}_${BAYES_SOFT_FUSION_MODE}"
  if [[ "$BAYES_TAG_EFFECTIVE" != *"$SOFT_TAG"* ]]; then
    BAYES_TAG_EFFECTIVE="${BAYES_TAG_EFFECTIVE}_${SOFT_TAG}"
  fi
fi

mkdir -p logs

for TARGET in "${TARGETS[@]}"; do
  RESULT_FILE="${OUTPUT_ROOT}/${MODEL_NAME}/distilbert/${TARGET}_top${TOP_K}_${ALPHA}_${NFRAMES_TAG}_bayes_${BAYES_TAG_EFFECTIVE}.json"
  if [ "$FORCE_EVAL" = "1" ] || [ ! -f "$RESULT_FILE" ]; then
    echo ""
    echo "================ VIB BAYES EVAL: target=${TARGET} ================"
    python eval/eval_bayes.py \
      --model_path "$MODEL_PATH" \
      --router_model distilbert \
      --target "$TARGET" \
      --top_k "$TOP_K" \
      --alpha "$ALPHA" \
      --nframes "$NFRAMES" \
      --route_dir "$ROUTE_DIR" \
      --output_root "$OUTPUT_ROOT" \
      --query_bge_dir "$QUERY_BGE_DIR" \
      --query_internvideo_dir "$QUERY_INTERNVIDEO_DIR" \
      --bayes_tag "$BAYES_TAG_EFFECTIVE" \
      --alpha_prior "$BAYES_ALPHA_PRIOR" \
      --tau "$BAYES_TAU" \
      --beta_cost "$BAYES_BETA_COST" \
      --modality_costs "$BAYES_MODALITY_COSTS" \
      --default_confidence "$BAYES_DEFAULT_CONFIDENCE" \
      --uncertainty_threshold "$BAYES_UNCERTAINTY_THRESHOLD" \
      --decision_mode "$BAYES_DECISION_MODE" \
      --fallback_when_uncertain "$BAYES_FALLBACK_WHEN_UNCERTAIN" \
      --online_update "$BAYES_ONLINE_UPDATE" \
      --eta "$BAYES_ETA" \
      --rho "$BAYES_RHO" \
      --penalty "${BAT_PENALTY:-0.5}" \
      --spread "${BAT_SPREAD:-0.25}" \
      --use_penalty_update "${BAT_USE_PENALTY_UPDATE:-1}" \
      --soft_top_n "$BAYES_SOFT_TOP_N" \
      --soft_weight_mode "$BAYES_SOFT_WEIGHT_MODE" \
      --soft_fusion_mode "$BAYES_SOFT_FUSION_MODE" \
      --soft_fusion_max_new_tokens "$BAYES_SOFT_FUSION_MAX_NEW_TOKENS" \
      --soft_store_candidates "$BAYES_SOFT_STORE_CANDIDATES" \
      --seed "$BAYES_SEED" | tee "logs/eval_bayes_vib_${TARGET}.log"
  else
    echo "[SKIP] $RESULT_FILE already exists."
  fi
done

echo ""
echo "================ VIB BAYES SCORING SUMMARY ================"

python - <<PY
import os
import sys

project_root = r"$PROJECT_ROOT"
sys.path.insert(0, project_root)
from eval.score import score_file

model_name = r"$MODEL_NAME"
top_k = r"$TOP_K"
alpha = r"$ALPHA"
nframes_tag = r"$NFRAMES_TAG"
output_root = r"$OUTPUT_ROOT"
bayes_tag = r"$BAYES_TAG_EFFECTIVE"
targets = [x for x in "${TARGETS[*]}".split() if x]

row = {"Router": "distilbert_vib"}
for target in targets:
    result_file = os.path.join(
        project_root,
        output_root,
        model_name,
        "distilbert",
        f"{target}_top{top_k}_{alpha}_{nframes_tag}_bayes_{bayes_tag}.json",
    )
    if not os.path.isfile(result_file):
        row[target] = "MISS"
        continue
    result = score_file(result_file, target=target)
    if target == "mmlu":
        row[target] = f"Acc={result['Accuracy']}"
    elif target in {"squad", "natural_questions", "hotpotqa"}:
        row[target] = f"EM={result['EM']}, F1={result['F1']}"
    elif target == "webqa":
        if "BERTScore" in result:
            row[target] = f"RL={result['ROUGE-L']}, BS={result['BERTScore']}"
        else:
            row[target] = f"RL={result['ROUGE-L']}"

print("Router:", row["Router"])
for target in targets:
    print(f"{target}: {row.get(target, 'MISS')}")
PY
