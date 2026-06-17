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
BASE_ROUTE_DIR="${BASE_ROUTE_DIR:-route/results_bayes_probs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-eval/results_bayes_vib_hybrid}"
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
BAYES_ALPHA_PRIOR_BY_TARGET="${BAYES_ALPHA_PRIOR_BY_TARGET:-mmlu=10,1,1,0.2;squad=0.5,10,2,0.2;natural_questions=0.5,10,1,0.2;hotpotqa=0.5,2,10,0.2;webqa=0.2,1,2,10}"
BAYES_TAU="${BAYES_TAU:-8.0}"
BAYES_BETA_COST="${BAYES_BETA_COST:-0.1}"
BAYES_MODALITY_COSTS="${BAYES_MODALITY_COSTS:-0.0,0.25,0.45,0.60}"
BAYES_DEFAULT_CONFIDENCE="${BAYES_DEFAULT_CONFIDENCE:-0.72}"
BAYES_UNCERTAINTY_THRESHOLD="${BAYES_UNCERTAINTY_THRESHOLD:-0.35}"
BAYES_DECISION_MODE="${BAYES_DECISION_MODE:-mean}"
BAYES_FALLBACK_WHEN_UNCERTAIN="${BAYES_FALLBACK_WHEN_UNCERTAIN:-1}"
BAYES_ROUTER_PROBS_TEMPERATURE="${BAYES_ROUTER_PROBS_TEMPERATURE:-1.0}"
BAYES_ROUTER_PROBS_TEMPERATURE_BY_TARGET="${BAYES_ROUTER_PROBS_TEMPERATURE_BY_TARGET:-}"
BAYES_ROUTER_PROBS_BLEND_WITH_ORIGINAL="${BAYES_ROUTER_PROBS_BLEND_WITH_ORIGINAL:-0.0}"
BAYES_ROUTER_PROBS_BLEND_WITH_ORIGINAL_BY_TARGET="${BAYES_ROUTER_PROBS_BLEND_WITH_ORIGINAL_BY_TARGET:-}"
BAYES_ONLINE_UPDATE="${BAYES_ONLINE_UPDATE:-0}"
BAYES_ETA="${BAYES_ETA:-1.0}"
BAYES_RHO="${BAYES_RHO:-0.0}"
BAYES_SEED="${BAYES_SEED:-42}"
BAYES_TAG="${BAYES_TAG:-vibhybrid_tau${BAYES_TAU}_beta${BAYES_BETA_COST}}"
BAYES_TAG="${BAYES_TAG//./p}"
BAYES_SOFT_TOP_N="${BAYES_SOFT_TOP_N:-2}"
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

HYBRID_USE_BASE="${HYBRID_USE_BASE:-1}"
REUSE_BASE_RESULTS_FILE="${REUSE_BASE_RESULTS_FILE:-}"
REUSE_BASE_WHEN_SAME_PLAN="${REUSE_BASE_WHEN_SAME_PLAN:-0}"
VIB_PROB_FIELD="${VIB_PROB_FIELD:-auto}"
VIB_UNCERTAINTY_LOW="${VIB_UNCERTAINTY_LOW:-0.28}"
VIB_UNCERTAINTY_HIGH="${VIB_UNCERTAINTY_HIGH:-0.45}"
VIB_WEIGHT_LOW="${VIB_WEIGHT_LOW:-0.15}"
VIB_WEIGHT_HIGH="${VIB_WEIGHT_HIGH:-0.85}"
VIB_TARGET_PARAMS="${VIB_TARGET_PARAMS:-}"
DYNAMIC_TAU_MIN="${DYNAMIC_TAU_MIN:-0.35}"
DYNAMIC_TAU_MAX="${DYNAMIC_TAU_MAX:-1.15}"
EVIDENCE_SATURATION="${EVIDENCE_SATURATION:-8.0}"
PROTECT_BASE_MODALITIES="${PROTECT_BASE_MODALITIES:-}"
ALLOW_VIB_MODALITIES="${ALLOW_VIB_MODALITIES:-}"
PROTECT_BASE_SELECTED_MODALITIES="${PROTECT_BASE_SELECTED_MODALITIES:-}"
VIB_RISK_CONTROL="${VIB_RISK_CONTROL:-0}"
VIB_RISK_CONTROL_TARGETS="${VIB_RISK_CONTROL_TARGETS:-}"
VIB_RISK_CONTROL_BASE_MODALITIES="${VIB_RISK_CONTROL_BASE_MODALITIES:-no,paragraph}"
VIB_RISK_CONTROL_ALLOW_VIB_MODALITIES="${VIB_RISK_CONTROL_ALLOW_VIB_MODALITIES:-document,image}"
VIB_RISK_CONTROL_MAX_UNCERTAINTY="${VIB_RISK_CONTROL_MAX_UNCERTAINTY:-0.32}"
VIB_RISK_CONTROL_MIN_VIB_CONF="${VIB_RISK_CONTROL_MIN_VIB_CONF:-0.70}"
VIB_RISK_CONTROL_MIN_VIB_MARGIN="${VIB_RISK_CONTROL_MIN_VIB_MARGIN:-0.20}"
VIB_RISK_CONTROL_REQUIRE_DISAGREEMENT="${VIB_RISK_CONTROL_REQUIRE_DISAGREEMENT:-1}"

mkdir -p logs

for TARGET in "${TARGETS[@]}"; do
  RESULT_FILE="${OUTPUT_ROOT}/${MODEL_NAME}/t5-large/${TARGET}_top${TOP_K}_${ALPHA}_${NFRAMES_TAG}_bayes_${BAYES_TAG_EFFECTIVE}.json"
  if [ "$FORCE_EVAL" = "1" ] || [ ! -f "$RESULT_FILE" ]; then
    echo ""
    echo "================ T5-LARGE VIB-HYBRID BAYES EVAL: target=${TARGET} ================"
    python eval/eval_bayes_vib.py \
      --model_path "$MODEL_PATH" \
      --router_model t5-large \
      --target "$TARGET" \
      --top_k "$TOP_K" \
      --alpha "$ALPHA" \
      --nframes "$NFRAMES" \
      --route_dir "$ROUTE_DIR" \
      --base_route_dir "$BASE_ROUTE_DIR" \
      --output_root "$OUTPUT_ROOT" \
      --query_bge_dir "$QUERY_BGE_DIR" \
      --query_internvideo_dir "$QUERY_INTERNVIDEO_DIR" \
      --bayes_tag "$BAYES_TAG_EFFECTIVE" \
      --alpha_prior "$BAYES_ALPHA_PRIOR" \
      --alpha_prior_by_target "$BAYES_ALPHA_PRIOR_BY_TARGET" \
      --tau "$BAYES_TAU" \
      --beta_cost "$BAYES_BETA_COST" \
      --modality_costs "$BAYES_MODALITY_COSTS" \
      --default_confidence "$BAYES_DEFAULT_CONFIDENCE" \
      --uncertainty_threshold "$BAYES_UNCERTAINTY_THRESHOLD" \
      --decision_mode "$BAYES_DECISION_MODE" \
      --fallback_when_uncertain "$BAYES_FALLBACK_WHEN_UNCERTAIN" \
      --router_probs_temperature "$BAYES_ROUTER_PROBS_TEMPERATURE" \
      --router_probs_temperature_by_target "$BAYES_ROUTER_PROBS_TEMPERATURE_BY_TARGET" \
      --router_probs_blend_with_original "$BAYES_ROUTER_PROBS_BLEND_WITH_ORIGINAL" \
      --router_probs_blend_with_original_by_target "$BAYES_ROUTER_PROBS_BLEND_WITH_ORIGINAL_BY_TARGET" \
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
      --seed "$BAYES_SEED" \
      --hybrid_use_base "$HYBRID_USE_BASE" \
      --reuse_base_results_file "$REUSE_BASE_RESULTS_FILE" \
      --reuse_base_when_same_plan "$REUSE_BASE_WHEN_SAME_PLAN" \
      --vib_prob_field "$VIB_PROB_FIELD" \
      --vib_uncertainty_low "$VIB_UNCERTAINTY_LOW" \
      --vib_uncertainty_high "$VIB_UNCERTAINTY_HIGH" \
      --vib_weight_low "$VIB_WEIGHT_LOW" \
      --vib_weight_high "$VIB_WEIGHT_HIGH" \
      --vib_target_params "$VIB_TARGET_PARAMS" \
      --dynamic_tau_min "$DYNAMIC_TAU_MIN" \
      --dynamic_tau_max "$DYNAMIC_TAU_MAX" \
      --evidence_saturation "$EVIDENCE_SATURATION" \
      --protect_base_modalities "$PROTECT_BASE_MODALITIES" \
      --allow_vib_modalities "$ALLOW_VIB_MODALITIES" \
      --protect_base_selected_modalities "$PROTECT_BASE_SELECTED_MODALITIES" \
      --vib_risk_control "$VIB_RISK_CONTROL" \
      --vib_risk_control_targets "$VIB_RISK_CONTROL_TARGETS" \
      --vib_risk_control_base_modalities "$VIB_RISK_CONTROL_BASE_MODALITIES" \
      --vib_risk_control_allow_vib_modalities "$VIB_RISK_CONTROL_ALLOW_VIB_MODALITIES" \
      --vib_risk_control_max_uncertainty "$VIB_RISK_CONTROL_MAX_UNCERTAINTY" \
      --vib_risk_control_min_vib_conf "$VIB_RISK_CONTROL_MIN_VIB_CONF" \
      --vib_risk_control_min_vib_margin "$VIB_RISK_CONTROL_MIN_VIB_MARGIN" \
      --vib_risk_control_require_disagreement "$VIB_RISK_CONTROL_REQUIRE_DISAGREEMENT" | tee "logs/eval_bayes_vib_hybrid_t5large_${TARGET}.log"
  else
    echo "[SKIP] $RESULT_FILE already exists."
  fi
done

echo ""
echo "================ T5-LARGE VIB-HYBRID SCORING SUMMARY ================"

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

row = {"Router": "t5-large_vib_hybrid"}
for target in targets:
    result_file = os.path.join(
        project_root,
        output_root,
        model_name,
        "t5-large",
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
