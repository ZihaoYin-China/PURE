#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

MODEL_PATH="${MODEL_PATH:-qwen2.5vl:7b}"
MODEL_NAME="${MODEL_PATH##*/}"
ROUTER_MODEL="${ROUTER_MODEL:-distilbert}"
TOP_K="${TOP_K:-1}"
ALPHA="${ALPHA:-0.2}"
NFRAMES="${NFRAMES:-1}"
NFRAMES_TAG="${NFRAMES//,/_}"
NFRAMES_TAG="${NFRAMES_TAG//:/}"

ROUTE_DIR="${ROUTE_DIR:-route/results_vib}"
BASE_ROUTE_DIR="${BASE_ROUTE_DIR:-route/results_bayes_probs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-eval/results_bayes_vib_posterior}"
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
BAYES_ALPHA_PRIOR_BY_TARGET="${BAYES_ALPHA_PRIOR_BY_TARGET:-mmlu=10,1,1,0.2;squad=0.5,10,2,0.2;natural_questions=0.5,10,1,0.2;hotpotqa=0.5,2,10,0.2;webqa=0.2,1,2,10;triviaqa=0.5,10,2,0.2;lara=0.5,1,10,0.2;truthfulqa=10,1,1,0.2;visual_rag=0.2,1,2,10}"
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
BAYES_TAG="${BAYES_TAG:-vibposteriorverifier_tau${BAYES_TAU}_beta${BAYES_BETA_COST}}"
BAYES_TAG="${BAYES_TAG//./p}"
BAYES_SOFT_TOP_N="${BAYES_SOFT_TOP_N:-2}"
BAYES_SOFT_WEIGHT_MODE="${BAYES_SOFT_WEIGHT_MODE:-theta}"
BAYES_SOFT_STORE_CANDIDATES="${BAYES_SOFT_STORE_CANDIDATES:-1}"
SELECTIVE_NO_RETRIEVAL="${SELECTIVE_NO_RETRIEVAL:-0}"
SELECTIVE_NO_TARGETS="${SELECTIVE_NO_TARGETS:-}"
SELECTIVE_NO_THETA_MIN="${SELECTIVE_NO_THETA_MIN:-0.55}"
SELECTIVE_NO_UTILITY_MARGIN="${SELECTIVE_NO_UTILITY_MARGIN:-0.03}"
SELECTIVE_NO_UNCERTAINTY_MAX="${SELECTIVE_NO_UNCERTAINTY_MAX:--1.0}"
SELECTIVE_SINGLE_BRANCH="${SELECTIVE_SINGLE_BRANCH:-0}"
SELECTIVE_SINGLE_BRANCH_TARGETS="${SELECTIVE_SINGLE_BRANCH_TARGETS:-}"
SELECTIVE_SINGLE_BRANCH_THETA_MIN="${SELECTIVE_SINGLE_BRANCH_THETA_MIN:-0.62}"
SELECTIVE_SINGLE_BRANCH_UTILITY_MARGIN="${SELECTIVE_SINGLE_BRANCH_UTILITY_MARGIN:-0.08}"
SELECTIVE_SINGLE_BRANCH_UNCERTAINTY_MAX="${SELECTIVE_SINGLE_BRANCH_UNCERTAINTY_MAX:--1.0}"
SELECTIVE_INCLUDE_NO_CANDIDATE="${SELECTIVE_INCLUDE_NO_CANDIDATE:-0}"
SELECTIVE_INCLUDE_NO_TARGETS="${SELECTIVE_INCLUDE_NO_TARGETS:-}"
SELECTIVE_INCLUDE_NO_UTILITY_GAP_MAX="${SELECTIVE_INCLUDE_NO_UTILITY_GAP_MAX:--1.0}"
BAYES_TAG_EFFECTIVE="$BAYES_TAG"
if [ "${BAYES_SOFT_TOP_N}" -gt 1 ]; then
  SOFT_TAG="softtop${BAYES_SOFT_TOP_N}_${BAYES_SOFT_WEIGHT_MODE}_posteriorverifier"
  if [[ "$BAYES_TAG_EFFECTIVE" != *"$SOFT_TAG"* ]]; then
    BAYES_TAG_EFFECTIVE="${BAYES_TAG_EFFECTIVE}_${SOFT_TAG}"
  fi
fi
if [ "$SELECTIVE_NO_RETRIEVAL" = "1" ]; then
  SELECTIVE_NO_TAG="selectiveno_t${SELECTIVE_NO_THETA_MIN}_m${SELECTIVE_NO_UTILITY_MARGIN}"
  SELECTIVE_NO_TAG="${SELECTIVE_NO_TAG//./p}"
  if [[ "$BAYES_TAG_EFFECTIVE" != *"$SELECTIVE_NO_TAG"* ]]; then
    BAYES_TAG_EFFECTIVE="${BAYES_TAG_EFFECTIVE}_${SELECTIVE_NO_TAG}"
  fi
fi
if [ "$SELECTIVE_SINGLE_BRANCH" = "1" ]; then
  SELECTIVE_SINGLE_TAG="selective1_t${SELECTIVE_SINGLE_BRANCH_THETA_MIN}_m${SELECTIVE_SINGLE_BRANCH_UTILITY_MARGIN}"
  SELECTIVE_SINGLE_TAG="${SELECTIVE_SINGLE_TAG//./p}"
  if [[ "$BAYES_TAG_EFFECTIVE" != *"$SELECTIVE_SINGLE_TAG"* ]]; then
    BAYES_TAG_EFFECTIVE="${BAYES_TAG_EFFECTIVE}_${SELECTIVE_SINGLE_TAG}"
  fi
fi
if [ "$SELECTIVE_INCLUDE_NO_CANDIDATE" = "1" ]; then
  SELECTIVE_INCLUDE_NO_TAG="includeno"
  if [[ "$SELECTIVE_INCLUDE_NO_UTILITY_GAP_MAX" != "-1.0" && "$SELECTIVE_INCLUDE_NO_UTILITY_GAP_MAX" != "-1" ]]; then
    SELECTIVE_INCLUDE_NO_TAG="${SELECTIVE_INCLUDE_NO_TAG}_g${SELECTIVE_INCLUDE_NO_UTILITY_GAP_MAX}"
  fi
  SELECTIVE_INCLUDE_NO_TAG="${SELECTIVE_INCLUDE_NO_TAG//./p}"
  if [[ "$BAYES_TAG_EFFECTIVE" != *"$SELECTIVE_INCLUDE_NO_TAG"* ]]; then
    BAYES_TAG_EFFECTIVE="${BAYES_TAG_EFFECTIVE}_${SELECTIVE_INCLUDE_NO_TAG}"
  fi
fi

HYBRID_USE_BASE="${HYBRID_USE_BASE:-1}"
VIB_PROB_FIELD="${VIB_PROB_FIELD:-auto}"
VIB_UNCERTAINTY_LOW="${VIB_UNCERTAINTY_LOW:-0.28}"
VIB_UNCERTAINTY_HIGH="${VIB_UNCERTAINTY_HIGH:-0.45}"
VIB_WEIGHT_LOW="${VIB_WEIGHT_LOW:-0.15}"
VIB_WEIGHT_HIGH="${VIB_WEIGHT_HIGH:-0.85}"
DYNAMIC_TAU_MIN="${DYNAMIC_TAU_MIN:-0.35}"
DYNAMIC_TAU_MAX="${DYNAMIC_TAU_MAX:-1.15}"
EVIDENCE_SATURATION="${EVIDENCE_SATURATION:-8.0}"

POSTERIOR_AGREEMENT_WEIGHT="${POSTERIOR_AGREEMENT_WEIGHT:-1.0}"
POSTERIOR_CONFLICT_WEIGHT="${POSTERIOR_CONFLICT_WEIGHT:-0.35}"
POSTERIOR_ROUTE_WEIGHT="${POSTERIOR_ROUTE_WEIGHT:-0.15}"
POSTERIOR_EVIDENCE_WEIGHT="${POSTERIOR_EVIDENCE_WEIGHT:-0.05}"
POSTERIOR_EMPTY_PENALTY="${POSTERIOR_EMPTY_PENALTY:-1.0}"
POSTERIOR_NON_ANSWER_PENALTY="${POSTERIOR_NON_ANSWER_PENALTY:-0.85}"
POSTERIOR_VERIFIER="${POSTERIOR_VERIFIER:-1}"
POSTERIOR_VERIFIER_CHOICE_ONLY="${POSTERIOR_VERIFIER_CHOICE_ONLY:-0}"
POSTERIOR_VERIFIER_MAX_NEW_TOKENS="${POSTERIOR_VERIFIER_MAX_NEW_TOKENS:-64}"
POSTERIOR_EVIDENCE_MAX_CHARS="${POSTERIOR_EVIDENCE_MAX_CHARS:-1200}"
POSTERIOR_SAFE_FALLBACK_FILE="${POSTERIOR_SAFE_FALLBACK_FILE:-}"
POSTERIOR_ACCEPT_MAX_SCORE_GAP="${POSTERIOR_ACCEPT_MAX_SCORE_GAP:--1.0}"
POSTERIOR_ACCEPT_MIN_SCORE_GAP="${POSTERIOR_ACCEPT_MIN_SCORE_GAP:--1.0}"
POSTERIOR_ACCEPT_MODALITY_PAIRS="${POSTERIOR_ACCEPT_MODALITY_PAIRS:-}"

mkdir -p logs

for TARGET in "${TARGETS[@]}"; do
  RESULT_FILE="${OUTPUT_ROOT}/${MODEL_NAME}/${ROUTER_MODEL}/${TARGET}_top${TOP_K}_${ALPHA}_${NFRAMES_TAG}_bayes_${BAYES_TAG_EFFECTIVE}.json"
  if [ "$FORCE_EVAL" = "1" ] || [ ! -f "$RESULT_FILE" ]; then
    echo ""
    echo "================ ${ROUTER_MODEL} VIB-BAYES POSTERIOR GEN EVAL: target=${TARGET} ================"
    python eval/eval_bayes_vib_posterior.py \
      --model_path "$MODEL_PATH" \
      --router_model "$ROUTER_MODEL" \
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
      --online_update "$BAYES_ONLINE_UPDATE" \
      --eta "$BAYES_ETA" \
      --rho "$BAYES_RHO" \
      --penalty "${BAT_PENALTY:-0.5}" \
      --spread "${BAT_SPREAD:-0.25}" \
      --use_penalty_update "${BAT_USE_PENALTY_UPDATE:-1}" \
      --soft_top_n "$BAYES_SOFT_TOP_N" \
      --soft_weight_mode "$BAYES_SOFT_WEIGHT_MODE" \
      --soft_store_candidates "$BAYES_SOFT_STORE_CANDIDATES" \
      --selective_no_retrieval "$SELECTIVE_NO_RETRIEVAL" \
      --selective_no_targets "$SELECTIVE_NO_TARGETS" \
      --selective_no_theta_min "$SELECTIVE_NO_THETA_MIN" \
      --selective_no_utility_margin "$SELECTIVE_NO_UTILITY_MARGIN" \
      --selective_no_uncertainty_max "$SELECTIVE_NO_UNCERTAINTY_MAX" \
      --selective_single_branch "$SELECTIVE_SINGLE_BRANCH" \
      --selective_single_branch_targets "$SELECTIVE_SINGLE_BRANCH_TARGETS" \
      --selective_single_branch_theta_min "$SELECTIVE_SINGLE_BRANCH_THETA_MIN" \
      --selective_single_branch_utility_margin "$SELECTIVE_SINGLE_BRANCH_UTILITY_MARGIN" \
      --selective_single_branch_uncertainty_max "$SELECTIVE_SINGLE_BRANCH_UNCERTAINTY_MAX" \
      --selective_include_no_candidate "$SELECTIVE_INCLUDE_NO_CANDIDATE" \
      --selective_include_no_targets "$SELECTIVE_INCLUDE_NO_TARGETS" \
      --selective_include_no_utility_gap_max "$SELECTIVE_INCLUDE_NO_UTILITY_GAP_MAX" \
      --seed "$BAYES_SEED" \
      --hybrid_use_base "$HYBRID_USE_BASE" \
      --vib_prob_field "$VIB_PROB_FIELD" \
      --vib_uncertainty_low "$VIB_UNCERTAINTY_LOW" \
      --vib_uncertainty_high "$VIB_UNCERTAINTY_HIGH" \
      --vib_weight_low "$VIB_WEIGHT_LOW" \
      --vib_weight_high "$VIB_WEIGHT_HIGH" \
      --dynamic_tau_min "$DYNAMIC_TAU_MIN" \
      --dynamic_tau_max "$DYNAMIC_TAU_MAX" \
      --evidence_saturation "$EVIDENCE_SATURATION" \
      --posterior_agreement_weight "$POSTERIOR_AGREEMENT_WEIGHT" \
      --posterior_conflict_weight "$POSTERIOR_CONFLICT_WEIGHT" \
      --posterior_route_weight "$POSTERIOR_ROUTE_WEIGHT" \
      --posterior_evidence_weight "$POSTERIOR_EVIDENCE_WEIGHT" \
      --posterior_empty_penalty "$POSTERIOR_EMPTY_PENALTY" \
      --posterior_non_answer_penalty "$POSTERIOR_NON_ANSWER_PENALTY" \
      --posterior_verifier "$POSTERIOR_VERIFIER" \
      --posterior_verifier_choice_only "$POSTERIOR_VERIFIER_CHOICE_ONLY" \
      --posterior_verifier_max_new_tokens "$POSTERIOR_VERIFIER_MAX_NEW_TOKENS" \
      --posterior_evidence_max_chars "$POSTERIOR_EVIDENCE_MAX_CHARS" \
      --posterior_safe_fallback_file "$POSTERIOR_SAFE_FALLBACK_FILE" \
      --posterior_accept_max_score_gap "$POSTERIOR_ACCEPT_MAX_SCORE_GAP" \
      --posterior_accept_min_score_gap "$POSTERIOR_ACCEPT_MIN_SCORE_GAP" \
      --posterior_accept_modality_pairs "$POSTERIOR_ACCEPT_MODALITY_PAIRS" | tee "logs/eval_bayes_vib_posterior_${ROUTER_MODEL}_${TARGET}.log"
  else
    echo "[SKIP] $RESULT_FILE already exists."
  fi
done

echo ""
echo "================ ${ROUTER_MODEL} VIB-BAYES POSTERIOR GEN SCORING SUMMARY ================"

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

router_model = r"$ROUTER_MODEL"
row = {"Router": f"{router_model}_vib_bayes_posterior_gen"}
for target in targets:
    result_file = os.path.join(
        project_root,
        output_root,
        model_name,
        router_model,
        f"{target}_top{top_k}_{alpha}_{nframes_tag}_bayes_{bayes_tag}.json",
    )
    if not os.path.isfile(result_file):
        row[target] = "MISS"
        continue
    result = score_file(result_file, target=target)
    if target in {"mmlu", "truthfulqa"}:
        row[target] = f"Acc={result['Accuracy']}"
    elif target in {"squad", "natural_questions", "hotpotqa", "triviaqa"}:
        row[target] = f"EM={result['EM']}, F1={result['F1']}"
    elif target in {"webqa", "lara", "visual_rag"}:
        if "BERTScore" in result:
            row[target] = f"RL={result['ROUGE-L']}, BS={result['BERTScore']}"
        else:
            row[target] = f"RL={result['ROUGE-L']}"

print("Router:", row["Router"])
for target in targets:
    print(f"{target}: {row.get(target, 'MISS')}")
PY
