#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-eval/results_crossgen_mmlu_noonly_fixed}"
MODEL_PATHS="${MODEL_PATHS:-dmxapi:gpt-4o,glm:glm-4.6v,deepseek:deepseek-v4-pro}"
ROUTERS="${ROUTERS:-distilbert,t5-large}"
FORCE_EVAL="${FORCE_EVAL:-0}"

export EVAL_RESUME_PARTIAL="${EVAL_RESUME_PARTIAL:-1}"
export EVAL_PARTIAL_SAVE_EVERY="${EVAL_PARTIAL_SAVE_EVERY:-10}"
export EVAL_GC_EVERY="${EVAL_GC_EVERY:-5}"
export EVAL_SINGLE_RETRIEVER_CACHE="${EVAL_SINGLE_RETRIEVER_CACHE:-1}"
export QWEN_FORCE_FINAL_ANSWER_PASS_MCQ="${QWEN_FORCE_FINAL_ANSWER_PASS_MCQ:-1}"
export GLM_THINKING="${GLM_THINKING:-disabled}"
export GLM_API_THINKING="${GLM_API_THINKING:-$GLM_THINKING}"
export DEEPSEEK_THINKING="${DEEPSEEK_THINKING:-enabled}"
export DEEPSEEK_API_THINKING="${DEEPSEEK_API_THINKING:-$DEEPSEEK_THINKING}"
export DEEPSEEK_REASONING_EFFORT="${DEEPSEEK_REASONING_EFFORT:-low}"
export DEEPSEEK_MCQ_MAX_TOKENS="${DEEPSEEK_MCQ_MAX_TOKENS:-256}"
export GLM_EMPTY_RESPONSE_FALLBACK="${GLM_EMPTY_RESPONSE_FALLBACK:-0}"
export DEEPSEEK_EMPTY_RESPONSE_FALLBACK="${DEEPSEEK_EMPTY_RESPONSE_FALLBACK:-0}"
export DMXAPI_EMPTY_RESPONSE_FALLBACK="${DMXAPI_EMPTY_RESPONSE_FALLBACK:-0}"
export GLM_INVALID_MCQ_FALLBACK="${GLM_INVALID_MCQ_FALLBACK:-0}"
export DEEPSEEK_INVALID_MCQ_FALLBACK="${DEEPSEEK_INVALID_MCQ_FALLBACK:-0}"
export DMXAPI_INVALID_MCQ_FALLBACK="${DMXAPI_INVALID_MCQ_FALLBACK:-0}"

route_dir_for() {
  case "$1" in
    distilbert) echo "route/results_vib_strict_d40_distilbert_test" ;;
    t5-large) echo "route/results_vib_strict_d40_test_t5_v2" ;;
    *) echo "[ERROR] unsupported router: $1" >&2; exit 1 ;;
  esac
}

base_route_dir_for() {
  case "$1" in
    distilbert) echo "route/results_bayes_probs_strict_d40_test_s30_hybrid" ;;
    t5-large) echo "route/results_bayes_probs_strict_d40_test_t5_s31_hybrid" ;;
    *) echo "[ERROR] unsupported router: $1" >&2; exit 1 ;;
  esac
}

router_tag_for() {
  local router="$1"
  router="${router//-/_}"
  echo "$router"
}

old_ifs="$IFS"
IFS=, read -r -a models <<< "$MODEL_PATHS"
IFS=, read -r -a routers <<< "$ROUTERS"
IFS="$old_ifs"

for model_path in "${models[@]}"; do
  model_path="${model_path#${model_path%%[![:space:]]*}}"
  model_path="${model_path%${model_path##*[![:space:]]}}"
  [[ -z "$model_path" ]] && continue

  model_name="${model_path##*/}"

  for router in "${routers[@]}"; do
    router="${router#${router%%[![:space:]]*}}"
    router="${router%${router##*[![:space:]]}}"
    [[ -z "$router" ]] && continue

    route_dir="$(route_dir_for "$router")"
    base_route_dir="$(base_route_dir_for "$router")"
    router_tag="$(router_tag_for "$router")"
    tag="mmlu_noonly_crossgen_fixed_${router_tag}_tau10_beta0p1"
    result_file="${OUTPUT_ROOT}/${model_name}/${router}/mmlu_top1_0.2_1_bayes_${tag}.json"

    if [[ "$FORCE_EVAL" != "1" && -f "$result_file" ]]; then
      echo "[SKIP] $result_file already exists. Set FORCE_EVAL=1 to overwrite."
    else
      echo ""
      echo "================ MMLU noonly fixed: model=${model_path}, router=${router} ================"
      "$PYTHON_BIN" eval/eval_bayes_vib_posterior.py \
        --model_path "$model_path" \
        --router_model "$router" \
        --target mmlu \
        --top_k 1 \
        --alpha 0.2 \
        --nframes 1 \
        --route_dir "$route_dir" \
        --base_route_dir "$base_route_dir" \
        --output_root "$OUTPUT_ROOT" \
        --query_bge_dir eval/features/query_test_d40/bge-large \
        --query_internvideo_dir eval/features/query_test_d40/internvideo \
        --bayes_tag "$tag" \
        --alpha_prior 1,1,1,1 \
        --alpha_prior_by_target mmlu=30,0.2,0.2,0.05 \
        --tau 10.0 \
        --beta_cost 0.1 \
        --modality_costs 0.0,2.0,3.0,4.0 \
        --default_confidence 0.72 \
        --uncertainty_threshold 0.35 \
        --decision_mode mean \
        --fallback_when_uncertain 1 \
        --online_update 0 \
        --eta 1.0 \
        --rho 0.0 \
        --penalty 0.5 \
        --spread 0.25 \
        --use_penalty_update 1 \
        --soft_top_n 1 \
        --soft_weight_mode theta \
        --soft_store_candidates 1 \
        --hybrid_use_base 1 \
        --vib_prob_field probs \
        --vib_uncertainty_low 0.28 \
        --vib_uncertainty_high 0.45 \
        --vib_weight_low 0.35 \
        --vib_weight_high 0.85 \
        --dynamic_tau_min 0.35 \
        --dynamic_tau_max 1.8 \
        --evidence_saturation 8.0 \
        --posterior_verifier 0 \
        --posterior_verifier_choice_only 0 \
        --posterior_verifier_max_new_tokens 8 \
        --posterior_evidence_max_chars 1200
    fi

    if [[ -f "$result_file" ]]; then
      "$PYTHON_BIN" eval/score.py --target mmlu --result_file "$result_file"
    else
      echo "[WARN] Missing result file after run: $result_file"
    fi
  done
done
