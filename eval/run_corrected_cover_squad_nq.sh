#!/usr/bin/env bash
set -euo pipefail

# One-shot runner for the SQuAD/NQ cells marked "rerun pending" in the paper tables.
# It runs the corrected-document-store variants for:
#   1) Hard top-2 + Ver. / classifier no-Bayes verifier
#   2) VIB + Ver. / VIB no-Bayes verifier
#   3) COVER / Bayes + VIB + posterior verifier

MODEL_PATH=${MODEL_PATH:-qwen-api:qwen3.6-plus}
MODEL_NAME=${MODEL_PATH##*/}
PYTHON=${PYTHON:-python}
TOP_K=${TOP_K:-1}
ALPHA=${ALPHA:-0.2}
NFRAMES=${NFRAMES:-1}
DRY_RUN=${DRY_RUN:-0}
FORCE=${FORCE:-0}
RUN_VERIFIER=${RUN_VERIFIER:-1}
RUN_COVER=${RUN_COVER:-1}
COVER_FROM_FIXED=${COVER_FROM_FIXED:-1}
RUN_SCORE=${RUN_SCORE:-1}

QUERY_BGE_DIR=${QUERY_BGE_DIR:-eval/features/query_test_d40/bge-large}
QUERY_INTERNVIDEO_DIR=${QUERY_INTERNVIDEO_DIR:-eval/features/query_test_d40/internvideo}

CLASSIFIER_VERIFIER_ROOT=${CLASSIFIER_VERIFIER_ROOT:-eval/results_ablation_classifier_verifier_no_bayes_corrected}
VIB_VERIFIER_ROOT=${VIB_VERIFIER_ROOT:-eval/results_ablation_vib_verifier_no_bayes_corrected}
COVER_DISTILBERT_ROOT=${COVER_DISTILBERT_ROOT:-eval/results_qwen36plus_api_all3_strict_d40_test_distilbert_corrected}
COVER_T5_ROOT=${COVER_T5_ROOT:-eval/results_qwen36plus_api_all3_strict_d40_test_t5_corrected}

FIXED_ROOT_TEMPLATE=${FIXED_ROOT_TEMPLATE:-}
if [[ -z "$FIXED_ROOT_TEMPLATE" ]]; then
  FIXED_ROOT_TEMPLATE="eval/results_qwen36plus_api_compare_fixed_{modality}/${MODEL_NAME}/t5-large"
fi
FIXED_DOCUMENT_CORRECTED_ROOT=${FIXED_DOCUMENT_CORRECTED_ROOT:-eval/results_qwen36plus_api_compare_fixed_document_corrected/${MODEL_NAME}/t5-large}
FIXED_ROOT_OVERRIDES=${FIXED_ROOT_OVERRIDES:-document=${FIXED_DOCUMENT_CORRECTED_ROOT}}

export EVAL_PARTIAL_SAVE_EVERY=${EVAL_PARTIAL_SAVE_EVERY:-25}
export EVAL_RESUME_PARTIAL=${EVAL_RESUME_PARTIAL:-1}
# Keep both paragraph/document text retrievers cached; SQuAD/NQ corrected document uses the paragraph store.
export EVAL_SINGLE_RETRIEVER_CACHE=${EVAL_SINGLE_RETRIEVER_CACHE:-0}

ALPHA_PRIOR_BY_TARGET=${ALPHA_PRIOR_BY_TARGET:-mmlu=3,1,1,0.8;squad=0.5,10,2,0.2;natural_questions=0.5,10,1,0.2;hotpotqa=0.5,2,10,0.2;webqa=1.8,0.6,0.2,3}

if [[ "$#" -gt 0 ]]; then
  TARGETS=("$@")
else
  TARGETS=(squad natural_questions)
fi

nframes_tag() {
  local tag="${NFRAMES//,/_}"
  tag="${tag//:/}"
  printf "%s" "$tag"
}

need_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "[ERROR] Missing required file: $path" >&2
    exit 1
  fi
}

need_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "[ERROR] Missing required directory: $path" >&2
    exit 1
  fi
}

run() {
  printf "\n[RUN]"
  printf " %q" "$@"
  printf "\n"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  "$@"
}

maybe_skip_complete() {
  local path="$1"
  if [[ "$FORCE" != "1" && -f "$path" ]]; then
    echo "[SKIP] Existing complete result: $path"
    return 0
  fi
  return 1
}

score_result() {
  local target="$1"
  local result_file="$2"
  if [[ "$RUN_SCORE" != "1" ]]; then
    return 0
  fi
  if [[ ! -f "$result_file" ]]; then
    echo "[WARN] Cannot score missing result: $result_file" >&2
    return 0
  fi
  run "$PYTHON" eval/score.py --result_file "$result_file" --target "$target"
}

check_model_credentials() {
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  case "$MODEL_PATH" in
    qwen-api:*|dashscope:*)
      if [[ -z "${DASHSCOPE_API_KEY:-}" && -z "${QWEN_API_KEY:-}" ]]; then
        echo "[ERROR] $MODEL_PATH requires DASHSCOPE_API_KEY or QWEN_API_KEY." >&2
        exit 2
      fi
      ;;
    openai:*|gpt:*|gpt-*)
      if [[ -z "${OPENAI_API_KEY:-}" ]]; then
        echo "[ERROR] $MODEL_PATH requires OPENAI_API_KEY." >&2
        exit 2
      fi
      ;;
    dmxapi:*)
      if [[ -z "${DMXAPI_API_KEY:-}" && -z "${DMX_API_KEY:-}" ]]; then
        echo "[ERROR] $MODEL_PATH requires DMXAPI_API_KEY or DMX_API_KEY." >&2
        exit 2
      fi
      ;;
    deepseek:*|deepseek-*)
      if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
        echo "[ERROR] $MODEL_PATH requires DEEPSEEK_API_KEY." >&2
        exit 2
      fi
      ;;
    glm:*|zhipu:*|glm-*)
      if [[ -z "${GLM_API_KEY:-}" && -z "${ZHIPU_API_KEY:-}" && -z "${BIGMODEL_API_KEY:-}" ]]; then
        echo "[ERROR] $MODEL_PATH requires GLM_API_KEY, ZHIPU_API_KEY, or BIGMODEL_API_KEY." >&2
        exit 2
      fi
      ;;
  esac
}

validate_target() {
  case "$1" in
    squad|natural_questions) ;;
    *) echo "[ERROR] This corrected pending runner only accepts squad and natural_questions, got: $1" >&2; exit 1 ;;
  esac
}

validate_inputs() {
  need_dir route/results_bayes_probs_strict_d40_test_s30_hybrid
  need_dir route/results_bayes_probs_strict_d40_test_t5_s31_hybrid
  need_dir route/results_vib_strict_d40_distilbert_test
  need_dir route/results_vib_strict_d40_test_t5_v2

  for target in "${TARGETS[@]}"; do
    validate_target "$target"
    need_file "${QUERY_BGE_DIR}/${target}.pkl"
    need_file "route/results_bayes_probs_strict_d40_test_s30_hybrid/distilbert/${target}.json"
    need_file "route/results_bayes_probs_strict_d40_test_t5_s31_hybrid/t5-large/${target}.json"
    need_file "route/results_vib_strict_d40_distilbert_test/distilbert/${target}.json"
    need_file "route/results_vib_strict_d40_test_t5_v2/t5-large/${target}.json"
    need_file "eval/results_qwen36plus_api_compare_fixed_no/${MODEL_NAME}/t5-large/${target}_top${TOP_K}_${ALPHA}_$(nframes_tag).json"
    need_file "eval/results_qwen36plus_api_compare_fixed_paragraph/${MODEL_NAME}/t5-large/${target}_top${TOP_K}_${ALPHA}_$(nframes_tag).json"
    need_file "${FIXED_DOCUMENT_CORRECTED_ROOT}/${target}_top${TOP_K}_${ALPHA}_$(nframes_tag).json"
  done
}

run_no_bayes_verifier() {
  local target="$1"
  local router="$2"
  local route_dir="$3"
  local output_root="$4"
  local tag="$5"
  local prob_field="$6"
  local output_file="${output_root}/${MODEL_NAME}/${router}/${target}_top${TOP_K}_${ALPHA}_$(nframes_tag)_${tag}.json"

  if maybe_skip_complete "$output_file"; then
    score_result "$target" "$output_file"
    return 0
  fi

  local resume=1
  if [[ "$FORCE" == "1" ]]; then
    resume=0
  fi

  run "$PYTHON" eval/eval_topk_verifier_from_fixed.py \
    --model_path "$MODEL_PATH" \
    --router_model "$router" \
    --target "$target" \
    --route_dir "$route_dir" \
    --output_root "$output_root" \
    --fixed_root_template "$FIXED_ROOT_TEMPLATE" \
    --fixed_root_overrides "$FIXED_ROOT_OVERRIDES" \
    --mode topk_verifier \
    --tag "$tag" \
    --prob_field "$prob_field" \
    --top_n 2 \
    --top_k "$TOP_K" \
    --alpha "$ALPHA" \
    --nframes "$NFRAMES" \
    --posterior_verifier 1 \
    --posterior_verifier_choice_only 1 \
    --resume "$resume" \
    --partial_save_every "$EVAL_PARTIAL_SAVE_EVERY"

  score_result "$target" "$output_file"
}

run_cover() {
  local target="$1"
  local router="$2"
  local route_dir="$3"
  local base_route_dir="$4"
  local output_root="$5"
  local tag="$6"
  local output_file="${output_root}/${MODEL_NAME}/${router}/${target}_top${TOP_K}_${ALPHA}_$(nframes_tag)_bayes_${tag}.json"

  if maybe_skip_complete "$output_file"; then
    score_result "$target" "$output_file"
    return 0
  fi

  if [[ "$COVER_FROM_FIXED" == "1" ]]; then
    run "$PYTHON" eval/eval_bayes_vib_posterior_from_fixed.py \
      --model_path "$MODEL_PATH" \
      --top_k "$TOP_K" \
      --alpha "$ALPHA" \
      --nframes "$NFRAMES" \
      --alpha_prior_by_target "$ALPHA_PRIOR_BY_TARGET" \
      --tau 10 \
      --beta_cost 0.1 \
      --modality_costs 0.0,0.25,0.45,0.60 \
      --soft_top_n 2 \
      --soft_weight_mode theta \
      --soft_store_candidates 1 \
      --hybrid_use_base 1 \
      --vib_prob_field probs \
      --vib_weight_low 0.35 \
      --vib_weight_high 0.85 \
      --dynamic_tau_max 1.8 \
      --posterior_conflict_weight 0.3 \
      --posterior_route_weight 0.12 \
      --posterior_verifier 1 \
      --posterior_verifier_choice_only 1 \
      --router_model "$router" \
      --target "$target" \
      --route_dir "$route_dir" \
      --base_route_dir "$base_route_dir" \
      --output_root "$output_root" \
      --fixed_root_template "$FIXED_ROOT_TEMPLATE" \
      --fixed_root_overrides "$FIXED_ROOT_OVERRIDES" \
      --bayes_tag "$tag"
  else
    run "$PYTHON" eval/eval_bayes_vib_posterior.py \
      --model_path "$MODEL_PATH" \
      --top_k "$TOP_K" \
      --alpha "$ALPHA" \
      --nframes "$NFRAMES" \
      --query_bge_dir "$QUERY_BGE_DIR" \
      --query_internvideo_dir "$QUERY_INTERNVIDEO_DIR" \
      --alpha_prior_by_target "$ALPHA_PRIOR_BY_TARGET" \
      --tau 10 \
      --beta_cost 0.1 \
      --modality_costs 0.0,0.25,0.45,0.60 \
      --soft_top_n 2 \
      --soft_weight_mode theta \
      --soft_store_candidates 1 \
      --hybrid_use_base 1 \
      --vib_prob_field probs \
      --vib_weight_low 0.35 \
      --vib_weight_high 0.85 \
      --dynamic_tau_max 1.8 \
      --posterior_conflict_weight 0.3 \
      --posterior_route_weight 0.12 \
      --posterior_verifier 1 \
      --posterior_verifier_choice_only 0 \
      --router_model "$router" \
      --target "$target" \
      --route_dir "$route_dir" \
      --base_route_dir "$base_route_dir" \
      --output_root "$output_root" \
      --bayes_tag "$tag"
  fi

  score_result "$target" "$output_file"
}

check_model_credentials
validate_inputs
run "$PYTHON" -B -c "import importlib.util; missing=[m for m in ('numpy','openai','torch','tqdm') if importlib.util.find_spec(m) is None]; raise SystemExit('Missing Python modules: '+', '.join(missing) if missing else 0)"

echo "[INFO] Model: $MODEL_PATH"
echo "[INFO] Python: $PYTHON"
echo "[INFO] Targets: ${TARGETS[*]}"
echo "[INFO] Fixed root template: $FIXED_ROOT_TEMPLATE"
echo "[INFO] Fixed root overrides: $FIXED_ROOT_OVERRIDES"
echo "[INFO] Verifier outputs: $CLASSIFIER_VERIFIER_ROOT ; $VIB_VERIFIER_ROOT"
echo "[INFO] COVER outputs: $COVER_DISTILBERT_ROOT ; $COVER_T5_ROOT"
echo "[INFO] COVER_FROM_FIXED: $COVER_FROM_FIXED"

if [[ "$RUN_VERIFIER" == "1" ]]; then
  for target in "${TARGETS[@]}"; do
    run_no_bayes_verifier \
      "$target" distilbert route/results_bayes_probs_strict_d40_test_s30_hybrid \
      "$CLASSIFIER_VERIFIER_ROOT" classifier_verifier_no_bayes_top2 auto
    run_no_bayes_verifier \
      "$target" t5-large route/results_bayes_probs_strict_d40_test_t5_s31_hybrid \
      "$CLASSIFIER_VERIFIER_ROOT" classifier_verifier_no_bayes_top2 auto
    run_no_bayes_verifier \
      "$target" distilbert route/results_vib_strict_d40_distilbert_test \
      "$VIB_VERIFIER_ROOT" vib_verifier_no_bayes_top2 probs
    run_no_bayes_verifier \
      "$target" t5-large route/results_vib_strict_d40_test_t5_v2 \
      "$VIB_VERIFIER_ROOT" vib_verifier_no_bayes_top2 probs
  done
fi

if [[ "$RUN_COVER" == "1" ]]; then
  for target in "${TARGETS[@]}"; do
    run_cover \
      "$target" distilbert route/results_vib_strict_d40_distilbert_test \
      route/results_bayes_probs_strict_d40_test_s30_hybrid \
      "$COVER_DISTILBERT_ROOT" \
      all3_qwen36plus_api_distilbert_corrected_docstore_tau10_beta0p1_softtop2_theta_posteriorverifier
    run_cover \
      "$target" t5-large route/results_vib_strict_d40_test_t5_v2 \
      route/results_bayes_probs_strict_d40_test_t5_s31_hybrid \
      "$COVER_T5_ROOT" \
      all3_qwen36plus_api_t5_corrected_docstore_tau10_beta0p1_softtop2_theta_posteriorverifier
  done
fi

echo "[INFO] Corrected SQuAD/NQ pending runner completed."
