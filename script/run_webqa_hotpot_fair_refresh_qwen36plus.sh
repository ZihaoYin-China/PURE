#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/opt/conda/envs/universalrag/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python3"
fi

MODEL_PATH="${MODEL_PATH:-qwen-api:qwen3.6-plus}"
MODEL_NAME="${MODEL_PATH##*/}"
RESULTS_PREFIX="${RESULTS_PREFIX:-eval/results_webqa_hotpot_fair_refresh_20260528}"
ENV_SITE_PACKAGES="${ENV_SITE_PACKAGES:-/opt/conda/envs/universalrag/lib/python3.10/site-packages}"
BASE_SITE_PACKAGES="${BASE_SITE_PACKAGES:-/opt/conda/lib/python3.10/site-packages}"
PYTHONPATH_PREFIX=()
if [[ -d "$ENV_SITE_PACKAGES" ]]; then
  PYTHONPATH_PREFIX+=("$ENV_SITE_PACKAGES")
fi
if [[ -d "$BASE_SITE_PACKAGES" ]]; then
  PYTHONPATH_PREFIX+=("$BASE_SITE_PACKAGES")
fi
if [[ ${#PYTHONPATH_PREFIX[@]} -gt 0 ]]; then
  old_ifs="$IFS"
  IFS=:
  export PYTHONPATH="${PYTHONPATH_PREFIX[*]}${PYTHONPATH:+:$PYTHONPATH}"
  IFS="$old_ifs"
fi
QUERY_BGE_DIR="${QUERY_BGE_DIR:-eval/features/query/bge-large}"
QUERY_INTERNVIDEO_DIR="${QUERY_INTERNVIDEO_DIR:-eval/features/query/internvideo}"
HOTPOTQA_TEXT_FEATS="${HOTPOTQA_TEXT_FEATS:-eval/features/text/hotpotqa_raw_context.pkl}"
TOP_K="${TOP_K:-1}"
ALPHA="${ALPHA:-0.2}"
NFRAMES="${NFRAMES:-1}"
DRY_RUN="${DRY_RUN:-0}"
RUN_FIXED="${RUN_FIXED:-1}"
RUN_SINGLE="${RUN_SINGLE:-1}"
RUN_VERIFIER="${RUN_VERIFIER:-1}"
RUN_COVER="${RUN_COVER:-1}"
EVAL_SAVE_EVERY="${EVAL_SAVE_EVERY:-50}"
EVAL_RESUME="${EVAL_RESUME:-1}"
EVAL_SINGLE_RETRIEVER_CACHE="${EVAL_SINGLE_RETRIEVER_CACHE:-1}"
ALPHA_PRIOR_BY_TARGET="${ALPHA_PRIOR_BY_TARGET:-mmlu=10,1,1,0.2;squad=0.5,10,2,0.2;natural_questions=0.5,10,1,0.2;hotpotqa=0.5,2,10,0.2;webqa=0.2,1,2,10;triviaqa=0.5,10,2,0.2;lara=0.5,1,10,0.2;truthfulqa=10,1,1,0.2;visual_rag=0.2,1,2,10}"

if [[ "$#" -gt 0 ]]; then
  TARGETS=("$@")
else
  TARGETS=(webqa hotpotqa)
fi

has_target() {
  local want="$1"
  local target
  for target in "${TARGETS[@]}"; do
    [[ "$target" == "$want" ]] && return 0
  done
  return 1
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

for target in "${TARGETS[@]}"; do
  case "$target" in
    webqa|hotpotqa) ;;
    *) echo "[ERROR] This fair-refresh script only accepts webqa and hotpotqa, got: $target" >&2; exit 1 ;;
  esac
done

if [[ "$DRY_RUN" != "1" ]]; then
  case "$MODEL_PATH" in
    qwen-api:*|dashscope:*)
      if [[ -z "${DASHSCOPE_API_KEY:-}" && -z "${QWEN_API_KEY:-}" ]]; then
        echo "[ERROR] $MODEL_PATH requires DASHSCOPE_API_KEY or QWEN_API_KEY in the environment." >&2
        echo "        The script stopped before launching any paid/remote generations." >&2
        exit 2
      fi
      ;;
    openai:*|gpt:*|gpt-*)
      if [[ -z "${OPENAI_API_KEY:-}" ]]; then
        echo "[ERROR] $MODEL_PATH requires OPENAI_API_KEY in the environment." >&2
        echo "        The script stopped before launching any paid/remote generations." >&2
        exit 2
      fi
      ;;
    dmxapi:*)
      if [[ -z "${DMXAPI_API_KEY:-}" && -z "${DMX_API_KEY:-}" ]]; then
        echo "[ERROR] $MODEL_PATH requires DMXAPI_API_KEY or DMX_API_KEY in the environment." >&2
        echo "        The script stopped before launching any paid/remote generations." >&2
        exit 2
      fi
      ;;
    deepseek:*|deepseek-*)
      if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
        echo "[ERROR] $MODEL_PATH requires DEEPSEEK_API_KEY in the environment." >&2
        echo "        The script stopped before launching any paid/remote generations." >&2
        exit 2
      fi
      ;;
    glm:*|zhipu:*|glm-*)
      if [[ -z "${GLM_API_KEY:-}" && -z "${ZHIPU_API_KEY:-}" && -z "${BIGMODEL_API_KEY:-}" ]]; then
        echo "[ERROR] $MODEL_PATH requires GLM_API_KEY, ZHIPU_API_KEY, or BIGMODEL_API_KEY in the environment." >&2
        echo "        The script stopped before launching any paid/remote generations." >&2
        exit 2
      fi
      ;;
    openai-compatible:*)
      if [[ -z "${GENERATOR_API_KEY:-}" && -z "${OPENAI_COMPATIBLE_API_KEY:-}" ]]; then
        echo "[ERROR] $MODEL_PATH requires GENERATOR_API_KEY or OPENAI_COMPATIBLE_API_KEY in the environment." >&2
        echo "        The script stopped before launching any paid/remote generations." >&2
        exit 2
      fi
      ;;
  esac
fi

need_file "$HOTPOTQA_TEXT_FEATS"
need_file "$QUERY_BGE_DIR/hotpotqa.pkl"
if has_target webqa; then
  need_file "$QUERY_BGE_DIR/webqa.pkl"
  need_file "eval/features/image/webqa_bge_captions.pkl"
fi

need_dir "route/fixed_route_baselines_strict_d40_test"
need_dir "route/results_universalrag_qwen36plus_test"
need_dir "route/results_large_strict_d40_test"
need_dir "route/results_vib_strict_d40_distilbert_test"
need_dir "route/results_vib_strict_d40_test_t5_v2"

export HOTPOTQA_TEXT_FEATS
export EVAL_SAVE_EVERY
export EVAL_RESUME
export EVAL_SINGLE_RETRIEVER_CACHE

common_eval_args=(
  --model_path "$MODEL_PATH"
  --top_k "$TOP_K"
  --alpha "$ALPHA"
  --nframes "$NFRAMES"
  --query_bge_dir "$QUERY_BGE_DIR"
  --query_internvideo_dir "$QUERY_INTERNVIDEO_DIR"
)

nframes_tag() {
  local tag="${NFRAMES//,/_}"
  tag="${tag//:/}"
  printf "%s" "$tag"
}

result_complete() {
  local path="$1"
  [[ -f "$path" ]]
}

skip_if_complete() {
  local path="$1"
  if result_complete "$path"; then
    echo "[SKIP] Existing complete result: $path"
    return 0
  fi
  return 1
}

run_eval_py() {
  local target="$1"
  local router_model="$2"
  local route_dir="$3"
  local output_root="$4"
  local output_file="${output_root}/${MODEL_NAME}/${router_model}/${target}_top${TOP_K}_${ALPHA}_$(nframes_tag).json"
  if skip_if_complete "$output_file"; then
    return 0
  fi
  local extra=()
  if [[ "$target" == "webqa" ]]; then
    extra+=(--bge_image_retrieval)
  fi
  run "$PYTHON" eval/eval.py \
    "${common_eval_args[@]}" \
    --router_model "$router_model" \
    --target "$target" \
    --route_dir "$route_dir" \
    --output_root "$output_root" \
    "${extra[@]}"
}

run_fixed() {
  local target="$1"
  local modality="$2"
  run_eval_py "$target" t5-large \
    "route/fixed_route_baselines_strict_d40_test/${modality}" \
    "${RESULTS_PREFIX}_fixed_${modality}"
}

run_verifier_from_fixed() {
  local target="$1"
  local router_model="$2"
  local route_dir="$3"
  local tag="$4"
  local prob_field="$5"
  local output_file="${RESULTS_PREFIX}_no_bayes_verifier/${MODEL_NAME}/${router_model}/${target}_top${TOP_K}_${ALPHA}_$(nframes_tag)_${tag}.json"
  if skip_if_complete "$output_file"; then
    return 0
  fi
  run "$PYTHON" eval/eval_topk_verifier_from_fixed.py \
    --model_path "$MODEL_PATH" \
    --router_model "$router_model" \
    --target "$target" \
    --route_dir "$route_dir" \
    --output_root "${RESULTS_PREFIX}_no_bayes_verifier" \
    --fixed_root_template "${RESULTS_PREFIX}_fixed_{modality}/${MODEL_NAME}/t5-large" \
    --mode topk_verifier \
    --tag "$tag" \
    --prob_field "$prob_field" \
    --top_n 2 \
    --top_k "$TOP_K" \
    --alpha "$ALPHA" \
    --nframes "$NFRAMES" \
    --posterior_verifier 1 \
    --posterior_verifier_choice_only 1 \
    --partial_save_every "$EVAL_SAVE_EVERY"
}

run_cover() {
  local target="$1"
  local router_model="$2"
  local route_dir="$3"
  local tag="$4"
  local output_file="${RESULTS_PREFIX}_cover/${MODEL_NAME}/${router_model}/${target}_top${TOP_K}_${ALPHA}_$(nframes_tag)_bayes_${tag}.json"
  if skip_if_complete "$output_file"; then
    return 0
  fi
  local extra=()
  if [[ "$target" == "webqa" ]]; then
    extra+=(--bge_image_retrieval)
  fi
  run "$PYTHON" eval/eval_bayes_vib_posterior.py \
    --model_path "$MODEL_PATH" \
    --router_model "$router_model" \
    --target "$target" \
    --top_k "$TOP_K" \
    --alpha "$ALPHA" \
    --nframes "$NFRAMES" \
    --route_dir "$route_dir" \
    --base_route_dir route/results_universalrag_qwen36plus_test \
    --output_root "${RESULTS_PREFIX}_cover" \
    --query_bge_dir "$QUERY_BGE_DIR" \
    --query_internvideo_dir "$QUERY_INTERNVIDEO_DIR" \
    --bayes_tag "$tag" \
    --alpha_prior_by_target "$ALPHA_PRIOR_BY_TARGET" \
    --tau 10.0 \
    --beta_cost 0.1 \
    --modality_costs 0.0,0.25,0.45,0.60 \
    --soft_top_n 2 \
    --soft_weight_mode theta \
    --soft_store_candidates 1 \
    --hybrid_use_base 1 \
    --vib_prob_field probs \
    --posterior_agreement_weight 1.0 \
    --posterior_conflict_weight 0.35 \
    --posterior_route_weight 0.15 \
    --posterior_evidence_weight 0.05 \
    --posterior_empty_penalty 1.0 \
    --posterior_non_answer_penalty 0.85 \
    --posterior_verifier 1 \
    --posterior_verifier_choice_only 1 \
    --posterior_verifier_max_new_tokens 64 \
    --posterior_evidence_max_chars 1200 \
    "${extra[@]}"
}

run "$PYTHON" -B -c "import importlib.util; missing=[m for m in (\"numpy\",\"openai\",\"torch\",\"tqdm\") if importlib.util.find_spec(m) is None]; raise SystemExit(\"Missing Python modules: \"+\", \".join(missing) if missing else 0)"

echo "[INFO] Model: $MODEL_PATH"
echo "[INFO] Results prefix: $RESULTS_PREFIX"
echo "[INFO] Python: $PYTHON"
echo "[INFO] PYTHONPATH fallback: $ENV_SITE_PACKAGES:$BASE_SITE_PACKAGES"
echo "[INFO] HotpotQA text features: $HOTPOTQA_TEXT_FEATS"
echo "[INFO] WebQA image retrieval: BGE caption features"
echo "[INFO] Targets: ${TARGETS[*]}"

if [[ "$RUN_FIXED" == "1" ]]; then
  for target in "${TARGETS[@]}"; do
    run_fixed "$target" no
    run_fixed "$target" paragraph
    run_fixed "$target" document
    if [[ "$target" == "webqa" ]]; then
      run_fixed "$target" image
    fi
    run_fixed "$target" oracle
  done
fi

if [[ "$RUN_SINGLE" == "1" ]]; then
  for target in "${TARGETS[@]}"; do
    run_eval_py "$target" distilbert route/results_universalrag_qwen36plus_test "${RESULTS_PREFIX}_universalrag"
    run_eval_py "$target" t5-large route/results_universalrag_qwen36plus_test "${RESULTS_PREFIX}_universalrag"
    run_eval_py "$target" distilbert route/results_large_strict_d40_test "${RESULTS_PREFIX}_hard"
    run_eval_py "$target" t5-large route/results_large_strict_d40_test "${RESULTS_PREFIX}_hard"
    run_eval_py "$target" adaptive_rag route/results_universalrag_qwen36plus_test "${RESULTS_PREFIX}_adaptive_self"
    run_eval_py "$target" selfrag route/results_universalrag_qwen36plus_test "${RESULTS_PREFIX}_adaptive_self"
    run_eval_py "$target" distilbert route/results_vib_strict_d40_distilbert_test "${RESULTS_PREFIX}_vib_only"
    run_eval_py "$target" t5-large route/results_vib_strict_d40_test_t5_v2 "${RESULTS_PREFIX}_vib_only"
  done
fi

if [[ "$RUN_VERIFIER" == "1" ]]; then
  for target in "${TARGETS[@]}"; do
    run_verifier_from_fixed "$target" distilbert route/results_large_strict_d40_test classifier_distilbert_verifier_no_bayes_top2_refresh auto
    run_verifier_from_fixed "$target" t5-large route/results_large_strict_d40_test classifier_t5large_verifier_no_bayes_top2_refresh auto
    run_verifier_from_fixed "$target" distilbert route/results_vib_strict_d40_distilbert_test vib_distilbert_verifier_no_bayes_top2_refresh probs
    run_verifier_from_fixed "$target" t5-large route/results_vib_strict_d40_test_t5_v2 vib_t5large_verifier_no_bayes_top2_refresh probs
  done
fi

if [[ "$RUN_COVER" == "1" ]]; then
  for target in "${TARGETS[@]}"; do
    run_cover "$target" distilbert route/results_vib_strict_d40_distilbert_test cover_distilbert_candidate_verifier_softtop2_theta_posteriorverifier_refresh
    run_cover "$target" t5-large route/results_vib_strict_d40_test_t5_v2 cover_t5_candidate_verifier_softtop2_theta_posteriorverifier_refresh
  done
fi

echo "[INFO] Fair WebQA/HotpotQA refresh commands completed."
echo "[INFO] Summarize with: $PYTHON analysis/summarize_webqa_hotpot_fair_refresh.py --results_prefix $RESULTS_PREFIX"
