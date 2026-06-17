#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

MODEL_PATH="${MODEL_PATH:-qwen-api:qwen3.6-plus}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/universalrag/bin/python}"
ROUTER_MODEL="${ROUTER_MODEL:-distilbert}"
ROUTE_DIR="${ROUTE_DIR:-route/results_ood_vib_full}"
BASE_ROUTE_DIR="${BASE_ROUTE_DIR:-route/results_universalrag_qwen36plus_ood_full}"
QUERY_BGE_DIR="${QUERY_BGE_DIR:-eval/features/query_ood/bge-large}"
QUERY_INTERNVIDEO_DIR="${QUERY_INTERNVIDEO_DIR:-eval/features/query_ood/internvideo}"

TOP_K_DEFAULT="${TOP_K_DEFAULT:-1}"
VISUAL_TOP_K="${VISUAL_TOP_K:-5}"
ALPHA="${ALPHA:-0.2}"
NFRAMES="${NFRAMES:-1}"
TAU="${TAU:-10.0}"
BETA_COST="${BETA_COST:-0.1}"
MODALITY_COSTS="${MODALITY_COSTS:-0.0,0.25,0.45,0.60}"
FORCE_EVAL="${FORCE_EVAL:-0}"

export EVAL_RESUME_PARTIAL="${EVAL_RESUME_PARTIAL:-1}"
export EVAL_PARTIAL_SAVE_EVERY="${EVAL_PARTIAL_SAVE_EVERY:-10}"
export EVAL_GC_EVERY="${EVAL_GC_EVERY:-5}"
export EVAL_SINGLE_RETRIEVER_CACHE="${EVAL_SINGLE_RETRIEVER_CACHE:-1}"
export EVAL_MAX_NEW_ROWS="${EVAL_MAX_NEW_ROWS:-100}"
EVAL_CHUNK_MAX_RUNS="${EVAL_CHUNK_MAX_RUNS:-50}"

TARGETS=(truthfulqa triviaqa lara visual_rag)
TARGET_PRIORS="mmlu=10,1,1,0.2;squad=0.5,10,2,0.2;natural_questions=0.5,10,1,0.2;hotpotqa=0.5,2,10,0.2;webqa=0.2,1,2,10;triviaqa=0.5,10,2,0.2;lara=0.5,1,10,0.2;truthfulqa=10,1,1,0.2;visual_rag=0.2,1,2,10"

run_one() {
  local target="$1"
  local top_k="$2"
  local output_root="$3"
  local tag="$4"
  local prior_by_target="$5"
  local verifier_choice_only="$6"

  local model_name="${MODEL_PATH##*/}"
  local nframes_tag="${NFRAMES//,/_}"
  nframes_tag="${nframes_tag//:/}"
  local result_file="${output_root}/${model_name}/${ROUTER_MODEL}/${target}_top${top_k}_${ALPHA}_${nframes_tag}_bayes_${tag}.json"

  if [[ "$FORCE_EVAL" != "1" && -f "$result_file" ]]; then
    echo "[SKIP] $result_file already exists."
    return
  fi

  local partial_file="${result_file}.partial"
  local attempt=1

  while [[ ! -f "$result_file" ]]; do
    if (( attempt > EVAL_CHUNK_MAX_RUNS )); then
      echo "[ERROR] Reached EVAL_CHUNK_MAX_RUNS=${EVAL_CHUNK_MAX_RUNS} before completing ${result_file}."
      return 1
    fi

    echo ""
    echo "================ COVER-DistilBERT OOD: target=${target}, top_k=${top_k}, tag=${tag}, attempt=${attempt}, max_new_rows=${EVAL_MAX_NEW_ROWS} ================"
    "$PYTHON_BIN" eval/eval_bayes_vib_posterior.py \
      --model_path "$MODEL_PATH" \
      --router_model "$ROUTER_MODEL" \
      --target "$target" \
      --top_k "$top_k" \
      --alpha "$ALPHA" \
      --nframes "$NFRAMES" \
      --route_dir "$ROUTE_DIR" \
      --base_route_dir "$BASE_ROUTE_DIR" \
      --output_root "$output_root" \
      --query_bge_dir "$QUERY_BGE_DIR" \
      --query_internvideo_dir "$QUERY_INTERNVIDEO_DIR" \
      --bayes_tag "$tag" \
      --alpha_prior "1,1,1,1" \
      --alpha_prior_by_target "$prior_by_target" \
      --tau "$TAU" \
      --beta_cost "$BETA_COST" \
      --modality_costs "$MODALITY_COSTS" \
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
      --soft_top_n 2 \
      --soft_weight_mode theta \
      --soft_store_candidates 1 \
      --hybrid_use_base 1 \
      --vib_prob_field probs \
      --vib_uncertainty_low 0.28 \
      --vib_uncertainty_high 0.45 \
      --vib_weight_low 0.15 \
      --vib_weight_high 0.85 \
      --dynamic_tau_min 0.35 \
      --dynamic_tau_max 1.15 \
      --evidence_saturation 8.0 \
      --posterior_agreement_weight 1.0 \
      --posterior_conflict_weight 0.35 \
      --posterior_route_weight 0.15 \
      --posterior_evidence_weight 0.05 \
      --posterior_empty_penalty 1.0 \
      --posterior_non_answer_penalty 0.85 \
      --posterior_verifier 1 \
      --posterior_verifier_choice_only "$verifier_choice_only" \
      --posterior_verifier_max_new_tokens 64 \
      --posterior_evidence_max_chars 1200

    if [[ -f "$result_file" ]]; then
      break
    fi
    if [[ ! -f "$partial_file" ]]; then
      echo "[ERROR] ${result_file} was not completed and ${partial_file} was not found."
      return 1
    fi
    echo "[INFO] ${result_file} not complete yet; continuing from partial (${partial_file})."
    attempt=$((attempt + 1))
  done
}

for target in "${TARGETS[@]}"; do
  top_k="$TOP_K_DEFAULT"
  if [[ "$target" == "visual_rag" ]]; then
    top_k="$VISUAL_TOP_K"
  fi
  run_one "$target" "$top_k" \
    "eval/results_cover_distilbert_qwen36plus_ood_noprior" \
    "cover_distilbert_ood_noprior_tau10_beta0p1_softtop2_theta_posteriorverifier" \
    "" \
    1
  run_one "$target" "$top_k" \
    "eval/results_cover_distilbert_qwen36plus_ood_prior_diag" \
    "cover_distilbert_ood_prior_diag_tau10_beta0p1_softtop2_theta_posteriorverifier" \
    "$TARGET_PRIORS" \
    0
done
