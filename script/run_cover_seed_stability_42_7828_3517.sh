#!/usr/bin/env bash
set -euo pipefail

# Seed-stability supplement for the COVER main-table routers.
#
# Seed 42 is the existing main-table run:
#   DistilBERT checkpoint: route/train/checkpoints/distilbert_vib_strict_d40
#   T5-large checkpoint  : route/train/checkpoints/t5-large_vib_large_v2
#
# This script only trains/routes/evaluates the two supplemental seeds by default:
#   7828, 3517

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PYTHON:-/opt/conda/envs/universalrag/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="${PYTHON_FALLBACK:-python}"
fi

MODEL_PATH="${MODEL_PATH:-qwen-api:qwen3.6-plus}"
MODEL_NAME="${MODEL_PATH##*/}"
SEEDS="${SEEDS-7828 3517}"
TARGETS_ALL="${TARGETS_ALL:-mmlu,squad,natural_questions,hotpotqa,webqa}"
TARGETS_CORRECTED=(${TARGETS_CORRECTED:-squad natural_questions})
TARGETS_DIRECT=(${TARGETS_DIRECT:-mmlu hotpotqa webqa})

FORCE_TRAIN="${FORCE_TRAIN:-0}"
FORCE_ROUTE="${FORCE_ROUTE:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"
RUN_DISTILBERT="${RUN_DISTILBERT:-1}"
RUN_T5="${RUN_T5:-1}"
RUN_SCORE="${RUN_SCORE:-1}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"

TOP_K="${TOP_K:-1}"
ALPHA="${ALPHA:-0.2}"
NFRAMES="${NFRAMES:-1}"
NFRAMES_TAG="${NFRAMES//,/_}"
NFRAMES_TAG="${NFRAMES_TAG//:/}"

QUERY_BGE_DIR="${QUERY_BGE_DIR:-eval/features/query/bge-large}"
QUERY_INTERNVIDEO_DIR="${QUERY_INTERNVIDEO_DIR:-eval/features/query/internvideo}"
QUERY_BGE_D40_DIR="${QUERY_BGE_D40_DIR:-eval/features/query_test_d40/bge-large}"
QUERY_INTERNVIDEO_D40_DIR="${QUERY_INTERNVIDEO_D40_DIR:-eval/features/query_test_d40/internvideo}"
HOTPOTQA_TEXT_FEATS="${HOTPOTQA_TEXT_FEATS:-eval/features/text/hotpotqa_raw_context.pkl}"
export HOTPOTQA_TEXT_FEATS

FIXED_ROOT_TEMPLATE="${FIXED_ROOT_TEMPLATE:-eval/results_qwen36plus_api_compare_fixed_{modality}/${MODEL_NAME}/t5-large}"
FIXED_DOCUMENT_CORRECTED_ROOT="${FIXED_DOCUMENT_CORRECTED_ROOT:-eval/results_qwen36plus_api_compare_fixed_document_corrected/${MODEL_NAME}/t5-large}"
FIXED_ROOT_OVERRIDES="${FIXED_ROOT_OVERRIDES:-document=${FIXED_DOCUMENT_CORRECTED_ROOT}}"

BASE_ROUTE_DISTILBERT="${BASE_ROUTE_DISTILBERT:-route/results_bayes_probs_strict_d40_test_s30_hybrid}"
BASE_ROUTE_T5="${BASE_ROUTE_T5:-route/results_bayes_probs_strict_d40_test_t5_s31_hybrid}"
BASE_ROUTE_UNIVERSAL="${BASE_ROUTE_UNIVERSAL:-route/results_universalrag_qwen36plus_test}"

ALPHA_PRIOR_BY_TARGET_CORRECTED="${ALPHA_PRIOR_BY_TARGET_CORRECTED:-mmlu=3,1,1,0.8;squad=0.5,10,2,0.2;natural_questions=0.5,10,1,0.2;hotpotqa=0.5,2,10,0.2;webqa=1.8,0.6,0.2,3}"
ALPHA_PRIOR_BY_TARGET_DIRECT="${ALPHA_PRIOR_BY_TARGET_DIRECT:-mmlu=10,1,1,0.2;squad=0.5,10,2,0.2;natural_questions=0.5,10,1,0.2;hotpotqa=0.5,2,10,0.2;webqa=0.2,1,2,10;triviaqa=0.5,10,2,0.2;lara=0.5,1,10,0.2;truthfulqa=10,1,1,0.2;visual_rag=0.2,1,2,10}"

export EVAL_PARTIAL_SAVE_EVERY="${EVAL_PARTIAL_SAVE_EVERY:-25}"
export EVAL_RESUME_PARTIAL="${EVAL_RESUME_PARTIAL:-1}"
export EVAL_SAVE_EVERY="${EVAL_SAVE_EVERY:-50}"
export EVAL_RESUME="${EVAL_RESUME:-1}"
export EVAL_SINGLE_RETRIEVER_CACHE="${EVAL_SINGLE_RETRIEVER_CACHE:-1}"
export EVAL_GC_EVERY="${EVAL_GC_EVERY:-100}"

mkdir -p logs

log_step() {
  echo ""
  echo "============================================================"
  echo "$*"
  echo "============================================================"
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

check_api_key() {
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
  esac
}

check_cuda() {
  if [[ "$REQUIRE_CUDA" != "1" ]]; then
    return 0
  fi
  "$PYTHON" - <<PY
import sys
import torch
if not torch.cuda.is_available():
    print("[ERROR] CUDA is not available. Set REQUIRE_CUDA=0 only if you intentionally want a CPU run.", file=sys.stderr)
    raise SystemExit(3)
print(f"[INFO] CUDA available: {torch.cuda.get_device_name(0)}")
PY
}

validate_inputs() {
  need_file "$HOTPOTQA_TEXT_FEATS"
  need_file "$QUERY_BGE_DIR/mmlu.pkl"
  need_file "$QUERY_BGE_DIR/hotpotqa.pkl"
  need_file "$QUERY_BGE_DIR/webqa.pkl"
  need_file "$QUERY_BGE_D40_DIR/squad.pkl"
  need_file "$QUERY_BGE_D40_DIR/natural_questions.pkl"
  need_file "eval/features/image/webqa_bge_captions.pkl"
  need_dir "$BASE_ROUTE_DISTILBERT"
  need_dir "$BASE_ROUTE_T5"
  need_dir "$BASE_ROUTE_UNIVERSAL"
  need_dir "route/train/checkpoints_strict_d40/distilbert"
  need_dir "route/train/checkpoints_large_genmetric_bf16/t5-large"
  need_dir "route/results_vib_strict_d40_distilbert_test"
  need_dir "route/results_vib_strict_d40_test_t5_v2"
  need_dir "route/results_bayes_probs_strict_d40_test_s30_hybrid"
  need_dir "route/results_bayes_probs_strict_d40_test_t5_s31_hybrid"
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
  if ! "$PYTHON" eval/score.py --result_file "$result_file" --target "$target"; then
    echo "[WARN] Scoring failed but experiment runner continues: $result_file" >&2
  fi
}

train_distilbert_seed() {
  local seed="$1"
  local ckpt="route/train/checkpoints/distilbert_vib_strict_d40_s${seed}"
  if [[ "$FORCE_TRAIN" != "1" && -f "$ckpt/router_model.pt" ]]; then
    echo "[SKIP] DistilBERT seed $seed checkpoint exists: $ckpt"
    return 0
  fi
  log_step "TRAIN DistilBERT VIB seed=$seed"
  SEED="$seed" \
  MODEL_NAME=distilbert-base-uncased \
  INPUT_PATH=dataset/query_nonvideo_large_strict_d40/router_train_fit_4class.json \
  CHECKPOINT_DIR="$ckpt" \
  INIT_CHECKPOINT_DIR=route/train/checkpoints_strict_d40/distilbert \
  TRAIN_SIZE=0.9 \
  MAX_INPUT_LENGTH=512 \
  NUM_TRAIN_EPOCHS=8 \
  LEARNING_RATE=2e-5 \
  WEIGHT_DECAY=0.01 \
  TRAIN_BATCH_SIZE=32 \
  EVAL_BATCH_SIZE=64 \
  GRADIENT_CLIP_NORM=1.0 \
  WARMUP_RATIO=0.1 \
  LATENT_DIM=128 \
  HIDDEN_DROPOUT_PROB=0.1 \
  PROTOTYPE_TEMPERATURE=1.0 \
  PROTOTYPE_MARGIN=0.24 \
  KL_WEIGHT=0.0005 \
  PROTO_WEIGHT=0.12 \
  EVI_WEIGHT=0.12 \
  PROTO_LOGIT_SCALE=0.6 \
  CLASS_WEIGHT_MODE=none \
  LABEL_SMOOTHING=0.0 \
  SELECT_METRIC=accuracy \
  MIXED_PRECISION=no \
  bash script/14_train_distilbert_vib.sh
}

train_t5_seed() {
  local seed="$1"
  local ckpt="route/train/checkpoints/t5-large_vib_large_v2_s${seed}"
  if [[ "$FORCE_TRAIN" != "1" && -f "$ckpt/router_model.pt" ]]; then
    echo "[SKIP] T5-large seed $seed checkpoint exists: $ckpt"
    return 0
  fi
  log_step "TRAIN T5-large VIB seed=$seed"
  SEED="$seed" \
  MODEL_NAME=route/train/checkpoints_large_genmetric_bf16/t5-large \
  INPUT_PATH=dataset/query_nonvideo_large/router_train_4class.json \
  CHECKPOINT_DIR="$ckpt" \
  INIT_CHECKPOINT_DIR="" \
  TRAIN_SIZE=0.9 \
  MAX_INPUT_LENGTH=512 \
  NUM_TRAIN_EPOCHS=6 \
  LEARNING_RATE=2e-5 \
  WEIGHT_DECAY=0.01 \
  TRAIN_BATCH_SIZE=1 \
  EVAL_BATCH_SIZE=1 \
  GRADIENT_ACCUMULATION_STEPS=16 \
  GRADIENT_CLIP_NORM=1.0 \
  WARMUP_RATIO=0.1 \
  LATENT_DIM=128 \
  HIDDEN_DROPOUT_PROB=0.1 \
  PROTOTYPE_TEMPERATURE=1.0 \
  PROTOTYPE_MARGIN=0.28 \
  KL_WEIGHT=0.0005 \
  PROTO_WEIGHT=0.18 \
  EVI_WEIGHT=0.10 \
  PROTO_LOGIT_SCALE=0.7 \
  CLASS_WEIGHT_MODE=sqrt_balanced \
  CLASS_WEIGHT_CLIP_MIN=0.5 \
  CLASS_WEIGHT_CLIP_MAX=2.5 \
  LABEL_SMOOTHING=0.03 \
  SELECT_METRIC=macro_f1 \
  MIXED_PRECISION=bf16 \
  bash script/17_train_t5_vib.sh
}

route_seed() {
  local seed="$1"
  local router="$2"
  local ckpt="$3"
  local out_dir="$4"
  local script_path="$5"
  local expected="$out_dir/$router/webqa.json"
  if [[ "$FORCE_ROUTE" != "1" && -f "$expected" ]]; then
    echo "[SKIP] Route seed $seed router=$router exists: $out_dir/$router"
    return 0
  fi
  log_step "ROUTE router=$router seed=$seed"
  CHECKPOINT_DIR="$ckpt" \
  OUTPUT_DIR="$out_dir" \
  ROUTER_NAME="$router" \
  INCLUDE_TARGETS="$TARGETS_ALL" \
  "$script_path"
}

run_cover_from_fixed() {
  local seed="$1"
  local target="$2"
  local router="$3"
  local route_dir="$4"
  local base_route_dir="$5"
  local output_root="$6"
  local tag="$7"
  local output_file="${output_root}/${MODEL_NAME}/${router}/${target}_top${TOP_K}_${ALPHA}_${NFRAMES_TAG}_bayes_${tag}.json"
  if [[ "$FORCE_EVAL" != "1" && -f "$output_file" ]]; then
    echo "[SKIP] Existing corrected COVER result: $output_file"
    score_result "$target" "$output_file"
    return 0
  fi
  log_step "EVAL corrected COVER target=$target router=$router seed=$seed"
  "$PYTHON" eval/eval_bayes_vib_posterior_from_fixed.py \
    --model_path "$MODEL_PATH" \
    --top_k "$TOP_K" \
    --alpha "$ALPHA" \
    --nframes "$NFRAMES" \
    --alpha_prior_by_target "$ALPHA_PRIOR_BY_TARGET_CORRECTED" \
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
    --seed "$seed" \
    --bayes_tag "$tag"
  score_result "$target" "$output_file"
}

run_cover_direct() {
  local seed="$1"
  local target="$2"
  local router="$3"
  local route_dir="$4"
  local output_root="$5"
  local tag="$6"
  local output_file="${output_root}/${MODEL_NAME}/${router}/${target}_top${TOP_K}_${ALPHA}_${NFRAMES_TAG}_bayes_${tag}.json"
  if [[ "$FORCE_EVAL" != "1" && -f "$output_file" ]]; then
    echo "[SKIP] Existing direct COVER result: $output_file"
    score_result "$target" "$output_file"
    return 0
  fi
  local extra=()
  if [[ "$target" == "webqa" ]]; then
    extra+=(--bge_image_retrieval)
  fi
  log_step "EVAL direct COVER target=$target router=$router seed=$seed"
  "$PYTHON" eval/eval_bayes_vib_posterior.py \
    --model_path "$MODEL_PATH" \
    --router_model "$router" \
    --target "$target" \
    --top_k "$TOP_K" \
    --alpha "$ALPHA" \
    --nframes "$NFRAMES" \
    --route_dir "$route_dir" \
    --base_route_dir "$BASE_ROUTE_UNIVERSAL" \
    --output_root "$output_root" \
    --query_bge_dir "$QUERY_BGE_DIR" \
    --query_internvideo_dir "$QUERY_INTERNVIDEO_DIR" \
    --bayes_tag "$tag" \
    --alpha_prior_by_target "$ALPHA_PRIOR_BY_TARGET_DIRECT" \
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
    --seed "$seed" \
    "${extra[@]}"
  score_result "$target" "$output_file"
}

run_seed() {
  local seed="$1"
  local distilbert_ckpt="route/train/checkpoints/distilbert_vib_strict_d40_s${seed}"
  local t5_ckpt="route/train/checkpoints/t5-large_vib_large_v2_s${seed}"
  local distilbert_route="route/results_vib_strict_d40_distilbert_test_s${seed}"
  local t5_route="route/results_vib_strict_d40_test_t5_v2_s${seed}"
  local corrected_distilbert_root="eval/results_seed${seed}_cover_distilbert_corrected"
  local corrected_t5_root="eval/results_seed${seed}_cover_t5_corrected"
  local direct_root="eval/results_webqa_hotpot_mmlu_seed${seed}_cover"

  log_step "START supplemental seed=$seed"

  if [[ "$RUN_DISTILBERT" == "1" ]]; then
    train_distilbert_seed "$seed"
    route_seed "$seed" distilbert "$distilbert_ckpt" "$distilbert_route" "script/15_route_distilbert_vib.sh"
    for target in "${TARGETS_CORRECTED[@]}"; do
      run_cover_from_fixed "$seed" "$target" distilbert "$distilbert_route" "$BASE_ROUTE_DISTILBERT" "$corrected_distilbert_root" "all3_qwen36plus_api_distilbert_s${seed}_corrected_docstore_tau10_beta0p1_softtop2_theta_posteriorverifier"
    done
    for target in "${TARGETS_DIRECT[@]}"; do
      run_cover_direct "$seed" "$target" distilbert "$distilbert_route" "$direct_root" "cover_distilbert_s${seed}_candidate_verifier_softtop2_theta_posteriorverifier"
    done
  fi

  if [[ "$RUN_T5" == "1" ]]; then
    train_t5_seed "$seed"
    route_seed "$seed" t5-large "$t5_ckpt" "$t5_route" "script/18_route_t5_vib.sh"
    for target in "${TARGETS_CORRECTED[@]}"; do
      run_cover_from_fixed "$seed" "$target" t5-large "$t5_route" "$BASE_ROUTE_T5" "$corrected_t5_root" "all3_qwen36plus_api_t5_s${seed}_corrected_docstore_tau10_beta0p1_softtop2_theta_posteriorverifier"
    done
    for target in "${TARGETS_DIRECT[@]}"; do
      run_cover_direct "$seed" "$target" t5-large "$t5_route" "$direct_root" "cover_t5_s${seed}_candidate_verifier_softtop2_theta_posteriorverifier"
    done
  fi

  log_step "DONE supplemental seed=$seed"
}

check_api_key
check_cuda
validate_inputs

log_step "COVER seed-stability supplement"
echo "Existing main-table seed : 42"
echo "Supplemental seeds       : $SEEDS"
echo "Model path               : $MODEL_PATH"
echo "Python                   : $PYTHON"
echo "Run DistilBERT           : $RUN_DISTILBERT"
echo "Run T5-large             : $RUN_T5"
echo "Force train/route/eval   : $FORCE_TRAIN / $FORCE_ROUTE / $FORCE_EVAL"

for seed in $SEEDS; do
  run_seed "$seed"
done

log_step "ALL SUPPLEMENTAL SEEDS FINISHED"
echo "Use seed 42 existing main-table results plus supplemental seeds: 7828, 3517."
