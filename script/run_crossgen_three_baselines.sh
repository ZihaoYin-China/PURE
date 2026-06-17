#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON:-/opt/conda/envs/universalrag/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

GENERATOR_MODEL_PATHS="${GENERATOR_MODEL_PATHS:-dmxapi:gpt-4o,glm:glm-4.6v,deepseek:deepseek-v4-pro}"
TARGETS="${TARGETS:-mmlu,squad,natural_questions,hotpotqa,webqa}"
TOP_K="${TOP_K:-1}"
ALPHA="${ALPHA:-0.2}"
NFRAMES="${NFRAMES:-1}"
QUERY_BGE_DIR="${QUERY_BGE_DIR:-eval/features/query/bge-large}"
QUERY_INTERNVIDEO_DIR="${QUERY_INTERNVIDEO_DIR:-eval/features/query/internvideo}"

export HOTPOTQA_TEXT_FEATS="${HOTPOTQA_TEXT_FEATS:-eval/features/text/hotpotqa_raw_context.pkl}"
export EVAL_SAVE_EVERY="${EVAL_SAVE_EVERY:-50}"
export EVAL_RESUME="${EVAL_RESUME:-1}"
export EVAL_SINGLE_RETRIEVER_CACHE="${EVAL_SINGLE_RETRIEVER_CACHE:-1}"

check_key() {
  local model="$1"
  case "$model" in
    dmxapi:*)
      [[ -n "${DMXAPI_API_KEY:-}${DMX_API_KEY:-}" ]] || {
        echo "[ERROR] Missing DMXAPI_API_KEY or DMX_API_KEY for $model" >&2
        exit 2
      }
      ;;
    glm:*|zhipu:*|glm-*)
      [[ -n "${GLM_API_KEY:-}${ZHIPU_API_KEY:-}${BIGMODEL_API_KEY:-}" ]] || {
        echo "[ERROR] Missing GLM_API_KEY, ZHIPU_API_KEY, or BIGMODEL_API_KEY for $model" >&2
        exit 2
      }
      ;;
    deepseek:*|deepseek-*)
      [[ -n "${DEEPSEEK_API_KEY:-}" ]] || {
        echo "[ERROR] Missing DEEPSEEK_API_KEY for $model" >&2
        exit 2
      }
      ;;
  esac
}

run_eval() {
  local label="$1"
  local model="$2"
  local router="$3"
  local route_dir="$4"
  local output_root="$5"
  local target="$6"
  local model_name="${model##*/}"
  local output_file="${output_root}/${model_name}/${router}/${target}_top${TOP_K}_${ALPHA}_${NFRAMES}.json"

  if [[ -f "$output_file" ]]; then
    echo "[SKIP] ${label} model=${model} target=${target}"
    return 0
  fi

  echo "[RUN] ${label} model=${model} target=${target}"
  local extra=()
  if [[ "$target" == "webqa" ]]; then
    extra+=(--bge_image_retrieval)
  fi

  "$PYTHON_BIN" eval/eval.py \
    --model_path "$model" \
    --router_model "$router" \
    --target "$target" \
    --top_k "$TOP_K" \
    --alpha "$ALPHA" \
    --nframes "$NFRAMES" \
    --query_bge_dir "$QUERY_BGE_DIR" \
    --query_internvideo_dir "$QUERY_INTERNVIDEO_DIR" \
    --route_dir "$route_dir" \
    --output_root "$output_root" \
    "${extra[@]}"
}

old_ifs="$IFS"
IFS=',' read -r -a generators <<< "$GENERATOR_MODEL_PATHS"
IFS="$old_ifs"
IFS=',' read -r -a targets <<< "$TARGETS"
IFS="$old_ifs"

for model in "${generators[@]}"; do
  model="${model#${model%%[![:space:]]*}}"
  model="${model%${model##*[![:space:]]}}"
  [[ -z "$model" ]] && continue
  check_key "$model"

  for target in "${targets[@]}"; do
    target="${target#${target%%[![:space:]]*}}"
    target="${target%${target##*[![:space:]]}}"
    [[ -z "$target" ]] && continue

    run_eval \
      "Hard-T5-large" \
      "$model" \
      "t5-large" \
      "route/results_large_strict_d40_test" \
      "eval/results_crossgen_baselines_20260604_hard" \
      "$target"

    run_eval \
      "UniversalRAG-T5-large" \
      "$model" \
      "t5-large" \
      "route/results_universalrag_qwen36plus_test" \
      "eval/results_crossgen_baselines_20260604_universalrag" \
      "$target"

    run_eval \
      "Self-RAG" \
      "$model" \
      "selfrag" \
      "route/results_fair_test_selfrag_targetmap" \
      "eval/results_crossgen_baselines_20260604_selfrag" \
      "$target"
  done
done

echo "[DONE] Cross-generator baseline generation finished."
