#!/usr/bin/env bash
set -euo pipefail

# Run the PURE-style baseline under multiple shared generators.
# Route files are generator-independent, so the same ROUTE_OUTPUT_DIR is reused.

GENERATOR_MODEL_PATHS="${GENERATOR_MODEL_PATHS:-qwen-api:qwen3.6-plus,dmxapi:gpt-4o,deepseek:deepseek-chat,glm:glm-4-plus}"
RESULTS_ROOT_BASE="${RESULTS_ROOT_BASE:-eval/results_universalrag_multigen_full}"
BASE_SCRIPT="${BASE_SCRIPT:-script/run_universalrag_qwen36plus_baseline.sh}"
ROUTE_OUTPUT_DIR="${ROUTE_OUTPUT_DIR:-route/results_universalrag_qwen36plus_full}"

slugify_model() {
  local value="$1"
  value="${value//\//_}"
  value="${value//:/_}"
  value="${value//./p}"
  value="${value//-/_}"
  echo "$value"
}

old_ifs="$IFS"
IFS=',' read -r -a generators <<< "$GENERATOR_MODEL_PATHS"
IFS="$old_ifs"

for model_path in "${generators[@]}"; do
  model_path="${model_path#${model_path%%[![:space:]]*}}"
  model_path="${model_path%${model_path##*[![:space:]]}}"
  [[ -z "$model_path" ]] && continue

  slug="$(slugify_model "$model_path")"
  results_root="${RESULTS_ROOT_BASE}_${slug}"

  echo ""
  echo "================ GENERATOR: $model_path ================"
  echo "[INFO] Results root: $results_root"
  echo "[INFO] Route output dir: $ROUTE_OUTPUT_DIR"

  env MODEL_PATH="$model_path" RESULTS_ROOT="$results_root" ROUTE_OUTPUT_DIR="$ROUTE_OUTPUT_DIR" bash "$BASE_SCRIPT"
done
