#!/usr/bin/env bash
set -euo pipefail

# Run the WebQA/HotpotQA fair-refresh experiment under multiple shared generators.
# Defaults keep Qwen as the primary generator and add one GPT, DeepSeek, and GLM run.
# Override GENERATOR_MODEL_PATHS with a comma-separated list when you want exact versions.

GENERATOR_MODEL_PATHS="${GENERATOR_MODEL_PATHS:-qwen-api:qwen3.6-plus,dmxapi:gpt-4o,deepseek:deepseek-chat,glm:glm-4-plus}"
RESULTS_PREFIX_BASE="${RESULTS_PREFIX_BASE:-eval/results_webqa_hotpot_fair_refresh_multigen}"
BASE_SCRIPT="${BASE_SCRIPT:-script/run_webqa_hotpot_fair_refresh_qwen36plus.sh}"
DRY_RUN="${DRY_RUN:-0}"

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

if [[ ${#generators[@]} -eq 0 ]]; then
  echo "[ERROR] GENERATOR_MODEL_PATHS is empty." >&2
  exit 1
fi

for model_path in "${generators[@]}"; do
  model_path="${model_path#${model_path%%[![:space:]]*}}"
  model_path="${model_path%${model_path##*[![:space:]]}}"
  [[ -z "$model_path" ]] && continue

  slug="$(slugify_model "$model_path")"
  prefix="${RESULTS_PREFIX_BASE}_${slug}"

  echo ""
  echo "================ GENERATOR: $model_path ================"
  echo "[INFO] Results prefix: $prefix"

  run_env=(MODEL_PATH="$model_path" RESULTS_PREFIX="$prefix" DRY_RUN="$DRY_RUN")
  case "$model_path" in
    deepseek:*|deepseek-*)
      run_env=(GENERATOR_API_IMAGE_MODE="${DEEPSEEK_API_IMAGE_MODE:-caption}" "${run_env[@]}")
      ;;
  esac

  env "${run_env[@]}" bash "$BASE_SCRIPT" "$@"
done

cat <<EOF
[INFO] Multi-generator fair-refresh commands completed.
[INFO] Summarize each run with:
  python analysis/summarize_webqa_hotpot_fair_refresh.py --results_prefix <prefix> --model_name <model_path>
EOF
