#!/bin/bash
set -euo pipefail

# End-to-end non-video pipeline:
# 1. train distilbert and t5-large routers if needed
# 2. route with qwen / distilbert / t5-large
# 3. run generation eval
# 4. score and print a summary

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

MODEL_PATH="${MODEL_PATH:-qwen3-vl:8b}"
QWEN_ROUTER_MODEL="${QWEN_ROUTER_MODEL:-$MODEL_PATH}"

FORCE_TRAIN="${FORCE_TRAIN:-0}"
FORCE_ROUTE="${FORCE_ROUTE:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"
PREFER_LOCAL_INIT="${PREFER_LOCAL_INIT:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_DISTILBERT="${RUN_DISTILBERT:-1}"
RUN_T5_LARGE="${RUN_T5_LARGE:-1}"
RUN_QWEN="${RUN_QWEN:-1}"

DISTILBERT_CKPT="route/train/checkpoints/distilbert"
T5_CKPT="route/train/checkpoints/t5-large"

resolve_model_init_path() {
  local explicit_value="$1"
  local local_ckpt_dir="$2"
  local default_remote="$3"
  local local_ready="0"

  if [ -f "$local_ckpt_dir/config.json" ] && { [ -f "$local_ckpt_dir/pytorch_model.bin" ] || [ -f "$local_ckpt_dir/model.safetensors" ]; }; then
    local_ready="1"
  fi

  if [ -n "$explicit_value" ] && ! { [ "$PREFER_LOCAL_INIT" = "1" ] && [ "$local_ready" = "1" ] && [ "$explicit_value" = "$default_remote" ]; }; then
    echo "$explicit_value"
    return
  fi

  if [ "$local_ready" = "1" ]; then
    echo "$local_ckpt_dir"
  else
    echo "$default_remote"
  fi
}

DISTILBERT_MODEL_NAME="$(resolve_model_init_path "${DISTILBERT_MODEL_NAME:-}" "$DISTILBERT_CKPT" "distilbert-base-uncased")"
T5_MODEL_NAME="$(resolve_model_init_path "${T5_MODEL_NAME:-}" "$T5_CKPT" "google/flan-t5-large")"

required_files=(
  "script/2_train.sh"
  "script/6_eval_all_routers_qwen.sh"
  "route/train/data/train_data_distilbert_4class.json"
  "route/train/data/train_data_t5_4class.json"
)

for path in "${required_files[@]}"; do
  if [ ! -e "$path" ]; then
    echo "[ERROR] Missing required file: $path"
    exit 1
  fi
done

echo "============================================================"
echo "Project root       : $PROJECT_ROOT"
echo "Generation model   : $MODEL_PATH"
echo "Qwen router model  : $QWEN_ROUTER_MODEL"
echo "FORCE_TRAIN        : $FORCE_TRAIN"
echo "FORCE_ROUTE        : $FORCE_ROUTE"
echo "FORCE_EVAL         : $FORCE_EVAL"
echo "PREFER_LOCAL_INIT  : $PREFER_LOCAL_INIT"
echo "CUDA_ALLOC_CONF    : $PYTORCH_CUDA_ALLOC_CONF"
echo "RUN_DISTILBERT     : $RUN_DISTILBERT"
echo "RUN_T5_LARGE       : $RUN_T5_LARGE"
echo "RUN_QWEN           : $RUN_QWEN"
echo "DISTILBERT_INIT    : $DISTILBERT_MODEL_NAME"
echo "T5_INIT            : $T5_MODEL_NAME"
echo "============================================================"

mkdir -p logs

train_router_if_needed() {
  local router_name="$1"
  local ckpt_dir="$2"
  local init_model="$3"

  if [ "$FORCE_TRAIN" = "1" ] || [ ! -d "$ckpt_dir" ]; then
    echo ""
    echo "================ TRAIN: $router_name ================"
    bash script/2_train.sh "$router_name" "$init_model" | tee "logs/train_${router_name}.log"
  else
    echo "[SKIP] $router_name checkpoint already exists at $ckpt_dir"
  fi
}

if [ "$RUN_DISTILBERT" = "1" ]; then
  train_router_if_needed "distilbert" "$DISTILBERT_CKPT" "$DISTILBERT_MODEL_NAME"
fi

if [ "$RUN_T5_LARGE" = "1" ]; then
  train_router_if_needed "t5-large" "$T5_CKPT" "$T5_MODEL_NAME"
fi

declare -a enabled_routers=()
if [ "$RUN_DISTILBERT" = "1" ]; then
  enabled_routers+=("distilbert")
fi
if [ "$RUN_T5_LARGE" = "1" ]; then
  enabled_routers+=("t5-large")
fi
if [ "$RUN_QWEN" = "1" ]; then
  enabled_routers+=("qwen")
fi

if [ "${#enabled_routers[@]}" -eq 0 ]; then
  echo "[ERROR] No routers enabled. Set at least one of RUN_DISTILBERT/RUN_T5_LARGE/RUN_QWEN to 1."
  exit 1
fi

echo ""
echo "Enabled routers    : ${enabled_routers[*]}"
echo ""

ROUTERS_CSV="$(IFS=,; echo "${enabled_routers[*]}")"

ROUTERS="$ROUTERS_CSV" \
MODEL_PATH="$MODEL_PATH" \
QWEN_ROUTER_MODEL="$QWEN_ROUTER_MODEL" \
FORCE_ROUTE="$FORCE_ROUTE" \
FORCE_EVAL="$FORCE_EVAL" \
bash script/6_eval_all_routers_qwen.sh
