#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# Reproduce the PURE-style hard-router comparison with a shared
# generator. The default is DeepSeek V4 Pro, but MODEL_PATH may point to
# any supported OpenAI-compatible backend such as qwen-api:, gpt:, or glm:.

MODEL_PATH="${MODEL_PATH:-deepseek:deepseek-v4-pro}"
INPUT_DIR="${INPUT_DIR:-dataset/query_nonvideo_large/full}"
ROUTE_OUTPUT_DIR="${ROUTE_OUTPUT_DIR:-route/results_universalrag_qwen36plus_full}"
RESULTS_ROOT="${RESULTS_ROOT:-eval/results_deepseek_v4_pro_universalrag_full}"
ROUTERS="${ROUTERS:-distilbert,t5-large}"
TARGETS="${TARGETS:-mmlu,squad,natural_questions,hotpotqa,webqa}"
TOP_K="${TOP_K:-1}"
ALPHA="${ALPHA:-0.2}"
NFRAMES="${NFRAMES:-1}"
QUERY_BGE_DIR="${QUERY_BGE_DIR:-eval/features/query_full/bge-large}"
QUERY_INTERNVIDEO_DIR="${QUERY_INTERNVIDEO_DIR:-eval/features/query_full/internvideo}"
FORCE_ROUTE="${FORCE_ROUTE:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"
ROUTE_ONLY="${ROUTE_ONLY:-0}"
EVAL_ONLY="${EVAL_ONLY:-0}"

if [ ! -d "$INPUT_DIR" ]; then
  echo "Missing INPUT_DIR: $INPUT_DIR"
  exit 1
fi

if [ ! -d route/train/checkpoints/distilbert ]; then
  echo "Missing DistilBERT checkpoint: route/train/checkpoints/distilbert"
  echo "Train it first with: bash script/2_train.sh distilbert"
  exit 1
fi

if [ ! -d route/train/checkpoints/t5-large ]; then
  echo "Missing T5-Large checkpoint: route/train/checkpoints/t5-large"
  echo "Train it first with: bash script/2_train.sh t5-large"
  exit 1
fi

read -r -a ROUTERS_ARR <<< "$(echo "$ROUTERS" | tr ',' ' ')"

route_one() {
  local router="$1"
  case "$router" in
    distilbert)
      python route/train/route_distilbert.py \
        --checkpoint_dir route/train/checkpoints/distilbert \
        --input_dir "$INPUT_DIR" \
        --batch_size 256 \
        --output_dir "$ROUTE_OUTPUT_DIR"
      ;;
    t5-large)
      python route/train/route_t5.py \
        --checkpoint_dir route/train/checkpoints/t5-large \
        --input_dir "$INPUT_DIR" \
        --batch_size 16 \
        --output_dir "$ROUTE_OUTPUT_DIR"
      ;;
    *)
      echo "Unsupported router for this baseline: $router"
      exit 1
      ;;
  esac
}

if [ "$EVAL_ONLY" != "1" ]; then
  for router in "${ROUTERS_ARR[@]}"; do
    first_target="${TARGETS%%,*}"
    expected="$ROUTE_OUTPUT_DIR/$router/$first_target.json"
    if [ "$FORCE_ROUTE" = "1" ] || [ ! -f "$expected" ]; then
      echo "================ ROUTE PURE router=$router ================"
      route_one "$router"
    else
      echo "[SKIP] Existing route files for $router under $ROUTE_OUTPUT_DIR"
    fi
  done
fi

if [ "$ROUTE_ONLY" = "1" ]; then
  echo "ROUTE_ONLY=1, skip generation evaluation."
  exit 0
fi

if [ ! -d "$QUERY_BGE_DIR" ]; then
  echo "Missing QUERY_BGE_DIR: $QUERY_BGE_DIR"
  echo "Generate full BGE query features with:"
  echo "  python preprocess/extract_query_feats_bge.py --input_path $INPUT_DIR --output_path $QUERY_BGE_DIR"
  exit 1
fi

if [ ! -d "$QUERY_INTERNVIDEO_DIR" ]; then
  echo "Missing QUERY_INTERNVIDEO_DIR: $QUERY_INTERNVIDEO_DIR"
  echo "Generate full InternVideo query features with:"
  echo "  INTERNVIDEO_PATH=/path/to/InternVideo python preprocess/extract_query_feats_internvideo.py --input_path $INPUT_DIR --output_path $QUERY_INTERNVIDEO_DIR"
  exit 1
fi

case "$MODEL_PATH" in
  qwen-api:*|dashscope:*)
    if [[ -z "${DASHSCOPE_API_KEY:-}" && -z "${QWEN_API_KEY:-}" ]]; then
      echo "Missing API key. Set DASHSCOPE_API_KEY or QWEN_API_KEY for $MODEL_PATH."
      exit 1
    fi
    ;;
  openai:*|gpt:*|gpt-*)
    if [[ -z "${OPENAI_API_KEY:-}" ]]; then
      echo "Missing API key. Set OPENAI_API_KEY for $MODEL_PATH."
      exit 1
    fi
    ;;
  dmxapi:*)
    if [[ -z "${DMXAPI_API_KEY:-}" && -z "${DMX_API_KEY:-}" ]]; then
      echo "Missing API key. Set DMXAPI_API_KEY or DMX_API_KEY for $MODEL_PATH."
      exit 1
    fi
    ;;
  deepseek:*|deepseek-*)
    if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
      echo "Missing API key. Set DEEPSEEK_API_KEY for $MODEL_PATH."
      exit 1
    fi
    ;;
  glm:*|zhipu:*|glm-*)
    if [[ -z "${GLM_API_KEY:-}" && -z "${ZHIPU_API_KEY:-}" && -z "${BIGMODEL_API_KEY:-}" ]]; then
      echo "Missing API key. Set GLM_API_KEY, ZHIPU_API_KEY, or BIGMODEL_API_KEY for $MODEL_PATH."
      exit 1
    fi
    ;;
  openai-compatible:*)
    if [[ -z "${GENERATOR_API_KEY:-}" && -z "${OPENAI_COMPATIBLE_API_KEY:-}" ]]; then
      echo "Missing API key. Set GENERATOR_API_KEY or OPENAI_COMPATIBLE_API_KEY for $MODEL_PATH."
      exit 1
    fi
    ;;
esac

echo "================ EVAL PURE shared-generator baseline ================"
echo "MODEL_PATH=$MODEL_PATH"
echo "ROUTE_DIR=$ROUTE_OUTPUT_DIR"
echo "RESULTS_ROOT=$RESULTS_ROOT"
echo "ROUTERS=$ROUTERS"
echo "TARGETS=$TARGETS"
echo "QUERY_BGE_DIR=$QUERY_BGE_DIR"
echo "QUERY_INTERNVIDEO_DIR=$QUERY_INTERNVIDEO_DIR"

MODEL_PATH="$MODEL_PATH" \
ROUTE_DIR="$ROUTE_OUTPUT_DIR" \
RESULTS_ROOT="$RESULTS_ROOT" \
ROUTERS="$ROUTERS" \
TARGETS="$TARGETS" \
TOP_K="$TOP_K" \
ALPHA="$ALPHA" \
NFRAMES="$NFRAMES" \
QUERY_BGE_DIR="$QUERY_BGE_DIR" \
QUERY_INTERNVIDEO_DIR="$QUERY_INTERNVIDEO_DIR" \
FORCE_EVAL="$FORCE_EVAL" \
bash script/23_eval_large_baseline_all.sh
