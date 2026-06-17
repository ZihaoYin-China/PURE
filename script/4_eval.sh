#!/bin/bash

# Default values. Environment variables override these so batch scripts can
# pass settings without repeating every CLI flag.
MODEL_PATH="${MODEL_PATH:-OpenGVLab/InternVL2_5-8B}"
ROUTER_MODEL="${ROUTER_MODEL:-distilbert}"
TARGET="${TARGET:-mmlu}"
TOP_K="${TOP_K:-1}"
ALPHA="${ALPHA:-0.2}"
NFRAMES="${NFRAMES:-1}"
QUERY_BGE_DIR="${QUERY_BGE_DIR:-eval/features/query/bge-large}"
QUERY_INTERNVIDEO_DIR="${QUERY_INTERNVIDEO_DIR:-eval/features/query/internvideo}"
ROUTE_DIR="${ROUTE_DIR:-route/results}"
RESULTS_ROOT="${RESULTS_ROOT:-eval/results}"
ALLOW_NONVISUAL_IMAGE_RETRIEVAL="${ALLOW_NONVISUAL_IMAGE_RETRIEVAL:-0}"
STRICT_IMAGE_GROUNDING="${STRICT_IMAGE_GROUNDING:-0}"
EVAL_PYTHON_BIN="${EVAL_PYTHON_BIN:-${PYTHON_BIN:-python}}"

show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --model_path PATH      Path or name of the model (default: $MODEL_PATH)"
    echo "                         Choices:"
    echo "                           OpenGVLab/InternVL2_5-8B"
    echo "                           Qwen/Qwen2.5-VL-7B-Instruct"
    echo "                           qwen2.5vl:32b"
    echo "                           qwen-api:qwen3.6-plus"
    echo "                           gpt:gpt-4o"
    echo "                           dmxapi:gpt-4o"
    echo "                           deepseek:deepseek-chat"
    echo "                           glm:glm-4.6v"
    echo "                           glm:glm-4-plus"
    echo "                           microsoft/Phi-3.5-vision-instruct"
    echo "  --router_model NAME    Router model to use (default: $ROUTER_MODEL)"
    echo "                         Choices: gpt, qwen, t5-large, distilbert, selfrag, adaptive_rag, crag"
    echo "  --target NAME          Target dataset for evaluation (default: $TARGET)"
    echo "                         Choices: mmlu, squad, natural_questions, hotpotqa, webqa, truthfulqa, triviaqa, lara, visual_rag"
    echo "  --top_k INT            Number of top retrievals to use (default: $TOP_K)"
    echo "  --alpha FLOAT          Weight for image caption features (default: $ALPHA, range: 0 to 1)"
    echo "  --nframes STR          Frame setting passed to eval.py (default: $NFRAMES)"
    echo "  --query_bge_dir PATH   BGE query feature directory (default: $QUERY_BGE_DIR)"
    echo "  --query_internvideo_dir PATH   InternVideo query feature directory (default: $QUERY_INTERNVIDEO_DIR)"
    echo "  --route_dir PATH       Route result directory (default: $ROUTE_DIR)"
    echo "  --output_root PATH     Eval output root (default: $RESULTS_ROOT)"
    echo "  --allow_nonvisual_image_retrieval"
    echo "                         Force image retrieval even on non-visual targets"
    echo "  --strict_image_grounding"
    echo "                         Require image answers to be grounded in visible image evidence"
    echo "  -h, --help             Show this help message and exit"
}

if [[ $# -eq 0 ]]; then
    echo "[INFO] Using default arguments."
fi

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    show_help
    exit 0
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --router_model)
            ROUTER_MODEL="$2"
            shift 2
            ;;
        --target)
            TARGET="$2"
            shift 2
            ;;
        --top_k)
            TOP_K="$2"
            shift 2
            ;;
        --alpha)
            ALPHA="$2"
            shift 2
            ;;
        --nframes)
            NFRAMES="$2"
            shift 2
            ;;
        --query_bge_dir)
            QUERY_BGE_DIR="$2"
            shift 2
            ;;
        --query_internvideo_dir)
            QUERY_INTERNVIDEO_DIR="$2"
            shift 2
            ;;
        --route_dir)
            ROUTE_DIR="$2"
            shift 2
            ;;
        --output_root)
            RESULTS_ROOT="$2"
            shift 2
            ;;
        --allow_nonvisual_image_retrieval)
            ALLOW_NONVISUAL_IMAGE_RETRIEVAL="1"
            shift
            ;;
        --strict_image_grounding)
            STRICT_IMAGE_GROUNDING="1"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage."
            exit 1
            ;;
    esac
done

echo "[INFO] Running evaluation with:"
echo "  MODEL_PATH   = $MODEL_PATH"
echo "  ROUTER_MODEL = $ROUTER_MODEL"
echo "  TARGET       = $TARGET"
echo "  TOP_K        = $TOP_K"
echo "  ALPHA        = $ALPHA"
echo "  NFRAMES      = $NFRAMES"
echo "  QUERY_BGE_DIR = $QUERY_BGE_DIR"
echo "  QUERY_INTERNVIDEO_DIR = $QUERY_INTERNVIDEO_DIR"
echo "  ROUTE_DIR = $ROUTE_DIR"
echo "  RESULTS_ROOT = $RESULTS_ROOT"
echo "  ALLOW_NONVISUAL_IMAGE_RETRIEVAL = $ALLOW_NONVISUAL_IMAGE_RETRIEVAL"
echo "  STRICT_IMAGE_GROUNDING = $STRICT_IMAGE_GROUNDING"
echo "  EVAL_PYTHON_BIN = $EVAL_PYTHON_BIN"

EXTRA_ARGS=()
if [[ "$ALLOW_NONVISUAL_IMAGE_RETRIEVAL" == "1" ]]; then
    EXTRA_ARGS+=(--allow_nonvisual_image_retrieval)
fi
if [[ "$STRICT_IMAGE_GROUNDING" == "1" ]]; then
    EXTRA_ARGS+=(--strict_image_grounding)
fi

"$EVAL_PYTHON_BIN" eval/eval.py \
    --model_path "$MODEL_PATH" \
    --router_model "$ROUTER_MODEL" \
    --target "$TARGET" \
    --top_k "$TOP_K" \
    --alpha "$ALPHA" \
    --nframes "$NFRAMES" \
    --query_bge_dir "$QUERY_BGE_DIR" \
    --query_internvideo_dir "$QUERY_INTERNVIDEO_DIR" \
    --route_dir "$ROUTE_DIR" \
    --output_root "$RESULTS_ROOT" \
    "${EXTRA_ARGS[@]}"
