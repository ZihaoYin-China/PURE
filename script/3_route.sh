#!/bin/bash
set -euo pipefail

if [ -z "${1:-}" ]; then
    echo "Error: No router specified. Use 'gpt', 'qwen', 't5-large', or 'distilbert'."
    exit 1
fi

ROUTER="$1"
QWEN_MODEL="${2:-${QWEN_ROUTER_MODEL:-qwen3-vl:8b}}"
INPUT_DIR="${3:-dataset/query}"

if [ ! -e "$INPUT_DIR" ]; then
    echo "Error: INPUT_DIR does not exist: $INPUT_DIR"
    exit 1
fi

if [ "$ROUTER" = "gpt" ]; then
    echo "Running GPT routing..."
    echo "Input dir/file: $INPUT_DIR"
    python route/gpt/route_gpt.py \
        --input_dir "$INPUT_DIR" \
        --output_dir route/results/gpt

elif [ "$ROUTER" = "qwen" ]; then
    echo "Running Qwen routing with model: $QWEN_MODEL"
    echo "Input dir/file: $INPUT_DIR"
    python route/qwen/route_qwen.py \
        --input_dir "$INPUT_DIR" \
        --output_dir route/results/qwen \
        --model_name "$QWEN_MODEL"

elif [ "$ROUTER" = "t5-large" ]; then
    echo "Running T5-Large routing..."
    echo "Input dir/file: $INPUT_DIR"
    python route/train/route_t5.py \
        --checkpoint_dir route/train/checkpoints/t5-large \
        --input_dir "$INPUT_DIR" \
        --batch_size 16 \
        --output_dir route/results

elif [ "$ROUTER" = "distilbert" ]; then
    echo "Running DistilBERT routing..."
    echo "Input dir/file: $INPUT_DIR"
    python route/train/route_distilbert.py \
        --checkpoint_dir route/train/checkpoints/distilbert \
        --input_dir "$INPUT_DIR" \
        --batch_size 256 \
        --output_dir route/results

else
    echo "Error: Unknown router '$ROUTER'. Use 'gpt', 'qwen', 't5-large', or 'distilbert'."
    exit 1
fi
