#!/bin/bash

if [ -z "$1" ]; then
    echo "Error: No model specified. Use 't5-large' or 'distilbert'."
    exit 1
fi

MODEL_OVERRIDE="${2:-}"
T5_MODEL_NAME="${MODEL_OVERRIDE:-${T5_MODEL_NAME:-google/flan-t5-large}}"
DISTILBERT_MODEL_NAME="${MODEL_OVERRIDE:-${DISTILBERT_MODEL_NAME:-distilbert-base-uncased}}"

if [ "$1" = "t5-large" ]; then
    echo "Running T5-Large training..."
    echo "Init model: $T5_MODEL_NAME"
    python route/train/train_t5.py \
        --model_name "$T5_MODEL_NAME" \
        --input_dir route/train/data/train_data_t5_4class.json \
        --num_train_epochs 10 \
        --learning_rate 3e-5 \
        --train_batch_size 1 \
        --eval_batch_size 2 \
        --gradient_accumulation_steps 8 \
        --gradient_checkpointing \
        --mixed_precision auto \
        --checkpoint_dir route/train/checkpoints/

elif [ "$1" = "distilbert" ]; then
    echo "Running DistilBERT training..."
    echo "Init model: $DISTILBERT_MODEL_NAME"
    python route/train/train_distilbert.py \
        --model_name "$DISTILBERT_MODEL_NAME" \
        --input_dir route/train/data/train_data_distilbert_4class.json \
        --num_train_epochs 5 \
        --learning_rate 2e-5 \
        --train_batch_size 64 \
        --eval_batch_size 64 \
        --checkpoint_dir route/train/checkpoints/

else
    echo "Error: Unknown model '$1'. Use 't5-large' or 'distilbert'."
    exit 1
fi
