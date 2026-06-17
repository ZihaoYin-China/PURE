#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL_NAME_DEFAULT="${MODEL_NAME:-route/train/checkpoints/t5-large}"
if [ "$MODEL_NAME_DEFAULT" = "route/train/checkpoints/t5-large" ] && [ ! -d "$MODEL_NAME_DEFAULT" ]; then
  MODEL_NAME_DEFAULT="google/flan-t5-large"
fi

python route/train/train_distilbert_vib.py \
  --model_name "$MODEL_NAME_DEFAULT" \
  --input_path "${INPUT_PATH:-route/train/data/train_data_t5_4class.json}" \
  --checkpoint_dir "${CHECKPOINT_DIR:-route/train/checkpoints/t5-large_vib}" \
  --init_checkpoint_dir "${INIT_CHECKPOINT_DIR:-}" \
  --train_size "${TRAIN_SIZE:-0.9}" \
  --max_input_length "${MAX_INPUT_LENGTH:-256}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-5}" \
  --learning_rate "${LEARNING_RATE:-2e-5}" \
  --weight_decay "${WEIGHT_DECAY:-0.01}" \
  --train_batch_size "${TRAIN_BATCH_SIZE:-1}" \
  --eval_batch_size "${EVAL_BATCH_SIZE:-2}" \
  --gradient_clip_norm "${GRADIENT_CLIP_NORM:-1.0}" \
  --warmup_ratio "${WARMUP_RATIO:-0.1}" \
  --seed "${SEED:-42}" \
  --device "${DEVICE:-auto}" \
  --latent_dim "${LATENT_DIM:-128}" \
  --hidden_dropout_prob "${HIDDEN_DROPOUT_PROB:-0.1}" \
  --prototype_temperature "${PROTOTYPE_TEMPERATURE:-1.0}" \
  --prototype_margin "${PROTOTYPE_MARGIN:-0.2}" \
  --kl_weight "${KL_WEIGHT:-0.001}" \
  --proto_weight "${PROTO_WEIGHT:-0.1}" \
  --evi_weight "${EVI_WEIGHT:-0.2}" \
  --proto_logit_scale "${PROTO_LOGIT_SCALE:-0.5}" \
  --class_weight_mode "${CLASS_WEIGHT_MODE:-none}" \
  --class_weight_clip_min "${CLASS_WEIGHT_CLIP_MIN:-0.25}" \
  --class_weight_clip_max "${CLASS_WEIGHT_CLIP_MAX:-4.0}" \
  --label_smoothing "${LABEL_SMOOTHING:-0.0}" \
  --select_metric "${SELECT_METRIC:-accuracy}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-8}" \
  --mixed_precision "${MIXED_PRECISION:-auto}" \
  --gradient_checkpointing
