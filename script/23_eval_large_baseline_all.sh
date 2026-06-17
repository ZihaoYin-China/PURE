#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

MODEL_PATH="${MODEL_PATH:-qwen2.5vl:7b}"
MODEL_NAME="${MODEL_PATH##*/}"
ROUTE_DIR="${ROUTE_DIR:-route/results_large}"
RESULTS_ROOT="${RESULTS_ROOT:-eval/results_large_baseline}"
QUERY_BGE_DIR="${QUERY_BGE_DIR:-eval/features/query/bge-large}"
QUERY_INTERNVIDEO_DIR="${QUERY_INTERNVIDEO_DIR:-eval/features/query/internvideo}"
TOP_K="${TOP_K:-1}"
ALPHA="${ALPHA:-0.2}"
NFRAMES="${NFRAMES:-1}"
FORCE_EVAL="${FORCE_EVAL:-0}"

TARGETS_ENV="${TARGETS:-mmlu,squad,natural_questions,hotpotqa,webqa}"
ROUTERS_ENV="${ROUTERS:-distilbert,t5-large,qwen}"
read -r -a TARGETS_ARR <<< "$(echo "$TARGETS_ENV" | tr ',' ' ')"
read -r -a ROUTERS_ARR <<< "$(echo "$ROUTERS_ENV" | tr ',' ' ')"

if [ ! -d "$ROUTE_DIR" ]; then
  echo "Missing ROUTE_DIR: $ROUTE_DIR"
  echo "Run routers on dataset/query_nonvideo_large/test first."
  exit 1
fi

if [ ! -d "$QUERY_BGE_DIR" ]; then
  echo "Missing QUERY_BGE_DIR: $QUERY_BGE_DIR"
  echo "Run preprocess/extract_query_feats_bge.py first."
  exit 1
fi

mkdir -p "$RESULTS_ROOT" logs

ROUTE_DEFAULT="route/results"
ROUTE_BACKUP=""
RESULTS_DEFAULT="eval/results"
RESULTS_BACKUP=""

cleanup() {
  rm -f "$ROUTE_DEFAULT"
  if [ -n "$ROUTE_BACKUP" ] && [ -e "$ROUTE_BACKUP" ]; then
    mv "$ROUTE_BACKUP" "$ROUTE_DEFAULT"
  fi

  rm -f "$RESULTS_DEFAULT"
  if [ -n "$RESULTS_BACKUP" ] && [ -e "$RESULTS_BACKUP" ]; then
    mv "$RESULTS_BACKUP" "$RESULTS_DEFAULT"
  fi
}
trap cleanup EXIT

if [ -L "$ROUTE_DEFAULT" ]; then
  rm "$ROUTE_DEFAULT"
elif [ -e "$ROUTE_DEFAULT" ]; then
  ROUTE_BACKUP="route/results.__baseline_large_backup__"
  if [ -e "$ROUTE_BACKUP" ]; then
    echo "Backup path already exists: $ROUTE_BACKUP"
    exit 1
  fi
  mv "$ROUTE_DEFAULT" "$ROUTE_BACKUP"
fi
ln -s "$(realpath "$ROUTE_DIR")" "$ROUTE_DEFAULT"

if [ -L "$RESULTS_DEFAULT" ]; then
  rm "$RESULTS_DEFAULT"
elif [ -e "$RESULTS_DEFAULT" ]; then
  RESULTS_BACKUP="eval/results.__baseline_large_backup__"
  if [ -e "$RESULTS_BACKUP" ]; then
    echo "Backup path already exists: $RESULTS_BACKUP"
    exit 1
  fi
  mv "$RESULTS_DEFAULT" "$RESULTS_BACKUP"
fi
ln -s "$(realpath "$RESULTS_ROOT")" "$RESULTS_DEFAULT"

echo "================ LARGE BASELINE EVAL ================"
echo "MODEL_PATH=$MODEL_PATH"
echo "ROUTE_DIR=$ROUTE_DIR"
echo "RESULTS_ROOT=$RESULTS_ROOT"
echo "QUERY_BGE_DIR=$QUERY_BGE_DIR"
echo "QUERY_INTERNVIDEO_DIR=$QUERY_INTERNVIDEO_DIR"
echo "ROUTERS=${ROUTERS_ARR[*]}"
echo "TARGETS=${TARGETS_ARR[*]}"

for ROUTER in "${ROUTERS_ARR[@]}"; do
  for TARGET in "${TARGETS_ARR[@]}"; do
    ROUTE_FILE="${ROUTE_DIR}/${ROUTER}/${TARGET}.json"
    if [ ! -f "$ROUTE_FILE" ]; then
      echo "[SKIP] Missing route file: $ROUTE_FILE"
      continue
    fi

    RESULT_FILE="${RESULTS_ROOT}/${MODEL_NAME}/${ROUTER}/${TARGET}_top${TOP_K}_${ALPHA}_${NFRAMES}.json"
    if [ "$FORCE_EVAL" != "1" ] && [ -f "$RESULT_FILE" ]; then
      echo "[SKIP] Existing result: $RESULT_FILE"
      continue
    fi

    echo ""
    echo "================ EVAL router=${ROUTER} target=${TARGET} ================"
    bash script/4_eval.sh \
      --model_path "$MODEL_PATH" \
      --router_model "$ROUTER" \
      --target "$TARGET" \
      --top_k "$TOP_K" \
      --alpha "$ALPHA" \
      --nframes "$NFRAMES" \
      --query_bge_dir "$QUERY_BGE_DIR" \
      --query_internvideo_dir "$QUERY_INTERNVIDEO_DIR" 2>&1 | tee "logs/eval_large_baseline_${ROUTER}_${TARGET}.log"
  done
done

echo ""
echo "================ LARGE BASELINE SCORING SUMMARY ================"
python - <<PY
import os
import sys

project_root = r"$PROJECT_ROOT"
sys.path.insert(0, project_root)
from eval.score import score_file

model_name = r"$MODEL_NAME"
results_root = r"$RESULTS_ROOT"
top_k = r"$TOP_K"
alpha = r"$ALPHA"
nframes = r"$NFRAMES"
routers = [x for x in "${ROUTERS_ARR[*]}".split() if x]
targets = [x for x in "${TARGETS_ARR[*]}".split() if x]

for router in routers:
    print(f"Router: {router}")
    for target in targets:
        result_file = os.path.join(
            project_root,
            results_root,
            model_name,
            router,
            f"{target}_top{top_k}_{alpha}_{nframes}.json",
        )
        if not os.path.isfile(result_file):
            print(f"  {target}: MISS")
            continue
        result = score_file(result_file, target=target)
        if target == "mmlu":
            msg = f"Acc={result['Accuracy']}"
        elif target in {"squad", "natural_questions", "hotpotqa"}:
            msg = f"EM={result['EM']}, F1={result['F1']}"
        elif target == "webqa":
            msg = f"RL={result['ROUGE-L']}"
            if "BERTScore" in result:
                msg += f", BS={result['BERTScore']}"
        else:
            msg = str(result)
        print(f"  {target}: {msg}")
PY
