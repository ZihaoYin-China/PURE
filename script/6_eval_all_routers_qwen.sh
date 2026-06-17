#!/bin/bash
set -euo pipefail

# ====== 基本配置 ======
# 生成模型
MODEL_PATH="${MODEL_PATH:-qwen3-vl:8b}"
MODEL_NAME="${MODEL_PATH##*/}"

# Qwen 路由器使用的模型，可与生成模型不同
QWEN_ROUTER_MODEL="${QWEN_ROUTER_MODEL:-$MODEL_PATH}"

TOP_K="${TOP_K:-1}"
ALPHA="${ALPHA:-0.2}"
NFRAMES="${NFRAMES:-1}"

if [ -n "${ROUTERS:-}" ]; then
  read -r -a ROUTERS <<< "$(echo "$ROUTERS" | tr ',' ' ')"
else
  ROUTERS=("distilbert" "t5-large" "qwen")
fi
TARGETS=("mmlu" "squad" "natural_questions" "hotpotqa" "webqa")

# 是否强制重跑
# 0: 若结果已存在则跳过
# 1: 无论是否存在都重跑
FORCE_ROUTE="${FORCE_ROUTE:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p logs

# ====== 只给路由阶段准备 5 个非视频 json ======
TMP_QUERY_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "$TMP_QUERY_DIR"
}
trap cleanup EXIT

for TARGET in "${TARGETS[@]}"; do
  SRC="dataset/query/${TARGET}.json"
  if [ ! -f "$SRC" ]; then
    echo "[ERROR] Missing query file: $SRC"
    exit 1
  fi
  cp "$SRC" "$TMP_QUERY_DIR/"
done

echo "============================================================"
echo "Project root       : $PROJECT_ROOT"
echo "Generation model   : $MODEL_PATH"
echo "Qwen router model  : $QWEN_ROUTER_MODEL"
echo "Routers            : ${ROUTERS[*]}"
echo "Targets            : ${TARGETS[*]}"
echo "Top-k              : $TOP_K"
echo "Alpha              : $ALPHA"
echo "NFrames            : $NFRAMES"
echo "FORCE_ROUTE        : $FORCE_ROUTE"
echo "FORCE_EVAL         : $FORCE_EVAL"
echo "Route query dir    : $TMP_QUERY_DIR"
echo "============================================================"

# ====== Step 1: 跑路由（只跑 5 个非视频 target） ======
for ROUTER in "${ROUTERS[@]}"; do
  NEED_ROUTE=0

  if [ "$FORCE_ROUTE" = "1" ]; then
    NEED_ROUTE=1
  else
    for TARGET in "${TARGETS[@]}"; do
      if [ ! -f "route/results/${ROUTER}/${TARGET}.json" ]; then
        NEED_ROUTE=1
        break
      fi
    done
  fi

  if [ "$NEED_ROUTE" = "1" ]; then
    echo ""
    echo "================ ROUTING: $ROUTER ================"
    if [ "$ROUTER" = "qwen" ]; then
      bash script/3_route.sh "$ROUTER" "$QWEN_ROUTER_MODEL" "$TMP_QUERY_DIR" | tee "logs/route_${ROUTER}.log"
    else
      bash script/3_route.sh "$ROUTER" "" "$TMP_QUERY_DIR" | tee "logs/route_${ROUTER}.log"
    fi
  else
    echo "[SKIP] route/results/${ROUTER} already exists for all non-video targets."
  fi
done

# ====== Step 2: 跑生成评估 ======
for ROUTER in "${ROUTERS[@]}"; do
  for TARGET in "${TARGETS[@]}"; do
    RESULT_FILE="eval/results/${MODEL_NAME}/${ROUTER}/${TARGET}_top${TOP_K}_${ALPHA}_${NFRAMES}.json"

    if [ "$FORCE_EVAL" = "1" ] || [ ! -f "$RESULT_FILE" ]; then
      echo ""
      echo "================ EVAL: router=${ROUTER}, target=${TARGET} ================"
      bash script/4_eval.sh \
        --model_path "$MODEL_PATH" \
        --router_model "$ROUTER" \
        --target "$TARGET" \
        --top_k "$TOP_K" \
        --alpha "$ALPHA" \
        --nframes "$NFRAMES" | tee "logs/eval_${ROUTER}_${TARGET}.log"
    else
      echo "[SKIP] $RESULT_FILE already exists."
    fi
  done
done

# ====== Step 3: 打分并汇总 ======
echo ""
echo "================ SCORING SUMMARY ================"

python - <<PY
import os
import sys
from tabulate import tabulate

project_root = r"$PROJECT_ROOT"
sys.path.insert(0, project_root)

from eval.score import score_file

MODEL_NAME = r"$MODEL_NAME"
TOP_K = r"$TOP_K"
ALPHA = r"$ALPHA"
NFRAMES = r"$NFRAMES"

routers = [x for x in "${ROUTERS[*]}".split() if x]
targets = ["mmlu", "squad", "natural_questions", "hotpotqa", "webqa"]

rows = []

for router in routers:
    row = {"Router": router}

    for target in targets:
        result_file = os.path.join(
            project_root,
            "eval", "results", MODEL_NAME, router,
            f"{target}_top{TOP_K}_{ALPHA}_{NFRAMES}.json"
        )

        if not os.path.isfile(result_file):
            row[target] = "MISS"
            continue

        try:
            result = score_file(result_file, target=target)

            if target == "mmlu":
                row[target] = f"Acc={result['Accuracy']}"
            elif target in {"squad", "natural_questions", "hotpotqa"}:
                row[target] = f"EM={result['EM']}, F1={result['F1']}"
            elif target == "webqa":
                if "BERTScore" in result:
                    row[target] = f"RL={result['ROUGE-L']}, BS={result['BERTScore']}"
                else:
                    row[target] = f"RL={result['ROUGE-L']}"
            else:
                row[target] = str(result)

        except Exception as e:
            row[target] = f"ERR: {str(e)[:60]}"

    rows.append(row)

print(tabulate(rows, headers="keys", tablefmt="fancy_grid"))
PY

echo ""
echo "Done."
