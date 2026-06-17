#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CHECK_ONLY=0
if [[ "${1:-}" == "--check-only" ]]; then
  CHECK_ONLY=1
  shift
fi

TARGETS=("$@")
if [[ "${#TARGETS[@]}" -eq 0 ]]; then
  TARGETS=(hotpotqa webqa)
fi

export PYTHON="${PYTHON:-/opt/conda/envs/universalrag/bin/python}"
export MODEL_PATH="${MODEL_PATH:-qwen-api:qwen3.6-plus}"
export ROUTER_MODEL="${ROUTER_MODEL:-t5-large}"
export ROUTE_DIR="${ROUTE_DIR:-route/fixed_route_baselines_strict_d40_test/no}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-eval/results_clean_naive_qwen36plus_20260602_fixed_no}"
export DATASET_TEST_DIR="${DATASET_TEST_DIR:-dataset/query_nonvideo_large_strict_d40/test}"
export HOTPOT_RAW_TEST_DIR="${HOTPOT_RAW_TEST_DIR:-dataset/query_hotpotqa_raw_context/query_nonvideo_large_strict_d40/test}"
export QUERY_BGE_DIR="${QUERY_BGE_DIR:-eval/features/query_test_d40/bge-large}"
export QUERY_INTERNVIDEO_DIR="${QUERY_INTERNVIDEO_DIR:-eval/features/query_test_d40/internvideo}"

export PYTHONPATH="/opt/conda/envs/universalrag/lib/python3.10/site-packages:/opt/conda/lib/python3.10/site-packages:${PYTHONPATH:-}"

# Conservative closed-book generator settings.
export QWEN_ENABLE_THINKING=0
export QWEN_FORCE_FINAL_ANSWER_PASS_MCQ=0
export QWEN_FORCE_FINAL_ANSWER_PASS_EXACT_SHORT=0
export QWEN_API_CANONICALIZE_EXACT_SHORT=0
export EVAL_RESUME=0
export EVAL_SAVE_EVERY="${EVAL_SAVE_EVERY:-25}"

# Harmless for Fixed-No, but prevents HotpotQA from silently using the wrong
# document feature set if the route is changed by accident.
export HOTPOTQA_TEXT_FEATS="${HOTPOTQA_TEXT_FEATS:-eval/features/text/hotpotqa_raw_context.pkl}"

if [[ "$CHECK_ONLY" -eq 0 ]]; then
  if [[ -z "${DASHSCOPE_API_KEY:-}${QWEN_API_KEY:-}${OPENAI_API_KEY:-}${GENERATOR_API_KEY:-}" ]]; then
    echo "[ERROR] No API key found. Set DASHSCOPE_API_KEY, QWEN_API_KEY, OPENAI_API_KEY, or GENERATOR_API_KEY." >&2
    exit 2
  fi
fi

mkdir -p logs

"$PYTHON" -B -c '
import collections
import hashlib
import json
import os
import sys

route_dir = os.environ["ROUTE_DIR"]
router = os.environ["ROUTER_MODEL"]
dataset_dir = os.environ["DATASET_TEST_DIR"]
hotpot_raw_dir = os.environ["HOTPOT_RAW_TEST_DIR"]
targets = sys.argv[1:]

def key_hash(rows):
    keys = [
        (
            str(row.get("source", "")),
            str(row.get("index", "")),
            str(row.get("question", "")),
        )
        for row in rows
    ]
    return hashlib.md5(repr(keys).encode()).hexdigest()

for target in targets:
    path = os.path.join(route_dir, router, f"{target}.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    counts = collections.Counter(str(row.get("retrieval", "")).lower() for row in data)
    route_hash = key_hash(data)
    print(f"[CHECK] {target}: rows={len(data)} retrieval_counts={dict(counts)} hash={route_hash} path={path}")
    if set(counts) != {"no"}:
        raise SystemExit(f"[ERROR] {target} is not a pure Fixed-No route: {dict(counts)}")

    dataset_path = os.path.join(dataset_dir, f"{target}.json")
    with open(dataset_path, encoding="utf-8") as f:
        dataset_data = json.load(f)
    dataset_hash = key_hash(dataset_data)
    print(f"[CHECK] {target}: modified_test_rows={len(dataset_data)} hash={dataset_hash} path={dataset_path}")
    if route_hash != dataset_hash:
        raise SystemExit(
            f"[ERROR] {target} route does not match modified d40 test set: "
            f"route_hash={route_hash}, dataset_hash={dataset_hash}"
        )

    if target == "hotpotqa":
        raw_path = os.path.join(hotpot_raw_dir, "hotpotqa.json")
        with open(raw_path, encoding="utf-8") as f:
            raw_data = json.load(f)
        raw_hash = key_hash(raw_data)
        print(f"[CHECK] hotpotqa: raw_context_test_rows={len(raw_data)} hash={raw_hash} path={raw_path}")
        if route_hash != raw_hash:
            raise SystemExit(
                f"[ERROR] HotpotQA route does not match raw-context modified test set: "
                f"route_hash={route_hash}, raw_hash={raw_hash}"
            )
' "${TARGETS[@]}"

echo "[CONFIG] MODEL_PATH=$MODEL_PATH"
echo "[CONFIG] ROUTE_DIR=$ROUTE_DIR"
echo "[CONFIG] OUTPUT_ROOT=$OUTPUT_ROOT"
echo "[CONFIG] DATASET_TEST_DIR=$DATASET_TEST_DIR"
echo "[CONFIG] HOTPOT_RAW_TEST_DIR=$HOTPOT_RAW_TEST_DIR"
echo "[CONFIG] QUERY_BGE_DIR=$QUERY_BGE_DIR"
echo "[CONFIG] QUERY_INTERNVIDEO_DIR=$QUERY_INTERNVIDEO_DIR"
echo "[CONFIG] TARGETS=${TARGETS[*]}"

if [[ "$CHECK_ONLY" -eq 1 ]]; then
  exit 0
fi

for target in "${TARGETS[@]}"; do
  log_file="logs/clean_naive_qwen36plus_${target}_20260602.log"
  echo "[RUN] $target -> $log_file"
  "$PYTHON" eval/eval.py \
    --model_path "$MODEL_PATH" \
    --router_model "$ROUTER_MODEL" \
    --target "$target" \
    --top_k 1 \
    --alpha 0.2 \
    --nframes 1 \
    --route_dir "$ROUTE_DIR" \
    --output_root "$OUTPUT_ROOT" \
    --query_bge_dir "$QUERY_BGE_DIR" \
    --query_internvideo_dir "$QUERY_INTERNVIDEO_DIR" \
    2>&1 | tee "$log_file"
done

echo "[DONE] Clean Naive outputs are under: $OUTPUT_ROOT/$MODEL_PATH/$ROUTER_MODEL"
