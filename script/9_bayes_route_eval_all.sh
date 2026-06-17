#!/bin/bash
set -euo pipefail

# End-to-end non-video pipeline with Bayesian-Dirichlet dynamic routing at eval time:
# 1) (optional) run routing for enabled routers
# 2) run bayes-aware evaluation using route/results as input
# 3) score and print a summary

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

MODEL_PATH="${MODEL_PATH:-qwen2.5vl:7b}"
MODEL_NAME="${MODEL_PATH##*/}"
QWEN_ROUTER_MODEL="${QWEN_ROUTER_MODEL:-$MODEL_PATH}"

TOP_K="${TOP_K:-1}"
ALPHA="${ALPHA:-0.2}"
NFRAMES="${NFRAMES:-1}"
NFRAMES_TAG="${NFRAMES//,/_}"
NFRAMES_TAG="${NFRAMES_TAG//:/}"

ROUTE_DIR="${ROUTE_DIR:-route/results}"
OUTPUT_ROOT="${OUTPUT_ROOT:-eval/results_bayes}"
QUERY_BGE_DIR="${QUERY_BGE_DIR:-eval/features/query/bge-large}"
QUERY_INTERNVIDEO_DIR="${QUERY_INTERNVIDEO_DIR:-eval/features/query/internvideo}"

ROUTERS_ENV="${ROUTERS:-}"
TARGETS_ENV="${TARGETS:-}"

if [ -n "$ROUTERS_ENV" ]; then
  read -r -a ROUTERS <<< "$(echo "$ROUTERS_ENV" | tr ',' ' ')"
else
  ROUTERS=("distilbert" "t5-large" "qwen")
fi

if [ -n "$TARGETS_ENV" ]; then
  read -r -a TARGETS <<< "$(echo "$TARGETS_ENV" | tr ',' ' ')"
else
  TARGETS=("mmlu" "squad" "natural_questions" "hotpotqa" "webqa")
fi

FORCE_ROUTE="${FORCE_ROUTE:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"

# Bayesian controls
BAYES_ALPHA_PRIOR="${BAYES_ALPHA_PRIOR:-1,1,1,1}"
BAYES_TAU="${BAYES_TAU:-8.0}"
BAYES_BETA_COST="${BAYES_BETA_COST:-0.1}"
BAYES_MODALITY_COSTS="${BAYES_MODALITY_COSTS:-0.0,0.25,0.45,0.60}"
BAYES_ROUTER_PROBS_TEMPERATURE="${BAYES_ROUTER_PROBS_TEMPERATURE:-1.0}"
BAYES_ROUTER_PROBS_TEMPERATURE_BY_TARGET="${BAYES_ROUTER_PROBS_TEMPERATURE_BY_TARGET:-}"
BAYES_ROUTER_PROBS_BLEND_WITH_ORIGINAL="${BAYES_ROUTER_PROBS_BLEND_WITH_ORIGINAL:-0.0}"
BAYES_ROUTER_PROBS_BLEND_WITH_ORIGINAL_BY_TARGET="${BAYES_ROUTER_PROBS_BLEND_WITH_ORIGINAL_BY_TARGET:-}"
BAYES_DEFAULT_CONFIDENCE="${BAYES_DEFAULT_CONFIDENCE:-0.72}"
BAYES_UNCERTAINTY_THRESHOLD="${BAYES_UNCERTAINTY_THRESHOLD:-0.35}"
BAYES_DECISION_MODE="${BAYES_DECISION_MODE:-mean}"
BAYES_FALLBACK_WHEN_UNCERTAIN="${BAYES_FALLBACK_WHEN_UNCERTAIN:-1}"
BAYES_ONLINE_UPDATE="${BAYES_ONLINE_UPDATE:-0}"
BAYES_ETA="${BAYES_ETA:-1.0}"
BAYES_RHO="${BAYES_RHO:-0.0}"
BAYES_SEED="${BAYES_SEED:-42}"
BAYES_TAG="${BAYES_TAG:-tau${BAYES_TAU}_beta${BAYES_BETA_COST}}"
BAYES_TAG="${BAYES_TAG//./p}"
BAYES_SOFT_TOP_N="${BAYES_SOFT_TOP_N:-1}"
BAYES_SOFT_WEIGHT_MODE="${BAYES_SOFT_WEIGHT_MODE:-theta}"
BAYES_SOFT_FUSION_MODE="${BAYES_SOFT_FUSION_MODE:-auto}"
BAYES_SOFT_FUSION_MAX_NEW_TOKENS="${BAYES_SOFT_FUSION_MAX_NEW_TOKENS:-64}"
BAYES_SOFT_STORE_CANDIDATES="${BAYES_SOFT_STORE_CANDIDATES:-1}"
BAYES_TAG_EFFECTIVE="$BAYES_TAG"
if [ "${BAYES_SOFT_TOP_N}" -gt 1 ]; then
  SOFT_TAG="softtop${BAYES_SOFT_TOP_N}_${BAYES_SOFT_WEIGHT_MODE}_${BAYES_SOFT_FUSION_MODE}"
  if [[ "$BAYES_TAG_EFFECTIVE" != *"$SOFT_TAG"* ]]; then
    BAYES_TAG_EFFECTIVE="${BAYES_TAG_EFFECTIVE}_${SOFT_TAG}"
  fi
fi
BAT_TARGET_PRIORS="${BAT_TARGET_PRIORS:-}"
BAT_PENALTY="${BAT_PENALTY:-0.5}"
BAT_SPREAD="${BAT_SPREAD:-0.25}"
BAT_USE_PENALTY_UPDATE="${BAT_USE_PENALTY_UPDATE:-1}"

# Optional sidecar probability export for Bayes.
BAYES_EXPORT_PROBS="${BAYES_EXPORT_PROBS:-0}"
BAYES_FORCE_EXPORT_PROBS="${BAYES_FORCE_EXPORT_PROBS:-0}"
BAYES_PROB_ROUTE_DIR="${BAYES_PROB_ROUTE_DIR:-route/results_bayes_probs}"
BAYES_PROB_ROUTERS="${BAYES_PROB_ROUTERS:-distilbert,t5-large}"
BAYES_PROB_BATCH_SIZE_DISTILBERT="${BAYES_PROB_BATCH_SIZE_DISTILBERT:-256}"
BAYES_PROB_BATCH_SIZE_T5="${BAYES_PROB_BATCH_SIZE_T5:-128}"
BAYES_PROB_DEVICE="${BAYES_PROB_DEVICE:-auto}"

mkdir -p logs

check_ollama_model() {
  local model_name="$1"
  if ! command -v ollama >/dev/null 2>&1; then
    echo "[WARN] 'ollama' command not found. Skipping Bayes-side local model existence check."
    return
  fi

  if ! ollama list | awk 'NR>1 {print $1}' | grep -Fxq "$model_name"; then
    echo "[ERROR] Requested generation model is not installed in local Ollama: $model_name"
    echo "        Installed models:"
    ollama list
    echo ""
    echo "        Example fix:"
    echo "        MODEL_PATH=qwen2.5vl:7b bash script/9_bayes_route_eval_all.sh"
    exit 1
  fi
}

if [[ "$MODEL_PATH" == qwen-api:* || "$MODEL_PATH" == dashscope:* ]]; then
  echo "[INFO] API generation model detected; skipping local Ollama model check: $MODEL_PATH"
else
  check_ollama_model "$MODEL_PATH"
fi

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
echo "Project root                : $PROJECT_ROOT"
echo "Generation model            : $MODEL_PATH"
echo "Qwen router model           : $QWEN_ROUTER_MODEL"
echo "Routers                     : ${ROUTERS[*]}"
echo "Targets                     : ${TARGETS[*]}"
echo "Top-k                       : $TOP_K"
echo "Alpha                       : $ALPHA"
echo "NFrames                     : $NFRAMES"
echo "Route dir                   : $ROUTE_DIR"
echo "Output root                 : $OUTPUT_ROOT"
echo "Query BGE dir               : $QUERY_BGE_DIR"
echo "Query InternVideo dir       : $QUERY_INTERNVIDEO_DIR"
echo "FORCE_ROUTE                 : $FORCE_ROUTE"
echo "FORCE_EVAL                  : $FORCE_EVAL"
echo "BAYES_ALPHA_PRIOR           : $BAYES_ALPHA_PRIOR"
echo "BAYES_TAU                   : $BAYES_TAU"
echo "BAYES_BETA_COST             : $BAYES_BETA_COST"
echo "BAYES_MODALITY_COSTS        : $BAYES_MODALITY_COSTS"
echo "BAYES_ROUTER_PROBS_TEMPERATURE : $BAYES_ROUTER_PROBS_TEMPERATURE"
echo "BAYES_ROUTER_PROBS_TEMPERATURE_BY_TARGET : ${BAYES_ROUTER_PROBS_TEMPERATURE_BY_TARGET:-<none>}"
echo "BAYES_ROUTER_PROBS_BLEND_WITH_ORIGINAL : $BAYES_ROUTER_PROBS_BLEND_WITH_ORIGINAL"
echo "BAYES_ROUTER_PROBS_BLEND_WITH_ORIGINAL_BY_TARGET : ${BAYES_ROUTER_PROBS_BLEND_WITH_ORIGINAL_BY_TARGET:-<none>}"
echo "BAYES_DEFAULT_CONFIDENCE    : $BAYES_DEFAULT_CONFIDENCE"
echo "BAYES_UNCERTAINTY_THRESHOLD : $BAYES_UNCERTAINTY_THRESHOLD"
echo "BAYES_DECISION_MODE         : $BAYES_DECISION_MODE"
echo "BAYES_FALLBACK_WHEN_UNCERTAIN : $BAYES_FALLBACK_WHEN_UNCERTAIN"
echo "BAYES_ONLINE_UPDATE         : $BAYES_ONLINE_UPDATE"
echo "BAYES_ETA                   : $BAYES_ETA"
echo "BAYES_RHO                   : $BAYES_RHO"
echo "BAYES_SEED                  : $BAYES_SEED"
echo "BAYES_TAG                   : $BAYES_TAG"
echo "BAYES_SOFT_TOP_N            : $BAYES_SOFT_TOP_N"
echo "BAYES_SOFT_WEIGHT_MODE      : $BAYES_SOFT_WEIGHT_MODE"
echo "BAYES_SOFT_FUSION_MODE      : $BAYES_SOFT_FUSION_MODE"
echo "BAYES_SOFT_FUSION_MAX_NEW_TOKENS : $BAYES_SOFT_FUSION_MAX_NEW_TOKENS"
echo "BAYES_SOFT_STORE_CANDIDATES : $BAYES_SOFT_STORE_CANDIDATES"
echo "BAYES_TAG_EFFECTIVE         : $BAYES_TAG_EFFECTIVE"
echo "BAT_TARGET_PRIORS           : ${BAT_TARGET_PRIORS:-<none>}"
echo "BAT_PENALTY                 : $BAT_PENALTY"
echo "BAT_SPREAD                  : $BAT_SPREAD"
echo "BAT_USE_PENALTY_UPDATE      : $BAT_USE_PENALTY_UPDATE"
echo "BAYES_EXPORT_PROBS          : $BAYES_EXPORT_PROBS"
echo "BAYES_FORCE_EXPORT_PROBS    : $BAYES_FORCE_EXPORT_PROBS"
echo "BAYES_PROB_ROUTE_DIR        : $BAYES_PROB_ROUTE_DIR"
echo "BAYES_PROB_ROUTERS          : $BAYES_PROB_ROUTERS"
echo "BAYES_PROB_BATCH_SIZE_DISTILBERT : $BAYES_PROB_BATCH_SIZE_DISTILBERT"
echo "BAYES_PROB_BATCH_SIZE_T5    : $BAYES_PROB_BATCH_SIZE_T5"
echo "BAYES_PROB_DEVICE           : $BAYES_PROB_DEVICE"
echo "============================================================"

# Step 1) Build route files (reusing existing routers)
for ROUTER in "${ROUTERS[@]}"; do
  NEED_ROUTE=0
  if [ "$FORCE_ROUTE" = "1" ]; then
    NEED_ROUTE=1
  else
    for TARGET in "${TARGETS[@]}"; do
      if [ ! -f "$ROUTE_DIR/${ROUTER}/${TARGET}.json" ]; then
        NEED_ROUTE=1
        break
      fi
    done
  fi

  if [ "$NEED_ROUTE" = "1" ]; then
    if [ "$ROUTE_DIR" != "route/results" ]; then
      echo "[ERROR] Automatic routing currently writes to route/results. Set ROUTE_DIR=route/results or pre-generate route files."
      exit 1
    fi
    echo ""
    echo "================ ROUTING: $ROUTER ================"
    if [ "$ROUTER" = "qwen" ]; then
      bash script/3_route.sh "$ROUTER" "$QWEN_ROUTER_MODEL" "$TMP_QUERY_DIR" | tee "logs/route_${ROUTER}.log"
    else
      bash script/3_route.sh "$ROUTER" "" "$TMP_QUERY_DIR" | tee "logs/route_${ROUTER}.log"
    fi
  else
    echo "[SKIP] $ROUTE_DIR/${ROUTER} already exists for all non-video targets."
  fi
done

# Step 1.5) Optional probability export for supported routers
TARGETS_CSV="$(IFS=,; echo "${TARGETS[*]}")"
if [ "$BAYES_EXPORT_PROBS" = "1" ]; then
  read -r -a PROB_ROUTERS <<< "$(echo "$BAYES_PROB_ROUTERS" | tr ',' ' ')"
  for ROUTER in "${PROB_ROUTERS[@]}"; do
    case "$ROUTER" in
      distilbert)
        PROB_BATCH_SIZE="$BAYES_PROB_BATCH_SIZE_DISTILBERT"
        ;;
      t5-large)
        PROB_BATCH_SIZE="$BAYES_PROB_BATCH_SIZE_T5"
        ;;
      *)
        echo "[SKIP] Probability export is not supported for router=$ROUTER"
        continue
        ;;
    esac

    NEED_PROB_EXPORT=0
    if [ "$BAYES_FORCE_EXPORT_PROBS" = "1" ]; then
      NEED_PROB_EXPORT=1
    else
      for TARGET in "${TARGETS[@]}"; do
        if [ ! -f "$BAYES_PROB_ROUTE_DIR/${ROUTER}/${TARGET}.json" ]; then
          NEED_PROB_EXPORT=1
          break
        fi
      done
    fi

    if [ "$NEED_PROB_EXPORT" = "1" ]; then
      echo ""
      echo "================ BAYES PROBS: $ROUTER ================"
      python route/export_router_probs_sidecar.py \
        --router_model "$ROUTER" \
        --base_route_dir "$ROUTE_DIR" \
        --output_dir "$BAYES_PROB_ROUTE_DIR" \
        --targets "$TARGETS_CSV" \
        --batch_size "$PROB_BATCH_SIZE" \
        --device "$BAYES_PROB_DEVICE" | tee "logs/bayes_probs_${ROUTER}.log"
    else
      echo "[SKIP] $BAYES_PROB_ROUTE_DIR/${ROUTER} already exists for all requested targets."
    fi
  done
fi

# Step 2) Bayes evaluation
for ROUTER in "${ROUTERS[@]}"; do
  for TARGET in "${TARGETS[@]}"; do
    RESULT_FILE="${OUTPUT_ROOT}/${MODEL_NAME}/${ROUTER}/${TARGET}_top${TOP_K}_${ALPHA}_${NFRAMES_TAG}_bayes_${BAYES_TAG_EFFECTIVE}.json"
    ROUTE_DIR_FOR_RUN="$ROUTE_DIR"
    if [ -f "$BAYES_PROB_ROUTE_DIR/${ROUTER}/${TARGET}.json" ]; then
      ROUTE_DIR_FOR_RUN="$BAYES_PROB_ROUTE_DIR"
    fi

    if [ "$FORCE_EVAL" = "1" ] || [ ! -f "$RESULT_FILE" ]; then
      echo ""
      echo "================ BAYES EVAL: router=${ROUTER}, target=${TARGET} ================"
      if [ "$ROUTE_DIR_FOR_RUN" != "$ROUTE_DIR" ]; then
        echo "[INFO] Using probability-enriched route file: $ROUTE_DIR_FOR_RUN/${ROUTER}/${TARGET}.json"
      fi
      python eval/eval_bayes.py \
        --model_path "$MODEL_PATH" \
        --router_model "$ROUTER" \
        --target "$TARGET" \
        --top_k "$TOP_K" \
        --alpha "$ALPHA" \
        --nframes "$NFRAMES" \
        --route_dir "$ROUTE_DIR_FOR_RUN" \
        --output_root "$OUTPUT_ROOT" \
        --query_bge_dir "$QUERY_BGE_DIR" \
        --query_internvideo_dir "$QUERY_INTERNVIDEO_DIR" \
        --bayes_tag "$BAYES_TAG_EFFECTIVE" \
        --alpha_prior "$BAYES_ALPHA_PRIOR" \
        --alpha_prior_by_target "$BAT_TARGET_PRIORS" \
        --tau "$BAYES_TAU" \
        --beta_cost "$BAYES_BETA_COST" \
        --modality_costs "$BAYES_MODALITY_COSTS" \
        --router_probs_temperature "$BAYES_ROUTER_PROBS_TEMPERATURE" \
        --router_probs_temperature_by_target "$BAYES_ROUTER_PROBS_TEMPERATURE_BY_TARGET" \
        --router_probs_blend_with_original "$BAYES_ROUTER_PROBS_BLEND_WITH_ORIGINAL" \
        --router_probs_blend_with_original_by_target "$BAYES_ROUTER_PROBS_BLEND_WITH_ORIGINAL_BY_TARGET" \
        --default_confidence "$BAYES_DEFAULT_CONFIDENCE" \
        --uncertainty_threshold "$BAYES_UNCERTAINTY_THRESHOLD" \
        --decision_mode "$BAYES_DECISION_MODE" \
        --fallback_when_uncertain "$BAYES_FALLBACK_WHEN_UNCERTAIN" \
        --online_update "$BAYES_ONLINE_UPDATE" \
        --eta "$BAYES_ETA" \
        --rho "$BAYES_RHO" \
        --penalty "$BAT_PENALTY" \
        --spread "$BAT_SPREAD" \
        --use_penalty_update "$BAT_USE_PENALTY_UPDATE" \
        --soft_top_n "$BAYES_SOFT_TOP_N" \
        --soft_weight_mode "$BAYES_SOFT_WEIGHT_MODE" \
        --soft_fusion_mode "$BAYES_SOFT_FUSION_MODE" \
        --soft_fusion_max_new_tokens "$BAYES_SOFT_FUSION_MAX_NEW_TOKENS" \
        --soft_store_candidates "$BAYES_SOFT_STORE_CANDIDATES" \
        --seed "$BAYES_SEED" | tee "logs/eval_bayes_${ROUTER}_${TARGET}.log"
    else
      echo "[SKIP] $RESULT_FILE already exists."
    fi
  done
done

# Step 3) Score summary
echo ""
echo "================ BAYES SCORING SUMMARY ================"

python - <<PY
import os
import sys
from tabulate import tabulate

project_root = r"$PROJECT_ROOT"
sys.path.insert(0, project_root)
from eval.score import score_file

model_name = r"$MODEL_NAME"
top_k = r"$TOP_K"
alpha = r"$ALPHA"
nframes_tag = r"$NFRAMES_TAG"
output_root = r"$OUTPUT_ROOT"
bayes_tag = r"$BAYES_TAG_EFFECTIVE"

routers = [x for x in "${ROUTERS[*]}".split() if x]
targets = ["mmlu", "squad", "natural_questions", "hotpotqa", "webqa"]

rows = []
for router in routers:
    row = {"Router": router}
    for target in targets:
        result_file = os.path.join(
            project_root,
            output_root,
            model_name,
            router,
            f"{target}_top{top_k}_{alpha}_{nframes_tag}_bayes_{bayes_tag}.json",
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
