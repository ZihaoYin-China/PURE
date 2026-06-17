#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ -n "${PYTHON_BIN:-}" ]; then
  :
elif [ -x "/opt/conda/envs/universalrag/bin/python" ]; then
  PYTHON_BIN="/opt/conda/envs/universalrag/bin/python"
else
  PYTHON_BIN="python"
fi

MODEL_PATH="${MODEL_PATH:-qwen-api:qwen3.6-plus}"
ROUTER_MODEL="${ROUTER_MODEL:-distilbert}"
SPLIT="${SPLIT:-full}"
TOP_K="${TOP_K:-1}"
ALPHA="${ALPHA:-0.2}"
NFRAMES="${NFRAMES:-1}"
VISUAL_RAG_POOL="${VISUAL_RAG_POOL:-full}"

if [ "$ROUTER_MODEL" != "distilbert" ]; then
  echo "This script currently supports the DistilBERT VIB/COVER router for VisualRAG."
  echo "Set ROUTER_MODEL=distilbert or route a T5-compatible OOD file separately."
  exit 1
fi

case "$SPLIT" in
  full)
    INPUT_DIR="${INPUT_DIR:-dataset/query_ood}"
    ROUTE_DIR="${ROUTE_DIR:-route/results_ood_vib_strict_d40_full}"
    OUTPUT_ROOT="${OUTPUT_ROOT:-eval/results_cover_visualrag_qwen36plus_full}"
    case "$VISUAL_RAG_POOL" in
      full)
        VISUAL_RAG_IMAGE_FEATS="${VISUAL_RAG_IMAGE_FEATS:-eval/features/image/visual_rag_full.pkl}"
        VISUAL_RAG_IMGCAP_FEATS="${VISUAL_RAG_IMGCAP_FEATS:-eval/features/image/visual_rag_full_imgcap.pkl}"
        ;;
      sample5|sample5_seed42)
        VISUAL_RAG_POOL="sample5_seed42"
        VISUAL_RAG_IMAGE_FEATS="${VISUAL_RAG_IMAGE_FEATS:-eval/features/image/visual_rag_sample5_seed42.pkl}"
        VISUAL_RAG_IMGCAP_FEATS="${VISUAL_RAG_IMGCAP_FEATS:-eval/features/image/visual_rag_sample5_seed42_imgcap.pkl}"
        ;;
      sample5_gt)
        VISUAL_RAG_IMAGE_FEATS="${VISUAL_RAG_IMAGE_FEATS:-eval/features/image/visual_rag_sample5_gt.pkl}"
        VISUAL_RAG_IMGCAP_FEATS="${VISUAL_RAG_IMGCAP_FEATS:-eval/features/image/visual_rag_sample5_gt_imgcap.pkl}"
        ;;
      *)
        echo "Unsupported VISUAL_RAG_POOL=$VISUAL_RAG_POOL for SPLIT=full."
        echo "Use full, sample5_seed42, or sample5_gt."
        exit 1
        ;;
    esac
    ;;
  holdout)
    if [ "$VISUAL_RAG_POOL" != "full" ]; then
      echo "VISUAL_RAG_POOL=$VISUAL_RAG_POOL is only prepared for SPLIT=full."
      echo "Use SPLIT=full or build a matching holdout pool first."
      exit 1
    fi
    INPUT_DIR="${INPUT_DIR:-dataset/query_ood_holdout}"
    ROUTE_DIR="${ROUTE_DIR:-route/results_ood_vib_strict_d40_holdout}"
    OUTPUT_ROOT="${OUTPUT_ROOT:-eval/results_cover_visualrag_qwen36plus_holdout}"
    VISUAL_RAG_IMAGE_FEATS="${VISUAL_RAG_IMAGE_FEATS:-eval/features/image/visual_rag.pkl}"
    VISUAL_RAG_IMGCAP_FEATS="${VISUAL_RAG_IMGCAP_FEATS:-eval/features/image/visual_rag_imgcap.pkl}"
    ;;
  *)
    echo "Unsupported SPLIT=$SPLIT. Use SPLIT=full or SPLIT=holdout."
    exit 1
    ;;
esac

CHECKPOINT_DIR="${CHECKPOINT_DIR:-route/train/checkpoints/distilbert_vib_strict_d40}"
QUERY_BGE_DIR="${QUERY_BGE_DIR:-eval/features/query_ood/bge-large}"
QUERY_INTERNVIDEO_DIR="${QUERY_INTERNVIDEO_DIR:-eval/features/query_ood/internvideo}"
ROUTE_ONLY="${ROUTE_ONLY:-0}"
EVAL_ONLY="${EVAL_ONLY:-0}"
FORCE_ROUTE="${FORCE_ROUTE:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"
if [ "$VISUAL_RAG_POOL" = "full" ]; then
  DEFAULT_BAYES_TAG="visualrag_${SPLIT}_tau10_beta0p1_softtop2_theta_posteriorverifier"
else
  DEFAULT_BAYES_TAG="visualrag_${SPLIT}_${VISUAL_RAG_POOL}_tau10_beta0p1_softtop2_theta_posteriorverifier"
fi
BAYES_TAG="${BAYES_TAG:-$DEFAULT_BAYES_TAG}"
EVAL_SINGLE_RETRIEVER_CACHE="${EVAL_SINGLE_RETRIEVER_CACHE:-1}"
EVAL_PARTIAL_SAVE_EVERY="${EVAL_PARTIAL_SAVE_EVERY:-10}"
EVAL_GC_EVERY="${EVAL_GC_EVERY:-5}"

echo "================ COVER VisualRAG shared-generator ================"
echo "MODEL_PATH=$MODEL_PATH"
echo "SPLIT=$SPLIT"
echo "VISUAL_RAG_POOL=$VISUAL_RAG_POOL"
echo "INPUT_DIR=$INPUT_DIR"
echo "ROUTE_DIR=$ROUTE_DIR"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "QUERY_BGE_DIR=$QUERY_BGE_DIR"
echo "QUERY_INTERNVIDEO_DIR=$QUERY_INTERNVIDEO_DIR"
echo "VISUAL_RAG_IMAGE_FEATS=$VISUAL_RAG_IMAGE_FEATS"
echo "VISUAL_RAG_IMGCAP_FEATS=$VISUAL_RAG_IMGCAP_FEATS"
echo "EVAL_SINGLE_RETRIEVER_CACHE=$EVAL_SINGLE_RETRIEVER_CACHE"
echo "EVAL_PARTIAL_SAVE_EVERY=$EVAL_PARTIAL_SAVE_EVERY"
echo "EVAL_GC_EVERY=$EVAL_GC_EVERY"

if [ "$EVAL_ONLY" != "1" ]; then
  ROUTE_FILE="$ROUTE_DIR/$ROUTER_MODEL/visual_rag.json"
  if [ "$FORCE_ROUTE" = "1" ] || [ ! -f "$ROUTE_FILE" ]; then
    echo ""
    echo "================ ROUTE VisualRAG ================"
    CHECKPOINT_DIR="$CHECKPOINT_DIR" \
    INPUT_DIR="$INPUT_DIR" \
    OUTPUT_DIR="$ROUTE_DIR" \
    ROUTER_NAME="$ROUTER_MODEL" \
    INCLUDE_TARGETS="visual_rag" \
    bash script/15_route_distilbert_vib.sh
  else
    echo "[SKIP] Existing route file: $ROUTE_FILE"
  fi
fi

if [ "$ROUTE_ONLY" = "1" ]; then
  exit 0
fi

case "$MODEL_PATH" in
  qwen-api:*|dashscope:*)
    if [[ -z "${DASHSCOPE_API_KEY:-}" && -z "${QWEN_API_KEY:-}" ]]; then
      echo "Missing API key. Set DASHSCOPE_API_KEY or QWEN_API_KEY before Qwen API evaluation."
      exit 1
    fi
    ;;
  glm:*|zhipu:*|glm-*)
    if [[ -z "${GLM_API_KEY:-}" && -z "${ZHIPU_API_KEY:-}" && -z "${BIGMODEL_API_KEY:-}" ]]; then
      echo "Missing API key. Set GLM_API_KEY, ZHIPU_API_KEY, or BIGMODEL_API_KEY before GLM evaluation."
      exit 1
    fi
    ;;
esac

if [ ! -f "$VISUAL_RAG_IMAGE_FEATS" ] || [ ! -f "$VISUAL_RAG_IMGCAP_FEATS" ]; then
  echo "Missing VisualRAG image features."
  echo "For SPLIT=full, extract them with:"
  echo "  INTERNVIDEO_PATH=/path/to/InternVideo python preprocess/extract_image_feats.py --input_path dataset/visual_rag/images.json --output_path eval/features/image/visual_rag_full.pkl --num_splits 4 --split_index 0"
  echo "  INTERNVIDEO_PATH=/path/to/InternVideo python preprocess/extract_image_feats.py --input_path dataset/visual_rag/images.json --output_path eval/features/image/visual_rag_full.pkl --num_splits 4 --split_index 1"
  echo "  INTERNVIDEO_PATH=/path/to/InternVideo python preprocess/extract_image_feats.py --input_path dataset/visual_rag/images.json --output_path eval/features/image/visual_rag_full.pkl --num_splits 4 --split_index 2"
  echo "  INTERNVIDEO_PATH=/path/to/InternVideo python preprocess/extract_image_feats.py --input_path dataset/visual_rag/images.json --output_path eval/features/image/visual_rag_full.pkl --num_splits 4 --split_index 3"
  echo "  python preprocess/merge_pickle_splits.py --inputs eval/features/image/visual_rag_full_split*.pkl --output_path eval/features/image/visual_rag_full.pkl"
  echo "Or build the paper-style sampled pool with:"
  echo "  python preprocess/sample_visual_rag_pool.py --include_gt 0"
  exit 1
fi

MODEL_NAME="${MODEL_PATH##*/}"
NFRAMES_TAG="${NFRAMES//,/_}"
NFRAMES_TAG="${NFRAMES_TAG//:/}"
RESULT_FILE="$OUTPUT_ROOT/$MODEL_NAME/$ROUTER_MODEL/visual_rag_top${TOP_K}_${ALPHA}_${NFRAMES_TAG}_bayes_${BAYES_TAG}.json"

if [ "$FORCE_EVAL" = "1" ] || [ ! -f "$RESULT_FILE" ]; then
  echo ""
  echo "================ EVAL VisualRAG ================"
  VISUAL_RAG_IMAGE_FEATS="$VISUAL_RAG_IMAGE_FEATS" \
  VISUAL_RAG_IMGCAP_FEATS="$VISUAL_RAG_IMGCAP_FEATS" \
  EVAL_SINGLE_RETRIEVER_CACHE="$EVAL_SINGLE_RETRIEVER_CACHE" \
  EVAL_PARTIAL_SAVE_EVERY="$EVAL_PARTIAL_SAVE_EVERY" \
  EVAL_GC_EVERY="$EVAL_GC_EVERY" \
  "$PYTHON_BIN" eval/eval_bayes_vib_posterior.py \
    --model_path "$MODEL_PATH" \
    --router_model "$ROUTER_MODEL" \
    --target visual_rag \
    --top_k "$TOP_K" \
    --alpha "$ALPHA" \
    --nframes "$NFRAMES" \
    --route_dir "$ROUTE_DIR" \
    --output_root "$OUTPUT_ROOT" \
    --query_bge_dir "$QUERY_BGE_DIR" \
    --query_internvideo_dir "$QUERY_INTERNVIDEO_DIR" \
    --bayes_tag "$BAYES_TAG" \
    --alpha_prior_by_target "visual_rag=1.8,0.6,0.2,3.0" \
    --tau 10.0 \
    --beta_cost 0.1 \
    --modality_costs "0.0,0.25,0.45,0.60" \
    --default_confidence 0.72 \
    --uncertainty_threshold 0.35 \
    --fallback_when_uncertain 1 \
    --hybrid_use_base 0 \
    --vib_prob_field probs \
    --vib_uncertainty_low 0.28 \
    --vib_uncertainty_high 0.45 \
    --vib_weight_low 0.35 \
    --vib_weight_high 0.85 \
    --dynamic_tau_min 0.35 \
    --dynamic_tau_max 1.8 \
    --evidence_saturation 8.0 \
    --soft_top_n 2 \
    --soft_weight_mode theta \
    --posterior_verifier 1 \
    --posterior_agreement_weight 1.0 \
    --posterior_conflict_weight 0.3 \
    --posterior_route_weight 0.12 \
    --posterior_evidence_weight 0.05 \
    --posterior_empty_penalty 1.0 \
    --posterior_non_answer_penalty 0.85
else
  echo "[SKIP] Existing result: $RESULT_FILE"
fi

echo ""
echo "================ SCORE VisualRAG ================"
"$PYTHON_BIN" eval/score.py --result_file "$RESULT_FILE" --target visual_rag
