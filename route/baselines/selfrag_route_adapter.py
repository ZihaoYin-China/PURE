#!/usr/bin/env python
"""
Self-RAG routing adapter: binary retrieval decisions → COVER action space.

Uses Self-RAG's fine-tuned Llama-2 to make retrieval decisions, then maps
them to COVER's 4-class action space. The resulting route files are consumed
by COVER's eval.py, which uses the SAME generator (Qwen3.6-plus) and SAME
retrievers (BGE/InternVideo) as all other baselines.

Architecture:
  Self-RAG model  ──→  binary decision (retrieve / no retrieve)
                           │
                           ▼
              TARGET_GRANULARITY mapping
                           │
                           ▼
              COVER route JSON  ──→  eval.py (Qwen3.6-plus + BGE)

This isolates routing quality: same generator, same retriever, different
routing logic (Self-RAG reflection tokens vs COVER VIB+EDL).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from typing import Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACTIONS = ["no", "paragraph", "document", "image"]

# When Self-RAG says "retrieve", which granularity should we use?
# Self-RAG only does binary routing, so this mapping picks the dataset-appropriate
# retrieval granularity based on task type.
TARGET_GRANULARITY: Dict[str, str] = {
    "mmlu": "paragraph",          # knowledge QA → paragraph for rare retrievals
    "squad": "paragraph",         # reading comprehension → paragraph
    "natural_questions": "paragraph",  # open-domain short-answer QA -> paragraph
    "hotpotqa": "document",       # multi-hop → document
    "triviaqa": "document",       # trivia → document
    "truthfulqa": "paragraph",    # factual QA → paragraph
    "lara": "document",           # long-form → document
    "webqa": "paragraph",         # text-based WebQA → paragraph
    "visual_rag": "paragraph",    # text-based VisRAG → paragraph
}

# Self-RAG prompt format (matches training-time format from paper)
SELFRAG_PROMPT_NO_INPUT = "### Instruction:\n{instruction}\n\n### Response:\n"

# Special reflection tokens Self-RAG uses for retrieval decisions
RETRIEVAL_TOKENS = ["[No Retrieval]", "[Retrieval]", "[Continue to Use Evidence]"]

# Targets with multiple-choice format that need special formatting
MCQ_TARGETS = {"mmlu", "truthfulqa"}

# Targets where image retrieval is meaningful
IMAGE_TARGETS = {"webqa", "visual_rag"}


def parse_retrieval_map(raw: Optional[str], fallback: Dict[str, str]) -> Dict[str, str]:
    """Parse Self-RAG retrieve->PURE action mapping."""
    mapping = dict(fallback)
    if not raw:
        return mapping

    for item in re.split(r"[,\s;]+", raw.strip()):
        if not item:
            continue
        if ":" in item:
            target, action = item.split(":", 1)
        elif "=" in item:
            target, action = item.split("=", 1)
        else:
            target, action = "*", item
        target = target.strip()
        action = action.strip().lower()
        if action not in ACTIONS:
            expected = ", ".join(ACTIONS)
            raise ValueError(
                f"Invalid retrieval action {action!r} in mapping {raw!r}. "
                f"Expected one of: {expected}"
            )
        if target == "*":
            for key in list(mapping):
                if key != "*":
                    mapping[key] = action
            mapping["*"] = action
        else:
            mapping[target] = action
    return mapping


def lookup_retrieval(mapping: Dict[str, str], target: str, default: str) -> str:
    return mapping.get(target, mapping.get("*", default))


# ---------------------------------------------------------------------------
# Query formatting (mirrors COVER eval.py reformat but simpler)
# ---------------------------------------------------------------------------

def format_query_for_selfrag(row: dict, target: str) -> str:
    """Format a query row into a string suitable for Self-RAG routing."""
    question = str(row.get("question", "")).strip()

    # For MCQ targets, keep choices in the instruction (Self-RAG sees full question)
    # The raw question from COVER already includes choices in "A) B) C) D)" format
    return question


# ---------------------------------------------------------------------------
# Self-RAG Router wrapper
# ---------------------------------------------------------------------------

class SelfRAGRouter:
    """
    Wraps Self-RAG's vllm model for routing-only inference.

    Only extracts the retrieval decision tokens; does NOT generate answers.
    """

    def __init__(
        self,
        model_path: str,
        download_dir: str = ".cache",
        dtype: str = "half",
        world_size: int = 1,
    ):
        self.model_path = model_path
        self._check_vllm()

        from vllm import LLM

        self.tokenizer = self._load_tokenizer(model_path, download_dir)
        self.model = LLM(
            model=model_path,
            download_dir=download_dir,
            dtype=dtype,
            tensor_parallel_size=world_size,
            max_logprobs=32016,
        )
        self.ret_token_ids = self._build_token_map(RETRIEVAL_TOKENS)

    @staticmethod
    def _check_vllm():
        try:
            import vllm  # noqa: F401
        except ImportError:
            print(
                "ERROR: vllm is required for Self-RAG routing.\n"
                "Install it with: pip install vllm\n"
                "vllm requires a CUDA-capable GPU.",
                file=sys.stderr,
            )
            sys.exit(1)

    def _load_tokenizer(self, model_path: str, cache_dir: Optional[str] = None):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            model_path, padding_side="left", cache_dir=cache_dir
        )

    def _build_token_map(self, tokens: List[str]) -> Dict[str, int]:
        token_map = {}
        for token in tokens:
            try:
                token_map[token] = self.tokenizer.convert_tokens_to_ids(token)
            except Exception:
                print(f"[WARN] Could not convert token '{token}' — skipping.")
        return token_map

    def route_batch(
        self,
        queries: List[str],
        threshold: Optional[float] = None,
        max_tokens: int = 10,
    ) -> List[dict]:
        """
        Run Self-RAG routing on a batch of queries.

        Args:
            queries: List of query strings.
            threshold: Retrieval probability threshold (Self-RAG default=0.2).
                       Lower = more aggressive retrieval.

        Returns:
            List of dicts with keys: decision, confidence, retrieval_logp,
            no_retrieval_logp, raw_text.
        """
        from vllm import SamplingParams

        prompts = [
            SELFRAG_PROMPT_NO_INPUT.format(instruction=q) for q in queries
        ]

        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=max_tokens,
            logprobs=32016,
            skip_special_tokens=False,
        )

        preds = self.model.generate(prompts, sampling_params)
        results = []

        for pred in preds:
            pred_log_probs = pred.outputs[0].logprobs
            pred_text = pred.outputs[0].text

            # Extract log-probabilities for retrieval decision tokens
            score_dict: Dict[str, float] = {}
            for tok, tok_id in self.ret_token_ids.items():
                logprob_entry = pred_log_probs[0].get(tok_id) if pred_log_probs else None
                if logprob_entry is None:
                    score_dict[tok] = float("-inf")
                else:
                    score_dict[tok] = float(getattr(logprob_entry, "logprob", logprob_entry))

            # Compute P(retrieve) from log-probabilities
            retrieval_logp = score_dict.get("[Retrieval]", float("-inf"))
            no_retrieval_logp = score_dict.get("[No Retrieval]", float("-inf"))

            if retrieval_logp > -50 and no_retrieval_logp > -50:
                # Both tokens have valid probabilities → normalize
                retrieval_prob = np.exp(retrieval_logp)
                no_retrieval_prob = np.exp(no_retrieval_logp)
                confidence = retrieval_prob / (retrieval_prob + no_retrieval_prob)
            elif "[Retrieval]" in str(pred_text):
                # Token appeared in text but not in logprobs → estimate
                confidence = 0.85
            elif "[No Retrieval]" in str(pred_text):
                confidence = 0.15
            else:
                # Neither token generated — check if model is generating an answer
                # (which implies it thinks retrieval isn't needed)
                confidence = 0.15

            threshold_val = threshold if threshold is not None else 0.2
            do_retrieve = confidence > threshold_val

            results.append({
                "decision": "retrieve" if do_retrieve else "no_retrieve",
                "confidence": float(confidence),
                "retrieval_logp": retrieval_logp,
                "no_retrieval_logp": no_retrieval_logp,
                "raw_text": pred_text,
            })

        return results


# ---------------------------------------------------------------------------
# Route file builder
# ---------------------------------------------------------------------------

def build_selfrag_routes(
    router: SelfRAGRouter,
    source_file: str,
    target: str,
    threshold: Optional[float],
    batch_size: int,
    skip_image_queries: bool,
    retrieval_map: Dict[str, str],
) -> List[dict]:
    """
    Read COVER query data, run Self-RAG routing, and produce COVER-format
    route entries.
    """
    with open(source_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {source_file}")

    # Extract and format queries
    raw_queries = [format_query_for_selfrag(row, target) for row in data]

    # Route in batches
    all_decisions: List[dict] = []
    n = len(raw_queries)
    for i in range(0, n, batch_size):
        batch = raw_queries[i : i + batch_size]
        decisions = router.route_batch(batch, threshold=threshold)
        all_decisions.extend(decisions)

    # Map to COVER format
    output_rows = []
    image_fallback_count = 0

    for row, decision in zip(data, all_decisions):
        gt_modality = str(row.get("gt_retrieval", "")).strip().lower()

        if decision["decision"] == "retrieve":
            modality = lookup_retrieval(retrieval_map, target, "paragraph")
        else:
            modality = "no"

        # For multimodal datasets: Self-RAG can't route to image retrieval,
        # so image-dependent queries fall back to the base router's decision.
        if (
            skip_image_queries
            and target in IMAGE_TARGETS
            and gt_modality == "image"
        ):
            modality = row.get("retrieval", gt_modality)
            image_fallback_count += 1

        out_row = dict(row)
        out_row["retrieval"] = modality
        out_row["retrieval_conf"] = decision["confidence"]
        out_row["retrieval_logp_retrieve"] = decision["retrieval_logp"]
        out_row["retrieval_logp_no_retrieve"] = decision["no_retrieval_logp"]
        out_row["router_model"] = "selfrag"
        out_row["selfrag_decision"] = decision["decision"]
        out_row["selfrag_raw_text"] = decision["raw_text"]
        output_rows.append(out_row)

    return output_rows, image_fallback_count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Self-RAG routing → COVER eval pipeline adapter"
    )
    parser.add_argument(
        "--model_name", type=str, default="selfrag/selfrag_llama2_7b",
        help="Self-RAG model path or HuggingFace ID."
    )
    parser.add_argument(
        "--source_route_dir", type=str, default="route/results/distilbert",
        help="Directory with COVER route files (read queries from here)."
    )
    parser.add_argument(
        "--output_dir", type=str, default="route/results/selfrag",
        help="Output directory for Self-RAG route files."
    )
    parser.add_argument(
        "--targets", type=str,
        default="mmlu,squad,natural_questions,hotpotqa,webqa",
        help="Comma-separated target datasets."
    )
    parser.add_argument(
        "--threshold", type=float, default=0.2,
        help="Self-RAG retrieval threshold. 0.2 = paper default (more retrieval), "
             "0.5 = balanced, 0.8 = conservative."
    )
    parser.add_argument(
        "--retrieval_map", type=str, default=None,
        help=(
            "Map Self-RAG retrieve decisions to PURE actions. "
            "Use one action for all targets, e.g. paragraph, or target:action pairs."
        )
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for vllm inference."
    )
    parser.add_argument(
        "--download_dir", type=str, default=".cache",
        help="vllm model download cache directory."
    )
    parser.add_argument(
        "--dtype", type=str, default="half",
        help="Model dtype for vllm (half, float16, bfloat16)."
    )
    parser.add_argument(
        "--world_size", type=int, default=1,
        help="Number of GPUs for tensor parallelism."
    )
    parser.add_argument(
        "--skip_image_queries", action="store_true",
        help="Fall back to base router for image-only queries in multimodal datasets."
    )
    args = parser.parse_args()

    targets = [t.strip() for t in args.targets.split(",")]
    retrieval_map = parse_retrieval_map(args.retrieval_map, TARGET_GRANULARITY)

    # --- Load Self-RAG model ---
    print(f"[1/3] Loading Self-RAG model: {args.model_name}")
    router = SelfRAGRouter(
        model_path=args.model_name,
        download_dir=args.download_dir,
        dtype=args.dtype,
        world_size=args.world_size,
    )
    print("      Model loaded successfully.")

    # --- Route each target ---
    os.makedirs(args.output_dir, exist_ok=True)

    for target in targets:
        source_file = os.path.join(args.source_route_dir, f"{target}.json")
        if not os.path.exists(source_file):
            print(f"\n[SKIP] {target}: no source file at {source_file}")
            continue

        print(f"\n[2/3] Routing {target}...")
        output_rows, img_fallback = build_selfrag_routes(
            router=router,
            source_file=source_file,
            target=target,
            threshold=args.threshold,
            batch_size=args.batch_size,
            skip_image_queries=args.skip_image_queries,
            retrieval_map=retrieval_map,
        )

        # --- Stats ---
        decisions = [r["retrieval"] for r in output_rows]
        counts = Counter(decisions)
        total = len(decisions)

        print(f"      Total queries: {total}")
        for action in ACTIONS:
            c = counts.get(action, 0)
            if c > 0:
                print(f"        {action}: {c} ({100 * c / total:.1f}%)")
        if img_fallback:
            print(f"        [image fallback]: {img_fallback}")

        # --- Save ---
        output_file = os.path.join(args.output_dir, f"{target}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_rows, f, indent=4, ensure_ascii=False)
        print(f"      Saved → {output_file}")

    # --- Next steps ---
    print(f"\n[3/3] Done. Route files written to: {args.output_dir}/")
    print()
    print("Next — run evaluation with unified Qwen generator:")
    print()
    for target in targets[:3]:
        print(
            f"  bash script/4_eval.sh \\\n"
            f"    --model_path qwen-api:qwen3.6-plus \\\n"
            f"    --router_model selfrag \\\n"
            f"    --target {target}"
        )
        print()
    print("Or run all at once:")
    print(f"  bash script/6_eval_all_routers_qwen.sh  # (add selfrag to the router list)")


if __name__ == "__main__":
    main()
