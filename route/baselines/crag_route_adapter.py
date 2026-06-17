#!/usr/bin/env python
"""
CRAG evaluator adapter for PURE evaluation.

CRAG judges whether retrieved evidence is useful. This adapter keeps the
generator and retrievers fixed by converting CRAG's evidence scores into the
PURE route action space:

  no / paragraph / document / image

The output files can be consumed by eval/eval.py exactly like the existing
Self-RAG and Adaptive-RAG route files.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from retrieve.retrieve_text import BGETextRetriever  # noqa: E402


ACTIONS = ["no", "paragraph", "document", "image"]
TEXT_ACTIONS = ["paragraph", "document"]
IMAGE_TARGETS = {"webqa", "visual_rag"}

TARGET_PREFERRED_TEXT: Dict[str, str] = {
    "mmlu": "paragraph",
    "squad": "paragraph",
    "natural_questions": "paragraph",
    "hotpotqa": "document",
    "webqa": "paragraph",
    "truthfulqa": "paragraph",
    "triviaqa": "document",
    "lara": "document",
    "visual_rag": "paragraph",
}


def parse_targets(raw: str) -> List[str]:
    return [t for t in re.split(r"[,\s]+", raw.strip()) if t]


def parse_actions(raw: str, target: str) -> List[str]:
    raw = raw.strip().lower()
    if raw in {"target", "target_preferred", "preferred"}:
        return [TARGET_PREFERRED_TEXT.get(target, "paragraph")]
    actions = []
    for action in re.split(r"[,\s]+", raw):
        if not action:
            continue
        if action not in TEXT_ACTIONS:
            raise ValueError(
                f"Invalid CRAG candidate action {action!r}. "
                f"Expected one of: {', '.join(TEXT_ACTIONS)}, or 'target'."
            )
        if action not in actions:
            actions.append(action)
    if not actions:
        raise ValueError("At least one candidate action is required.")
    return actions


def parse_action_map(
    raw: Optional[str],
    default: str,
    extra_actions: Optional[Sequence[str]] = None,
) -> Dict[str, str]:
    mapping: Dict[str, str] = {"*": default}
    allowed = set(ACTIONS)
    if extra_actions:
        allowed.update(extra_actions)
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
        if action not in allowed:
            raise ValueError(
                f"Invalid action {action!r} in map {raw!r}. "
                f"Expected one of: {', '.join(sorted(allowed))}"
            )
        mapping[target] = action
    return mapping


def lookup_map(mapping: Dict[str, str], target: str) -> str:
    return mapping.get(target, mapping.get("*", "no"))


def get_text_feature_paths(target: str, modality: str) -> List[str]:
    if modality == "paragraph":
        if target == "triviaqa":
            return ["eval/features/text/triviaqa.pkl"]
        return [
            "eval/features/text/squad.pkl",
            "eval/features/text/natural_questions.pkl",
        ]

    if modality == "document":
        if target == "lara":
            return ["eval/features/text/lara.pkl"]
        if target == "hotpotqa":
            return [os.environ.get("HOTPOTQA_TEXT_FEATS", "eval/features/text/hotpotqa.pkl")]
        return ["eval/features/text/hotpotqa.pkl"]

    raise ValueError(f"Invalid text modality: {modality}")


def load_json(path: str) -> object:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: object) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def sigmoid(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def score_to_flag(score: Optional[float], upper_threshold: float, lower_threshold: float) -> str:
    if score is None:
        return "incorrect"
    if score >= upper_threshold:
        return "correct"
    if score >= lower_threshold:
        return "ambiguous"
    return "incorrect"


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    clean = sorted(v for v in values if math.isfinite(float(v)))
    if not clean:
        return None
    q = min(1.0, max(0.0, q))
    idx = int(round((len(clean) - 1) * q))
    return float(clean[idx])


def auto_similarity_thresholds(
    best_scores: Dict[int, Dict[str, dict]],
    fallback_upper: float,
    fallback_lower: float,
) -> Tuple[float, float]:
    row_scores = []
    for per_action in best_scores.values():
        scores = [safe_float(info.get("score")) for info in per_action.values()]
        scores = [s for s in scores if s is not None]
        if scores:
            row_scores.append(max(scores))
    lower = percentile(row_scores, 0.25)
    upper = percentile(row_scores, 0.75)
    if lower is None or upper is None:
        return fallback_upper, fallback_lower
    if upper < lower:
        upper, lower = lower, upper
    return upper, lower


def read_text(path: str, max_chars: int) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read(max_chars).strip()


class CRAGEvaluator:
    """Batch scorer for CRAG's T5-style evidence evaluator."""

    def __init__(self, evaluator_path: str, device: str = "auto", max_seq_length: int = 512):
        import torch
        from transformers import AutoTokenizer, T5ForSequenceClassification

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(evaluator_path, use_fast=False)
        self.model = T5ForSequenceClassification.from_pretrained(
            evaluator_path,
            num_labels=1,
        )
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.max_seq_length = max_seq_length
        self.model.to(self.device)
        self.model.eval()

    def score_batch(self, inputs: Sequence[str]) -> List[float]:
        batch = self.tokenizer(
            list(inputs),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
        ).to(self.device)
        with self.torch.no_grad():
            outputs = self.model(
                batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
            )
        logits = outputs["logits"].detach().cpu().float().view(-1)
        return [float(x) for x in logits.tolist()]


def iter_target_files(source_route_dir: str, targets: Iterable[str]) -> Iterable[Tuple[str, str]]:
    for target in targets:
        source_file = os.path.join(source_route_dir, f"{target}.json")
        if not os.path.exists(source_file):
            print(f"\n[SKIP] {target}: no source file at {source_file}")
            continue
        yield target, source_file


def build_retriever(target: str, modality: str, query_bge_dir: str) -> BGETextRetriever:
    return BGETextRetriever(
        queryfeats_path=os.path.join(query_bge_dir, f"{target}.pkl"),
        textfeats_path=get_text_feature_paths(target, modality),
    )


def collect_examples(
    rows: List[dict],
    target: str,
    modality: str,
    retriever: BGETextRetriever,
    ndocs: int,
    max_chars: int,
    skip_image_queries: bool,
    allow_missing_query_features: bool,
) -> Tuple[List[dict], int]:
    examples: List[dict] = []
    errors = 0

    candidate_query_ids = []
    for row in rows:
        gt_modality = str(row.get("gt_retrieval", "")).strip().lower()
        if skip_image_queries and target in IMAGE_TARGETS and gt_modality == "image":
            continue
        query_id = row.get("index")
        if query_id is not None:
            candidate_query_ids.append(str(query_id))

    missing_query_ids = [
        query_id for query_id in candidate_query_ids
        if query_id not in retriever.queryfeats
    ]
    if missing_query_ids and not allow_missing_query_features:
        preview = ", ".join(missing_query_ids[:10])
        raise ValueError(
            f"{target}/{modality}: {len(missing_query_ids)}/{len(candidate_query_ids)} "
            "route rows are missing query features. This means --source_route_dir "
            "and --query_bge_dir are from different splits. First missing ids: "
            f"{preview}. For route/results, try --query_bge_dir "
            "eval/features/query/bge-large_old; for route/results_large, use "
            "eval/features/query/bge-large; for strict test, use "
            "eval/features/query_test_d40/bge-large."
        )
    if missing_query_ids:
        print(
            f"      [WARN] {target}/{modality}: missing query features for "
            f"{len(missing_query_ids)}/{len(candidate_query_ids)} rows; skipped."
        )

    for row_idx, row in enumerate(rows):
        gt_modality = str(row.get("gt_retrieval", "")).strip().lower()
        if skip_image_queries and target in IMAGE_TARGETS and gt_modality == "image":
            continue

        query_id = row.get("index")
        if query_id is None:
            errors += 1
            continue
        if str(query_id) not in retriever.queryfeats:
            errors += 1
            continue

        try:
            retrieved, retriever_scores = retriever.retrieve(query_id, top_k=ndocs)
        except Exception as exc:  # Keep one bad row from killing a full sweep.
            errors += 1
            print(f"      [WARN] {target}/{modality}: retrieve failed for {query_id}: {exc}")
            continue

        for rank, path in enumerate(retrieved):
            try:
                passage = read_text(path, max_chars=max_chars)
            except OSError as exc:
                errors += 1
                print(f"      [WARN] {target}/{modality}: cannot read {path}: {exc}")
                continue
            if not passage:
                continue

            retriever_score = None
            if rank < len(retriever_scores):
                retriever_score = safe_float(retriever_scores[rank])

            examples.append(
                {
                    "row_idx": row_idx,
                    "modality": modality,
                    "rank": rank,
                    "path": path,
                    "retriever_score": retriever_score,
                    "input": f"{str(row.get('question', '')).strip()} [SEP] {passage}",
                }
            )

    return examples, errors


def score_examples(
    evaluator: CRAGEvaluator,
    examples: List[dict],
    batch_size: int,
) -> Dict[int, Dict[str, dict]]:
    best: Dict[int, Dict[str, dict]] = defaultdict(dict)

    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        scores = evaluator.score_batch([ex["input"] for ex in batch])
        for ex, score in zip(batch, scores):
            row_idx = ex["row_idx"]
            modality = ex["modality"]
            current = best[row_idx].get(modality)
            if current is None or score > current["score"]:
                best[row_idx][modality] = {
                    "score": float(score),
                    "rank": int(ex["rank"]),
                    "path": ex["path"],
                    "retriever_score": ex["retriever_score"],
                    "score_source": "crag_evaluator",
                }

    return best


def score_examples_from_similarity(examples: List[dict]) -> Dict[int, Dict[str, dict]]:
    """Checkpoint-free CRAG-style fallback using retriever similarity."""
    best: Dict[int, Dict[str, dict]] = defaultdict(dict)
    for ex in examples:
        score = safe_float(ex.get("retriever_score"))
        if score is None:
            continue
        row_idx = ex["row_idx"]
        modality = ex["modality"]
        current = best[row_idx].get(modality)
        if current is None or score > current["score"]:
            best[row_idx][modality] = {
                "score": float(score),
                "rank": int(ex["rank"]),
                "path": ex["path"],
                "retriever_score": ex["retriever_score"],
                "score_source": "retriever_similarity",
            }
    return best


def merge_best_scores(
    score_maps: Iterable[Dict[int, Dict[str, dict]]]
) -> Dict[int, Dict[str, dict]]:
    merged: Dict[int, Dict[str, dict]] = defaultdict(dict)
    for score_map in score_maps:
        for row_idx, per_action in score_map.items():
            for action, info in per_action.items():
                current = merged[row_idx].get(action)
                if current is None or info["score"] > current["score"]:
                    merged[row_idx][action] = dict(info)
    return merged


def choose_action(
    per_action: Dict[str, dict],
    target: str,
    upper_threshold: float,
    lower_threshold: float,
    ambiguous_action_map: Dict[str, str],
    incorrect_action_map: Dict[str, str],
) -> Tuple[str, Optional[str], Optional[float], str]:
    best_action = None
    best_score = None

    for action, info in per_action.items():
        score = safe_float(info.get("score"))
        if score is None:
            continue
        if best_score is None or score > best_score:
            best_score = score
            best_action = action

    flag = score_to_flag(best_score, upper_threshold, lower_threshold)
    if flag == "correct":
        return best_action or "no", best_action, best_score, flag
    if flag == "ambiguous":
        mapped = lookup_map(ambiguous_action_map, target)
        return (best_action if mapped == "best" else mapped), best_action, best_score, flag
    return lookup_map(incorrect_action_map, target), best_action, best_score, flag


def route_rows(
    rows: List[dict],
    target: str,
    best_scores: Dict[int, Dict[str, dict]],
    upper_threshold: float,
    lower_threshold: float,
    skip_image_queries: bool,
    ambiguous_action_map: Dict[str, str],
    incorrect_action_map: Dict[str, str],
) -> Tuple[List[dict], int]:
    output_rows: List[dict] = []
    image_fallback_count = 0

    for row_idx, row in enumerate(rows):
        gt_modality = str(row.get("gt_retrieval", "")).strip().lower()

        if skip_image_queries and target in IMAGE_TARGETS and gt_modality == "image":
            modality = row.get("retrieval", gt_modality)
            best_action = modality
            best_score = None
            flag = "image_fallback"
            image_fallback_count += 1
        else:
            modality, best_action, best_score, flag = choose_action(
                best_scores.get(row_idx, {}),
                target=target,
                upper_threshold=upper_threshold,
                lower_threshold=lower_threshold,
                ambiguous_action_map=ambiguous_action_map,
                incorrect_action_map=incorrect_action_map,
            )

        out_row = dict(row)
        out_row["retrieval"] = modality
        out_row["retrieval_conf"] = sigmoid(best_score)
        out_row["router_model"] = "crag"
        out_row["crag_flag"] = flag
        out_row["crag_best_action"] = best_action
        out_row["crag_score"] = best_score
        out_row["crag_thresholds"] = {
            "upper": upper_threshold,
            "lower": lower_threshold,
        }
        out_row["crag_evidence"] = best_scores.get(row_idx, {})
        output_rows.append(out_row)

    return output_rows, image_fallback_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CRAG evidence evaluator -> PURE route adapter"
    )
    parser.add_argument(
        "--evaluator_path",
        type=str,
        default="similarity",
        help=(
            "CRAG evaluator checkpoint, e.g. a T5ForSequenceClassification path. "
            "Use similarity/none to fall back to BGE retrieval similarity when no "
            "official evaluator checkpoint is available."
        ),
    )
    parser.add_argument(
        "--source_route_dir",
        type=str,
        default="route/results/distilbert",
        help="Directory with PURE route files used as query source.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="route/results/crag",
        help="Output directory for PURE-compatible CRAG routes.",
    )
    parser.add_argument(
        "--targets",
        type=str,
        default="mmlu,squad,natural_questions,hotpotqa,webqa",
        help="Comma- or space-separated target datasets.",
    )
    parser.add_argument(
        "--query_bge_dir",
        type=str,
        default="eval/features/query/bge-large",
        help="Directory containing BGE query feature pickles.",
    )
    parser.add_argument(
        "--candidate_actions",
        type=str,
        default="paragraph,document",
        help=(
            "Text actions to let CRAG score. Use 'paragraph,document' for the "
            "main fair setting, or 'target' for target-preferred single action."
        ),
    )
    parser.add_argument("--ndocs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--max_chars", type=int, default=3000)
    parser.add_argument("--upper_threshold", type=float, default=0.592)
    parser.add_argument("--lower_threshold", type=float, default=-0.995)
    parser.add_argument(
        "--ambiguous_action_map",
        type=str,
        default="best",
        help=(
            "Action for CRAG ambiguous rows. Use 'best' to keep the best-scored "
            "text action, or an action / target:action map."
        ),
    )
    parser.add_argument(
        "--incorrect_action_map",
        type=str,
        default="no",
        help="Action for CRAG incorrect rows. Default: no.",
    )
    parser.add_argument(
        "--skip_image_queries",
        action="store_true",
        help="For image-labeled WebQA/VisualRAG rows, keep the source router decision.",
    )
    parser.add_argument(
        "--allow_missing_query_features",
        action="store_true",
        help="Skip rows whose query ids are absent from --query_bge_dir instead of failing.",
    )
    args = parser.parse_args()

    if args.ndocs <= 0:
        raise ValueError("--ndocs must be positive.")

    targets = parse_targets(args.targets)
    ambiguous_action_map = parse_action_map(
        args.ambiguous_action_map,
        "best",
        extra_actions=["best"],
    )
    incorrect_action_map = parse_action_map(args.incorrect_action_map, "no")

    evaluator = None
    use_similarity_fallback = str(args.evaluator_path).strip().lower() in {
        "",
        "none",
        "similarity",
        "retriever",
        "bge",
    }
    if use_similarity_fallback:
        print("[1/3] Using BGE retrieval similarity as CRAG-style evidence score.")
        print("      This is a checkpoint-free fallback, not the official CRAG evaluator.")
    else:
        print(f"[1/3] Loading CRAG evaluator: {args.evaluator_path}")
        evaluator = CRAGEvaluator(
            evaluator_path=args.evaluator_path,
            device=args.device,
            max_seq_length=args.max_seq_length,
        )
        print("      Evaluator loaded successfully.")

    os.makedirs(args.output_dir, exist_ok=True)

    for target, source_file in iter_target_files(args.source_route_dir, targets):
        print(f"\n[2/3] Routing {target}...")
        rows = load_json(source_file)
        if not isinstance(rows, list):
            raise ValueError(f"Expected a JSON array in {source_file}")

        actions = parse_actions(args.candidate_actions, target)
        print(f"      Candidate actions: {', '.join(actions)}")

        score_maps = []
        total_errors = 0
        for action in actions:
            print(f"      Retrieving/scoring {action} evidence...")
            retriever = build_retriever(
                target=target,
                modality=action,
                query_bge_dir=args.query_bge_dir,
            )
            examples, errors = collect_examples(
                rows=rows,
                target=target,
                modality=action,
                retriever=retriever,
                ndocs=args.ndocs,
                max_chars=args.max_chars,
                skip_image_queries=args.skip_image_queries,
                allow_missing_query_features=args.allow_missing_query_features,
            )
            total_errors += errors
            print(f"        Evidence pairs: {len(examples)}")
            if use_similarity_fallback:
                score_maps.append(score_examples_from_similarity(examples))
            else:
                score_maps.append(score_examples(evaluator, examples, args.batch_size))

        best_scores = merge_best_scores(score_maps)
        upper_threshold = args.upper_threshold
        lower_threshold = args.lower_threshold
        if (
            use_similarity_fallback
            and args.upper_threshold == 0.592
            and args.lower_threshold == -0.995
        ):
            upper_threshold, lower_threshold = auto_similarity_thresholds(
                best_scores,
                fallback_upper=args.upper_threshold,
                fallback_lower=args.lower_threshold,
            )
            print(
                "      Auto similarity thresholds: "
                f"upper={upper_threshold:.4f}, lower={lower_threshold:.4f}"
            )

        output_rows, image_fallback = route_rows(
            rows=rows,
            target=target,
            best_scores=best_scores,
            upper_threshold=upper_threshold,
            lower_threshold=lower_threshold,
            skip_image_queries=args.skip_image_queries,
            ambiguous_action_map=ambiguous_action_map,
            incorrect_action_map=incorrect_action_map,
        )

        counts = Counter(row["retrieval"] for row in output_rows)
        flag_counts = Counter(row["crag_flag"] for row in output_rows)
        total = len(output_rows)
        print(f"      Total queries: {total}")
        print(
            "      CRAG flags: "
            + ", ".join(
                f"{flag}={flag_counts.get(flag, 0)}"
                for flag in ["correct", "ambiguous", "incorrect", "image_fallback"]
            )
        )
        for action in ACTIONS:
            count = counts.get(action, 0)
            if count:
                print(f"        {action}: {count} ({100 * count / total:.1f}%)")
        if image_fallback:
            print(f"        [image fallback]: {image_fallback}")
        if total_errors:
            print(f"        [warnings]: {total_errors} retrieval/read errors")

        output_file = os.path.join(args.output_dir, f"{target}.json")
        save_json(output_file, output_rows)
        print(f"      Saved -> {output_file}")

    print(f"\n[3/3] Done. Route files written to: {args.output_dir}/")


if __name__ == "__main__":
    main()
