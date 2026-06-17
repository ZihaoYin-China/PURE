#!/usr/bin/env python
"""
Adaptive-RAG classifier adapter for PURE evaluation.

Adaptive-RAG predicts query complexity:
  A = no retrieval
  B = single-step retrieval
  C = multi-step retrieval

PURE evaluates one route action per query:
  no / paragraph / document / image

This adapter keeps the generator and retrievers fixed, reads existing PURE
route files for the queries, predicts A/B/C with a trained Adaptive-RAG T5
classifier, and writes PURE-compatible route JSON files.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional


ACTIONS = ["no", "paragraph", "document", "image"]
LABEL_TO_STRATEGY = {
    "A": "zero",
    "B": "single",
    "C": "multi",
}

# Adaptive-RAG only predicts retrieval complexity, not PURE granularity.
# These mappings choose the closest PURE retrieval action for B/C.
SINGLE_RETRIEVAL: Dict[str, str] = {
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

MULTI_RETRIEVAL: Dict[str, str] = {
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

IMAGE_TARGETS = {"webqa", "visual_rag"}


def parse_targets(raw: str) -> List[str]:
    return [t for t in re.split(r"[,\s]+", raw.strip()) if t]


def normalize_label(text: object) -> str:
    """Normalize classifier output into A/B/C, defaulting to B if malformed."""
    if text is None:
        return "B"
    match = re.search(r"[ABC]", str(text).strip().upper())
    return match.group(0) if match else "B"


def parse_retrieval_map(raw: Optional[str], fallback: Dict[str, str]) -> Dict[str, str]:
    """Parse mapping specs for Adaptive-RAG B/C labels."""
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


def load_json(path: str) -> object:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: object) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_prediction_file(path: str) -> Dict[str, str]:
    """
    Load Adaptive-RAG classifier predictions.

    Supports:
      - official dict_id_pred_results.json: {id: {"prediction": "A", ...}}
      - simple dict: {id: "A"}
      - list: [{"id": "...", "prediction": "A"}]
    """
    raw = load_json(path)
    preds: Dict[str, str] = {}

    if isinstance(raw, dict):
        for key, value in raw.items():
            if isinstance(value, dict):
                pred = value.get("prediction", value.get("label"))
            else:
                pred = value
            preds[str(key)] = normalize_label(pred)
        return preds

    if isinstance(raw, list):
        for row in raw:
            if not isinstance(row, dict):
                continue
            key = row.get("id", row.get("index", row.get("qid")))
            if key is None:
                continue
            preds[str(key)] = normalize_label(row.get("prediction", row.get("label")))
        return preds

    raise ValueError(f"Unsupported prediction file format: {path}")


def load_prediction_dir(path: str, target: str) -> Optional[Dict[str, str]]:
    candidates = [
        os.path.join(path, f"{target}.json"),
        os.path.join(path, f"{target}_predictions.json"),
        os.path.join(path, target, "dict_id_pred_results.json"),
        os.path.join(path, "dict_id_pred_results.json"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return load_prediction_file(candidate)
    return None


class AdaptiveRAGClassifier:
    """Direct HuggingFace inference wrapper for the trained T5 classifier."""

    def __init__(self, model_path: str, device: str = "auto", max_seq_length: int = 384):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.max_seq_length = max_seq_length
        self.model.to(self.device)
        self.model.eval()

    def predict_batch(self, questions: List[str], max_new_tokens: int = 4) -> List[str]:
        batch = self.tokenizer(
            [q.strip() for q in questions],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
        ).to(self.device)
        with self.torch.no_grad():
            outputs = self.model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                do_sample=False,
            )
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return [normalize_label(x) for x in decoded]


def map_label_to_retrieval(
    label: str,
    target: str,
    single_retrieval: Dict[str, str],
    multi_retrieval: Dict[str, str],
) -> str:
    label = normalize_label(label)
    if label == "A":
        return "no"
    if label == "B":
        return lookup_retrieval(single_retrieval, target, "paragraph")
    single_default = lookup_retrieval(single_retrieval, target, "paragraph")
    return lookup_retrieval(multi_retrieval, target, single_default)


def route_rows(
    rows: List[dict],
    labels: List[str],
    target: str,
    skip_image_queries: bool,
    single_retrieval: Dict[str, str],
    multi_retrieval: Dict[str, str],
) -> tuple[List[dict], int]:
    output_rows: List[dict] = []
    image_fallback_count = 0

    for row, label in zip(rows, labels):
        label = normalize_label(label)
        modality = map_label_to_retrieval(
            label,
            target,
            single_retrieval=single_retrieval,
            multi_retrieval=multi_retrieval,
        )
        gt_modality = str(row.get("gt_retrieval", "")).strip().lower()

        # Adaptive-RAG has no image class. For image-only cases, keep the source
        # router's image decision so multimodal OOD datasets remain evaluable.
        if skip_image_queries and target in IMAGE_TARGETS and gt_modality == "image":
            modality = row.get("retrieval", gt_modality)
            image_fallback_count += 1

        out_row = dict(row)
        out_row["retrieval"] = modality
        out_row["retrieval_conf"] = None
        out_row["router_model"] = "adaptive_rag"
        out_row["adaptive_rag_label"] = label
        out_row["adaptive_rag_strategy"] = LABEL_TO_STRATEGY[label]
        output_rows.append(out_row)

    return output_rows, image_fallback_count


def infer_labels(
    rows: List[dict],
    classifier: Optional[AdaptiveRAGClassifier],
    predictions: Optional[Dict[str, str]],
    batch_size: int,
) -> List[str]:
    if predictions is not None:
        labels = []
        missing = 0
        for row in rows:
            key = str(row.get("index", row.get("id", "")))
            label = predictions.get(key)
            if label is None:
                missing += 1
                label = "B"
            labels.append(normalize_label(label))
        if missing:
            print(f"      [WARN] {missing} rows missing predictions; defaulted to B.")
        return labels

    if classifier is None:
        raise ValueError("Either --classifier_model_path or predictions must be provided.")

    labels: List[str] = []
    questions = [str(row.get("question", "")) for row in rows]
    for start in range(0, len(questions), batch_size):
        labels.extend(classifier.predict_batch(questions[start : start + batch_size]))
    return labels


def iter_target_files(source_route_dir: str, targets: Iterable[str]) -> Iterable[tuple[str, str]]:
    for target in targets:
        source_file = os.path.join(source_route_dir, f"{target}.json")
        if not os.path.exists(source_file):
            print(f"\n[SKIP] {target}: no source file at {source_file}")
            continue
        yield target, source_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adaptive-RAG classifier -> PURE route adapter"
    )
    parser.add_argument(
        "--classifier_model_path",
        type=str,
        default=None,
        help="Trained Adaptive-RAG T5 classifier checkpoint. Optional if predictions are supplied.",
    )
    parser.add_argument(
        "--prediction_dir",
        type=str,
        default=None,
        help="Directory containing per-target predictions or dict_id_pred_results.json.",
    )
    parser.add_argument(
        "--prediction_file",
        type=str,
        default=None,
        help="Single prediction file used for all targets.",
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
        default="route/results/adaptive_rag",
        help="Output directory for PURE-compatible Adaptive-RAG routes.",
    )
    parser.add_argument(
        "--targets",
        type=str,
        default="mmlu,squad,natural_questions,hotpotqa,webqa",
        help="Comma- or space-separated target datasets.",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--single_retrieval_map",
        type=str,
        default=None,
        help=(
            "Map Adaptive-RAG B/single labels to PURE actions. "
            "Use an action for all targets, e.g. no, or target:action pairs."
        ),
    )
    parser.add_argument(
        "--multi_retrieval_map",
        type=str,
        default=None,
        help=(
            "Map Adaptive-RAG C/multi labels to PURE actions. "
            "Use an action for all targets, e.g. no, or target:action pairs."
        ),
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max_seq_length", type=int, default=384)
    parser.add_argument(
        "--skip_image_queries",
        action="store_true",
        help="For image-labeled WebQA/VisualRAG rows, keep the source router decision.",
    )
    args = parser.parse_args()

    targets = parse_targets(args.targets)
    single_retrieval = parse_retrieval_map(args.single_retrieval_map, SINGLE_RETRIEVAL)
    multi_retrieval = parse_retrieval_map(args.multi_retrieval_map, MULTI_RETRIEVAL)

    shared_predictions = None
    if args.prediction_file:
        shared_predictions = load_prediction_file(args.prediction_file)

    classifier = None
    if shared_predictions is None and not args.prediction_dir:
        if not args.classifier_model_path:
            raise ValueError(
                "Provide --classifier_model_path, --prediction_file, or --prediction_dir."
            )
        print(f"[1/3] Loading Adaptive-RAG classifier: {args.classifier_model_path}")
        classifier = AdaptiveRAGClassifier(
            model_path=args.classifier_model_path,
            device=args.device,
            max_seq_length=args.max_seq_length,
        )
        print("      Classifier loaded successfully.")
    else:
        print("[1/3] Using precomputed Adaptive-RAG predictions.")

    os.makedirs(args.output_dir, exist_ok=True)

    for target, source_file in iter_target_files(args.source_route_dir, targets):
        print(f"\n[2/3] Routing {target}...")
        rows = load_json(source_file)
        if not isinstance(rows, list):
            raise ValueError(f"Expected a JSON array in {source_file}")

        predictions = shared_predictions
        if predictions is None and args.prediction_dir:
            predictions = load_prediction_dir(args.prediction_dir, target)
            if predictions is None and classifier is None:
                raise FileNotFoundError(
                    f"No predictions found for {target} under {args.prediction_dir}"
                )

        labels = infer_labels(
            rows=rows,
            classifier=classifier,
            predictions=predictions,
            batch_size=args.batch_size,
        )
        output_rows, image_fallback = route_rows(
            rows=rows,
            labels=labels,
            target=target,
            skip_image_queries=args.skip_image_queries,
            single_retrieval=single_retrieval,
            multi_retrieval=multi_retrieval,
        )

        counts = Counter(row["retrieval"] for row in output_rows)
        label_counts = Counter(row["adaptive_rag_label"] for row in output_rows)
        total = len(output_rows)
        print(f"      Total queries: {total}")
        print(
            "      Labels: "
            + ", ".join(f"{label}={label_counts.get(label, 0)}" for label in ["A", "B", "C"])
        )
        for action in ACTIONS:
            count = counts.get(action, 0)
            if count:
                print(f"        {action}: {count} ({100 * count / total:.1f}%)")
        if image_fallback:
            print(f"        [image fallback]: {image_fallback}")

        output_file = os.path.join(args.output_dir, f"{target}.json")
        save_json(output_file, output_rows)
        print(f"      Saved -> {output_file}")

    print(f"\n[3/3] Done. Route files written to: {args.output_dir}/")


if __name__ == "__main__":
    main()
