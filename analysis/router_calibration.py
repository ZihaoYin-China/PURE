#!/usr/bin/env python
"""Compute router accuracy and calibration metrics for PURE outputs.

The script accepts method specifications in the form:

    --method "Baseline=route/results_large_strict_d40_test/distilbert"

When a method uses a different path for one target, append target-specific
overrides separated by semicolons:

    --method "Final=eval/results/main/t5-large;hotpotqa=eval/results/hotpot/t5-large"

Each path can be either:
  - a directory containing target JSON files, e.g. mmlu.json; or
  - a directory containing eval JSON files, e.g. mmlu_top1_...json; or
  - a file/template containing "{target}".
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


LABELS = ["no", "paragraph", "document", "image"]
TEXT_EQUIV_TARGETS = {"squad", "nq", "natural_questions"}
TEXT_ACTIONS = {"paragraph", "document"}
COLLAPSED_LABELS = ["no", "text", "image"]


def parse_targets(text: str) -> List[str]:
    return [x.strip() for x in str(text).replace(",", " ").split() if x.strip()]


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_target_file(root_or_template: str, target: str) -> str:
    path = root_or_template.format(target=target)
    if os.path.isfile(path):
        return path

    if os.path.isdir(path):
        direct = os.path.join(path, f"{target}.json")
        if os.path.isfile(direct):
            return direct
        candidates = sorted(
            p
            for p in glob.glob(os.path.join(path, f"{target}_*.json"))
            if not p.endswith(".meta.json")
        )
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise ValueError(
                f"Multiple JSON files for target={target} under {path}. "
                f"Use an explicit file template. Candidates: {candidates[:5]}"
            )

    candidates = sorted(p for p in glob.glob(path) if not p.endswith(".meta.json"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No JSON file found for target={target}: {root_or_template}")
    raise ValueError(
        f"Multiple JSON files for target={target}: {candidates[:5]}. "
        "Use a more specific path."
    )


def load_rows(root_or_template: str, target: str) -> Tuple[str, List[dict]]:
    path = find_target_file(root_or_template, target)
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON rows, got {type(data).__name__}: {path}")
    return path, data


def row_key(row: dict, target: str) -> Tuple[str, str]:
    source = str(row.get("source") or target)
    index = str(row.get("index"))
    return source, index


def normalize_label(label: object) -> str:
    return str(label or "").strip().lower()


def is_text_equiv_target(target: object) -> bool:
    return normalize_label(target) in TEXT_EQUIV_TARGETS


def collapse_label(target: object, label: object) -> str:
    label = normalize_label(label)
    if is_text_equiv_target(target) and label in TEXT_ACTIONS:
        return "text"
    return label


def route_match(target: object, pred: object, gold: object) -> bool:
    pred = normalize_label(pred)
    gold = normalize_label(gold)
    if is_text_equiv_target(target) and pred in TEXT_ACTIONS and gold in TEXT_ACTIONS:
        return True
    return pred == gold


def collapse_probs(target: object, probs: Optional[Sequence[float]]) -> Optional[List[float]]:
    if probs is None:
        return None
    if not is_text_equiv_target(target):
        return [float(x) for x in probs]
    if len(probs) != len(LABELS):
        return None
    return [float(probs[0]), float(probs[1]) + float(probs[2]), float(probs[3])]


def labels_for_targets(targets: Sequence[object], collapsed: bool) -> List[str]:
    if not collapsed:
        return list(LABELS)
    has_text_equiv = any(is_text_equiv_target(target) for target in targets)
    has_regular = any(not is_text_equiv_target(target) for target in targets)
    if has_text_equiv and has_regular:
        return ["no", "paragraph", "document", "image", "text"]
    if has_text_equiv:
        return list(COLLAPSED_LABELS)
    return list(LABELS)


def label_index(label: str, labels: Sequence[str] = LABELS) -> Optional[int]:
    label = normalize_label(label)
    try:
        return labels.index(label)
    except ValueError:
        return None


def normalize_probs(values: Iterable[object]) -> Optional[List[float]]:
    try:
        probs = [max(0.0, float(x)) for x in values]
    except Exception:
        return None
    if len(probs) != len(LABELS):
        return None
    total = sum(probs)
    if total <= 0:
        return None
    return [x / total for x in probs]


def reorder_probs(row: dict, probs: Sequence[float]) -> Optional[List[float]]:
    order = row.get("retrieval_probs_order") or LABELS
    order = [normalize_label(x) for x in order]
    if len(order) != len(probs):
        return None
    out = [0.0] * len(LABELS)
    for src_idx, label in enumerate(order):
        dst_idx = label_index(label)
        if dst_idx is None:
            return None
        out[dst_idx] = float(probs[src_idx])
    return normalize_probs(out)


def probability_vector(row: dict, mode: str) -> Optional[List[float]]:
    mode = str(mode or "auto").lower()
    if mode in {"auto", "bayes"} and isinstance(row.get("retrieval_bayes_theta"), list):
        probs = normalize_probs(row["retrieval_bayes_theta"])
        if probs is not None:
            return probs
    if mode in {"auto", "dirichlet"} and isinstance(row.get("retrieval_dirichlet_mean"), list):
        probs = normalize_probs(row["retrieval_dirichlet_mean"])
        if probs is not None:
            return probs
    if mode in {"auto", "probs", "softmax"} and isinstance(row.get("retrieval_probs"), list):
        probs = normalize_probs(row["retrieval_probs"])
        if probs is not None:
            reordered = reorder_probs(row, probs)
            return reordered or probs
    if mode in {"auto", "conf"}:
        pred_idx = label_index(row.get("retrieval"))
        conf = row.get("retrieval_conf")
        try:
            conf = float(conf)
        except Exception:
            conf = None
        if pred_idx is not None and conf is not None and math.isfinite(conf):
            conf = min(max(conf, 1.0 / len(LABELS)), 1.0)
            rest = (1.0 - conf) / (len(LABELS) - 1)
            probs = [rest] * len(LABELS)
            probs[pred_idx] = conf
            return probs
    return None


def prediction_confidence(row: dict, probs: Optional[Sequence[float]]) -> Optional[float]:
    pred_idx = label_index(row.get("retrieval"))
    if probs is not None and pred_idx is not None:
        return float(probs[pred_idx])
    try:
        conf = float(row.get("retrieval_conf"))
    except Exception:
        return None
    if math.isfinite(conf):
        return min(max(conf, 0.0), 1.0)
    return None


def uncertainty_score(row: dict, conf: Optional[float]) -> Optional[float]:
    for key in ("retrieval_bayes_uncertainty", "retrieval_uncertainty"):
        if key in row:
            try:
                value = float(row[key])
            except Exception:
                continue
            if math.isfinite(value):
                return value
    if conf is not None:
        return 1.0 - conf
    return None


def macro_f1(golds: Sequence[str], preds: Sequence[str], labels: Sequence[str] = LABELS) -> float:
    values = []
    for label in labels:
        tp = sum(1 for g, p in zip(golds, preds) if g == label and p == label)
        fp = sum(1 for g, p in zip(golds, preds) if g != label and p == label)
        fn = sum(1 for g, p in zip(golds, preds) if g == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        values.append(f1)
    return sum(values) / len(values) if values else float("nan")


def ece_score(correct: Sequence[int], confs: Sequence[float], bins: int) -> float:
    n = len(correct)
    if n == 0:
        return float("nan")
    total = 0.0
    for b in range(bins):
        low = b / bins
        high = (b + 1) / bins
        idxs = [
            i
            for i, conf in enumerate(confs)
            if (conf >= low and (conf < high or (b == bins - 1 and conf <= high)))
        ]
        if not idxs:
            continue
        acc = sum(correct[i] for i in idxs) / len(idxs)
        avg_conf = sum(confs[i] for i in idxs) / len(idxs)
        total += len(idxs) / n * abs(acc - avg_conf)
    return total


def brier_score(
    golds: Sequence[str],
    prob_vectors: Sequence[Optional[Sequence[float]]],
    labels: Sequence[str],
) -> float:
    values = []
    for gold, probs in zip(golds, prob_vectors):
        if probs is None or len(probs) != len(labels):
            continue
        try:
            gold_idx = labels.index(gold)
        except ValueError:
            continue
        target = [0.0] * len(labels)
        target[gold_idx] = 1.0
        values.append(sum((float(p) - t) ** 2 for p, t in zip(probs, target)))
    if not values:
        return float("nan")
    return sum(values) / len(values)


def auroc_binary(labels: Sequence[int], scores: Sequence[float]) -> float:
    pairs = [(float(score), int(label)) for label, score in zip(labels, scores)]
    positives = sum(label == 1 for _, label in pairs)
    negatives = sum(label == 0 for _, label in pairs)
    if positives == 0 or negatives == 0:
        return float("nan")
    pairs.sort(key=lambda x: x[0])
    rank_sum_pos = 0.0
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        rank_sum_pos += avg_rank * sum(label == 1 for _, label in pairs[i:j])
        i = j
    return (rank_sum_pos - positives * (positives + 1) / 2.0) / (positives * negatives)


def mean(values: Sequence[float]) -> float:
    values = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return sum(values) / len(values) if values else float("nan")


def compute_metrics(rows: Sequence[dict], target: str, confidence_mode: str, bins: int) -> Dict[str, object]:
    row_targets = [row.get("__target", target) for row in rows]
    golds_raw = [normalize_label(row.get("gt_retrieval")) for row in rows]
    preds_raw = [normalize_label(row.get("retrieval")) for row in rows]
    valid = [
        i
        for i, (gold, pred) in enumerate(zip(golds_raw, preds_raw))
        if gold in LABELS and pred in LABELS
    ]
    golds = [golds_raw[i] for i in valid]
    preds = [preds_raw[i] for i in valid]
    row_targets = [row_targets[i] for i in valid]
    rows = [rows[i] for i in valid]

    prob_vectors_4class = [probability_vector(row, confidence_mode) for row in rows]
    prob_vectors_main = [
        collapse_probs(row_target, probs)
        for row_target, probs in zip(row_targets, prob_vectors_4class)
    ]

    confs = []
    for row, row_target, probs_main, probs_4class in zip(rows, row_targets, prob_vectors_main, prob_vectors_4class):
        pred_main = collapse_label(row_target, row.get("retrieval"))
        labels_main = labels_for_targets([row_target], collapsed=True)
        if probs_main is not None and pred_main in labels_main:
            confs.append(float(probs_main[labels_main.index(pred_main)]))
        else:
            confs.append(prediction_confidence(row, probs_4class))

    uncertainties = [
        uncertainty_score(row, conf) for row, conf in zip(rows, confs)
    ]
    keep_conf = [i for i, conf in enumerate(confs) if conf is not None and math.isfinite(conf)]

    correct_4class = [1 if gold == pred else 0 for gold, pred in zip(golds, preds)]
    correct_main = [
        1 if route_match(row_target, pred, gold) else 0
        for row_target, gold, pred in zip(row_targets, golds, preds)
    ]
    golds_main = [collapse_label(row_target, gold) for row_target, gold in zip(row_targets, golds)]
    preds_main = [collapse_label(row_target, pred) for row_target, pred in zip(row_targets, preds)]
    labels_main = labels_for_targets(row_targets, collapsed=True)

    error_scores = []
    error_labels = []
    for is_correct, unc, conf in zip(correct_main, uncertainties, confs):
        score = unc if unc is not None and math.isfinite(float(unc)) else None
        if score is None and conf is not None:
            score = 1.0 - conf
        if score is not None and math.isfinite(float(score)):
            error_labels.append(0 if is_correct else 1)
            error_scores.append(float(score))

    by_gold = defaultdict(int)
    by_pred = defaultdict(int)
    for gold in golds:
        by_gold[gold] += 1
    for pred in preds:
        by_pred[pred] += 1

    conf_correct = [confs[i] for i, c in enumerate(correct_main) if c == 1 and confs[i] is not None]
    conf_wrong = [confs[i] for i, c in enumerate(correct_main) if c == 0 and confs[i] is not None]
    unc_correct = [
        uncertainties[i]
        for i, c in enumerate(correct_main)
        if c == 1 and uncertainties[i] is not None
    ]
    unc_wrong = [
        uncertainties[i]
        for i, c in enumerate(correct_main)
        if c == 0 and uncertainties[i] is not None
    ]

    acc_4class = sum(correct_4class) / len(correct_4class) if correct_4class else float("nan")
    acc_main = sum(correct_main) / len(correct_main) if correct_main else float("nan")
    brier_4class = brier_score(golds, prob_vectors_4class, LABELS)
    brier_main = brier_score(golds_main, prob_vectors_main, labels_main)

    return {
        "n": len(rows),
        "acc": acc_main,
        "acc_4class": acc_4class,
        "text_equiv_acc": acc_main,
        "text_equiv_applied": any(is_text_equiv_target(row_target) for row_target in row_targets),
        "macro_f1": macro_f1(golds_main, preds_main, labels_main) if correct_main else float("nan"),
        "macro_f1_4class": macro_f1(golds, preds, LABELS) if correct_main else float("nan"),
        "macro_f1_text_equiv": macro_f1(golds_main, preds_main, labels_main) if correct_main else float("nan"),
        "ece": ece_score([correct_main[i] for i in keep_conf], [confs[i] for i in keep_conf], bins),
        "brier": brier_main,
        "brier_4class": brier_4class,
        "brier_text_equiv": brier_main,
        "auroc_error": auroc_binary(error_labels, error_scores),
        "avg_conf_correct": mean(conf_correct),
        "avg_conf_wrong": mean(conf_wrong),
        "avg_unc_correct": mean(unc_correct),
        "avg_unc_wrong": mean(unc_wrong),
        "gold_counts": dict(by_gold),
        "pred_counts": dict(by_pred),
    }


def format_float(value: object) -> str:
    if not isinstance(value, (float, int)) or not math.isfinite(float(value)):
        return ""
    return f"{float(value):.4f}"


def parse_method(text: str) -> Tuple[str, str, Dict[str, str]]:
    if "=" not in text:
        raise ValueError(f"Method spec must look like NAME=PATH, got: {text}")
    parts = [part.strip() for part in text.split(";") if part.strip()]
    name, path = parts[0].split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise ValueError(f"Invalid method spec: {text}")
    overrides: Dict[str, str] = {}
    for part in parts[1:]:
        if "=" not in part:
            raise ValueError(
                "Target-specific overrides must look like target=PATH, "
                f"got: {part}"
            )
        target, target_path = part.split("=", 1)
        target = target.strip()
        target_path = target_path.strip()
        if not target or not target_path:
            raise ValueError(f"Invalid target override: {part}")
        overrides[target] = target_path
    return name, path, overrides


def write_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = [
        "method",
        "target",
        "n",
        "acc",
        "acc_4class",
        "text_equiv_acc",
        "text_equiv_applied",
        "macro_f1",
        "macro_f1_4class",
        "macro_f1_text_equiv",
        "ece",
        "brier",
        "brier_4class",
        "brier_text_equiv",
        "auroc_error",
        "avg_conf_correct",
        "avg_conf_wrong",
        "avg_unc_correct",
        "avg_unc_wrong",
        "source_file",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def print_markdown(rows: Sequence[Dict[str, object]]) -> None:
    fields = [
        ("method", "Method"),
        ("target", "Target"),
        ("n", "N"),
        ("acc", "Acc"),
        ("acc_4class", "4-class Acc"),
        ("text_equiv_acc", "Text-equiv Acc"),
        ("macro_f1", "Macro-F1"),
        ("ece", "ECE"),
        ("brier", "Brier"),
        ("auroc_error", "AUROC(error)"),
        ("avg_unc_correct", "Unc(correct)"),
        ("avg_unc_wrong", "Unc(wrong)"),
    ]
    print("| " + " | ".join(title for _, title in fields) + " |")
    print("| " + " | ".join(["---"] * len(fields)) + " |")
    for row in rows:
        values = []
        for key, _ in fields:
            value = row.get(key, "")
            values.append(str(value) if key in {"method", "target", "n"} else format_float(value))
        print("| " + " | ".join(values) + " |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", action="append", required=True, help="NAME=PATH")
    parser.add_argument(
        "--targets",
        default="mmlu,squad,natural_questions,hotpotqa,webqa",
        help="Comma/space separated target names.",
    )
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument(
        "--confidence_mode",
        default="auto",
        choices=["auto", "bayes", "dirichlet", "probs", "softmax", "conf"],
        help="Which probability source to use for calibration.",
    )
    parser.add_argument(
        "--align",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Align methods by (source,index) per target before computing metrics.",
    )
    parser.add_argument("--output_csv", default="analysis/results/router_calibration.csv")
    args = parser.parse_args()

    methods = [parse_method(spec) for spec in args.method]
    targets = parse_targets(args.targets)

    loaded: Dict[Tuple[str, str], Dict[str, object]] = {}
    by_target_method: Dict[str, Dict[str, Dict[Tuple[str, str], dict]]] = defaultdict(dict)
    source_files: Dict[Tuple[str, str], str] = {}

    for method_name, method_path, method_overrides in methods:
        for target in targets:
            source_root = method_overrides.get(target, method_path)
            source_file, rows = load_rows(source_root, target)
            keyed = {row_key(row, target): row for row in rows}
            by_target_method[target][method_name] = keyed
            source_files[(method_name, target)] = source_file

    out_rows: List[Dict[str, object]] = []
    pooled_rows_by_method: Dict[str, List[dict]] = defaultdict(list)

    for target in targets:
        method_maps = by_target_method[target]
        if args.align:
            common_keys = None
            for _, keyed in method_maps.items():
                keys = set(keyed.keys())
                common_keys = keys if common_keys is None else common_keys & keys
            common_keys = sorted(common_keys or [])
        else:
            common_keys = None

        for method_name, _, _ in methods:
            keyed = method_maps[method_name]
            rows = [keyed[key] for key in common_keys] if common_keys is not None else list(keyed.values())
            metrics = compute_metrics(rows, target, args.confidence_mode, args.bins)
            metrics.update(
                {
                    "method": method_name,
                    "target": target,
                    "source_file": source_files[(method_name, target)],
                }
            )
            out_rows.append(metrics)
            pooled_rows_by_method[method_name].extend(dict(row, __target=target) for row in rows)

    for method_name, _, _ in methods:
        metrics = compute_metrics(pooled_rows_by_method[method_name], "ALL", args.confidence_mode, args.bins)
        metrics.update({"method": method_name, "target": "ALL", "source_file": ""})
        out_rows.append(metrics)

    write_csv(args.output_csv, out_rows)
    print_markdown(out_rows)
    print(f"\nSaved CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
