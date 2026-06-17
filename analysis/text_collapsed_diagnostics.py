#!/usr/bin/env python
"""Appendix diagnostics for SQuAD/NQ text-collapsed routing and cost accounting.

The main paper reports text-collapsed routing/cost on SQuAD and NQ because
paragraph and document actions resolve to the same passage-level evidence store.
This script keeps the raw four-class view beside the collapsed view so the
appendix can report the size of the removed distinction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.summarize_qa_cost import DEFAULT_COSTS, cost_stats, find_result_file, parse_costs

TEXT_EQUIV_TARGETS = {"squad", "nq", "natural_questions"}
TEXT_ACTIONS = {"paragraph", "document"}
MODALITIES = ["no", "paragraph", "document", "image"]
DEFAULT_TARGETS = ["squad", "natural_questions"]


def parse_list(text: str, default: Iterable[str]) -> List[str]:
    if not text:
        return list(default)
    return [x.strip() for x in str(text).replace(",", " " ).split() if x.strip()]


def parse_method(text: str) -> Tuple[str, str, Dict[str, str]]:
    if "=" not in text:
        raise ValueError(f"Method spec must look like NAME=PATH, got: {text}")
    parts = [part.strip() for part in text.split(";") if part.strip()]
    name, path = parts[0].split("=", 1)
    overrides: Dict[str, str] = {}
    for part in parts[1:]:
        target, target_path = part.split("=", 1)
        overrides[target.strip()] = target_path.strip()
    return name.strip(), path.strip(), overrides


def load_rows(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON rows in {path}")
    return data


def normalize_action(value: object) -> str:
    action = str(value or "").strip().lower()
    return action if action in MODALITIES else "no"


def text_equiv_match(target: str, pred: object, gold: object) -> bool:
    pred = normalize_action(pred)
    gold = normalize_action(gold)
    if target in TEXT_EQUIV_TARGETS and pred in TEXT_ACTIONS and gold in TEXT_ACTIONS:
        return True
    return pred == gold


def route_stats(rows: Sequence[dict], target: str) -> Dict[str, float]:
    n = len(rows)
    counts = Counter(normalize_action(row.get("retrieval")) for row in rows)
    gold_counts = Counter(normalize_action(row.get("gt_retrieval")) for row in rows)
    raw_correct = sum(
        normalize_action(row.get("retrieval")) == normalize_action(row.get("gt_retrieval"))
        for row in rows
    )
    text_correct = sum(
        text_equiv_match(target, row.get("retrieval"), row.get("gt_retrieval"))
        for row in rows
    )
    out: Dict[str, float] = {
        "count": n,
        "raw_correct": raw_correct,
        "text_equiv_correct": text_correct,
        "acc_4class": raw_correct / n if n else float("nan"),
        "text_equiv_acc": text_correct / n if n else float("nan"),
    }
    for action in MODALITIES:
        out[f"pred_{action}"] = counts[action]
        out[f"gt_{action}"] = gold_counts[action]
        out[f"pred_{action}_rate"] = counts[action] / n if n else float("nan")
        out[f"{action}_rate"] = counts[action] / n if n else float("nan")
    return out


def fmt(value: object) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, (float, int)) and math.isfinite(float(value)):
        return f"{float(value):.6f}"
    return ""


def write_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = [
        "method",
        "target",
        "count",
        "acc_4class",
        "text_equiv_acc",
        "raw_correct",
        "text_equiv_correct",
        "avg_exec_cost",
        "avg_exec_cost_raw",
        "exec_cost_overhead",
        "avg_posterior_cost",
        "avg_posterior_cost_raw",
        "posterior_cost_overhead",
        "cost_accounting",
        "pred_no",
        "pred_paragraph",
        "pred_document",
        "pred_image",
        "paragraph_rate",
        "document_rate",
        "image_rate",
        "exec_paragraph_rate",
        "exec_document_rate",
        "exec_image_rate",
        "document_only_cost_overhead",
        "gt_no",
        "gt_paragraph",
        "gt_document",
        "gt_image",
        "file",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def print_markdown(rows: Sequence[Dict[str, object]]) -> None:
    print("| Method | Target | 4-class Acc | Text-equiv Acc | Doc Rate | Raw Cost | Collapsed Cost | Overhead |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        print(
            "| {method} | {target} | {raw} | {text} | {doc} | {raw_cost} | {cost} | {overhead} |".format(
                method=row.get("method", ""),
                target=row.get("target", ""),
                raw=fmt(row.get("acc_4class")),
                text=fmt(row.get("text_equiv_acc")),
                doc=fmt(row.get("document_rate")),
                raw_cost=fmt(row.get("avg_exec_cost_raw")),
                cost=fmt(row.get("avg_exec_cost")),
                overhead=fmt(row.get("exec_cost_overhead")),
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", action="append", required=True, help="NAME=RESULT_DIR")
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    parser.add_argument("--costs", default="no=0,paragraph=0.25,document=0.45,image=0.60")
    parser.add_argument("--expensive_threshold", type=float, default=0.45)
    parser.add_argument("--output_csv", default="analysis/results/text_collapsed_route_cost_diagnostics.csv")
    parser.add_argument("--allow_missing", action="store_true")
    args = parser.parse_args()

    costs = parse_costs(args.costs)
    targets = parse_list(args.targets, DEFAULT_TARGETS)
    rows_out: List[Dict[str, object]] = []
    for method_name, default_path, overrides in [parse_method(spec) for spec in args.method]:
        for target in targets:
            root = overrides.get(target, default_path)
            try:
                result_file = find_result_file(root, target)
            except FileNotFoundError:
                if args.allow_missing:
                    continue
                raise
            rows = load_rows(result_file)
            out: Dict[str, object] = {
                "method": method_name,
                "target": target,
                "file": result_file,
            }
            cstats = cost_stats(rows, target, costs, args.expensive_threshold)
            for action in MODALITIES:
                cstats[f"exec_{action}_rate"] = cstats.get(f"{action}_rate")
            out.update(cstats)
            out.update(route_stats(rows, target))
            out["document_only_cost_overhead"] = out.get("document_rate", 0.0) * (costs["document"] - costs["paragraph"])
            rows_out.append(out)

    write_csv(args.output_csv, rows_out)
    print_markdown(rows_out)
    print(f"\nSaved CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
