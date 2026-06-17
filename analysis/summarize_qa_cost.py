#!/usr/bin/env python
"""Summarize QA metrics together with cost-aware routing statistics.

Method specs use the same form as ``analysis/router_calibration.py``:

    --method "Name=RESULT_DIR"
    --method "Name=RESULT_DIR;hotpotqa=HOTPOT_RESULT_DIR"

Each directory may contain files named like:
    mmlu_top1_0.2_1.json
    mmlu_top1_0.2_1_bayes_xxx.json
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from typing import Dict, Iterable, List, Sequence, Tuple

from eval.score import score_file


TARGETS = ["mmlu", "squad", "natural_questions", "hotpotqa", "webqa"]
MODALITIES = ["no", "paragraph", "document", "image"]
TEXT_EQUIV_TARGETS = {"squad", "nq", "natural_questions"}
TEXT_ACTIONS = {"paragraph", "document"}
DEFAULT_COSTS = {
    "no": 0.0,
    "paragraph": 0.25,
    "document": 0.45,
    "image": 0.60,
}


def parse_list(text: str, default: Iterable[str]) -> List[str]:
    if not text:
        return list(default)
    return [x.strip() for x in str(text).replace(",", " ").split() if x.strip()]


def parse_costs(text: str) -> Dict[str, float]:
    if not text:
        return dict(DEFAULT_COSTS)
    parts = [x.strip() for x in str(text).replace(",", " ").split() if x.strip()]
    if len(parts) == 4 and all("=" not in x for x in parts):
        return {m: float(v) for m, v in zip(MODALITIES, parts)}
    out = dict(DEFAULT_COSTS)
    for part in parts:
        if "=" not in part:
            raise ValueError(f"Invalid cost entry: {part}")
        key, value = part.split("=", 1)
        key = key.strip().lower()
        if key not in out:
            raise ValueError(f"Unknown modality in cost entry: {part}")
        out[key] = float(value)
    return out


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
            raise ValueError(f"Override must look like target=PATH, got: {part}")
        target, target_path = part.split("=", 1)
        overrides[target.strip()] = target_path.strip()
    return name, path, overrides


def find_result_file(root_or_template: str, target: str) -> str:
    path = root_or_template.format(target=target)
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        candidates = sorted(
            p
            for p in glob.glob(os.path.join(path, f"{target}_*.json"))
            if not p.endswith(".meta.json") and not p.endswith(".partial")
        )
        direct = os.path.join(path, f"{target}.json")
        if os.path.isfile(direct):
            candidates = [direct] + candidates
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise ValueError(
                f"Multiple result files for target={target} under {path}. "
                f"Use an explicit template. First candidates: {candidates[:5]}"
            )
        raise FileNotFoundError(f"No result file found for target={target} under directory: {path}")
    candidates = sorted(
        p
        for p in glob.glob(path)
        if os.path.isfile(p) and not p.endswith(".meta.json") and not p.endswith(".partial")
    )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No result file found for target={target}: {root_or_template}")
    raise ValueError(f"Multiple files found for target={target}: {candidates[:5]}")


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_modality(value: object, target: str) -> str:
    modality = str(value or "no").strip().lower()
    if modality not in MODALITIES:
        modality = "no"
    if target not in {"webqa", "visual_rag"} and modality == "image":
        # eval.py maps image to document for non-visual targets.
        modality = "document"
    return modality


def is_text_equiv_target(target: str) -> bool:
    return str(target or "").strip().lower() in TEXT_EQUIV_TARGETS


def cost_modality(modality: str, target: str, text_collapsed: bool = True) -> str:
    if text_collapsed and is_text_equiv_target(target) and modality in TEXT_ACTIONS:
        return "paragraph"
    return modality


def modality_cost(
    modality: str,
    target: str,
    costs: Dict[str, float],
    text_collapsed: bool = True,
) -> float:
    return costs[cost_modality(modality, target, text_collapsed=text_collapsed)]

def executed_modalities(row: dict, target: str) -> List[str]:
    no_bayes_modalities = row.get("retrieval_no_bayes_top_modalities")
    if isinstance(no_bayes_modalities, list) and no_bayes_modalities:
        modalities = [normalize_modality(x, target) for x in no_bayes_modalities]
    else:
        soft_modalities = row.get("retrieval_bayes_soft_modalities")
        if isinstance(soft_modalities, list) and soft_modalities:
            modalities = [normalize_modality(x, target) for x in soft_modalities]
        elif row.get("retrieval_bayes"):
            modalities = [normalize_modality(row.get("retrieval_bayes"), target)]
        else:
            modalities = [normalize_modality(row.get("retrieval"), target)]

    out = []
    for modality in modalities:
        if modality not in out:
            out.append(modality)
    return out or ["no"]


def posterior_weighted_cost(
    row: dict,
    target: str,
    costs: Dict[str, float],
    text_collapsed: bool = True,
) -> float:
    no_bayes_modalities = row.get("retrieval_no_bayes_top_modalities")
    no_bayes_weights = row.get("retrieval_no_bayes_top_weights")
    if (
        isinstance(no_bayes_modalities, list)
        and isinstance(no_bayes_weights, list)
        and len(no_bayes_modalities) == len(no_bayes_weights)
    ):
        total = 0.0
        for modality, weight in zip(no_bayes_modalities, no_bayes_weights):
            try:
                w = float(weight)
            except Exception:
                w = 0.0
            normalized = normalize_modality(modality, target)
            total += w * modality_cost(normalized, target, costs, text_collapsed=text_collapsed)
        return total

    soft_modalities = row.get("retrieval_bayes_soft_modalities")
    soft_weights = row.get("retrieval_bayes_soft_weights")
    if (
        isinstance(soft_modalities, list)
        and isinstance(soft_weights, list)
        and len(soft_modalities) == len(soft_weights)
    ):
        total = 0.0
        for modality, weight in zip(soft_modalities, soft_weights):
            try:
                w = float(weight)
            except Exception:
                w = 0.0
            normalized = normalize_modality(modality, target)
            total += w * modality_cost(normalized, target, costs, text_collapsed=text_collapsed)
        return total

    if row.get("retrieval_bayes"):
        normalized = normalize_modality(row.get("retrieval_bayes"), target)
        return modality_cost(normalized, target, costs, text_collapsed=text_collapsed)

    mods = executed_modalities(row, target)
    return sum(modality_cost(m, target, costs, text_collapsed=text_collapsed) for m in mods)


def cost_stats(rows: Sequence[dict], target: str, costs: Dict[str, float], expensive_threshold: float) -> Dict[str, float]:
    accounting = "text-collapsed" if is_text_equiv_target(target) else "raw"
    if not rows:
        return {
            "avg_exec_cost": float("nan"),
            "avg_exec_cost_raw": float("nan"),
            "avg_posterior_cost": float("nan"),
            "avg_posterior_cost_raw": float("nan"),
            "exec_cost_overhead": float("nan"),
            "posterior_cost_overhead": float("nan"),
            "cost_accounting": accounting,
            "retrieval_rate": float("nan"),
            "multi_path_rate": float("nan"),
            "expensive_rate": float("nan"),
            "expensive_rate_raw": float("nan"),
            "avg_branches": float("nan"),
            "no_rate": float("nan"),
            "paragraph_rate": float("nan"),
            "document_rate": float("nan"),
            "image_rate": float("nan"),
        }

    exec_costs = []
    exec_costs_raw = []
    weighted_costs = []
    weighted_costs_raw = []
    retrieval_flags = []
    multi_flags = []
    expensive_flags = []
    expensive_flags_raw = []
    branches = []
    modality_counts = {m: 0 for m in MODALITIES}

    for row in rows:
        mods = executed_modalities(row, target)
        exec_costs.append(
            sum(modality_cost(m, target, costs, text_collapsed=True) for m in mods)
        )
        exec_costs_raw.append(
            sum(modality_cost(m, target, costs, text_collapsed=False) for m in mods)
        )
        weighted_costs.append(posterior_weighted_cost(row, target, costs, text_collapsed=True))
        weighted_costs_raw.append(posterior_weighted_cost(row, target, costs, text_collapsed=False))
        retrieval_flags.append(any(m != "no" for m in mods))
        multi_flags.append(len(mods) > 1)
        expensive_flags.append(
            any(
                modality_cost(m, target, costs, text_collapsed=True) >= expensive_threshold
                for m in mods
            )
        )
        expensive_flags_raw.append(
            any(
                modality_cost(m, target, costs, text_collapsed=False) >= expensive_threshold
                for m in mods
            )
        )
        branches.append(len(mods))
        for modality in set(mods):
            modality_counts[modality] += 1

    n = len(rows)
    avg_exec_cost = sum(exec_costs) / n
    avg_exec_cost_raw = sum(exec_costs_raw) / n
    avg_posterior_cost = sum(weighted_costs) / n
    avg_posterior_cost_raw = sum(weighted_costs_raw) / n
    return {
        "avg_exec_cost": avg_exec_cost,
        "avg_exec_cost_raw": avg_exec_cost_raw,
        "avg_posterior_cost": avg_posterior_cost,
        "avg_posterior_cost_raw": avg_posterior_cost_raw,
        "exec_cost_overhead": avg_exec_cost_raw - avg_exec_cost,
        "posterior_cost_overhead": avg_posterior_cost_raw - avg_posterior_cost,
        "cost_accounting": accounting,
        "retrieval_rate": sum(retrieval_flags) / n,
        "multi_path_rate": sum(multi_flags) / n,
        "expensive_rate": sum(expensive_flags) / n,
        "expensive_rate_raw": sum(expensive_flags_raw) / n,
        "avg_branches": sum(branches) / n,
        "no_rate": modality_counts["no"] / n,
        "paragraph_rate": modality_counts["paragraph"] / n,
        "document_rate": modality_counts["document"] / n,
        "image_rate": modality_counts["image"] / n,
    }


def quality_value(target: str, metrics: Dict[str, object]) -> float:
    if target == "mmlu":
        return float(metrics.get("Accuracy", float("nan")))
    if target in {"squad", "natural_questions", "hotpotqa"}:
        return float(metrics.get("F1", float("nan")))
    if target == "webqa":
        return float(metrics.get("BERTScore", metrics.get("ROUGE-L", float("nan"))))
    return float("nan")


def fmt(value: object) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, (float, int)) and math.isfinite(float(value)):
        return f"{float(value):.4f}"
    return ""


def write_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = [
        "method",
        "target",
        "count",
        "score",
        "metric",
        "avg_exec_cost",
        "avg_exec_cost_raw",
        "avg_posterior_cost",
        "avg_posterior_cost_raw",
        "exec_cost_overhead",
        "posterior_cost_overhead",
        "cost_accounting",
        "retrieval_rate",
        "multi_path_rate",
        "expensive_rate",
        "expensive_rate_raw",
        "avg_branches",
        "no_rate",
        "paragraph_rate",
        "document_rate",
        "image_rate",
        "quality_per_exec_cost",
        "file",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def print_table(rows: Sequence[Dict[str, object]]) -> None:
    fields = [
        ("method", "Method"),
        ("target", "Target"),
        ("score", "Score"),
        ("metric", "Metric"),
        ("avg_exec_cost", "AvgExecCost"),
        ("avg_posterior_cost", "AvgPostCost"),
        ("retrieval_rate", "RetrRate"),
        ("multi_path_rate", "MultiRate"),
        ("expensive_rate", "ExpRate"),
        ("avg_branches", "Branches"),
    ]
    print("| " + " | ".join(title for _, title in fields) + " |")
    print("| " + " | ".join(["---"] * len(fields)) + " |")
    for row in rows:
        vals = []
        for key, _ in fields:
            vals.append(str(row.get(key, "")) if key in {"method", "target", "metric"} else fmt(row.get(key)))
        print("| " + " | ".join(vals) + " |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", action="append", required=True, help="NAME=RESULT_DIR")
    parser.add_argument("--targets", default=",".join(TARGETS))
    parser.add_argument(
        "--costs",
        default="no=0,paragraph=0.25,document=0.45,image=0.60",
        help="Either four comma-separated costs or key=value entries.",
    )
    parser.add_argument("--expensive_threshold", type=float, default=0.45)
    parser.add_argument("--output_csv", default="analysis/results/qa_cost_summary.csv")
    parser.add_argument(
        "--allow_missing",
        action="store_true",
        help="Skip method-target pairs whose result file is absent.",
    )
    args = parser.parse_args()

    targets = parse_list(args.targets, TARGETS)
    methods = [parse_method(x) for x in args.method]
    costs = parse_costs(args.costs)

    rows_out: List[Dict[str, object]] = []
    for method_name, default_path, overrides in methods:
        for target in targets:
            root = overrides.get(target, default_path)
            try:
                result_file = find_result_file(root, target)
            except FileNotFoundError:
                if args.allow_missing:
                    continue
                raise
            data = load_json(result_file)
            metrics = score_file(result_file, target=target)
            cstats = cost_stats(data, target, costs, args.expensive_threshold)

            if target == "mmlu":
                metric_name = "Accuracy"
            elif target in {"squad", "natural_questions", "hotpotqa"}:
                metric_name = "F1"
            elif target == "webqa":
                metric_name = "BERTScore"
            else:
                metric_name = "Score"

            score = quality_value(target, metrics)
            denom = cstats["avg_exec_cost"] if cstats["avg_exec_cost"] > 0 else 1.0
            rows_out.append(
                {
                    "method": method_name,
                    "target": target,
                    "count": metrics.get("Count", len(data)),
                    "score": score,
                    "metric": metric_name,
                    "quality_per_exec_cost": score / denom,
                    "file": result_file,
                    **cstats,
                }
            )

    write_csv(args.output_csv, rows_out)
    print_table(rows_out)
    print(f"\nSaved CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
