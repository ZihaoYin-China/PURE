#!/usr/bin/env python
# Summarize refreshed WebQA and HotpotQA runs for a fair comparison table.

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
EXTRA_SITE_PACKAGES = [
    Path("/opt/conda/envs/universalrag/lib/python3.10/site-packages"),
    Path("/opt/conda/lib/python3.10/site-packages"),
]
for site_path in reversed(EXTRA_SITE_PACKAGES):
    site = str(site_path)
    if site_path.is_dir() and site not in sys.path:
        sys.path.insert(0, site)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.summarize_qa_cost import DEFAULT_COSTS, cost_stats
from eval.score import extract_references, get_prediction_text, rouge_l_max_batch, score_file

TARGETS = ["hotpotqa", "webqa"]


@dataclass(frozen=True)
class MethodSpec:
    name: str
    path_func: Callable[[str, argparse.Namespace], Path]
    webqa_only: bool = False


def top1_name(target: str) -> str:
    return f"{target}_top1_0.2_1.json"


def tagged_name(target: str, tag: str) -> str:
    return f"{target}_top1_0.2_1_{tag}.json"


def cover_name(target: str, tag: str) -> str:
    return f"{target}_top1_0.2_1_bayes_{tag}.json"


def fixed_path(modality: str) -> Callable[[str, argparse.Namespace], Path]:
    def _path(target: str, args: argparse.Namespace) -> Path:
        return Path(f"{args.results_prefix}_fixed_{modality}") / args.model_name / "t5-large" / top1_name(target)

    return _path


def single_path(suffix: str, router: str) -> Callable[[str, argparse.Namespace], Path]:
    def _path(target: str, args: argparse.Namespace) -> Path:
        return Path(f"{args.results_prefix}_{suffix}") / args.model_name / router / top1_name(target)

    return _path


def verifier_path(router: str, tag: str) -> Callable[[str, argparse.Namespace], Path]:
    def _path(target: str, args: argparse.Namespace) -> Path:
        return Path(f"{args.results_prefix}_no_bayes_verifier") / args.model_name / router / tagged_name(target, tag)

    return _path


def cover_path(router: str, tag: str) -> Callable[[str, argparse.Namespace], Path]:
    def _path(target: str, args: argparse.Namespace) -> Path:
        return Path(f"{args.results_prefix}_cover") / args.model_name / router / cover_name(target, tag)

    return _path


def default_methods() -> List[MethodSpec]:
    return [
        MethodSpec("Naive", fixed_path("no")),
        MethodSpec("ParagraphRAG", fixed_path("paragraph")),
        MethodSpec("DocumentRAG", fixed_path("document")),
        MethodSpec("ImageRAG", fixed_path("image"), webqa_only=True),
        MethodSpec("Oracle Action", fixed_path("oracle")),
        MethodSpec("UniversalRAG-DistilBERT", single_path("universalrag", "distilbert")),
        MethodSpec("UniversalRAG-T5-large", single_path("universalrag", "t5-large")),
        MethodSpec("Hard-DistilBERT", single_path("hard", "distilbert")),
        MethodSpec("Hard-T5-large", single_path("hard", "t5-large")),
        MethodSpec("Adaptive-RAG", single_path("adaptive_self", "adaptive_rag")),
        MethodSpec("Self-RAG", single_path("adaptive_self", "selfrag")),
        MethodSpec("CRAG", single_path("adaptive_self", "crag")),
        MethodSpec("VIB-only-DistilBERT", single_path("vib_only", "distilbert")),
        MethodSpec("VIB-only-T5-large", single_path("vib_only", "t5-large")),
        MethodSpec(
            "Hard-DistilBERT + Ver.",
            verifier_path("distilbert", "classifier_distilbert_verifier_no_bayes_top2_refresh"),
        ),
        MethodSpec(
            "Hard-T5-large + Ver.",
            verifier_path("t5-large", "classifier_t5large_verifier_no_bayes_top2_refresh"),
        ),
        MethodSpec(
            "VIB-DistilBERT + Ver.",
            verifier_path("distilbert", "vib_distilbert_verifier_no_bayes_top2_refresh"),
        ),
        MethodSpec(
            "VIB-T5-large + Ver.",
            verifier_path("t5-large", "vib_t5large_verifier_no_bayes_top2_refresh"),
        ),
        MethodSpec(
            "COVER-DistilBERT",
            cover_path("distilbert", "cover_distilbert_candidate_verifier_softtop2_theta_posteriorverifier_refresh"),
        ),
        MethodSpec(
            "COVER-T5-large",
            cover_path("t5-large", "cover_t5_candidate_verifier_softtop2_theta_posteriorverifier_refresh"),
        ),
    ]


def load_json(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}")
    return data


def score_webqa_rouge_only(path: Path) -> Dict[str, object]:
    data = load_json(path)
    preds = []
    refs_list = []
    for item in data:
        refs = extract_references(item, "webqa")
        if not refs:
            continue
        preds.append(get_prediction_text(item))
        refs_list.append(refs)
    if not preds:
        raise ValueError(f"No valid WebQA references in {path}")
    rouge_scores = rouge_l_max_batch(preds, refs_list)
    return {
        "target": "webqa",
        "file": str(path),
        "ROUGE-L": round(sum(rouge_scores) / len(rouge_scores) * 100.0, 2),
        "BERTScore": None,
        "Count": len(preds),
    }


def score_result(path: Path, target: str, quick_rouge_only: bool) -> Dict[str, object]:
    if target == "webqa" and quick_rouge_only:
        return score_webqa_rouge_only(path)
    try:
        return score_file(str(path), target=target)
    except ModuleNotFoundError as exc:
        if target == "webqa":
            print(
                f"[WARN] WebQA BERTScore dependency is missing ({exc}); "
                "falling back to ROUGE-L only for this file.",
                file=sys.stderr,
            )
            return score_webqa_rouge_only(path)
        raise


def metric_value(metrics: Optional[Dict[str, object]], key: str) -> Optional[float]:
    if not metrics:
        return None
    value = metrics.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: Optional[float]) -> str:
    if value is None:
        return "--"
    if isinstance(value, float) and not math.isfinite(value):
        return "--"
    return f"{float(value):.2f}"


def fmt_cost(value: Optional[float]) -> str:
    if value is None:
        return "--"
    if isinstance(value, float) and not math.isfinite(value):
        return "--"
    return f"{float(value):.3f}"


def tex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


def summarize(args: argparse.Namespace) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    costs = dict(DEFAULT_COSTS)
    for spec in default_methods():
        for target in TARGETS:
            if spec.webqa_only and target != "webqa":
                continue
            path = spec.path_func(target, args)
            if not path.is_file():
                if args.allow_missing:
                    rows.append({
                        "method": spec.name,
                        "target": target,
                        "file": str(path),
                        "missing": True,
                    })
                    continue
                raise FileNotFoundError(str(path))
            metrics = score_result(path, target, args.quick_rouge_only)
            data = load_json(path)
            cstats = cost_stats(data, target, costs, args.expensive_threshold)
            row = {
                "method": spec.name,
                "target": target,
                "file": str(path),
                "missing": False,
                "count": metrics.get("Count", len(data)),
                "em": metric_value(metrics, "EM"),
                "f1": metric_value(metrics, "F1"),
                "rouge_l": metric_value(metrics, "ROUGE-L"),
                "bertscore": metric_value(metrics, "BERTScore"),
            }
            row.update(cstats)
            rows.append(row)
    return rows


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "method",
        "target",
        "missing",
        "count",
        "em",
        "f1",
        "rouge_l",
        "bertscore",
        "avg_exec_cost",
        "avg_posterior_cost",
        "retrieval_rate",
        "multi_path_rate",
        "no_rate",
        "paragraph_rate",
        "document_rate",
        "image_rate",
        "file",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def index_rows(rows: List[Dict[str, object]]) -> Dict[str, Dict[str, Dict[str, object]]]:
    out: Dict[str, Dict[str, Dict[str, object]]] = {}
    for row in rows:
        out.setdefault(str(row["method"]), {})[str(row["target"])] = row
    return out


def mean_cost(items: List[Optional[Dict[str, object]]]) -> Optional[float]:
    values = []
    for item in items:
        if not item or item.get("missing"):
            continue
        value = item.get("avg_exec_cost")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    if not values:
        return None
    return sum(values) / len(values)


def write_tex(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_method = index_rows(rows)
    lines = [
        "% Generated by analysis/summarize_webqa_hotpot_fair_refresh.py",
        "% Uses refreshed HotpotQA raw-context text features and WebQA BGE caption retrieval.",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Method & HotpotQA EM & HotpotQA F1 & WebQA R-L & WebQA BERT & Avg. Cost \\\\",
        "\\midrule",
    ]
    for spec in default_methods():
        targets = by_method.get(spec.name, {})
        hotpot = targets.get("hotpotqa")
        webqa = targets.get("webqa")
        lines.append(
            "{} & {} & {} & {} & {} & {} \\\\".format(
                tex_escape(spec.name),
                fmt(metric_value(hotpot, "em")),
                fmt(metric_value(hotpot, "f1")),
                fmt(metric_value(webqa, "rouge_l")),
                fmt(metric_value(webqa, "bertscore")),
                fmt_cost(mean_cost([hotpot, webqa])),
            )
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def print_markdown(rows: List[Dict[str, object]]) -> None:
    by_method = index_rows(rows)
    print("| Method | Hotpot EM | Hotpot F1 | WebQA R-L | WebQA BERT | Avg Cost |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for spec in default_methods():
        targets = by_method.get(spec.name, {})
        hotpot = targets.get("hotpotqa")
        webqa = targets.get("webqa")
        print(
            "| {method} | {hem} | {hf1} | {wrl} | {wbert} | {cost} |".format(
                method=spec.name,
                hem=fmt(metric_value(hotpot, "em")),
                hf1=fmt(metric_value(hotpot, "f1")),
                wrl=fmt(metric_value(webqa, "rouge_l")),
                wbert=fmt(metric_value(webqa, "bertscore")),
                cost=fmt_cost(mean_cost([hotpot, webqa])),
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize refreshed WebQA and HotpotQA fair comparison runs.")
    parser.add_argument("--results_prefix", default="eval/results_webqa_hotpot_fair_refresh_20260528")
    parser.add_argument("--model_name", default="qwen-api:qwen3.6-plus")
    parser.add_argument("--output_csv", default="analysis/results/webqa_hotpot_fair_refresh_summary_20260528.csv")
    parser.add_argument("--output_tex", default="analysis/results/webqa_hotpot_fair_refresh_rows_20260528.tex")
    parser.add_argument("--allow_missing", action="store_true")
    parser.add_argument("--quick_rouge_only", action="store_true", help="Skip WebQA BERTScore for fast smoke checks.")
    parser.add_argument("--expensive_threshold", type=float, default=0.45)
    args = parser.parse_args()

    rows = summarize(args)
    write_csv(Path(args.output_csv), rows)
    write_tex(Path(args.output_tex), rows)
    print_markdown(rows)
    print(f"\nSaved CSV: {args.output_csv}")
    print(f"Saved TeX rows: {args.output_tex}")


if __name__ == "__main__":
    main()
