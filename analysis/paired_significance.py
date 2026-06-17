#!/usr/bin/env python3
"""Paired significance tests for PURE result JSON files."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCORE_PATH = ROOT / "eval" / "score.py"


def load_score_module():
    spec = importlib.util.spec_from_file_location("pure_score", SCORE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load scoring module from {SCORE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


score = load_score_module()


@dataclass
class TestResult:
    target: str
    metric: str
    baseline_file: str
    candidate_file: str
    n_aligned: int
    n_tested: int
    n_missing_candidate: int
    n_extra_candidate: int
    n_skipped_invalid_metric: int
    baseline_mean: float
    candidate_mean: float
    diff: float
    ci_level: float
    ci_low: Optional[float]
    ci_high: Optional[float]
    p_value: Optional[float]
    alternative: str
    permutation_samples: int
    bootstrap_samples: int
    stars: str


def canonical_metric(metric: str) -> str:
    value = metric.strip().lower().replace("_", "-")
    aliases = {
        "acc": "accuracy",
        "accuracy": "accuracy",
        "em": "em",
        "exact-match": "em",
        "exactmatch": "em",
        "f1": "f1",
        "rouge": "rouge-l",
        "rouge-l": "rouge-l",
        "rougel": "rouge-l",
        "bert": "bertscore",
        "bert-score": "bertscore",
        "bertscore": "bertscore",
    }
    if value not in aliases:
        raise ValueError(f"Unsupported metric: {metric}")
    return aliases[value]


def resolve_metrics(metrics: str, target: str) -> List[str]:
    if metrics.strip().lower() == "auto":
        if target in score.MC_TARGETS:
            return ["accuracy"]
        if target in score.SHORT_ANSWER_TARGETS:
            return ["em", "f1"]
        if target in score.LONG_ANSWER_TARGETS:
            return ["rouge-l"]
        raise ValueError(f"Unsupported target: {target}")
    return [canonical_metric(x) for x in metrics.split(",") if x.strip()]


def sample_key(item: Dict[str, Any]) -> Tuple[str, ...]:
    source = str(item.get("source", "")).strip()
    index = str(item.get("index", "")).strip()
    if index:
        return (source, index)
    return (
        source,
        str(item.get("question", "")).strip(),
        str(item.get("answer", item.get("gold", ""))).strip(),
    )


def unique_by_key(data: Sequence[Dict[str, Any]], path: str) -> Dict[Tuple[str, ...], Dict[str, Any]]:
    out: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    duplicates = []
    for item in data:
        key = sample_key(item)
        if key in out:
            duplicates.append(key)
        out[key] = item
    if duplicates:
        shown = ", ".join(str(x) for x in duplicates[:3])
        raise ValueError(f"{path} has duplicate sample keys, e.g. {shown}")
    return out


def load_and_align(
    baseline_file: str,
    candidate_file: str,
    target: Optional[str],
) -> Tuple[str, List[Tuple[Tuple[str, ...], Dict[str, Any], Dict[str, Any]]], int, int]:
    baseline = score.load_json(baseline_file)
    candidate = score.load_json(candidate_file)

    baseline_target = target or score.infer_target(baseline_file, baseline)
    candidate_target = target or score.infer_target(candidate_file, candidate)
    if baseline_target != candidate_target:
        raise ValueError(
            f"Target mismatch: baseline={baseline_target}, candidate={candidate_target}. "
            "Pass --target if the file names are ambiguous."
        )

    candidate_by_key = unique_by_key(candidate, candidate_file)
    baseline_keys = {sample_key(item) for item in baseline}

    aligned = []
    missing_candidate = 0
    for item in baseline:
        key = sample_key(item)
        other = candidate_by_key.get(key)
        if other is None:
            missing_candidate += 1
            continue
        aligned.append((key, item, other))

    extra_candidate = len(set(candidate_by_key) - baseline_keys)
    if not aligned:
        raise ValueError("No overlapping samples found between baseline and candidate.")
    return baseline_target, aligned, missing_candidate, extra_candidate


def item_score(item: Dict[str, Any], target: str, metric: str) -> Optional[float]:
    pred = score.get_prediction_text(item)

    if metric == "accuracy":
        gold = score.extract_mc_gold(item)
        if gold is None:
            return None
        return float(score.extract_choice_from_prediction(pred) == gold)

    refs = score.extract_references(item, target)
    if not refs:
        return None

    if metric == "em":
        return score.metric_max_over_ground_truths(score.exact_match_score, pred, refs)
    if metric == "f1":
        return score.metric_max_over_ground_truths(score.f1_score, pred, refs)

    raise ValueError(f"Metric {metric} requires batch scoring.")


def batch_scores(items: Sequence[Dict[str, Any]], target: str, metric: str) -> np.ndarray:
    if metric in {"accuracy", "em", "f1"}:
        values = [item_score(item, target, metric) for item in items]
        return np.asarray([np.nan if v is None else float(v) for v in values], dtype=float)

    preds = []
    refs_list = []
    valid_indices = []
    for idx, item in enumerate(items):
        refs = score.extract_references(item, target)
        if not refs:
            continue
        preds.append(score.get_prediction_text(item))
        refs_list.append(refs)
        valid_indices.append(idx)

    values = np.full(len(items), np.nan, dtype=float)
    if not valid_indices:
        return values

    if metric == "rouge-l":
        metric_values = score.rouge_l_max_batch(preds, refs_list)
    elif metric == "bertscore":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        metric_values = score.bertscore_max_batch(preds, refs_list, device=device)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    for idx, value in zip(valid_indices, metric_values):
        values[idx] = float(value)
    return values


def mean_with_bootstrap_ci(
    diffs: np.ndarray,
    samples: int,
    ci_level: float,
    seed: int,
    chunk_size: int = 512,
) -> Tuple[Optional[float], Optional[float]]:
    if samples <= 0:
        return None, None

    rng = np.random.default_rng(seed)
    n = len(diffs)
    means = []
    remaining = samples
    while remaining > 0:
        size = min(chunk_size, remaining)
        indices = rng.integers(0, n, size=(size, n))
        means.append(diffs[indices].mean(axis=1))
        remaining -= size

    boot = np.concatenate(means)
    tail = (1.0 - ci_level) / 2.0
    low, high = np.quantile(boot, [tail, 1.0 - tail])
    return float(low), float(high)


def permutation_p_value(
    diffs: np.ndarray,
    samples: int,
    alternative: str,
    seed: int,
    chunk_size: int = 512,
) -> Optional[float]:
    if samples <= 0:
        return None

    rng = np.random.default_rng(seed)
    observed = float(np.mean(diffs))
    count = 0
    remaining = samples
    while remaining > 0:
        size = min(chunk_size, remaining)
        signs = rng.integers(0, 2, size=(size, len(diffs)), dtype=np.int8) * 2 - 1
        permuted = (signs * diffs).mean(axis=1)
        if alternative == "greater":
            count += int(np.sum(permuted >= observed))
        elif alternative == "less":
            count += int(np.sum(permuted <= observed))
        else:
            count += int(np.sum(np.abs(permuted) >= abs(observed)))
        remaining -= size

    return float((count + 1) / (samples + 1))


def significance_stars(p_value: Optional[float]) -> str:
    if p_value is None or math.isnan(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def run_one_metric(
    baseline_file: str,
    candidate_file: str,
    target: Optional[str],
    metric: str,
    alternative: str,
    permutation_samples: int,
    bootstrap_samples: int,
    ci_level: float,
    seed: int,
) -> TestResult:
    resolved_target, aligned, missing_candidate, extra_candidate = load_and_align(
        baseline_file, candidate_file, target
    )
    metric = canonical_metric(metric)

    baseline_items = [x[1] for x in aligned]
    candidate_items = [x[2] for x in aligned]
    baseline_scores = batch_scores(baseline_items, resolved_target, metric)
    candidate_scores = batch_scores(candidate_items, resolved_target, metric)

    valid = ~(np.isnan(baseline_scores) | np.isnan(candidate_scores))
    if not np.any(valid):
        raise ValueError(f"No valid paired samples for metric={metric}")

    baseline_scores = baseline_scores[valid]
    candidate_scores = candidate_scores[valid]
    diffs = candidate_scores - baseline_scores

    ci_low, ci_high = mean_with_bootstrap_ci(
        diffs, bootstrap_samples, ci_level, seed=seed + 17
    )
    p_value = permutation_p_value(
        diffs, permutation_samples, alternative, seed=seed + 29
    )

    return TestResult(
        target=resolved_target,
        metric=metric,
        baseline_file=baseline_file,
        candidate_file=candidate_file,
        n_aligned=len(aligned),
        n_tested=int(len(diffs)),
        n_missing_candidate=missing_candidate,
        n_extra_candidate=extra_candidate,
        n_skipped_invalid_metric=int(len(aligned) - len(diffs)),
        baseline_mean=float(np.mean(baseline_scores)),
        candidate_mean=float(np.mean(candidate_scores)),
        diff=float(np.mean(diffs)),
        ci_level=ci_level,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        alternative=alternative,
        permutation_samples=permutation_samples,
        bootstrap_samples=bootstrap_samples,
        stars=significance_stars(p_value),
    )


def pct(value: Optional[float]) -> str:
    if value is None or math.isnan(value):
        return "NA"
    return f"{value * 100:.2f}"


def p_text(value: Optional[float]) -> str:
    if value is None or math.isnan(value):
        return "NA"
    if value < 0.0001:
        return "<0.0001"
    return f"{value:.4f}"


def render_text(results: Sequence[TestResult]) -> str:
    lines = []
    for result in results:
        lines.extend(
            [
                "=" * 72,
                f"Target    : {result.target}",
                f"Metric    : {result.metric}",
                f"Baseline  : {result.baseline_file}",
                f"Candidate : {result.candidate_file}",
                (
                    f"Samples   : aligned={result.n_aligned}, tested={result.n_tested}, "
                    f"skipped={result.n_skipped_invalid_metric}, "
                    f"missing_candidate={result.n_missing_candidate}, "
                    f"extra_candidate={result.n_extra_candidate}"
                ),
                f"Baseline  : {pct(result.baseline_mean)}",
                f"Candidate : {pct(result.candidate_mean)}",
                f"Delta     : {pct(result.diff)}",
                (
                    f"CI        : {int(result.ci_level * 100)}% bootstrap "
                    f"[{pct(result.ci_low)}, {pct(result.ci_high)}]"
                ),
                (
                    f"p-value   : {p_text(result.p_value)} "
                    f"({result.alternative}, paired permutation, "
                    f"{result.permutation_samples} samples) {result.stars}"
                ),
            ]
        )
    lines.append("=" * 72)
    return "\n".join(lines)


def render_latex(results: Sequence[TestResult]) -> str:
    rows = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Metric & Baseline & Candidate & $\Delta$ & $p$ & Sig. \\",
        r"\midrule",
    ]
    for result in results:
        rows.append(
            (
                f"{result.metric} & {pct(result.baseline_mean)} & "
                f"{pct(result.candidate_mean)} & {pct(result.diff)} & "
                f"{p_text(result.p_value)} & {result.stars} " + r"\\"
            )
        )
    rows.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(rows)


def write_csv(path: Path, results: Sequence[TestResult]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run paired bootstrap confidence intervals and paired permutation "
            "significance tests over two PURE result JSON files."
        )
    )
    parser.add_argument("--baseline", required=True, help="Baseline result JSON file.")
    parser.add_argument("--candidate", required=True, help="Candidate result JSON file.")
    parser.add_argument(
        "--target",
        default=None,
        choices=sorted(
            score.MC_TARGETS | score.SHORT_ANSWER_TARGETS | score.LONG_ANSWER_TARGETS
        ),
        help="Optional target override. By default it is inferred from the file.",
    )
    parser.add_argument(
        "--metrics",
        default="auto",
        help=(
            "Comma-separated metrics: accuracy,em,f1,rouge-l,bertscore. "
            "auto uses accuracy for MC, em/f1 for short answers, and rouge-l "
            "for long answers."
        ),
    )
    parser.add_argument(
        "--alternative",
        default="two-sided",
        choices=["two-sided", "greater", "less"],
        help="Permutation-test alternative for candidate - baseline.",
    )
    parser.add_argument("--permutation-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--format",
        default="text",
        choices=["text", "json", "csv", "latex"],
        help="Output format.",
    )
    parser.add_argument("--output", default=None, help="Optional output file.")
    args = parser.parse_args()

    target, _, _, _ = load_and_align(args.baseline, args.candidate, args.target)
    metrics = resolve_metrics(args.metrics, target)

    results = [
        run_one_metric(
            baseline_file=args.baseline,
            candidate_file=args.candidate,
            target=args.target,
            metric=metric,
            alternative=args.alternative,
            permutation_samples=args.permutation_samples,
            bootstrap_samples=args.bootstrap_samples,
            ci_level=args.ci_level,
            seed=args.seed + idx * 1009,
        )
        for idx, metric in enumerate(metrics)
    ]

    output_path = Path(args.output) if args.output else None
    if args.format == "json":
        payload = json.dumps([asdict(x) for x in results], indent=2)
        if output_path:
            output_path.write_text(payload + "\n", encoding="utf-8")
        else:
            print(payload)
    elif args.format == "csv":
        if output_path:
            write_csv(output_path, results)
        else:
            fieldnames = list(asdict(results[0]).keys())
            print(",".join(fieldnames))
            for result in results:
                print(",".join(str(asdict(result)[field]) for field in fieldnames))
    elif args.format == "latex":
        payload = render_latex(results)
        if output_path:
            output_path.write_text(payload + "\n", encoding="utf-8")
        else:
            print(payload)
    else:
        payload = render_text(results)
        if output_path:
            output_path.write_text(payload + "\n", encoding="utf-8")
        else:
            print(payload)


if __name__ == "__main__":
    main()
