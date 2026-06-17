#!/usr/bin/env python3
"""Routing-to-answer coupling diagnostics for the COVER in-domain runs.

The script computes per-dataset correlations between routing signals and
per-example answer utility, then Fisher-z averages the dataset-level
correlations. Binary routing signals use point-biserial correlation; continuous
uncertainty uses Spearman correlation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
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


RUN_FILES = {
    "COVER-DistilBERT": {
        "mmlu": "eval/results_cover_candidate_verifier_qwen36plus_indomain/qwen-api:qwen3.6-plus/distilbert/mmlu_top1_0.2_1_bayes_cover_distilbert_candidate_verifier_softtop2_theta_posteriorverifier.json",
        "squad": "eval/results_qwen36plus_api_all3_strict_d40_test_distilbert_corrected/qwen-api:qwen3.6-plus/distilbert/squad_top1_0.2_1_bayes_all3_qwen36plus_api_distilbert_corrected_docstore_tau10_beta0p1_softtop2_theta_posteriorverifier.json",
        "natural_questions": "eval/results_qwen36plus_api_all3_strict_d40_test_distilbert_corrected/qwen-api:qwen3.6-plus/distilbert/natural_questions_top1_0.2_1_bayes_all3_qwen36plus_api_distilbert_corrected_docstore_tau10_beta0p1_softtop2_theta_posteriorverifier.json",
        "hotpotqa": "eval/results_webqa_hotpot_fair_refresh_20260528_cover/qwen-api:qwen3.6-plus/distilbert/hotpotqa_top1_0.2_1_bayes_cover_distilbert_candidate_verifier_softtop2_theta_posteriorverifier_refresh.json",
        "webqa": "eval/results_webqa_hotpot_fair_refresh_20260528_cover/qwen-api:qwen3.6-plus/distilbert/webqa_top1_0.2_1_bayes_cover_distilbert_candidate_verifier_softtop2_theta_posteriorverifier_refresh.json",
    },
    "COVER-T5-large": {
        "mmlu": "eval/results_cover_candidate_verifier_qwen36plus_indomain/qwen-api:qwen3.6-plus/t5-large/mmlu_top1_0.2_1_bayes_cover_t5_candidate_verifier_softtop2_theta_posteriorverifier.json",
        "squad": "eval/results_qwen36plus_api_all3_strict_d40_test_t5_corrected/qwen-api:qwen3.6-plus/t5-large/squad_top1_0.2_1_bayes_all3_qwen36plus_api_t5_corrected_docstore_tau10_beta0p1_softtop2_theta_posteriorverifier.json",
        "natural_questions": "eval/results_qwen36plus_api_all3_strict_d40_test_t5_corrected/qwen-api:qwen3.6-plus/t5-large/natural_questions_top1_0.2_1_bayes_all3_qwen36plus_api_t5_corrected_docstore_tau10_beta0p1_softtop2_theta_posteriorverifier.json",
        "hotpotqa": "eval/results_webqa_hotpot_fair_refresh_20260528_cover/qwen-api:qwen3.6-plus/t5-large/hotpotqa_top1_0.2_1_bayes_cover_t5_candidate_verifier_softtop2_theta_posteriorverifier_refresh.json",
        "webqa": "eval/results_webqa_hotpot_fair_refresh_20260528_cover/qwen-api:qwen3.6-plus/t5-large/webqa_top1_0.2_1_bayes_cover_t5_candidate_verifier_softtop2_theta_posteriorverifier_refresh.json",
    },
}


TEXT_EQUIV_TARGETS = {"squad", "nq", "natural_questions"}
TEXT_ACTIONS = {"paragraph", "document"}


def load_json(path: str) -> List[Dict[str, Any]]:
    with (ROOT / path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}")
    return data


def action_match(target: str, predicted: Any, gold: Any) -> bool:
    pred = str(predicted or "").strip().lower()
    ref = str(gold or "").strip().lower()
    if target in TEXT_EQUIV_TARGETS and pred in TEXT_ACTIONS and ref in TEXT_ACTIONS:
        return True
    return pred == ref


def top_route(row: Dict[str, Any]) -> str:
    if row.get("retrieval_original"):
        return str(row["retrieval_original"])
    order = row.get("retrieval_probs_order") or ["no", "paragraph", "document", "image"]
    probs = row.get("retrieval_probs") or []
    if probs and len(probs) == len(order):
        return str(order[int(np.argmax(np.asarray(probs, dtype=float)))])
    return str(row.get("retrieval", ""))


def candidate_modalities(row: Dict[str, Any]) -> List[str]:
    modalities = row.get("retrieval_bayes_soft_modalities")
    if isinstance(modalities, list) and modalities:
        return [str(x) for x in modalities]
    candidates = row.get("retrieval_bayes_soft_candidates")
    if isinstance(candidates, list) and candidates:
        return [str(c.get("modality", "")) for c in candidates if isinstance(c, dict)]
    return [top_route(row)]


def selected_modality(row: Dict[str, Any]) -> str:
    modalities = candidate_modalities(row)
    idx = row.get("retrieval_bayes_posterior_generation_selected_index")
    if isinstance(idx, int) and 0 <= idx < len(modalities):
        return modalities[idx]
    return str(row.get("retrieval", top_route(row)))


def answer_utilities(items: Sequence[Dict[str, Any]], target: str) -> np.ndarray:
    if target in score.MC_TARGETS:
        values = []
        for item in items:
            pred = score.extract_choice_from_prediction(score.get_prediction_text(item))
            gold = score.extract_mc_gold(item)
            values.append(np.nan if gold is None else float(pred == gold))
        return np.asarray(values, dtype=float)

    if target in score.SHORT_ANSWER_TARGETS:
        values = []
        for item in items:
            pred = score.get_prediction_text(item)
            refs = score.extract_references(item, target)
            if not refs:
                values.append(np.nan)
            else:
                values.append(score.metric_max_over_ground_truths(score.f1_score, pred, refs))
        return np.asarray(values, dtype=float)

    preds: List[str] = []
    refs_list: List[List[str]] = []
    valid_indices: List[int] = []
    values = np.full(len(items), np.nan, dtype=float)
    for idx, item in enumerate(items):
        refs = score.extract_references(item, target)
        if not refs:
            continue
        preds.append(score.get_prediction_text(item))
        refs_list.append(refs)
        valid_indices.append(idx)
    rouge_values = score.rouge_l_max_batch(preds, refs_list)
    for idx, value in zip(valid_indices, rouge_values):
        values[idx] = float(value)
    return values


def brier_scores(items: Sequence[Dict[str, Any]]) -> np.ndarray:
    values = np.full(len(items), np.nan, dtype=float)
    for idx, row in enumerate(items):
        probs = row.get("retrieval_probs")
        order = row.get("retrieval_probs_order") or ["no", "paragraph", "document", "image"]
        gold = str(row.get("gt_retrieval", "")).strip().lower()
        if not isinstance(probs, list) or len(probs) != len(order):
            continue
        order = [str(x).strip().lower() for x in order]
        if gold not in order:
            continue
        p = np.asarray(probs, dtype=float)
        y = np.asarray([1.0 if action == gold else 0.0 for action in order], dtype=float)
        values[idx] = float(np.sum((p - y) ** 2))
    return values


def point_biserial(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(y)
    x = x[mask].astype(bool)
    y = y[mask].astype(float)
    if len(y) < 3 or len(np.unique(x)) < 2:
        return float("nan")
    y1 = y[x]
    y0 = y[~x]
    if len(y1) == 0 or len(y0) == 0:
        return float("nan")
    std = float(np.std(y, ddof=0))
    if std == 0.0:
        return float("nan")
    p = len(y1) / len(y)
    q = len(y0) / len(y)
    return float((np.mean(y1) - np.mean(y0)) / std * math.sqrt(p * q))


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0
        i = j
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask].astype(float)
    y = y[mask].astype(float)
    if len(y) < 3 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    rx = rankdata(x)
    ry = rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def binary_gap(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(y)
    x = x[mask].astype(bool)
    y = y[mask].astype(float)
    if len(np.unique(x)) < 2:
        return float("nan")
    return float(np.mean(y[x]) - np.mean(y[~x]))


def quartile_gap_low_minus_high(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask].astype(float)
    y = y[mask].astype(float)
    if len(y) < 4:
        return float("nan")
    lo = np.quantile(x, 0.25)
    hi = np.quantile(x, 0.75)
    low_y = y[x <= lo]
    high_y = y[x >= hi]
    if len(low_y) == 0 or len(high_y) == 0:
        return float("nan")
    return float(np.mean(low_y) - np.mean(high_y))


def fisher_average(rs: Iterable[float]) -> float:
    vals = [r for r in rs if math.isfinite(r) and abs(r) < 1.0]
    if not vals:
        return float("nan")
    zs = [math.atanh(r) for r in vals]
    return float(math.tanh(float(np.mean(zs))))


def mean_finite(xs: Iterable[float]) -> float:
    vals = [x for x in xs if math.isfinite(x)]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def fmt(value: float, scale: float = 1.0) -> str:
    if not math.isfinite(value):
        return "--"
    return f"{value * scale:.2f}"


def compute_run(run_name: str, files: Dict[str, str]) -> Dict[str, Dict[str, float]]:
    per_signal: Dict[str, Dict[str, List[float]]] = {
        "top1": {"corr": [], "gap": []},
        "sel": {"corr": [], "gap": []},
        "unc": {"corr": [], "gap": []},
        "brier": {"corr": [], "gap": []},
    }

    for target, path in files.items():
        items = load_json(path)
        utility = answer_utilities(items, target)

        route_correct = np.asarray(
            [action_match(target, top_route(row), row.get("gt_retrieval")) for row in items],
            dtype=bool,
        )
        selected_correct = np.asarray(
            [action_match(target, selected_modality(row), row.get("gt_retrieval")) for row in items],
            dtype=bool,
        )
        uncertainty = np.asarray(
            [
                float(row.get("retrieval_uncertainty"))
                if row.get("retrieval_uncertainty") is not None
                else float("nan")
                for row in items
            ],
            dtype=float,
        )
        brier = brier_scores(items)

        for key, signal in [
            ("top1", route_correct),
            ("sel", selected_correct),
        ]:
            per_signal[key]["corr"].append(point_biserial(signal, utility))
            per_signal[key]["gap"].append(binary_gap(signal, utility))

        per_signal["unc"]["corr"].append(spearman(uncertainty, utility))
        per_signal["unc"]["gap"].append(quartile_gap_low_minus_high(uncertainty, utility))
        per_signal["brier"]["corr"].append(spearman(brier, utility))
        per_signal["brier"]["gap"].append(quartile_gap_low_minus_high(brier, utility))

    return {
        key: {
            "corr": fisher_average(values["corr"]),
            "gap": mean_finite(values["gap"]),
        }
        for key, values in per_signal.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latex", action="store_true", help="Print compact LaTeX table rows.")
    args = parser.parse_args()

    results = {name: compute_run(name, files) for name, files in RUN_FILES.items()}

    labels = {
        "top1": "Top-1 route correct",
        "sel": "Verifier-selected route correct",
        "unc": "Evidential uncertainty",
        "brier": r"Router Brier score $\downarrow$",
    }

    if args.latex:
        for key, label in labels.items():
            d = results["COVER-DistilBERT"][key]
            t = results["COVER-T5-large"][key]
            print(
                f"{label} & {fmt(d['corr'])} & {fmt(d['gap'], 100.0)} "
                f"& {fmt(t['corr'])} & {fmt(t['gap'], 100.0)} \\\\"
            )
        return

    for run_name, run_results in results.items():
        print(run_name)
        for key, label in labels.items():
            values = run_results[key]
            print(
                f"  {label:32s} r={fmt(values['corr'])} "
                f"gap={fmt(values['gap'], 100.0)} utility points"
            )


if __name__ == "__main__":
    main()
