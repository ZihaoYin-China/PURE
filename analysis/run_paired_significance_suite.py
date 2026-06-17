#!/usr/bin/env python3
"""Run the paired significance comparisons used by the paper tables."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List

from paired_significance import TestResult, run_one_metric


QWEN = "qwen-api:qwen3.6-plus"


PATHS: Dict[str, str] = {
    # Main-table COVER rows.
    "cover_distilbert_mmlu": f"eval/results_cover_candidate_verifier_qwen36plus_indomain/{QWEN}/distilbert/mmlu_top1_0.2_1_bayes_cover_distilbert_candidate_verifier_softtop2_theta_posteriorverifier.json",
    "cover_distilbert_squad": f"eval/results_qwen36plus_api_all3_strict_d40_test_distilbert_corrected/{QWEN}/distilbert/squad_top1_0.2_1_bayes_all3_qwen36plus_api_distilbert_corrected_docstore_tau10_beta0p1_softtop2_theta_posteriorverifier.json",
    "cover_distilbert_nq": f"eval/results_qwen36plus_api_all3_strict_d40_test_distilbert_corrected/{QWEN}/distilbert/natural_questions_top1_0.2_1_bayes_all3_qwen36plus_api_distilbert_corrected_docstore_tau10_beta0p1_softtop2_theta_posteriorverifier.json",
    "cover_distilbert_hotpotqa": f"eval/results_webqa_hotpot_fair_refresh_20260528_cover/{QWEN}/distilbert/hotpotqa_top1_0.2_1_bayes_cover_distilbert_candidate_verifier_softtop2_theta_posteriorverifier_refresh.json",
    "cover_distilbert_webqa": f"eval/results_webqa_hotpot_fair_refresh_20260528_cover/{QWEN}/distilbert/webqa_top1_0.2_1_bayes_cover_distilbert_candidate_verifier_softtop2_theta_posteriorverifier_refresh.json",
    "cover_t5_mmlu": f"eval/results_cover_candidate_verifier_qwen36plus_indomain/{QWEN}/t5-large/mmlu_top1_0.2_1_bayes_cover_t5_candidate_verifier_softtop2_theta_posteriorverifier.json",
    "cover_t5_squad": f"eval/results_qwen36plus_api_all3_strict_d40_test_t5_corrected/{QWEN}/t5-large/squad_top1_0.2_1_bayes_all3_qwen36plus_api_t5_corrected_docstore_tau10_beta0p1_softtop2_theta_posteriorverifier.json",
    "cover_t5_nq": f"eval/results_qwen36plus_api_all3_strict_d40_test_t5_corrected/{QWEN}/t5-large/natural_questions_top1_0.2_1_bayes_all3_qwen36plus_api_t5_corrected_docstore_tau10_beta0p1_softtop2_theta_posteriorverifier.json",
    "cover_t5_hotpotqa": f"eval/results_webqa_hotpot_fair_refresh_20260528_cover/{QWEN}/t5-large/hotpotqa_top1_0.2_1_bayes_cover_t5_candidate_verifier_softtop2_theta_posteriorverifier_refresh.json",
    "cover_t5_webqa": f"eval/results_webqa_hotpot_fair_refresh_20260528_cover/{QWEN}/t5-large/webqa_top1_0.2_1_bayes_cover_t5_candidate_verifier_softtop2_theta_posteriorverifier_refresh.json",
    # Strongest non-COVER baseline in the current main table for each primary metric.
    "best_main_mmlu": f"eval/results_qwen36plus_api_compare_hard_router/{QWEN}/distilbert/mmlu_top1_0.2_1.json",
    "best_main_squad": f"eval/results_qwen36plus_api_adaptive_self_large_corrected/{QWEN}/crag/squad_top1_0.2_1.json",
    "best_main_nq": f"eval/results_ablation_classifier_verifier_no_bayes_corrected/{QWEN}/t5-large/natural_questions_top1_0.2_1_classifier_verifier_no_bayes_top2.json",
    "best_main_hotpotqa": f"eval/results_webqa_hotpot_fair_refresh_20260528_adaptive_self/{QWEN}/adaptive_rag/hotpotqa_top1_0.2_1.json",
    "best_main_webqa": f"eval/results_hard_top2_no_verifier_qwen36plus_20260603/{QWEN}/t5-large/webqa_top1_0.2_1_hard_top2_no_verifier.json",
    # SQuAD/NQ ablation rows.
    "hard_distilbert_squad": f"eval/results_qwen36plus_api_compare_hard_router_corrected/{QWEN}/distilbert/squad_top1_0.2_1.json",
    "hard_distilbert_nq": f"eval/results_qwen36plus_api_compare_hard_router_corrected/{QWEN}/distilbert/natural_questions_top1_0.2_1.json",
    "hard_t5_squad": f"eval/results_qwen36plus_api_compare_hard_router_corrected/{QWEN}/t5-large/squad_top1_0.2_1.json",
    "hard_t5_nq": f"eval/results_qwen36plus_api_compare_hard_router_corrected/{QWEN}/t5-large/natural_questions_top1_0.2_1.json",
    "vib_only_distilbert_squad": f"eval/results_ablation_vib_only_from_fixed_corrected/{QWEN}/distilbert/squad_top1_0.2_1_vib_only_top1.json",
    "vib_only_distilbert_nq": f"eval/results_ablation_vib_only_from_fixed_corrected/{QWEN}/distilbert/natural_questions_top1_0.2_1_vib_only_top1.json",
    "vib_only_t5_squad": f"eval/results_ablation_vib_only_from_fixed_corrected/{QWEN}/t5-large/squad_top1_0.2_1_vib_only_top1.json",
    "vib_only_t5_nq": f"eval/results_ablation_vib_only_from_fixed_corrected/{QWEN}/t5-large/natural_questions_top1_0.2_1_vib_only_top1.json",
    "vib_ver_distilbert_squad": f"eval/results_ablation_vib_verifier_no_bayes_corrected/{QWEN}/distilbert/squad_top1_0.2_1_vib_verifier_no_bayes_top2.json",
    "vib_ver_distilbert_nq": f"eval/results_ablation_vib_verifier_no_bayes_corrected/{QWEN}/distilbert/natural_questions_top1_0.2_1_vib_verifier_no_bayes_top2.json",
    "vib_ver_t5_squad": f"eval/results_ablation_vib_verifier_no_bayes_corrected/{QWEN}/t5-large/squad_top1_0.2_1_vib_verifier_no_bayes_top2.json",
    "vib_ver_t5_nq": f"eval/results_ablation_vib_verifier_no_bayes_corrected/{QWEN}/t5-large/natural_questions_top1_0.2_1_vib_verifier_no_bayes_top2.json",
    "hard_top2_ver_distilbert_squad": f"eval/results_ablation_classifier_verifier_no_bayes_corrected/{QWEN}/distilbert/squad_top1_0.2_1_classifier_verifier_no_bayes_top2.json",
    "hard_top2_ver_distilbert_nq": f"eval/results_ablation_classifier_verifier_no_bayes_corrected/{QWEN}/distilbert/natural_questions_top1_0.2_1_classifier_verifier_no_bayes_top2.json",
    "hard_top2_ver_t5_squad": f"eval/results_ablation_classifier_verifier_no_bayes_corrected/{QWEN}/t5-large/squad_top1_0.2_1_classifier_verifier_no_bayes_top2.json",
    "hard_top2_ver_t5_nq": f"eval/results_ablation_classifier_verifier_no_bayes_corrected/{QWEN}/t5-large/natural_questions_top1_0.2_1_classifier_verifier_no_bayes_top2.json",
    # MMLU ablation rows.
    "hard_distilbert_mmlu": f"eval/results_qwen36plus_api_compare_hard_router/{QWEN}/distilbert/mmlu_top1_0.2_1.json",
    "hard_t5_mmlu": f"eval/results_qwen36plus_api_compare_hard_router/{QWEN}/t5-large/mmlu_top1_0.2_1.json",
    "vib_only_distilbert_mmlu": f"eval/results_ablation_vib_only_from_fixed/{QWEN}/distilbert/mmlu_top1_0.2_1_vib_only_top1.json",
    "vib_only_t5_mmlu": f"eval/results_ablation_vib_only_from_fixed/{QWEN}/t5-large/mmlu_top1_0.2_1_vib_only_top1.json",
    "vib_ver_distilbert_mmlu": f"eval/results_ablation_vib_verifier_no_bayes/{QWEN}/distilbert/mmlu_top1_0.2_1_vib_verifier_no_bayes_top2.json",
    "vib_ver_t5_mmlu": f"eval/results_ablation_vib_verifier_no_bayes/{QWEN}/t5-large/mmlu_top1_0.2_1_vib_verifier_no_bayes_top2.json",
    "hard_top2_ver_distilbert_mmlu": f"eval/results_ablation_classifier_verifier_no_bayes/{QWEN}/distilbert/mmlu_top1_0.2_1_classifier_verifier_no_bayes_top2.json",
    "hard_top2_ver_t5_mmlu": f"eval/results_ablation_classifier_verifier_no_bayes/{QWEN}/t5-large/mmlu_top1_0.2_1_classifier_verifier_no_bayes_top2.json",
}


for backbone in ("distilbert", "t5"):
    table_name = "distilbert" if backbone == "distilbert" else "t5-large"
    prefix = "distilbert" if backbone == "distilbert" else "t5large"
    for target in ("hotpotqa", "webqa"):
        PATHS[f"hard_{backbone}_{target}"] = (
            f"eval/results_webqa_hotpot_fair_refresh_20260528_hard/{QWEN}/{table_name}/{target}_top1_0.2_1.json"
        )
        PATHS[f"vib_only_{backbone}_{target}"] = (
            f"eval/results_webqa_hotpot_fair_refresh_20260528_vib_only/{QWEN}/{table_name}/{target}_top1_0.2_1.json"
        )
        PATHS[f"vib_ver_{backbone}_{target}"] = (
            f"eval/results_webqa_hotpot_fair_refresh_20260528_no_bayes_verifier/{QWEN}/{table_name}/{target}_top1_0.2_1_vib_{prefix}_verifier_no_bayes_top2_refresh.json"
        )
        PATHS[f"hard_top2_ver_{backbone}_{target}"] = (
            f"eval/results_webqa_hotpot_fair_refresh_20260528_no_bayes_verifier/{QWEN}/{table_name}/{target}_top1_0.2_1_classifier_{prefix}_verifier_no_bayes_top2_refresh.json"
        )


TARGET_METRIC = {
    "mmlu": "accuracy",
    "squad": "f1",
    "nq": "f1",
    "hotpotqa": "f1",
    "webqa": "rouge-l",
}

TARGET_NAME = {
    "mmlu": "mmlu",
    "squad": "squad",
    "nq": "natural_questions",
    "hotpotqa": "hotpotqa",
    "webqa": "webqa",
}


def main_comparisons() -> List[Dict[str, str]]:
    out = []
    for backbone in ("distilbert", "t5"):
        for target in ("mmlu", "squad", "nq", "hotpotqa", "webqa"):
            out.append(
                {
                    "suite": "main",
                    "comparison": f"COVER-{backbone} vs strongest main-table baseline",
                    "baseline_label": "strongest_main_baseline",
                    "candidate_label": f"COVER-{backbone}",
                    "target_key": target,
                    "metric": TARGET_METRIC[target],
                    "baseline": PATHS[f"best_main_{target}"],
                    "candidate": PATHS[f"cover_{backbone}_{target}"],
                }
            )
    return out


def ablation_comparisons() -> List[Dict[str, str]]:
    labels = {
        "hard": "Hard routing",
        "vib_only": "VIB-only",
        "vib_ver": "VIB+Ver. (no Bayes)",
        "hard_top2_ver": "Hard top-2+Ver.",
    }
    out = []
    for backbone in ("distilbert", "t5"):
        for target in ("mmlu", "squad", "nq", "hotpotqa", "webqa"):
            for variant, label in labels.items():
                out.append(
                    {
                        "suite": "ablation",
                        "comparison": f"COVER-{backbone} vs {label}",
                        "baseline_label": label,
                        "candidate_label": f"COVER-{backbone}",
                        "target_key": target,
                        "metric": TARGET_METRIC[target],
                        "baseline": PATHS[f"{variant}_{backbone}_{target}"],
                        "candidate": PATHS[f"cover_{backbone}_{target}"],
                    }
                )
    return out


def row_from_result(spec: Dict[str, str], result: TestResult) -> Dict[str, str]:
    row = asdict(result)
    row.update(
        {
            "suite": spec["suite"],
            "comparison": spec["comparison"],
            "baseline_label": spec["baseline_label"],
            "candidate_label": spec["candidate_label"],
            "target_key": spec["target_key"],
        }
    )
    return row


def write_rows(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "suite",
        "comparison",
        "target",
        "metric",
        "baseline_label",
        "candidate_label",
        "n_tested",
        "baseline_mean",
        "candidate_mean",
        "diff",
        "ci_low",
        "ci_high",
        "p_value",
        "stars",
        "alternative",
        "permutation_samples",
        "bootstrap_samples",
        "baseline_file",
        "candidate_file",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_specs(
    specs: Iterable[Dict[str, str]],
    permutation_samples: int,
    bootstrap_samples: int,
    ci_level: float,
    seed: int,
) -> List[Dict[str, str]]:
    rows = []
    for idx, spec in enumerate(specs):
        result = run_one_metric(
            baseline_file=spec["baseline"],
            candidate_file=spec["candidate"],
            target=TARGET_NAME[spec["target_key"]],
            metric=spec["metric"],
            alternative="greater",
            permutation_samples=permutation_samples,
            bootstrap_samples=bootstrap_samples,
            ci_level=ci_level,
            seed=seed + idx * 997,
        )
        rows.append(row_from_result(spec, result))
        print(
            "{suite}: {comparison} | {target} {metric} | delta={delta:.2f} p={p:.4g} {stars}".format(
                suite=spec["suite"],
                comparison=spec["comparison"],
                target=result.target,
                metric=result.metric,
                delta=result.diff * 100,
                p=result.p_value if result.p_value is not None else float("nan"),
                stars=result.stars,
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["main", "ablation", "all"], default="all")
    parser.add_argument("--permutation-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="analysis/results")
    args = parser.parse_args()

    suites = []
    if args.suite in {"main", "all"}:
        suites.append(("main", main_comparisons()))
    if args.suite in {"ablation", "all"}:
        suites.append(("ablation", ablation_comparisons()))

    for suite_name, specs in suites:
        rows = run_specs(
            specs,
            permutation_samples=args.permutation_samples,
            bootstrap_samples=args.bootstrap_samples,
            ci_level=args.ci_level,
            seed=args.seed,
        )
        write_rows(
            Path(args.output_dir) / f"paired_significance_{suite_name}_primary.csv",
            rows,
        )


if __name__ == "__main__":
    main()
