#!/usr/bin/env python3
"""Automatic failure-mode diagnostics for the primary COVER runs.

The analysis is intentionally conservative: it uses existing result caches,
gold route labels, candidate answers, selected branch indices, and known gold
evidence ids when available. The categories are diagnostic buckets rather than
manual annotations.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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


COVER_T5_FILES = {
    "mmlu": "eval/results_cover_candidate_verifier_qwen36plus_indomain/qwen-api:qwen3.6-plus/t5-large/mmlu_top1_0.2_1_bayes_cover_t5_candidate_verifier_softtop2_theta_posteriorverifier.json",
    "squad": "eval/results_qwen36plus_api_all3_strict_d40_test_t5_corrected/qwen-api:qwen3.6-plus/t5-large/squad_top1_0.2_1_bayes_all3_qwen36plus_api_t5_corrected_docstore_tau10_beta0p1_softtop2_theta_posteriorverifier.json",
    "natural_questions": "eval/results_qwen36plus_api_all3_strict_d40_test_t5_corrected/qwen-api:qwen3.6-plus/t5-large/natural_questions_top1_0.2_1_bayes_all3_qwen36plus_api_t5_corrected_docstore_tau10_beta0p1_softtop2_theta_posteriorverifier.json",
    "hotpotqa": "eval/results_webqa_hotpot_fair_refresh_20260528_cover/qwen-api:qwen3.6-plus/t5-large/hotpotqa_top1_0.2_1_bayes_cover_t5_candidate_verifier_softtop2_theta_posteriorverifier_refresh.json",
    "webqa": "eval/results_webqa_hotpot_fair_refresh_20260528_cover/qwen-api:qwen3.6-plus/t5-large/webqa_top1_0.2_1_bayes_cover_t5_candidate_verifier_softtop2_theta_posteriorverifier_refresh.json",
}


HARD_T5_FILES = {
    # These two files match the Hard-T5-large HotpotQA/WebQA rows in
    # analysis/results/webqa_hotpot_fair_refresh_summary_20260528.csv.
    "hotpotqa": "eval/results_webqa_hotpot_fair_refresh_20260528_hard/qwen-api:qwen3.6-plus/t5-large/hotpotqa_top1_0.2_1.json",
    "webqa": "eval/results_webqa_hotpot_fair_refresh_20260528_hard/qwen-api:qwen3.6-plus/t5-large/webqa_top1_0.2_1.json",
}


HOTPOT_RAW_GT = (
    "dataset/query_hotpotqa_raw_context/query_nonvideo_large_strict_d40/test/hotpotqa.json"
)

TEXT_ACTIONS = {"paragraph", "document"}


def load_json(rel_path: str) -> List[Dict[str, Any]]:
    with (ROOT / rel_path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {rel_path}")
    return data


def load_hotpot_raw_gt() -> Dict[str, List[str]]:
    path = ROOT / HOTPOT_RAW_GT
    if not path.exists():
        return {}
    data = load_json(HOTPOT_RAW_GT)
    return {
        str(row.get("index")): [str(x) for x in row.get("gt_texts", [])]
        for row in data
    }


def norm_action(value: Any) -> str:
    return str(value or "").strip().lower()


def action_equiv(target: str, predicted: Any, gold: Any) -> bool:
    pred = norm_action(predicted)
    ref = norm_action(gold)
    if target in {"squad", "natural_questions"} and pred in TEXT_ACTIONS and ref in TEXT_ACTIONS:
        return True
    return pred == ref


def candidate_modalities(row: Dict[str, Any]) -> List[str]:
    modalities = row.get("retrieval_bayes_soft_modalities")
    if isinstance(modalities, list) and modalities:
        return [norm_action(x) for x in modalities]
    candidates = row.get("retrieval_bayes_soft_candidates")
    if isinstance(candidates, list) and candidates:
        return [norm_action(c.get("modality")) for c in candidates if isinstance(c, dict)]
    return [norm_action(row.get("retrieval"))]


def selected_modality(row: Dict[str, Any]) -> str:
    modalities = candidate_modalities(row)
    idx = row.get("retrieval_bayes_posterior_generation_selected_index")
    if isinstance(idx, int) and 0 <= idx < len(modalities):
        return modalities[idx]
    return norm_action(row.get("retrieval"))


def selected_index(row: Dict[str, Any]) -> int:
    idx = row.get("retrieval_bayes_posterior_generation_selected_index")
    if isinstance(idx, int):
        return idx
    return 0


def threshold(target: str) -> float:
    if target in score.MC_TARGETS:
        return 1.0
    if target in score.SHORT_ANSWER_TARGETS:
        return 0.50
    return 0.30


def rouge_l_score(prediction: str, reference: str) -> float:
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        return float(scorer.score(reference, prediction)["rougeL"].fmeasure)
    except Exception:
        pred = score.normalize_answer(prediction).split()
        ref = score.normalize_answer(reference).split()
        if not pred or not ref:
            return 0.0
        prev = [0] * (len(ref) + 1)
        for token in pred:
            curr = [0]
            for j, ref_token in enumerate(ref, start=1):
                if token == ref_token:
                    curr.append(prev[j - 1] + 1)
                else:
                    curr.append(max(prev[j], curr[-1]))
            prev = curr
        lcs = prev[-1]
        if lcs == 0:
            return 0.0
        precision = lcs / len(pred)
        recall = lcs / len(ref)
        return 2 * precision * recall / (precision + recall)


def utility(row: Dict[str, Any], target: str, response: Optional[str] = None) -> float:
    pred = score.get_prediction_text(row) if response is None else str(response).strip()
    if target in score.MC_TARGETS:
        gold = score.extract_mc_gold(row)
        if gold is None:
            return 0.0
        return float(score.extract_choice_from_prediction(pred) == gold)
    refs = score.extract_references(row, target)
    if not refs:
        return 0.0
    if target in score.SHORT_ANSWER_TARGETS:
        return score.metric_max_over_ground_truths(score.f1_score, pred, refs)
    return max(rouge_l_score(pred, ref) for ref in refs)


def candidate_utilities(row: Dict[str, Any], target: str) -> List[float]:
    candidates = row.get("retrieval_bayes_soft_candidates")
    if not isinstance(candidates, list) or not candidates:
        return [utility(row, target)]
    return [utility(row, target, c.get("response", "")) for c in candidates if isinstance(c, dict)]


def canonical_candidate_answers(row: Dict[str, Any]) -> List[str]:
    scores = row.get("retrieval_bayes_posterior_generation_scores")
    if isinstance(scores, list) and scores:
        vals = [
            str(s.get("canonical_answer", "")).strip()
            for s in scores
            if isinstance(s, dict)
        ]
        if vals:
            return vals
    candidates = row.get("retrieval_bayes_soft_candidates")
    if isinstance(candidates, list) and candidates:
        return [
            score.normalize_answer(str(c.get("response", "")))
            for c in candidates
            if isinstance(c, dict)
        ]
    return [score.normalize_answer(str(row.get("response", "")))]


def branch_disagreement(row: Dict[str, Any]) -> bool:
    answers = [x for x in canonical_candidate_answers(row) if x]
    return len(set(answers)) > 1


def retrieved_items_for_selected(row: Dict[str, Any]) -> List[str]:
    candidates = row.get("retrieval_bayes_soft_candidates")
    idx = selected_index(row)
    if isinstance(candidates, list) and 0 <= idx < len(candidates):
        cand = candidates[idx]
        if isinstance(cand, dict):
            return [str(x) for x in cand.get("retrieved", [])]
    return [str(x) for x in row.get("retrieved", [])]


def evidence_hit(
    row: Dict[str, Any],
    target: str,
    hotpot_raw_gt: Dict[str, List[str]],
) -> Optional[bool]:
    selected = selected_modality(row)
    gold = norm_action(row.get("gt_retrieval"))
    if gold == "no":
        return selected == "no"
    if not action_equiv(target, selected, gold):
        return None

    retrieved = set(retrieved_items_for_selected(row))
    if target == "webqa":
        gold_items = set(str(x) for x in row.get("gt_images", []))
    elif target == "hotpotqa":
        gold_items = set(hotpot_raw_gt.get(str(row.get("index")), []))
    else:
        gold_items = set(str(x) for x in row.get("gt_texts", []))

    if not gold_items:
        return None
    return bool(retrieved & gold_items)


def failure_mode(
    row: Dict[str, Any],
    target: str,
    hotpot_raw_gt: Dict[str, List[str]],
) -> Tuple[str, str]:
    tau = threshold(target)
    final_u = utility(row, target)
    final_ok = final_u >= tau

    modalities = candidate_modalities(row)
    gold = row.get("gt_retrieval")
    contains_gold = any(action_equiv(target, m, gold) for m in modalities)
    selected_matches_gold = action_equiv(target, selected_modality(row), gold)
    cand_utils = candidate_utilities(row, target)
    best_cand = max(cand_utils) if cand_utils else final_u
    any_candidate_ok = best_cand >= tau
    ev_hit = evidence_hit(row, target, hotpot_raw_gt)

    if final_ok:
        if selected_matches_gold:
            return "success_gold_route", "selected gold/equivalent retrieval action and answered correctly"
        if contains_gold:
            return "success_verifier_override", "answered correctly after selecting a non-gold but safer branch"
        return "success_without_gold_route", "answered correctly although the gold retrieval action was absent"

    if not contains_gold:
        return "routing_candidate_miss", "gold retrieval action absent from executed candidates"
    if any_candidate_ok:
        return "verifier_selection_error", "a candidate answer passed the threshold but the selected answer did not"
    if not selected_matches_gold:
        return "wrong_branch_selected", "gold action was available but the selected branch used another action"
    if ev_hit is False:
        return "retrieval_evidence_miss", "selected gold/equivalent branch did not retrieve known gold evidence"
    if ev_hit is True or norm_action(gold) == "no":
        return "generation_error", "known gold evidence or no-retrieval route was selected but answer failed"
    return "unresolved_no_correct_candidate", "no candidate answer passed the threshold under available diagnostics"


def summarize_failure_modes() -> List[Dict[str, Any]]:
    hotpot_raw_gt = load_hotpot_raw_gt()
    rows: List[Dict[str, Any]] = []
    totals = {
        "dataset": "all",
        "n": 0,
        "low_utility_failures": 0,
        "success": 0,
        "routing_candidate_miss": 0,
        "verifier_selection_error": 0,
        "wrong_branch_selected": 0,
        "retrieval_evidence_miss": 0,
        "generation_error": 0,
        "unresolved_no_correct_candidate": 0,
    }

    for target, path in COVER_T5_FILES.items():
        data = load_json(path)
        counts = {k: 0 for k in totals if k not in {"dataset", "n"}}
        for row in data:
            mode, _ = failure_mode(row, target, hotpot_raw_gt)
            if mode.startswith("success"):
                counts["success"] += 1
            else:
                counts["low_utility_failures"] += 1
                counts[mode] += 1
        record = {"dataset": target, "n": len(data), **counts}
        rows.append(record)
        totals["n"] += len(data)
        for key in counts:
            totals[key] += counts[key]

    rows.append(totals)
    return rows


def aligned_hard_rows(target: str) -> Dict[str, Dict[str, Any]]:
    path = HARD_T5_FILES.get(target)
    if not path or not (ROOT / path).exists():
        return {}
    return {str(row.get("index")): row for row in load_json(path)}


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize_gain_buckets() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    hotpot_raw_gt = load_hotpot_raw_gt()
    deltas_by_bucket: Dict[str, List[float]] = {
        "branch agreement": [],
        "branch disagreement": [],
        "lowest uncertainty quartile": [],
        "highest uncertainty quartile": [],
        "known evidence hit": [],
        "known evidence miss": [],
    }

    for target, path in COVER_T5_FILES.items():
        cover = load_json(path)
        hard_by_id = aligned_hard_rows(target)
        if not hard_by_id:
            continue
        uncertainties = [
            float(row.get("retrieval_uncertainty", row.get("retrieval_bayes_uncertainty", 0.0)) or 0.0)
            for row in cover
        ]
        ordered = sorted(uncertainties)
        lo = ordered[int(0.25 * (len(ordered) - 1))]
        hi = ordered[int(0.75 * (len(ordered) - 1))]

        for row in cover:
            hard = hard_by_id.get(str(row.get("index")))
            if hard is None:
                continue
            delta = utility(row, target) - utility(hard, target)

            if branch_disagreement(row):
                deltas_by_bucket["branch disagreement"].append(delta)
            else:
                deltas_by_bucket["branch agreement"].append(delta)

            unc = float(row.get("retrieval_uncertainty", row.get("retrieval_bayes_uncertainty", 0.0)) or 0.0)
            if unc <= lo:
                deltas_by_bucket["lowest uncertainty quartile"].append(delta)
            if unc >= hi:
                deltas_by_bucket["highest uncertainty quartile"].append(delta)

            ev_hit = evidence_hit(row, target, hotpot_raw_gt)
            if ev_hit is True:
                deltas_by_bucket["known evidence hit"].append(delta)
            elif ev_hit is False:
                deltas_by_bucket["known evidence miss"].append(delta)

    for bucket, deltas in deltas_by_bucket.items():
        rows.append(
            {
                "bucket": bucket,
                "n": len(deltas),
                "cover_minus_hard_points": mean(deltas) * 100.0,
            }
        )
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def pct(part: int, whole: int) -> float:
    return 100.0 * part / whole if whole else 0.0


def failure_latex(rows: Sequence[Dict[str, Any]]) -> str:
    labels = {
        "routing_candidate_miss": "Candidate-route miss",
        "verifier_selection_error": "Verifier selection error",
        "wrong_branch_selected": "Wrong branch selected",
        "retrieval_evidence_miss": "Retrieval/evidence miss",
        "generation_error": "Generation error",
        "unresolved_no_correct_candidate": "No correct candidate",
    }
    all_row = next(row for row in rows if row["dataset"] == "all")
    denom = int(all_row["low_utility_failures"])
    lines = []
    for key, label in labels.items():
        count = int(all_row[key])
        lines.append(f"{label} & {count} & {pct(count, denom):.2f} \\\\")
    return "\n".join(lines)


def gain_latex(rows: Sequence[Dict[str, Any]]) -> str:
    label_map = {
        "branch agreement": "Branch answers agree",
        "branch disagreement": "Branch answers disagree",
        "lowest uncertainty quartile": "Lowest uncertainty quartile",
        "highest uncertainty quartile": "Highest uncertainty quartile",
        "known evidence hit": "Known evidence hit",
        "known evidence miss": "Known evidence miss",
    }
    out = []
    for row in rows:
        out.append(
            f"{label_map.get(row['bucket'], row['bucket'])} & {int(row['n'])} "
            f"& {float(row['cover_minus_hard_points']):+.2f} \\\\"
        )
    return "\n".join(out)


def print_failure_summary(rows: Sequence[Dict[str, Any]]) -> None:
    for row in rows:
        n = int(row["n"])
        failures = int(row["low_utility_failures"])
        print(
            f"{row['dataset']:18s} n={n:5d} "
            f"success={pct(int(row['success']), n):6.2f}% "
            f"low_utility_fail={pct(failures, n):6.2f}%"
        )
    print("\nFailure causes among low-utility COVER-T5 examples:")
    print(failure_latex(rows))


def print_gain_summary(rows: Sequence[Dict[str, Any]]) -> None:
    print("\nCOVER-T5 minus Hard-T5 utility by bucket:")
    print(gain_latex(rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="analysis/results")
    parser.add_argument("--latex", action="store_true")
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir
    failure_rows = summarize_failure_modes()
    gain_rows = summarize_gain_buckets()

    write_csv(out_dir / "failure_mode_distribution_cover_t5.csv", failure_rows)
    write_csv(out_dir / "cover_gain_by_diagnostic_bucket_t5.csv", gain_rows)
    (out_dir / "failure_mode_distribution_cover_t5.tex").write_text(
        failure_latex(failure_rows) + "\n", encoding="utf-8"
    )
    (out_dir / "cover_gain_by_diagnostic_bucket_t5.tex").write_text(
        gain_latex(gain_rows) + "\n", encoding="utf-8"
    )

    if args.latex:
        print(failure_latex(failure_rows))
        print()
        print(gain_latex(gain_rows))
    else:
        print_failure_summary(failure_rows)
        print_gain_summary(gain_rows)


if __name__ == "__main__":
    main()
