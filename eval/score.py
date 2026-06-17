# /Yin_zi_hao/code/PURE/eval/score.py

import argparse
import json
import os
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


MC_TARGETS = {"mmlu", "truthfulqa"}
SHORT_ANSWER_TARGETS = {"squad", "natural_questions", "hotpotqa", "triviaqa"}
LONG_ANSWER_TARGETS = {"webqa", "lara", "visual_rag"}

CHOICE_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of samples in {path}, but got {type(data)}")
    return data


def infer_target(result_file: str, data: List[Dict[str, Any]]) -> str:
    file_name = Path(result_file).name.lower()

    for target in [
        "mmlu",
        "squad",
        "natural_questions",
        "hotpotqa",
        "webqa",
        "truthfulqa",
        "triviaqa",
        "lara",
        "visual_rag",
    ]:
        if target in file_name:
            return target

    if data and isinstance(data[0], dict):
        src = str(data[0].get("source", "")).strip().lower()
        if src in MC_TARGETS | SHORT_ANSWER_TARGETS | LONG_ANSWER_TARGETS:
            return src

    raise ValueError(
        "Cannot infer target from file name or sample['source'].\n"
        "Please pass --target explicitly."
    )


def get_prediction_text(item: Dict[str, Any]) -> str:
    return str(item.get("response", "")).strip()


def flatten_to_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        out = []
        for v in value:
            out.extend(flatten_to_str_list(v))
        return [x for x in out if x != ""]
    if isinstance(value, dict):
        out = []
        for v in value.values():
            out.extend(flatten_to_str_list(v))
        return [x for x in out if x != ""]
    return [str(value).strip()]


def dedup_keep_order(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def normalize_answer(s: str) -> str:
    s = str(s).lower()

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    return white_space_fix(remove_articles(remove_punc(s)))


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths: List[str]) -> float:
    if not ground_truths:
        return 0.0
    return max(metric_fn(prediction, gt) for gt in ground_truths)


def safe_int_to_choice(v: Any) -> Optional[str]:
    if isinstance(v, int):
        if 0 <= v < len(CHOICE_LETTERS):
            return CHOICE_LETTERS[v]
        return None

    if isinstance(v, str):
        text = v.strip().upper()
        if text in CHOICE_LETTERS:
            return text

        m = re.search(r"\b([A-Z])\b", text)
        if m:
            return m.group(1)

        m = re.search(r"\(([A-Z])\)", text)
        if m:
            return m.group(1)

        if text.isdigit():
            idx = int(text)
            if 0 <= idx < len(CHOICE_LETTERS):
                return CHOICE_LETTERS[idx]

    return None


def extract_choice_from_prediction(text: str, allowed_choices: str = "ABCD") -> Optional[str]:
    text = str(text).strip().upper()
    allowed = {ch.upper() for ch in str(allowed_choices or "") if ch.strip()}

    patterns = [
        r"ANSWER\s*(?:IS)?\s*[:：]?\s*\(?([A-Z])\)?",
        r"CORRECT\s+ANSWER\s*(?:IS)?\s*[:：]?\s*\(?([A-Z])\)?",
        r"OPTION\s*[:：]?\s*\(?([A-Z])\)?",
        r"CHOICE\s*[:：]?\s*\(?([A-Z])\)?",
        r"\(([A-Z])\)",
        r"\b([A-Z])\b",
    ]

    for pat in patterns:
        matches = re.findall(pat, text)
        if matches:
            for match in reversed(matches):
                choice = str(match).upper()
                if choice in allowed:
                    return choice

    return None


def extract_mc_gold(item: Dict[str, Any]) -> Optional[str]:
    candidate_keys = [
        "answer",
        "gt_answer",
        "gold",
        "gold_answer",
        "label",
        "correct_answer",
        "target",
    ]

    for k in candidate_keys:
        if k in item:
            gold = safe_int_to_choice(item[k])
            if gold is not None:
                return gold

    # 有些数据会把标准答案写成 index
    for k in ["answer_idx", "label_idx", "gold_idx"]:
        if k in item:
            gold = safe_int_to_choice(item[k])
            if gold is not None:
                return gold

    return None


def extract_references(item: Dict[str, Any], target: str) -> List[str]:
    candidate_keys = [
        "answers",
        "answer",
        "gt_answers",
        "gt_answer",
        "ground_truths",
        "ground_truth",
        "gold_answers",
        "gold_answer",
        "gold",
        "label",
        "labels",
        "target",
        "targets",
        "reference",
        "references",
    ]

    refs = []
    for k in candidate_keys:
        if k in item:
            refs.extend(flatten_to_str_list(item[k]))

    refs = [x.strip() for x in refs if str(x).strip() != ""]

    # 多选题不走这里
    if target in MC_TARGETS:
        return refs

    # 有些数据会把整数标签带进来，短答/长答里尽量过滤掉孤立选项字母
    cleaned = []
    for x in refs:
        x_norm = x.strip()
        if len(x_norm) == 1 and x_norm.upper() in CHOICE_LETTERS:
            continue
        cleaned.append(x_norm)

    return dedup_keep_order(cleaned)


def score_mmlu(data: List[Dict[str, Any]]) -> Dict[str, float]:
    total = 0
    correct = 0

    for item in data:
        pred = extract_choice_from_prediction(get_prediction_text(item))
        gold = extract_mc_gold(item)

        if gold is None:
            continue

        total += 1
        if pred == gold:
            correct += 1

    if total == 0:
        raise ValueError("No valid MMLU gold answers were found in the result file.")

    return {
        "Accuracy": round(correct / total * 100, 2),
        "Count": total,
    }


def score_short_answers(data: List[Dict[str, Any]], target: str) -> Dict[str, float]:
    em_scores = []
    f1_scores = []

    for item in data:
        pred = get_prediction_text(item)
        refs = extract_references(item, target)
        if not refs:
            continue

        em_scores.append(metric_max_over_ground_truths(exact_match_score, pred, refs))
        f1_scores.append(metric_max_over_ground_truths(f1_score, pred, refs))

    if not em_scores:
        raise ValueError(f"No valid references were found for target={target}")

    return {
        "EM": round(float(np.mean(em_scores)) * 100, 2),
        "F1": round(float(np.mean(f1_scores)) * 100, 2),
        "Count": len(em_scores),
    }


def rouge_l_max_batch(preds: List[str], refs_list: List[List[str]]) -> List[float]:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []

    for pred, refs in zip(preds, refs_list):
        if not refs:
            scores.append(0.0)
            continue
        best = 0.0
        for ref in refs:
            score = scorer.score(ref, pred)["rougeL"].fmeasure
            best = max(best, score)
        scores.append(best)

    return scores


def bertscore_max_batch(preds: List[str], refs_list: List[List[str]], device: str) -> List[float]:
    from bert_score import score as bertscore_score

    flat_preds = []
    flat_refs = []
    group_sizes = []

    for pred, refs in zip(preds, refs_list):
        if not refs:
            refs = [""]
        group_sizes.append(len(refs))
        flat_preds.extend([pred] * len(refs))
        flat_refs.extend(refs)

    _, _, f1 = bertscore_score(
        flat_preds,
        flat_refs,
        lang="en",
        verbose=False,
        device=device,
        rescale_with_baseline=False,
    )

    f1 = f1.detach().cpu().numpy()

    out = []
    start = 0
    for size in group_sizes:
        out.append(float(np.max(f1[start:start + size])))
        start += size
    return out


def score_long_answers(data: List[Dict[str, Any]], target: str) -> Dict[str, float]:
    preds = []
    refs_list = []

    for item in data:
        pred = get_prediction_text(item)
        refs = extract_references(item, target)
        if not refs:
            continue
        preds.append(pred)
        refs_list.append(refs)

    if not preds:
        raise ValueError(f"No valid references were found for target={target}")

    rouge_scores = rouge_l_max_batch(preds, refs_list)

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bert_scores = bertscore_max_batch(preds, refs_list, device=device)

    return {
        "ROUGE-L": round(float(np.mean(rouge_scores)) * 100, 2),
        "BERTScore": round(float(np.mean(bert_scores)) * 100, 2),
        "Count": len(preds),
    }


def score_file(result_file: str, target: Optional[str] = None) -> Dict[str, Any]:
    data = load_json(result_file)
    target = target or infer_target(result_file, data)

    if target in MC_TARGETS:
        metrics = score_mmlu(data)
    elif target in SHORT_ANSWER_TARGETS:
        metrics = score_short_answers(data, target)
    elif target in LONG_ANSWER_TARGETS:
        metrics = score_long_answers(data, target)
    else:
        raise ValueError(f"Unsupported target: {target}")

    return {
        "target": target,
        "file": result_file,
        **metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Score PURE generated results.")
    parser.add_argument("--result_file", type=str, required=True, help="Path to eval/results/*.json")
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        choices=[
            "mmlu",
            "squad",
            "natural_questions",
            "hotpotqa",
            "webqa",
            "truthfulqa",
            "triviaqa",
            "lara",
            "visual_rag",
        ],
        help="Optional. If omitted, infer from filename or sample['source']."
    )
    args = parser.parse_args()

    result = score_file(args.result_file, args.target)

    print("=" * 60)
    print(f"File   : {result['file']}")
    print(f"Target : {result['target']}")
    for k, v in result.items():
        if k not in {"file", "target"}:
            print(f"{k:10s}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
