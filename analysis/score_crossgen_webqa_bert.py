#!/usr/bin/env python
"""Compute WebQA BERTScore for completed cross-generator baseline files.

This script is intentionally resumable: after each Generator/Method row is
scored, the output JSON is written. Re-running the script skips rows already
present unless --force is passed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.score import extract_references, get_prediction_text, load_json  # noqa: E402


MODELS = [
    ("GPT-4o", "dmxapi:gpt-4o"),
    ("GLM-4.6V", "glm:glm-4.6v"),
    ("DeepSeek-v4-pro", "deepseek:deepseek-v4-pro"),
]

METHODS = [
    ("Hard-T5-large", "eval/results_crossgen_baselines_20260604_hard", "t5-large"),
    (
        "UniversalRAG-T5-large",
        "eval/results_crossgen_baselines_20260604_universalrag",
        "t5-large",
    ),
    ("Self-RAG", "eval/results_crossgen_baselines_20260604_selfrag", "selfrag"),
]


def load_existing(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}")
    return data


def save_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def flatten_webqa(path: Path, limit: int | None = None):
    data = load_json(str(path))
    if limit is not None:
        data = data[:limit]

    flat_preds: list[str] = []
    flat_refs: list[str] = []
    group_sizes: list[int] = []

    kept = 0
    for item in data:
        refs = extract_references(item, "webqa")
        if not refs:
            continue
        pred = get_prediction_text(item)
        group_sizes.append(len(refs))
        flat_preds.extend([pred] * len(refs))
        flat_refs.extend(refs)
        kept += 1

    return flat_preds, flat_refs, group_sizes, kept, len(data)


def max_over_refs(f1_values, group_sizes: list[int]) -> list[float]:
    values = f1_values.detach().cpu().numpy()
    out: list[float] = []
    start = 0
    for size in group_sizes:
        out.append(float(np.max(values[start : start + size])))
        start += size
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="eval/results_crossgen_baselines_20260604_webqa_bert_gpt4o_glm.json",
    )
    parser.add_argument(
        "--merged-out",
        default="eval/results_crossgen_baselines_20260604_summary_gpt4o_glm_with_bert.json",
    )
    parser.add_argument(
        "--fast-summary",
        default="eval/results_crossgen_baselines_20260604_summary_gpt4o_glm_fast.json",
    )
    parser.add_argument("--device", default="auto", help="auto, cuda, cpu, or cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="debug: score first N rows only")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    import torch
    from bert_score import BERTScorer

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    out_path = ROOT / args.out
    rows = load_existing(out_path)
    done = {(row["Generator"], row["Method"]) for row in rows}

    print(f"[INFO] device={device} batch_size={args.batch_size}")
    print(f"[INFO] output={out_path}")

    scorer = BERTScorer(
        lang="en",
        rescale_with_baseline=False,
        device=device,
        batch_size=args.batch_size,
    )

    for generator_label, model_path in MODELS:
        for method, root, router in METHODS:
            key = (generator_label, method)
            if key in done and not args.force:
                print(f"[SKIP] {generator_label} / {method}")
                continue

            result_file = ROOT / root / model_path / router / "webqa_top1_0.2_1.json"
            if not result_file.exists():
                raise FileNotFoundError(result_file)

            flat_preds, flat_refs, group_sizes, kept, total = flatten_webqa(
                result_file, limit=args.limit
            )
            print(
                f"[RUN] {generator_label} / {method}: "
                f"{kept}/{total} examples, {len(flat_preds)} pred-ref pairs"
            )
            _, _, f1 = scorer.score(flat_preds, flat_refs, batch_size=args.batch_size)
            scores = max_over_refs(f1, group_sizes)
            bert = round(float(np.mean(scores)) * 100, 2)

            row = {
                "Generator": generator_label,
                "Method": method,
                "WebQA_BERT": bert,
                "Count": kept,
                "File": str(result_file.relative_to(ROOT)),
            }
            rows = [r for r in rows if (r["Generator"], r["Method"]) != key]
            rows.append(row)
            rows.sort(key=lambda r: (r["Generator"], r["Method"]))
            save_json(out_path, rows)
            done.add(key)
            print(f"[DONE] {generator_label} / {method}: WebQA_BERT={bert:.2f}")

    fast_path = ROOT / args.fast_summary
    if fast_path.exists():
        fast_rows = load_existing(fast_path)
        bert_by_key = {(r["Generator"], r["Method"]): r["WebQA_BERT"] for r in rows}
        for row in fast_rows:
            key = (row["Generator"], row["Method"])
            if key in bert_by_key:
                row["WebQA_BERT"] = bert_by_key[key]
        merged_path = ROOT / args.merged_out
        save_json(merged_path, fast_rows)
        print(f"[INFO] merged summary={merged_path}")

        print(
            "Generator | Method | MMLU | SQuAD_EM | SQuAD_F1 | "
            "NQ_EM | NQ_F1 | HotpotQA_EM | HotpotQA_F1 | WebQA_RL | WebQA_BERT"
        )
        cols = [
            "Generator",
            "Method",
            "MMLU",
            "SQuAD_EM",
            "SQuAD_F1",
            "NQ_EM",
            "NQ_F1",
            "HotpotQA_EM",
            "HotpotQA_F1",
            "WebQA_RL",
            "WebQA_BERT",
        ]
        for row in fast_rows:
            values = []
            for col in cols:
                value = row.get(col)
                if isinstance(value, str):
                    values.append(value)
                elif value is None:
                    values.append("--")
                else:
                    values.append(f"{value:.2f}")
            print(" | ".join(values))


if __name__ == "__main__":
    main()
