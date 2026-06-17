#!/usr/bin/env python
"""Create OOD router fine-tuning data and a held-out routing split."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List


VALID_ROUTES = {"no", "paragraph", "document", "image"}


def load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data)}")
    return data


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def to_train_record(row: Dict[str, Any]) -> Dict[str, Any]:
    route = str(row.get("gt_retrieval", "")).strip().lower()
    if route not in VALID_ROUTES:
        raise ValueError(f"Unsupported route label: {route}")
    return {
        "question": row.get("question", ""),
        "source": route,
        "gt_retrieval": route,
        "source_label": route,
        "origin_source": row.get("source", ""),
        "origin_index": row.get("index", ""),
    }


def split_rows(rows: List[Dict[str, Any]], train_ratio: float, rng: random.Random):
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row.get("gt_retrieval", "")).strip().lower()].append(row)

    train_rows = []
    holdout_rows = []
    for label, items in grouped.items():
        rng.shuffle(items)
        if len(items) <= 1:
            train_count = len(items)
        else:
            train_count = int(round(len(items) * train_ratio))
            train_count = max(1, min(train_count, len(items) - 1))
        train_rows.extend(items[:train_count])
        holdout_rows.extend(items[train_count:])

    rng.shuffle(train_rows)
    rng.shuffle(holdout_rows)
    return train_rows, holdout_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build OOD router fine-tuning split.")
    parser.add_argument("--query_dir", type=Path, default=Path("dataset/query_ood"))
    parser.add_argument("--holdout_dir", type=Path, default=Path("dataset/query_ood_holdout"))
    parser.add_argument("--ood_train_out", type=Path, default=Path("route/train/data/train_data_ood_router_4class.json"))
    parser.add_argument("--base_train", type=Path, default=Path("route/train/data/train_data_distilbert_4class.json"))
    parser.add_argument("--mixed_train_out", type=Path, default=Path("route/train/data/train_data_distilbert_4class_plus_ood.json"))
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    all_train_records = []
    manifest = []

    for path in sorted(args.query_dir.glob("*.json")):
        rows = load_json(path)
        rows = [row for row in rows if str(row.get("gt_retrieval", "")).strip().lower() in VALID_ROUTES]
        train_rows, holdout_rows = split_rows(rows, args.train_ratio, rng)

        dump_json(args.holdout_dir / path.name, holdout_rows)
        train_records = [to_train_record(row) for row in train_rows]
        all_train_records.extend(train_records)
        manifest.append(
            {
                "file": path.name,
                "train": len(train_rows),
                "holdout": len(holdout_rows),
                "train_labels": dict(Counter(row["gt_retrieval"] for row in train_records)),
                "holdout_labels": dict(Counter(row["gt_retrieval"] for row in holdout_rows)),
            }
        )

    rng.shuffle(all_train_records)
    dump_json(args.ood_train_out, all_train_records)

    mixed_records = None
    if args.base_train and args.base_train.is_file():
        base_records = load_json(args.base_train)
        mixed_records = base_records + all_train_records
        rng.shuffle(mixed_records)
        dump_json(args.mixed_train_out, mixed_records)

    summary = {
        "ood_train_out": str(args.ood_train_out),
        "ood_train_rows": len(all_train_records),
        "ood_train_labels": dict(Counter(row["gt_retrieval"] for row in all_train_records)),
        "holdout_dir": str(args.holdout_dir),
        "mixed_train_out": str(args.mixed_train_out) if mixed_records is not None else None,
        "mixed_train_rows": len(mixed_records) if mixed_records is not None else None,
        "manifest": manifest,
    }
    dump_json(args.holdout_dir.parent / f"{args.holdout_dir.name}_manifest.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
