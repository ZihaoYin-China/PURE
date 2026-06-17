#!/usr/bin/env python
import argparse
import json
import os
from collections import Counter, defaultdict


TARGETS = ["mmlu", "squad", "natural_questions", "hotpotqa", "webqa"]

DATASET_TO_ROUTE = {
    "mmlu": "no",
    "squad": "paragraph",
    "natural_questions": "paragraph",
    "hotpotqa": "document",
    "webqa": "image",
}

ROUTE_ID_TO_LABEL = {
    0: "no",
    1: "paragraph",
    2: "document",
    3: "image",
    "0": "no",
    "1": "paragraph",
    "2": "document",
    "3": "image",
}

VIDEO_SOURCES = {"lvbench", "videorag_wikihow", "videorag_synth"}


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def infer_target(sample):
    source = str(sample.get("source", "")).lower().strip()
    if source in VIDEO_SOURCES:
        return None
    if source in TARGETS:
        return source

    index = str(sample.get("index", "")).lower()
    for target in TARGETS:
        if target == "natural_questions":
            prefixes = ("nq_", "natural_questions")
        elif target == "webqa":
            prefixes = ("webqa_",)
        else:
            prefixes = (f"{target}_", target)
        if any(index.startswith(prefix) for prefix in prefixes):
            return target

    return None


def normalize_route(sample, target):
    route = sample.get("gt_retrieval")
    if route in ROUTE_ID_TO_LABEL:
        return ROUTE_ID_TO_LABEL[route]
    route = str(route or "").lower().strip()
    if route in DATASET_TO_ROUTE.values():
        return route
    return DATASET_TO_ROUTE[target]


def normalize_sample(sample, target, split):
    new_sample = dict(sample)
    new_sample["source"] = target
    new_sample["gt_retrieval"] = normalize_route(sample, target)
    new_sample["split"] = split

    if target == "hotpotqa" and "gt_texts" in new_sample:
        fixed = []
        for path in new_sample.get("gt_texts") or []:
            path = str(path)
            path = path.replace(
                "dataset/LongRAG/hotpot_qa_corpus/text/",
                "dataset/hotpotqa/text/",
            )
            fixed.append(path)
        new_sample["gt_texts"] = fixed

    return new_sample


def to_router_sample(sample, target):
    route = normalize_route(sample, target)
    router_sample = dict(sample)
    router_sample["dataset_source"] = target
    router_sample["source"] = route
    router_sample["gt_retrieval"] = route
    router_sample["source_label"] = route
    return router_sample


def is_eval_ready(sample, target):
    if target == "mmlu":
        return "choices:" in str(sample.get("question", "")).lower()
    if target in {"squad", "natural_questions", "hotpotqa"}:
        return bool(sample.get("gt_texts"))
    if target == "webqa":
        return bool(sample.get("gt_images"))
    return True


def dedupe_keep_order(rows):
    seen = set()
    out = []
    for row in rows:
        key = str(row.get("index", ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Build a non-video train/test/full query split without touching dataset/query."
    )
    parser.add_argument(
        "--train_data",
        default="route/train/data/train_data_distilbert.json",
        help="Original router training JSON containing the 30 percent train split.",
    )
    parser.add_argument(
        "--test_query_dir",
        default="dataset/query",
        help="Current query directory containing the 70 percent eval split.",
    )
    parser.add_argument(
        "--output_dir",
        default="dataset/query_nonvideo_split",
        help="New output directory. Existing files in this directory may be overwritten.",
    )
    args = parser.parse_args()

    train_source = read_json(args.train_data)
    train_by_target = defaultdict(list)
    for sample in train_source:
        target = infer_target(sample)
        if target not in TARGETS:
            continue
        train_by_target[target].append(normalize_sample(sample, target, "train"))

    test_by_target = {}
    for target in TARGETS:
        path = os.path.join(args.test_query_dir, f"{target}.json")
        test_rows = read_json(path)
        test_by_target[target] = [
            normalize_sample(sample, target, "test") for sample in test_rows
        ]

    manifest = {
        "description": (
            "Non-video split reconstructed from router train data and current "
            "dataset/query eval files. This keeps the original files untouched."
        ),
        "targets": TARGETS,
        "source_files": {
            "train": args.train_data,
            "test": args.test_query_dir,
        },
        "note": (
            "Use test/*.json for paper-style evaluation. full/*.json is for "
            "counting/accounting unless you intentionally want train+test evaluation."
        ),
        "counts": {},
    }

    router_train_rows = []
    router_test_rows = []
    router_full_rows = []

    for target in TARGETS:
        train_rows = dedupe_keep_order(train_by_target[target])
        test_rows = dedupe_keep_order(test_by_target[target])
        full_rows = dedupe_keep_order(train_rows + test_rows)

        train_ids = {str(row.get("index", "")) for row in train_rows}
        test_ids = {str(row.get("index", "")) for row in test_rows}

        for split, rows in [
            ("train", train_rows),
            ("test", test_rows),
            ("full", full_rows),
        ]:
            write_json(os.path.join(args.output_dir, split, f"{target}.json"), rows)

        router_train_rows.extend(to_router_sample(row, target) for row in train_rows)
        router_test_rows.extend(to_router_sample(row, target) for row in test_rows)
        router_full_rows.extend(to_router_sample(row, target) for row in full_rows)

        manifest["counts"][target] = {
            "train": len(train_rows),
            "test": len(test_rows),
            "full": len(full_rows),
            "train_ratio": round(len(train_rows) / max(1, len(full_rows)), 4),
            "test_ratio": round(len(test_rows) / max(1, len(full_rows)), 4),
            "train_test_overlap": len(train_ids & test_ids),
            "routes": {
                "train": dict(Counter(row["gt_retrieval"] for row in train_rows)),
                "test": dict(Counter(row["gt_retrieval"] for row in test_rows)),
            },
            "eval_ready": {
                "train": sum(is_eval_ready(row, target) for row in train_rows),
                "test": sum(is_eval_ready(row, target) for row in test_rows),
                "full": sum(is_eval_ready(row, target) for row in full_rows),
            },
        }

    write_json(os.path.join(args.output_dir, "manifest.json"), manifest)
    write_json(os.path.join(args.output_dir, "router_train_4class.json"), router_train_rows)
    write_json(os.path.join(args.output_dir, "router_test_4class.json"), router_test_rows)
    write_json(os.path.join(args.output_dir, "router_full_4class.json"), router_full_rows)

    readme = [
        "# Non-Video Query Split",
        "",
        "This directory is generated by `tools/build_nonvideo_query_split.py`.",
        "",
        "- `train/`: the 30 percent split used by router training.",
        "- `test/`: the current `dataset/query` split used for evaluation.",
        "- `full/`: train plus test, mainly for count checking.",
        "- `router_train_4class.json`: concatenated non-video 4-class router train file.",
        "- `router_test_4class.json`: concatenated non-video 4-class router test file.",
        "",
        "For paper-style evaluation, use `test/`, not `full/`.",
        "The original `dataset/query` files are not modified.",
        "",
    ]
    with open(os.path.join(args.output_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(readme))

    print(json.dumps(manifest["counts"], indent=2, ensure_ascii=False))
    print(f"Saved split to {args.output_dir}")


if __name__ == "__main__":
    main()
