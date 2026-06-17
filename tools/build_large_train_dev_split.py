#!/usr/bin/env python
import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from math import floor

try:
    from route.gpt.prompt import ROUTER_PROMPT
except Exception:
    ROUTER_PROMPT = None


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


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


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


def normalize_route(sample, target):
    value = sample.get("gt_retrieval")
    if value in ROUTE_ID_TO_LABEL:
        return ROUTE_ID_TO_LABEL[value]
    value = str(value or "").lower().strip()
    if value in {"no", "paragraph", "document", "image"}:
        return value
    return DATASET_TO_ROUTE[target]


def stable_sort_key(row, seed):
    key = str(row.get("index", "")) or str(row.get("question", ""))
    digest = hashlib.md5(f"{seed}:{key}".encode("utf-8", errors="ignore")).hexdigest()
    return digest, key


def safe_slug(text, max_len=48):
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")
    return text[:max_len] or "doc"


def mmlu_subject(row):
    subject = str(row.get("subject", "")).strip()
    if subject:
        return safe_slug(subject, 64)

    index = str(row.get("index", ""))
    match = re.match(r"mmlu_large_(.+)_\d+_[0-9a-f]{12}$", index)
    if match:
        return match.group(1)
    return "mmlu_unknown"


def stratified_train_dev_split(rows, dev_ratio=0.2, seed=42):
    groups = defaultdict(list)
    for row in rows:
        groups[mmlu_subject(row)].append(dict(row))

    total = sum(len(group_rows) for group_rows in groups.values())
    target_dev = int(round(total * dev_ratio))

    quotas = {}
    remainders = []
    for subject, group_rows in groups.items():
        exact = len(group_rows) * dev_ratio
        base = floor(exact)
        quotas[subject] = base
        remainders.append((exact - base, subject))

    diff = target_dev - sum(quotas.values())
    if diff > 0:
        for _, subject in sorted(remainders, key=lambda item: (-item[0], item[1]))[:diff]:
            quotas[subject] += 1
    elif diff < 0:
        for _, subject in sorted(remainders, key=lambda item: (item[0], item[1])):
            if diff == 0:
                break
            if quotas[subject] > 0:
                quotas[subject] -= 1
                diff += 1

    train_rows = []
    dev_rows = []
    for subject in sorted(groups):
        group_rows = sorted(groups[subject], key=lambda row: stable_sort_key(row, seed=seed))
        dev_n = quotas[subject]
        dev_rows.extend(group_rows[:dev_n])
        train_rows.extend(group_rows[dev_n:])

    train_rows = sorted(train_rows, key=lambda row: stable_sort_key(row, seed=seed))
    dev_rows = sorted(dev_rows, key=lambda row: stable_sort_key(row, seed=seed))
    return train_rows, dev_rows


def deterministic_train_dev_split(rows, target, dev_ratio=0.2, seed=42):
    rows = dedupe_keep_order(rows)
    if target == "mmlu":
        return stratified_train_dev_split(rows, dev_ratio=dev_ratio, seed=seed)

    ordered = sorted(rows, key=lambda row: stable_sort_key(row, seed=seed))
    dev_count = int(round(len(ordered) * dev_ratio))
    dev_rows = ordered[:dev_count]
    train_rows = ordered[dev_count:]
    return train_rows, dev_rows


def to_router_sample(sample, target):
    route = normalize_route(sample, target)
    row = dict(sample)
    row["dataset_source"] = target
    row["source"] = route
    row["gt_retrieval"] = route
    row["source_label"] = route
    return row


def to_t5_router_sample(sample, target):
    row = to_router_sample(sample, target)
    query = str(row.get("question", ""))
    if ROUTER_PROMPT is not None:
        row["question"] = ROUTER_PROMPT.format(query=query)
    else:
        row["question"] = (
            "Classify the following query into one of four categories: "
            "[No, Paragraph, Document, Image].\n"
            f"Query: {query}\n"
            "Provide only the category."
        )
    return row


def mark_split(rows, split_name):
    out = []
    for row in rows:
        copied = dict(row)
        copied["split"] = split_name
        out.append(copied)
    return out


def eval_ready(row, target):
    if target == "mmlu":
        return "choices:" in str(row.get("question", "")).lower()
    if target in {"squad", "natural_questions", "hotpotqa"}:
        return bool(row.get("gt_texts"))
    if target == "webqa":
        return bool(row.get("gt_images"))
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Split dataset/query_nonvideo_large train into train_fit/dev without touching test."
    )
    parser.add_argument("--input_root", default="dataset/query_nonvideo_large")
    parser.add_argument("--output_root", default="dataset/query_nonvideo_large_strict")
    parser.add_argument("--dev_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not (0.0 < args.dev_ratio < 1.0):
        raise ValueError("--dev_ratio must be in (0, 1).")

    manifest = {
        "description": (
            "Strict train_fit/dev/test split derived from query_nonvideo_large. "
            "Use dev for tuning; keep test untouched for final reporting."
        ),
        "input_root": args.input_root,
        "output_root": args.output_root,
        "dev_ratio": args.dev_ratio,
        "seed": args.seed,
        "targets": TARGETS,
        "counts": {},
        "notes": [
            "test is copied from input_root/test and must not be used for tuning.",
            "train_fit + dev is partitioned from input_root/train.",
            "router_train_fit_4class.json and router_train_fit_t5_4class.json are for training.",
        ],
    }

    router_train_fit = []
    router_train_fit_t5 = []
    router_dev = []
    router_dev_t5 = []
    router_test = []

    for target in TARGETS:
        train_path = os.path.join(args.input_root, "train", f"{target}.json")
        test_path = os.path.join(args.input_root, "test", f"{target}.json")
        if not os.path.isfile(train_path):
            raise FileNotFoundError(f"Missing train file: {train_path}")
        if not os.path.isfile(test_path):
            raise FileNotFoundError(f"Missing test file: {test_path}")

        train_rows = dedupe_keep_order(read_json(train_path))
        test_rows = dedupe_keep_order(read_json(test_path))

        train_fit_rows, dev_rows = deterministic_train_dev_split(
            rows=train_rows,
            target=target,
            dev_ratio=args.dev_ratio,
            seed=args.seed,
        )
        train_fit_rows = mark_split(train_fit_rows, "train_fit")
        dev_rows = mark_split(dev_rows, "dev")
        test_rows = mark_split(test_rows, "test")
        full_rows = dedupe_keep_order(train_fit_rows + dev_rows + test_rows)

        write_json(os.path.join(args.output_root, "train_fit", f"{target}.json"), train_fit_rows)
        write_json(os.path.join(args.output_root, "dev", f"{target}.json"), dev_rows)
        write_json(os.path.join(args.output_root, "test", f"{target}.json"), test_rows)
        write_json(os.path.join(args.output_root, "full", f"{target}.json"), full_rows)

        router_train_fit.extend(to_router_sample(row, target) for row in train_fit_rows)
        router_train_fit_t5.extend(to_t5_router_sample(row, target) for row in train_fit_rows)
        router_dev.extend(to_router_sample(row, target) for row in dev_rows)
        router_dev_t5.extend(to_t5_router_sample(row, target) for row in dev_rows)
        router_test.extend(to_router_sample(row, target) for row in test_rows)

        manifest["counts"][target] = {
            "train_input": len(train_rows),
            "train_fit": len(train_fit_rows),
            "dev": len(dev_rows),
            "test": len(test_rows),
            "full": len(full_rows),
            "routes": {
                "train_fit": dict(Counter(str(row.get("gt_retrieval", "")).lower() for row in train_fit_rows)),
                "dev": dict(Counter(str(row.get("gt_retrieval", "")).lower() for row in dev_rows)),
                "test": dict(Counter(str(row.get("gt_retrieval", "")).lower() for row in test_rows)),
            },
            "eval_ready": {
                "train_fit": sum(eval_ready(row, target) for row in train_fit_rows),
                "dev": sum(eval_ready(row, target) for row in dev_rows),
                "test": sum(eval_ready(row, target) for row in test_rows),
                "full": sum(eval_ready(row, target) for row in full_rows),
            },
        }

    write_json(os.path.join(args.output_root, "router_train_fit_4class.json"), router_train_fit)
    write_json(os.path.join(args.output_root, "router_train_fit_t5_4class.json"), router_train_fit_t5)
    write_json(os.path.join(args.output_root, "router_dev_4class.json"), router_dev)
    write_json(os.path.join(args.output_root, "router_dev_t5_4class.json"), router_dev_t5)
    write_json(os.path.join(args.output_root, "router_test_4class.json"), router_test)
    write_json(os.path.join(args.output_root, "manifest.json"), manifest)

    with open(os.path.join(args.output_root, "README.md"), "w", encoding="utf-8") as f:
        f.write(
            "# Strict Large Non-Video Split\n\n"
            "Generated by `tools/build_large_train_dev_split.py`.\n\n"
            "- `train_fit/`: router training split (used for fitting).\n"
            "- `dev/`: tuning split (used for parameter selection only).\n"
            "- `test/`: final evaluation split (do not tune on this).\n"
            "- `full/`: train_fit + dev + test for accounting only.\n\n"
            "- `router_train_fit_4class.json`: training data for distilbert/vib router.\n"
            "- `router_train_fit_t5_4class.json`: training data for t5/vib router.\n"
            "- `router_dev_4class.json`: dev labels for routing analysis.\n"
            "- `router_dev_t5_4class.json`: dev labels with T5 prompt format.\n\n"
            "Use dev for tuning and run test once for final report.\n"
        )

    print(json.dumps(manifest["counts"], indent=2, ensure_ascii=False))
    print(f"\nSaved strict split to {args.output_root}")


if __name__ == "__main__":
    main()
