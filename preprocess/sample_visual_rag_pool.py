#!/usr/bin/env python
"""Build a small VisualRAG image retrieval pool by sampling images per category."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def dump_pickle(obj: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def stable_rng(seed: int, key: str) -> random.Random:
    digest = hashlib.md5(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return random.Random(int(digest[:8], 16))


def image_category(image_id: str) -> str:
    return os.path.dirname(image_id)


def collect_gt_images(query_file: str) -> dict[str, list[str]]:
    rows = load_json(query_file)
    by_category: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        for image_id in row.get("gt_images") or []:
            category = image_category(image_id)
            if image_id not in by_category[category]:
                by_category[category].append(image_id)
    return by_category


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample a VisualRAG retrieval pool with a fixed number of images per "
            "category. By default the query GT image is kept when available, then "
            "the remaining slots are filled by deterministic random sampling."
        )
    )
    parser.add_argument("--image_meta", default="dataset/visual_rag/images.json")
    parser.add_argument("--query_file", default="dataset/query_ood/visual_rag.json")
    parser.add_argument("--image_feats", default="eval/features/image/visual_rag_full.pkl")
    parser.add_argument(
        "--imgcap_feats", default="eval/features/image/visual_rag_full_imgcap.pkl"
    )
    parser.add_argument(
        "--output_image_feats", default="eval/features/image/visual_rag_sample5_gt.pkl"
    )
    parser.add_argument(
        "--output_imgcap_feats",
        default="eval/features/image/visual_rag_sample5_gt_imgcap.pkl",
    )
    parser.add_argument(
        "--output_meta", default="dataset/visual_rag/images_sample5_gt.json"
    )
    parser.add_argument(
        "--output_manifest",
        default="eval/features/image/visual_rag_sample5_gt_manifest.json",
    )
    parser.add_argument("--num_per_category", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include_gt",
        type=int,
        default=1,
        help="Keep GT images first when they exist in the feature pool.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_per_category <= 0:
        raise ValueError("--num_per_category must be positive")

    image_meta = load_json(args.image_meta)
    image_feats = load_pickle(args.image_feats)
    imgcap_feats = load_pickle(args.imgcap_feats)

    available_ids = set(image_feats) & set(imgcap_feats)
    groups: dict[str, list[str]] = defaultdict(list)
    for image_id in sorted(available_ids):
        groups[image_category(image_id)].append(image_id)

    gt_by_category = collect_gt_images(args.query_file) if args.include_gt else {}

    selected_by_category: dict[str, list[str]] = {}
    missing_gt: list[str] = []
    truncated_gt_categories: list[str] = []

    for category in sorted(groups):
        candidates = groups[category]
        selected: list[str] = []

        if args.include_gt:
            for image_id in gt_by_category.get(category, []):
                if image_id in available_ids and image_id not in selected:
                    selected.append(image_id)
                elif image_id not in available_ids:
                    missing_gt.append(image_id)

        if len(selected) > args.num_per_category:
            selected = selected[: args.num_per_category]
            truncated_gt_categories.append(category)

        remaining = [image_id for image_id in candidates if image_id not in selected]
        rng = stable_rng(args.seed, category)
        rng.shuffle(remaining)
        selected.extend(remaining[: max(0, args.num_per_category - len(selected))])
        selected_by_category[category] = selected

    selected_ids = [
        image_id
        for category in sorted(selected_by_category)
        for image_id in selected_by_category[category]
    ]

    sampled_image_feats = {image_id: image_feats[image_id] for image_id in selected_ids}
    sampled_imgcap_feats = {image_id: imgcap_feats[image_id] for image_id in selected_ids}
    sampled_meta = {image_id: image_meta.get(image_id, {}) for image_id in selected_ids}

    dump_pickle(sampled_image_feats, args.output_image_feats)
    dump_pickle(sampled_imgcap_feats, args.output_imgcap_feats)
    dump_json(sampled_meta, args.output_meta)

    query_rows = load_json(args.query_file)
    selected_id_set = set(selected_ids)
    total_gt_refs = 0
    selected_gt_refs = 0
    unique_gt_ids: set[str] = set()
    queries_without_pool_overlap = []
    for row in query_rows:
        gt_images = row.get("gt_images") or []
        total_gt_refs += len(gt_images)
        selected_gt_refs += sum(image_id in selected_id_set for image_id in gt_images)
        unique_gt_ids.update(gt_images)
        candidate_images = set(row.get("candidate_images") or [])
        if candidate_images and not (candidate_images & selected_id_set):
            queries_without_pool_overlap.append(row.get("index"))

    manifest = {
        "image_meta": args.image_meta,
        "query_file": args.query_file,
        "image_feats": args.image_feats,
        "imgcap_feats": args.imgcap_feats,
        "output_image_feats": args.output_image_feats,
        "output_imgcap_feats": args.output_imgcap_feats,
        "output_meta": args.output_meta,
        "num_per_category": args.num_per_category,
        "seed": args.seed,
        "include_gt": bool(args.include_gt),
        "available_images": len(available_ids),
        "categories": len(groups),
        "selected_images": len(selected_ids),
        "min_selected_per_category": min(len(v) for v in selected_by_category.values()),
        "max_selected_per_category": max(len(v) for v in selected_by_category.values()),
        "total_gt_references": total_gt_refs,
        "unique_gt_images": len(unique_gt_ids),
        "selected_gt_references": selected_gt_refs,
        "unselected_gt_references": total_gt_refs - selected_gt_refs,
        "missing_gt_count": len(missing_gt),
        "missing_gt_images": missing_gt,
        "truncated_gt_category_count": len(truncated_gt_categories),
        "truncated_gt_categories": truncated_gt_categories,
        "queries_without_pool_overlap_count": len(queries_without_pool_overlap),
        "queries_without_pool_overlap": queries_without_pool_overlap,
        "selected_by_category": selected_by_category,
    }
    dump_json(manifest, args.output_manifest)

    print("Saved sampled VisualRAG pool:")
    print(f"  image feats : {args.output_image_feats}")
    print(f"  imgcap feats: {args.output_imgcap_feats}")
    print(f"  meta        : {args.output_meta}")
    print(f"  manifest    : {args.output_manifest}")
    print(f"  categories  : {len(groups)}")
    print(f"  images      : {len(selected_ids)}")
    print(f"  missing GT  : {len(missing_gt)}")


if __name__ == "__main__":
    main()
