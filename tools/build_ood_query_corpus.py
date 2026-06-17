#!/usr/bin/env python
"""Build PURE-style OOD query files and lightweight corpora.

This script converts the raw OOD datasets used by PURE into the
project's JSON query schema:

  index, source, question, answer, gt_retrieval

When retrieval is needed it also writes gt_texts or gt_images entries. It is
intended as a data preparation step before routing and feature extraction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


CHOICE_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def load_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def stable_hash(text: str, n: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:n]


def sanitize_name(text: str, max_len: int = 80) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")
    return (text or "item")[:max_len]


def flatten_strs(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = value.strip()
        return [value] if value else []
    if isinstance(value, dict):
        out: List[str] = []
        for v in value.values():
            out.extend(flatten_strs(v))
        return out
    if isinstance(value, (list, tuple, set)):
        out = []
        for v in value:
            out.extend(flatten_strs(v))
        return out
    value = str(value).strip()
    return [value] if value else []


def dedup_keep_order(values: Sequence[str]) -> List[str]:
    seen = set()
    out = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def reservoir_sample(
    rows: Iterable[Dict[str, Any]], k: int, seed: int
) -> Tuple[List[Dict[str, Any]], int]:
    rng = random.Random(seed)
    sample: List[Dict[str, Any]] = []
    total = 0
    for total, row in enumerate(rows, start=1):
        if len(sample) < k:
            sample.append(row)
            continue
        j = rng.randrange(total)
        if j < k:
            sample[j] = row
    return sample, total


def split_words(text: str, max_words: int) -> List[str]:
    words = str(text).split()
    if not words:
        return []
    return [
        " ".join(words[i : i + max_words])
        for i in range(0, len(words), max_words)
    ]


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    path.write_text(text, encoding="utf-8")


def build_truthfulqa(args: argparse.Namespace) -> Dict[str, Any]:
    input_path = args.raw_root / "truthfulqa" / "validation.jsonl"
    rows = list(load_jsonl(input_path))

    out = []
    for i, row in enumerate(rows):
        targets = row.get("mc1_targets") or row.get("mc0_targets") or {}
        choices = flatten_strs(targets.get("choices"))
        labels = targets.get("labels") or []
        if not choices or not labels:
            continue

        try:
            gold_idx = list(labels).index(1)
        except ValueError:
            continue
        if gold_idx >= len(choices) or gold_idx >= len(CHOICE_LETTERS):
            continue

        choice_lines = [
            f"{CHOICE_LETTERS[j]}) {choice}" for j, choice in enumerate(choices)
        ]
        out.append(
            {
                "index": f"truthfulqa_{i}",
                "source": "truthfulqa",
                "question": f"{row.get('question', '').strip()}\nChoices:\n"
                + "\n".join(choice_lines),
                "answer": CHOICE_LETTERS[gold_idx],
                "answer_text": choices[gold_idx],
                "answers": [choices[gold_idx]],
                "choices": choices,
                "gt_retrieval": "no",
            }
        )

    output_path = args.query_out / "truthfulqa.json"
    dump_json(output_path, out)
    return {"target": "truthfulqa", "rows": len(out), "output": str(output_path)}


def trivia_contexts(row: Dict[str, Any]) -> Iterator[Tuple[str, str, str]]:
    entity_pages = row.get("entity_pages") or {}
    for idx, text in enumerate(flatten_strs(entity_pages.get("wiki_context"))):
        titles = flatten_strs(entity_pages.get("title"))
        title = titles[idx] if idx < len(titles) else "entity_page"
        yield "entity", title, text

    search_results = row.get("search_results") or {}
    for idx, text in enumerate(flatten_strs(search_results.get("search_context"))):
        titles = flatten_strs(search_results.get("title"))
        title = titles[idx] if idx < len(titles) else "search_result"
        yield "search", title, text


def build_triviaqa(args: argparse.Namespace) -> Dict[str, Any]:
    input_path = args.raw_root / "triviaqa" / "validation_rc.jsonl"
    rows, total = reservoir_sample(load_jsonl(input_path), args.triviaqa_sample_size, args.seed)
    rows.sort(key=lambda x: str(x.get("question_id", "")))

    text_dir = args.corpus_out / "triviaqa" / "text"
    out = []
    written_chunks = set()
    skipped_no_context = 0

    for i, row in enumerate(rows):
        qid = sanitize_name(row.get("question_id") or f"row_{i}")
        gt_texts: List[str] = []
        for ctx_idx, (kind, title, text) in enumerate(trivia_contexts(row)):
            for chunk_idx, chunk in enumerate(split_words(text, args.triviaqa_chunk_words)):
                chunk_id = stable_hash(f"{qid}\n{kind}\n{title}\n{chunk}")
                rel = Path("dataset") / "triviaqa" / "text" / f"{qid}_{ctx_idx:02d}_{chunk_idx:03d}_{chunk_id}.txt"
                abs_path = args.project_root / rel
                if chunk_id not in written_chunks:
                    write_text(abs_path, chunk)
                    written_chunks.add(chunk_id)
                gt_texts.append(str(rel))

        if not gt_texts:
            skipped_no_context += 1
            continue

        answer = row.get("answer") or {}
        refs = dedup_keep_order(
            flatten_strs(answer.get("value"))
            + flatten_strs(answer.get("aliases"))
            + flatten_strs(answer.get("normalized_aliases"))
        )

        out.append(
            {
                "index": f"triviaqa_{qid}",
                "source": "triviaqa",
                "question": str(row.get("question", "")).strip(),
                "answer": refs[0] if refs else "",
                "answers": refs,
                "gt_texts": dedup_keep_order(gt_texts),
                "gt_retrieval": "paragraph",
            }
        )

    output_path = args.query_out / "triviaqa.json"
    dump_json(output_path, out)
    return {
        "target": "triviaqa",
        "raw_rows": total,
        "rows": len(out),
        "skipped_no_context": skipped_no_context,
        "text_files": len(list(text_dir.glob("*.txt"))),
        "output": str(output_path),
    }


def iter_lara_queries(raw_root: Path, exclude_comp: bool) -> Iterator[Dict[str, Any]]:
    query_dir = raw_root / "lara" / "datasets" / "query"
    for path in sorted(query_dir.glob("*.jsonl")):
        parts = path.stem.split("_")
        if len(parts) < 3:
            continue
        length, domain, task = parts[0], parts[1], parts[2]
        if exclude_comp and task == "comp":
            continue
        for row in load_jsonl(path):
            row = dict(row)
            row["_query_file"] = path.name
            row["_length"] = length
            row["_domain"] = domain
            row["_task"] = task
            yield row


def build_lara(args: argparse.Namespace) -> Dict[str, Any]:
    all_rows = list(iter_lara_queries(args.raw_root, exclude_comp=args.lara_exclude_comp))
    rng = random.Random(args.seed)
    if args.lara_sample_size > 0 and len(all_rows) > args.lara_sample_size:
        rows = rng.sample(all_rows, args.lara_sample_size)
    else:
        rows = all_rows
    rows.sort(key=lambda x: (x.get("_length", ""), x.get("_domain", ""), x.get("_task", ""), x.get("file", "")))

    out = []
    skipped_missing_doc = 0

    for i, row in enumerate(rows):
        file_name = str(row.get("file") or "").strip()
        if not file_name:
            skipped_missing_doc += 1
            continue

        src_doc = args.raw_root / "lara" / "datasets" / row["_length"] / row["_domain"] / file_name
        if not src_doc.is_file():
            skipped_missing_doc += 1
            continue

        doc_id = f"{stable_hash(str(src_doc))}_{sanitize_name(file_name)}"
        rel_doc = Path("dataset") / "lara" / "text" / doc_id
        dst_doc = args.project_root / rel_doc
        dst_doc.parent.mkdir(parents=True, exist_ok=True)
        if not dst_doc.exists():
            shutil.copyfile(src_doc, dst_doc)

        answer_refs = dedup_keep_order(flatten_strs(row.get("answer")))
        out.append(
            {
                "index": f"lara_{i:04d}",
                "source": "lara",
                "question": str(row.get("question", "")).strip(),
                "answer": answer_refs[0] if answer_refs else "",
                "answers": answer_refs,
                "gt_texts": [str(rel_doc)],
                "gt_retrieval": "document",
                "lara_length": row.get("_length"),
                "lara_domain": row.get("_domain"),
                "lara_task": row.get("_task"),
                "lara_file": file_name,
            }
        )

    output_path = args.query_out / "lara.json"
    dump_json(output_path, out)
    text_dir = args.corpus_out / "lara" / "text"
    return {
        "target": "lara",
        "eligible_rows": len(all_rows),
        "rows": len(out),
        "skipped_missing_doc": skipped_missing_doc,
        "text_files": len(list(text_dir.glob("*"))),
        "output": str(output_path),
    }


def build_visual_rag(args: argparse.Namespace) -> Dict[str, Any]:
    input_path = args.raw_root / "visual_rag" / "v2_anno.jsonl"
    rows = list(load_jsonl(input_path))
    image_meta: Dict[str, Dict[str, str]] = {}
    out = []

    for i, row in enumerate(rows):
        images = row.get("images") or {}
        if not isinstance(images, dict):
            continue

        candidate_images = []
        gt_images = []
        for rel_img, label in images.items():
            rel = Path("dataset") / "visual_rag" / "images" / rel_img
            rel_str = str(rel)
            candidate_images.append(rel_str)
            if int(label) == 1:
                gt_images.append(rel_str)
            image_meta[rel_str] = {"caption": str(row.get("sn") or rel_img)}

        answers = dedup_keep_order(flatten_strs(row.get("answer")))
        out.append(
            {
                "index": f"visual_rag_{i:04d}",
                "source": "visual_rag",
                "question": str(row.get("question", "")).strip(),
                "answer": answers[0] if answers else "",
                "answers": answers,
                "gt_images": gt_images,
                "candidate_images": candidate_images,
                "gt_retrieval": "image",
                "visual_rag_sn": row.get("sn"),
                "visual_rag_subset": row.get("subset"),
            }
        )

    query_path = args.query_out / "visual_rag.json"
    meta_path = args.project_root / "dataset" / "visual_rag" / "images.json"
    dump_json(query_path, out)
    dump_json(meta_path, image_meta)
    return {
        "target": "visual_rag",
        "rows": len(out),
        "image_refs": len(image_meta),
        "output": str(query_path),
        "image_meta": str(meta_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build OOD query files for PURE.")
    parser.add_argument("--project_root", type=Path, default=Path("."))
    parser.add_argument("--raw_root", type=Path, default=Path("dataset/ood_raw"))
    parser.add_argument("--query_out", type=Path, default=Path("dataset/query_ood"))
    parser.add_argument("--corpus_out", type=Path, default=Path("dataset"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--triviaqa_sample_size", type=int, default=661)
    parser.add_argument("--triviaqa_chunk_words", type=int, default=120)
    parser.add_argument("--lara_sample_size", type=int, default=112)
    parser.add_argument("--lara_exclude_comp", action="store_true", default=True)
    parser.add_argument(
        "--targets",
        type=str,
        default="truthfulqa,triviaqa,lara,visual_rag",
        help="Comma-separated targets to build.",
    )
    args = parser.parse_args()
    args.project_root = args.project_root.resolve()
    if not args.raw_root.is_absolute():
        args.raw_root = args.project_root / args.raw_root
    if not args.query_out.is_absolute():
        args.query_out = args.project_root / args.query_out
    if not args.corpus_out.is_absolute():
        args.corpus_out = args.project_root / args.corpus_out
    return args


def main() -> None:
    args = parse_args()
    builders = {
        "truthfulqa": build_truthfulqa,
        "triviaqa": build_triviaqa,
        "lara": build_lara,
        "visual_rag": build_visual_rag,
    }

    manifest = []
    for target in [x.strip() for x in args.targets.split(",") if x.strip()]:
        if target not in builders:
            raise ValueError(f"Unsupported target: {target}")
        print(f"[build] {target}")
        manifest.append(builders[target](args))

    manifest_path = args.query_out.parent / f"{args.query_out.name}_manifest.json"
    dump_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
