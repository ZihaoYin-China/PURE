"""
Build a local HotpotQA retrieval corpus from the bundled HotpotQA distractor
raw file.

This is the safer fallback when dataset/hotpotqa/text/*.txt IDs do not align
with the current query annotations and Wikipedia API downloads are unavailable
or map IDs to unrelated current pages.
"""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

DEFAULT_QUERY_PATHS = [
    "dataset/query/hotpotqa.json",
    "dataset/query_nonvideo_split/train/hotpotqa.json",
    "dataset/query_nonvideo_split/test/hotpotqa.json",
    "dataset/query_nonvideo_split/full/hotpotqa.json",
    "dataset/query_nonvideo_large_strict_d40/test/hotpotqa.json",
    "dataset/query_nonvideo_large_strict_d40/dev/hotpotqa.json",
    "dataset/query_nonvideo_large_strict_d40/train_fit/hotpotqa.json",
]


def _slug(title):
    base = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_").lower() or "untitled"
    base = base[:80]
    digest = hashlib.md5(title.encode("utf-8")).hexdigest()[:10]
    return f"{base}_{digest}.txt"


def _doc_text(title, sentences):
    text = " ".join(str(s).strip() for s in sentences if str(s).strip())
    return f"Title: {title}\nText: {text}\n"


def _support_titles(raw_row):
    titles = []
    for title, _sent_idx in raw_row.get("supporting_facts", []):
        if title not in titles:
            titles.append(title)
    return titles


def _mirror_query_path(query_path, output_query_dir):
    parts = Path(query_path).parts
    if parts and parts[0] == "dataset":
        parts = parts[1:]
    return Path(output_query_dir).joinpath(*parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_path",
        default="dataset/raw_query_sources/hotpot_dev_distractor_v1.json",
    )
    parser.add_argument(
        "--output_text_dir",
        default="dataset/hotpotqa/raw_context_text",
    )
    parser.add_argument(
        "--output_query_dir",
        default="dataset/query_hotpotqa_raw_context",
    )
    parser.add_argument(
        "--manifest_path",
        default="dataset/hotpotqa/raw_context_manifest.json",
    )
    parser.add_argument(
        "--query_paths",
        default=",".join(DEFAULT_QUERY_PATHS),
        help="Comma-separated query JSON paths to mirror with raw-context gt_texts.",
    )
    args = parser.parse_args()

    raw_path = Path(args.raw_path)
    output_text_dir = Path(args.output_text_dir)
    output_query_dir = Path(args.output_query_dir)
    manifest_path = Path(args.manifest_path)

    raw_data = json.loads(raw_path.read_text(encoding="utf-8"))
    output_text_dir.mkdir(parents=True, exist_ok=True)
    output_query_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    title_to_path = {}
    title_to_text = {}
    for row in raw_data:
        for title, sentences in row.get("context", []):
            if title not in title_to_path:
                rel_path = Path(args.output_text_dir).joinpath(_slug(title)).as_posix()
                title_to_path[title] = rel_path
                title_to_text[title] = _doc_text(title, sentences)

    for title, rel_path in title_to_path.items():
        Path(rel_path).write_text(title_to_text[title], encoding="utf-8")

    raw_by_question = {row.get("question"): row for row in raw_data}
    mirrored = []
    for query_path in [p for p in args.query_paths.split(",") if p.strip()]:
        qpath = Path(query_path.strip())
        if not qpath.exists():
            continue
        rows = json.loads(qpath.read_text(encoding="utf-8"))
        matched = 0
        with_support = 0
        updated_rows = []
        for row in rows:
            new_row = dict(row)
            raw_row = raw_by_question.get(row.get("question"))
            if raw_row:
                matched += 1
                gt_texts = [title_to_path[t] for t in _support_titles(raw_row) if t in title_to_path]
                if gt_texts:
                    with_support += 1
                    new_row["gt_texts"] = gt_texts
                new_row["hotpot_raw_context_available"] = bool(gt_texts)
            else:
                new_row["hotpot_raw_context_available"] = False
            updated_rows.append(new_row)

        out_path = _mirror_query_path(qpath.as_posix(), output_query_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(updated_rows, indent=2, ensure_ascii=False), encoding="utf-8")
        mirrored.append(
            {
                "source": qpath.as_posix(),
                "output": out_path.as_posix(),
                "rows": len(rows),
                "matched_questions": matched,
                "rows_with_support_gt": with_support,
            }
        )

    manifest = {
        "raw_path": raw_path.as_posix(),
        "output_text_dir": output_text_dir.as_posix(),
        "num_documents": len(title_to_path),
        "mirrored_queries": mirrored,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {len(title_to_path)} raw-context docs to {output_text_dir}")
    for item in mirrored:
        print(
            f"Mirrored {item['source']} -> {item['output']}: "
            f"matched {item['matched_questions']}/{item['rows']}, "
            f"support gt {item['rows_with_support_gt']}/{item['rows']}"
        )
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
