#!/usr/bin/env python
import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
from collections import Counter, defaultdict
from math import floor
from urllib.parse import urlparse

import requests

try:
    from route.gpt.prompt import ROUTER_PROMPT
except Exception:
    ROUTER_PROMPT = None


TARGETS = ["mmlu", "squad", "natural_questions", "hotpotqa", "webqa"]

TARGET_TOTALS = {
    "mmlu": 2500,
    "squad": 2500,
    "natural_questions": 2500,
    "hotpotqa": 2500,
    "webqa": 2000,
}

DATASET_TO_ROUTE = {
    "mmlu": "no",
    "squad": "paragraph",
    "natural_questions": "paragraph",
    "hotpotqa": "document",
    "webqa": "image",
}

SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
HOTPOT_DEV_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
MMLU_DATA_URL = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"

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


def download_file(url, output_path, allow_download):
    if os.path.isfile(output_path):
        return output_path
    if not allow_download:
        raise FileNotFoundError(
            f"Missing {output_path}. Re-run with --allow_download to fetch {url}"
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=os.path.basename(output_path) + ".",
        suffix=".tmp",
        dir=os.path.dirname(output_path),
    )
    os.close(fd)

    try:
        with requests.get(url, stream=True, timeout=30) as response:
            response.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        shutil.move(tmp_path, output_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return output_path


def read_json_maybe_gzip(path):
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    return read_json(path)


def normalize_route(sample, target):
    value = sample.get("gt_retrieval")
    if value in ROUTE_ID_TO_LABEL:
        return ROUTE_ID_TO_LABEL[value]
    value = str(value or "").lower().strip()
    if value in DATASET_TO_ROUTE.values():
        return value
    return DATASET_TO_ROUTE[target]


def infer_target(sample):
    source = str(sample.get("source", "")).lower().strip()
    if source in VIDEO_SOURCES:
        return None
    if source in TARGETS:
        return source

    index = str(sample.get("index", "")).lower()
    if index.startswith("nq_"):
        return "natural_questions"
    for target in ["mmlu", "squad", "hotpotqa", "webqa"]:
        if index.startswith(f"{target}_") or index.startswith(target):
            return target
    return None


def normalize_existing_sample(sample, target, split="full"):
    row = dict(sample)
    row["source"] = target
    row["gt_retrieval"] = normalize_route(sample, target)
    row["split"] = split
    if target == "hotpotqa" and "gt_texts" in row:
        row["gt_texts"] = [
            str(path).replace(
                "dataset/LongRAG/hotpot_qa_corpus/text/",
                "dataset/hotpotqa/text/",
            )
            for path in (row.get("gt_texts") or [])
        ]
    return row


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


def dedupe_keep_order(rows):
    seen = set()
    out = []
    for row in rows:
        key = str(row.get("index", ""))
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def merge_official_released_rows(split_dir, train_data_path):
    by_target = defaultdict(list)

    full_dir = os.path.join(split_dir, "full")
    if os.path.isdir(full_dir):
        for target in TARGETS:
            path = os.path.join(full_dir, f"{target}.json")
            if os.path.isfile(path):
                by_target[target].extend(
                    normalize_existing_sample(row, target) for row in read_json(path)
                )
        return by_target

    if os.path.isfile(train_data_path):
        for row in read_json(train_data_path):
            target = infer_target(row)
            if target in TARGETS:
                by_target[target].append(normalize_existing_sample(row, target, "train"))

    query_dir = "dataset/query"
    for target in TARGETS:
        path = os.path.join(query_dir, f"{target}.json")
        if os.path.isfile(path):
            by_target[target].extend(
                normalize_existing_sample(row, target, "test") for row in read_json(path)
            )

    return by_target


def first_answer(answers):
    if isinstance(answers, list):
        return str(answers[0]) if answers else ""
    return str(answers or "")


def write_ctx_if_missing(ctx, output_dir, id_keys):
    for key in id_keys:
        if key in ctx and ctx[key] is not None:
            doc_id = str(ctx[key])
            break
    else:
        raw = f"{ctx.get('title', '')}\n{ctx.get('text', '')}".encode(
            "utf-8",
            errors="ignore",
        )
        doc_id = hashlib.md5(raw).hexdigest()

    text = str(ctx.get("text", "")).strip()
    if not text:
        return None

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{doc_id}.txt")
    if not os.path.isfile(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    return path


def build_dpr_qa_rows(raw_rows, target, text_dir, existing_ids, limit):
    out = []
    prefix = "nq_" if target == "natural_questions" else "squad"
    id_keys = ("psg_id", "passage_id", "id")

    for idx, item in enumerate(raw_rows):
        row_id = f"{prefix}{idx}" if target == "squad" else f"{prefix}{idx}"
        if row_id in existing_ids:
            continue

        positive_ctxs = item.get("positive_ctxs") or []
        gt_texts = []
        for ctx in positive_ctxs:
            path = write_ctx_if_missing(ctx, text_dir, id_keys=id_keys)
            if path:
                gt_texts.append(path)

        question = str(item.get("question", "")).strip()
        if not question or not gt_texts:
            continue

        answers = item.get("answers") or []
        row = {
            "index": row_id,
            "question": question,
            "answer": first_answer(answers),
            "aliases": [str(x) for x in answers[1:]] if isinstance(answers, list) else [],
            "source": target,
            "gt_texts": gt_texts,
            "gt_retrieval": DATASET_TO_ROUTE[target],
            "split": "augmented",
        }
        out.append(row)
        existing_ids.add(row_id)
        if len(out) >= limit:
            break

    return out


def build_official_squad_rows(raw_data, text_dir, existing_ids, limit):
    out = []
    for article_idx, article in enumerate(raw_data.get("data", [])):
        title = str(article.get("title", "squad")).strip()
        for para_idx, paragraph in enumerate(article.get("paragraphs", [])):
            context = str(paragraph.get("context", "")).strip()
            if not context:
                continue

            digest = hashlib.md5(
                f"{title}\n{para_idx}\n{context}".encode("utf-8", errors="ignore")
            ).hexdigest()[:12]
            doc_id = f"large_squad_{safe_slug(title, 32)}_{para_idx}_{digest}"
            os.makedirs(text_dir, exist_ok=True)
            doc_path = os.path.join(text_dir, f"{doc_id}.txt")
            if not os.path.isfile(doc_path):
                with open(doc_path, "w", encoding="utf-8") as f:
                    f.write(context)

            for qa in paragraph.get("qas", []):
                raw_id = str(qa.get("id", "")).strip()
                row_id = f"squad_large_{raw_id}" if raw_id else f"squad_large_{article_idx}_{para_idx}_{len(out)}"
                if row_id in existing_ids:
                    continue
                question = str(qa.get("question", "")).strip()
                answers = qa.get("answers") or []
                if not question or not answers:
                    continue
                answer_texts = [
                    str(answer.get("text", "")).strip()
                    for answer in answers
                    if str(answer.get("text", "")).strip()
                ]
                if not answer_texts:
                    continue

                row = {
                    "index": row_id,
                    "question": question,
                    "answer": answer_texts[0],
                    "aliases": answer_texts[1:],
                    "source": "squad",
                    "gt_texts": [doc_path],
                    "gt_retrieval": "paragraph",
                    "split": "augmented",
                }
                out.append(row)
                existing_ids.add(row_id)
                if len(out) >= limit:
                    return out
    return out


def build_squad_rows(raw_data, text_dir, existing_ids, limit):
    if isinstance(raw_data, list):
        return build_dpr_qa_rows(raw_data, "squad", text_dir, existing_ids, limit)
    if isinstance(raw_data, dict) and "data" in raw_data:
        return build_official_squad_rows(raw_data, text_dir, existing_ids, limit)
    raise ValueError("Unsupported SQuAD raw format")


def download_mmlu_csvs(data_url, output_dir, allow_download):
    os.makedirs(output_dir, exist_ok=True)
    local_csvs = [
        os.path.join(output_dir, fname)
        for fname in sorted(os.listdir(output_dir))
        if fname.endswith(".csv")
    ]
    if local_csvs:
        return local_csvs
    if not allow_download:
        raise FileNotFoundError(
            f"Missing MMLU CSVs in {output_dir}. Re-run with --allow_download to fetch {data_url}"
        )

    tar_path = os.path.join(os.path.dirname(output_dir), "mmlu_data.tar")
    download_file(data_url, tar_path, allow_download=True)

    with tarfile.open(tar_path, "r") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            normalized_name = member.name.replace("\\", "/")
            if "/test/" not in normalized_name or not normalized_name.endswith(".csv"):
                continue
            member.name = os.path.basename(normalized_name)
            tar.extract(member, output_dir)

    return [
        os.path.join(output_dir, fname)
        for fname in sorted(os.listdir(output_dir))
        if fname.endswith(".csv")
    ]


def build_mmlu_rows(csv_paths, existing_ids, limit):
    import csv

    out = []
    for csv_path in sorted(csv_paths):
        subject = os.path.basename(csv_path).replace("_test.csv", "")
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            for row_idx, row in enumerate(reader):
                if len(row) < 6:
                    continue
                question, choice_a, choice_b, choice_c, choice_d, answer = row[:6]
                answer = str(answer).strip().upper()
                if answer not in {"A", "B", "C", "D"}:
                    continue
                digest = hashlib.md5(
                    f"{subject}\n{row_idx}\n{question}".encode("utf-8", errors="ignore")
                ).hexdigest()[:12]
                row_id = f"mmlu_large_{safe_slug(subject, 40)}_{row_idx}_{digest}"
                if row_id in existing_ids:
                    continue
                formatted_question = (
                    f"{str(question).strip()}\n"
                    "Choices:\n"
                    f"A) {str(choice_a).strip()}\n"
                    f"B) {str(choice_b).strip()}\n"
                    f"C) {str(choice_c).strip()}\n"
                    f"D) {str(choice_d).strip()}"
                )
                out.append(
                    {
                        "index": row_id,
                        "question": formatted_question,
                        "answer": answer,
                        "source": "mmlu",
                        "subject": subject,
                        "gt_retrieval": "no",
                        "split": "augmented",
                    }
                )
                existing_ids.add(row_id)
                if len(out) >= limit:
                    return out
    return out


def safe_slug(text, max_len=48):
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")
    return text[:max_len] or "doc"


def write_hotpot_context(raw_id, title, sentences, text_dir):
    text = " ".join(str(sentence).strip() for sentence in sentences if str(sentence).strip())
    if not text:
        return None
    digest = hashlib.md5(f"{raw_id}\n{title}".encode("utf-8", errors="ignore")).hexdigest()[:12]
    doc_id = f"large_hotpotqa_{safe_slug(raw_id, 32)}_{safe_slug(title, 32)}_{digest}"
    os.makedirs(text_dir, exist_ok=True)
    path = os.path.join(text_dir, f"{doc_id}.txt")
    if not os.path.isfile(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"{title}\n{text}")
    return path


def build_hotpot_rows(raw_rows, text_dir, existing_ids, limit):
    out = []
    for idx, item in enumerate(raw_rows):
        row_id = f"hotpotqa_{idx}"
        if row_id in existing_ids:
            continue

        context_map = {
            str(title): sentences
            for title, sentences in item.get("context", [])
        }
        support_titles = []
        for fact in item.get("supporting_facts", []):
            if fact and str(fact[0]) not in support_titles:
                support_titles.append(str(fact[0]))

        gt_texts = []
        for title in support_titles:
            if title not in context_map:
                continue
            path = write_hotpot_context(item.get("_id", row_id), title, context_map[title], text_dir)
            if path:
                gt_texts.append(path)

        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if not question or not answer or not gt_texts:
            continue

        row = {
            "index": row_id,
            "question": question,
            "answer": answer,
            "source": "hotpotqa",
            "gt_texts": gt_texts,
            "gt_retrieval": "document",
            "split": "augmented",
        }
        out.append(row)
        existing_ids.add(row_id)
        if len(out) >= limit:
            break

    return out


def top_up_target(by_target, target, candidates):
    rows = dedupe_keep_order(by_target[target])
    need = max(0, TARGET_TOTALS[target] - len(rows))
    if need <= 0:
        by_target[target] = rows[: TARGET_TOTALS[target]]
        return 0
    by_target[target] = dedupe_keep_order(rows + candidates[:need])
    return max(0, TARGET_TOTALS[target] - len(by_target[target]))


def stable_sort_key(row):
    key = str(row.get("index", "")) or str(row.get("question", ""))
    digest = hashlib.md5(key.encode("utf-8", errors="ignore")).hexdigest()
    return digest, key


def mmlu_subject(row):
    subject = str(row.get("subject", "")).strip()
    if subject:
        return safe_slug(subject, 64)

    index = str(row.get("index", ""))
    match = re.match(r"mmlu_large_(.+)_\d+_[0-9a-f]{12}$", index)
    if match:
        return match.group(1)

    return "released_mmlu"


def stratified_mmlu_split(rows, train_ratio=0.3):
    groups = defaultdict(list)
    for row in rows:
        groups[mmlu_subject(row)].append(dict(row))

    total = sum(len(group_rows) for group_rows in groups.values())
    target_train = int(round(total * train_ratio))

    quotas = {}
    remainders = []
    for subject, group_rows in groups.items():
        exact = len(group_rows) * train_ratio
        base = floor(exact)
        quotas[subject] = base
        remainders.append((exact - base, subject))

    diff = target_train - sum(quotas.values())
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
    test_rows = []
    for subject in sorted(groups):
        group_rows = sorted(groups[subject], key=stable_sort_key)
        train_n = quotas[subject]
        train_rows.extend(group_rows[:train_n])
        test_rows.extend(group_rows[train_n:])

    train_rows = sorted(train_rows, key=stable_sort_key)
    test_rows = sorted(test_rows, key=stable_sort_key)

    for row in train_rows:
        row["split"] = "train"
    for row in test_rows:
        row["split"] = "test"

    return train_rows, test_rows


def deterministic_split(rows, target):
    rows = dedupe_keep_order(rows)
    total = len(rows)

    if target == "mmlu":
        choice_rows = [
            dict(row) for row in rows
            if "choices:" in str(row.get("question", "")).lower()
        ]
        return stratified_mmlu_split(choice_rows, train_ratio=0.3)

    train_count = int(round(total * 0.3))
    train_rows = []
    test_rows = []

    for idx, row in enumerate(rows):
        copied = dict(row)
        if idx < train_count:
            copied["split"] = "train"
            train_rows.append(copied)
        else:
            copied["split"] = "test"
            test_rows.append(copied)
    return train_rows, test_rows


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
        description="Build a larger non-video query split from released PURE rows plus public QA sources."
    )
    parser.add_argument("--output_dir", default="dataset/query_nonvideo_large")
    parser.add_argument("--released_split_dir", default="dataset/query_nonvideo_split")
    parser.add_argument("--train_data", default="route/train/data/train_data_distilbert.json")
    parser.add_argument("--download_dir", default="dataset/raw_query_sources")
    parser.add_argument("--allow_download", action="store_true")
    parser.add_argument("--squad_url", default=SQUAD_URL)
    parser.add_argument("--hotpot_url", default=HOTPOT_DEV_URL)
    parser.add_argument("--mmlu_data_url", default=MMLU_DATA_URL)
    parser.add_argument("--nq_raw", default="dataset/natural_questions/biencoder-nq-dev.json")
    parser.add_argument("--squad_text_dir", default="dataset/squad/text")
    parser.add_argument("--nq_text_dir", default="dataset/natural_questions/text")
    parser.add_argument("--hotpot_text_dir", default="dataset/hotpotqa/text")
    args = parser.parse_args()

    by_target = merge_official_released_rows(args.released_split_dir, args.train_data)
    warnings = []

    # MMLU released rows use the small dev split; optionally top up from original MMLU test CSVs.
    try:
        existing = {str(row.get("index", "")) for row in by_target["mmlu"]}
        existing_eval_ready = sum(
            "choices:" in str(row.get("question", "")).lower()
            for row in by_target["mmlu"]
        )
        mmlu_dir = os.path.join(args.download_dir, "mmlu_test")
        csv_paths = download_mmlu_csvs(args.mmlu_data_url, mmlu_dir, args.allow_download)
        by_target["mmlu"].extend(
            build_mmlu_rows(
                csv_paths=csv_paths,
                existing_ids=existing,
                limit=max(0, TARGET_TOTALS["mmlu"] - existing_eval_ready),
            )
        )
    except Exception as exc:
        warnings.append(f"MMLU expansion skipped: {exc}")

    # NQ can be expanded from the local DPR dev file already present in this repo.
    if os.path.isfile(args.nq_raw):
        existing = {str(row.get("index", "")) for row in by_target["natural_questions"]}
        nq_rows = read_json(args.nq_raw)
        by_target["natural_questions"].extend(
            build_dpr_qa_rows(
                raw_rows=nq_rows,
                target="natural_questions",
                text_dir=args.nq_text_dir,
                existing_ids=existing,
                limit=max(0, TARGET_TOTALS["natural_questions"] - len(by_target["natural_questions"])),
            )
        )
    else:
        warnings.append(f"NQ raw file not found: {args.nq_raw}")

    # SQuAD raw file is downloaded by the original script and then removed; fetch it again if allowed.
    squad_name = os.path.basename(urlparse(args.squad_url).path)
    squad_path = os.path.join(args.download_dir, squad_name)
    try:
        download_file(args.squad_url, squad_path, args.allow_download)
        existing = {str(row.get("index", "")) for row in by_target["squad"]}
        squad_rows = read_json_maybe_gzip(squad_path)
        by_target["squad"].extend(
            build_squad_rows(
                raw_data=squad_rows,
                text_dir=args.squad_text_dir,
                existing_ids=existing,
                limit=max(0, TARGET_TOTALS["squad"] - len(by_target["squad"])),
            )
        )
    except Exception as exc:
        warnings.append(f"SQuAD expansion skipped: {exc}")

    # HotpotQA official dev-distractor contains answerable QA and supporting docs.
    hotpot_name = os.path.basename(urlparse(args.hotpot_url).path)
    hotpot_path = os.path.join(args.download_dir, hotpot_name)
    try:
        download_file(args.hotpot_url, hotpot_path, args.allow_download)
        existing = {str(row.get("index", "")) for row in by_target["hotpotqa"]}
        hotpot_rows = read_json(hotpot_path)
        by_target["hotpotqa"].extend(
            build_hotpot_rows(
                raw_rows=hotpot_rows,
                text_dir=args.hotpot_text_dir,
                existing_ids=existing,
                limit=max(0, TARGET_TOTALS["hotpotqa"] - len(by_target["hotpotqa"])),
            )
        )
    except Exception as exc:
        warnings.append(f"HotpotQA expansion skipped: {exc}")

    for target in TARGETS:
        rows = dedupe_keep_order(by_target[target])
        if target == "mmlu":
            rows = [
                row for row in rows
                if "choices:" in str(row.get("question", "")).lower()
            ]
        by_target[target] = rows[: TARGET_TOTALS[target]]

    manifest = {
        "description": (
            "Larger non-video PURE-style split. It keeps released rows first, "
            "then tops up NQ/SQuAD/HotpotQA from public QA sources when available."
        ),
        "targets": TARGETS,
        "target_totals": TARGET_TOTALS,
        "warnings": warnings,
        "source_files": {
            "released_split_dir": args.released_split_dir,
            "train_data": args.train_data,
            "nq_raw": args.nq_raw,
            "squad_url": args.squad_url,
            "hotpot_url": args.hotpot_url,
            "mmlu_data_url": args.mmlu_data_url,
        },
        "counts": {},
        "notes": [
            "MMLU released rows correspond to the small MMLU dev split; larger MMLU rows are topped up from original MMLU test CSVs when available.",
            "WebQA is kept at the released 2000 real QA rows and is not caption-augmented.",
            "For new SQuAD/NQ/HotpotQA questions, regenerate query features before evaluation.",
            "If HotpotQA was expanded, regenerate the HotpotQA text corpus features so new support documents are retrievable.",
        ],
    }

    router_train = []
    router_test = []
    router_full = []
    router_train_t5 = []

    for target in TARGETS:
        train_rows, test_rows = deterministic_split(by_target[target], target)
        full_rows = dedupe_keep_order(train_rows + test_rows)

        write_json(os.path.join(args.output_dir, "train", f"{target}.json"), train_rows)
        write_json(os.path.join(args.output_dir, "test", f"{target}.json"), test_rows)
        write_json(os.path.join(args.output_dir, "full", f"{target}.json"), full_rows)

        router_train.extend(to_router_sample(row, target) for row in train_rows)
        router_test.extend(to_router_sample(row, target) for row in test_rows)
        router_full.extend(to_router_sample(row, target) for row in full_rows)
        router_train_t5.extend(to_t5_router_sample(row, target) for row in train_rows)

        manifest["counts"][target] = {
            "train": len(train_rows),
            "test": len(test_rows),
            "full": len(full_rows),
            "routes": {
                "train": dict(Counter(row["gt_retrieval"] for row in train_rows)),
                "test": dict(Counter(row["gt_retrieval"] for row in test_rows)),
            },
            "eval_ready": {
                "train": sum(eval_ready(row, target) for row in train_rows),
                "test": sum(eval_ready(row, target) for row in test_rows),
                "full": sum(eval_ready(row, target) for row in full_rows),
            },
        }

    write_json(os.path.join(args.output_dir, "router_train_4class.json"), router_train)
    write_json(os.path.join(args.output_dir, "router_train_t5_4class.json"), router_train_t5)
    write_json(os.path.join(args.output_dir, "router_test_4class.json"), router_test)
    write_json(os.path.join(args.output_dir, "router_full_4class.json"), router_full)
    write_json(os.path.join(args.output_dir, "manifest.json"), manifest)

    with open(os.path.join(args.output_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(
            "# Larger Non-Video Query Split\n\n"
            "Generated by `tools/build_large_nonvideo_query_split.py`.\n\n"
            "- `train/`: router training split.\n"
            "- `test/`: final QA evaluation split.\n"
            "- `full/`: train + test for accounting only.\n"
            "- `router_train_4class.json`: concatenated 4-class router training file.\n\n"
            "- `router_train_t5_4class.json`: T5 prompt-formatted router training file.\n\n"
            "Do not report results on `full/` as final test results.\n"
        )

    print(json.dumps(manifest["counts"], indent=2, ensure_ascii=False))
    if warnings:
        print("\nWARNINGS:")
        for warning in warnings:
            print(f"- {warning}")
    print(f"\nSaved larger split to {args.output_dir}")


if __name__ == "__main__":
    main()
