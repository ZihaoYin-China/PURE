"""
Re-extract HotpotQA text features using BGE after the Wikipedia corpus fix.

Usage:
    python preprocess/re_extract_hotpotqa_features.py
"""

import os
import sys
import json
import pickle

BATCH_SIZE = 64

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "preprocess"))
from bge_encoder import encode, split_text_into_chunks

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "dataset/hotpotqa/text")
OUTPUT_PATH = os.path.join(BASE_DIR, "eval/features/text/hotpotqa.pkl")
MAX_TOKENS = 512


def main():
    all_files = sorted(
        [f for f in os.listdir(INPUT_PATH) if f.endswith(".txt")]
    )
    print(f"Found {len(all_files)} text files in {INPUT_PATH}")

    filepaths = []
    texts = []

    for filename in all_files:
        filepath = os.path.join(INPUT_PATH, filename)
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        chunks = split_text_into_chunks(text, max_tokens=MAX_TOKENS)
        for i, chunk in enumerate(chunks):
            chunk_filepath = f"{filepath}_part{i + 1}" if len(chunks) > 1 else filepath
            texts.append(chunk)
            filepaths.append(chunk_filepath)

    print(f"Total text chunks: {len(texts)}")
    print(f"Encoding (batch_size={BATCH_SIZE})...")

    encoded = encode(texts, batch_size=BATCH_SIZE, normalize=True, show_progress=True)

    features = {}
    for fpath, feat in zip(filepaths, encoded):
        features[fpath] = feat

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(features, f)

    print(f"Saved {len(features)} feature vectors to {OUTPUT_PATH}")

    # Verify GT coverage
    qpath = os.path.join(BASE_DIR, "dataset/query/hotpotqa.json")
    if os.path.exists(qpath):
        with open(qpath, "r") as f:
            data = json.load(f)

        feature_keys_base = set()
        for k in features:
            feature_keys_base.add(k.rsplit("_part", 1)[0])

        total_gt = 0
        matched_gt = 0
        for row in data:
            for gt_path in row.get("gt_texts", []):
                total_gt += 1
                if gt_path in feature_keys_base:
                    matched_gt += 1

        print(f"\nGT feature coverage: {matched_gt}/{total_gt} "
              f"({100 * matched_gt / max(total_gt, 1):.1f}%)")


if __name__ == "__main__":
    main()
