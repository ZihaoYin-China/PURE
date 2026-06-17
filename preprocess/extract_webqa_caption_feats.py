"""
Extract BGE text features for WebQA image captions (title + caption text).

This enables text-to-text retrieval between queries (BGE-encoded) and image
captions (BGE-encoded), replacing the broken InternVideo2 cross-modal retriever.

Usage:
    python preprocess/extract_webqa_caption_feats.py
"""

import json
import os
import sys
import pickle
import numpy as np
import torch

# Reduce memory pressure — 32GB container limit
BATCH_SIZE = 64

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "preprocess"))
from bge_encoder import encode

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "dataset/WebQA/webqa_images.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "eval/features/image/webqa_bge_captions.pkl")


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print(f"Loaded {len(metadata)} image metadata entries")

    img_paths = []
    texts = []

    for img_path, meta in metadata.items():
        title = meta.get("title", "")
        caption = meta.get("caption", "")
        text = f"{title}. {caption}" if title else caption
        if not text.strip():
            text = title or caption or os.path.basename(img_path)

        img_paths.append(img_path)
        texts.append(text)

    print(f"Encoding {len(texts)} caption texts (batch_size={BATCH_SIZE})...")

    encoded = encode(texts, batch_size=BATCH_SIZE, normalize=True, show_progress=True)

    features = {}
    for img_path, feat in zip(img_paths, encoded):
        features[img_path] = feat

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(features, f)

    print(f"Saved {len(features)} caption feature vectors to {OUTPUT_PATH}")

    # Quick diagnostic: check if GT images would be retrievable
    qpath = os.path.join(BASE_DIR, "dataset/query/webqa.json")
    bge_qfeat_path = os.path.join(
        BASE_DIR, "eval/features/query/bge-large/webqa.pkl"
    )

    if os.path.exists(qpath) and os.path.exists(bge_qfeat_path):
        with open(qpath, "r") as f:
            data = json.load(f)

        # Handle CUDA pickles on CPU
        import io as _io
        original = torch.storage._load_from_bytes
        numpy_aliases = {
            "numpy._core": np.core,
            "numpy._core.multiarray": np.core.multiarray,
            "numpy._core.numeric": np.core.numeric,
            "numpy._core.umath": np.core.umath,
        }
        def _load_from_bytes_cpu(b):
            return torch.load(_io.BytesIO(b), map_location="cpu", weights_only=False)
        torch.storage._load_from_bytes = _load_from_bytes_cpu
        for alias, module in numpy_aliases.items():
            if alias not in sys.modules:
                sys.modules[alias] = module
        with open(bge_qfeat_path, "rb") as f:
            bge_qfeats = pickle.load(f)
        torch.storage._load_from_bytes = original

        feat_matrix = np.stack([features[p] for p in img_paths])

        hits_1 = hits_5 = hits_10 = 0
        total = 0
        for row in data:
            qid = row["index"]
            if qid not in bge_qfeats:
                continue
            qvec = np.asarray(bge_qfeats[qid])
            sims = feat_matrix @ qvec
            top_idx = np.argsort(-sims)

            for gt_img in row.get("gt_images", []):
                if gt_img in features:
                    total += 1
                    gt_idx = img_paths.index(gt_img)
                    if gt_idx in top_idx[:1]:
                        hits_1 += 1
                    if gt_idx in top_idx[:5]:
                        hits_5 += 1
                    if gt_idx in top_idx[:10]:
                        hits_10 += 1

        print(f"\nBGE caption retrieval diagnostic:")
        print(f"  Test queries with GT in corpus: {total}")
        if total > 0:
            print(f"  R@1:  {hits_1}/{total} = {hits_1 / total:.4f}")
            print(f"  R@5:  {hits_5}/{total} = {hits_5 / total:.4f}")
            print(f"  R@10: {hits_10}/{total} = {hits_10 / total:.4f}")


if __name__ == "__main__":
    main()
