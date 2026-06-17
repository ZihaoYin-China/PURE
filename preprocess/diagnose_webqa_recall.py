"""
Diagnose WebQA image retrieval recall at different caption-blending alpha values
and top-k settings. Tests whether the cross-modal retrieval gap can be closed by
increasing the caption-text contribution to the image embeddings.
"""

import json
import sys
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "eval"))

from retrieve.retrieve_image import InternImgRetriever


def diagnose():
    queryfeats_path = os.path.join(BASE_DIR, "eval/features/query/internvideo/webqa.pkl")
    imgfeats_path = [os.path.join(BASE_DIR, "eval/features/image/webqa.pkl")]
    imgcapfeats_path = [os.path.join(BASE_DIR, "eval/features/image/webqa_imgcap.pkl")]

    gt_path = os.path.join(BASE_DIR, "dataset/query/webqa.json")
    with open(gt_path, "r") as f:
        data = json.load(f)
    gt_ranking = {row["index"]: row["gt_images"] for row in data}
    query_ids = list(gt_ranking.keys())
    print(f"WebQA queries: {len(query_ids)}")
    print(f"Image corpus size: 19960")

    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    k_values = [1, 5, 10, 20, 50, 100]

    header = f"{'Alpha':>8} " + " ".join(f"R@{k:>4}" for k in k_values)
    print(f"\n{header}")
    print("-" * len(header))

    for alpha in alphas:
        cap_path = imgcapfeats_path if alpha > 0 else None
        retriever = InternImgRetriever(
            queryfeats_path=queryfeats_path,
            imgfeats_path=imgfeats_path,
            imgcapfeats_path=cap_path,
            alpha=alpha,
        )
        results = retriever.score_recall(query_ids, gt_ranking, k_values=k_values)
        line = f"{alpha:>8.1f} " + " ".join(f"{results[f'recall@{k}']:>5.3f}" for k in k_values)
        print(line)

    print("\nDiagnostic notes:")
    print("- alpha=0.0: pure visual features (InternVideo2 vision encoder)")
    print("- alpha=0.2: current default (20% caption text, 80% visual)")
    print("- alpha=1.0: pure caption text (InternVideo2 text encoder on image captions)")
    print("- Higher alpha = text-text matching (question text vs caption text)")


if __name__ == "__main__":
    diagnose()
