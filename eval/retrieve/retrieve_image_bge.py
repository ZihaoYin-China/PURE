"""
BGE-based image retriever that matches query text embeddings (from BGE) to
image caption embeddings (from BGE), enabling text-to-text retrieval for images.

This replaces the InternVideo2 cross-modal retriever for datasets where
InternVideo2 embeddings fail to align queries with images.
"""

import numpy as np
import pickle
import sys
import io
import torch


def _pickle_load_cpu(path):
    original = torch.storage._load_from_bytes
    numpy_aliases = {
        "numpy._core": np.core,
        "numpy._core.multiarray": np.core.multiarray,
        "numpy._core.numeric": np.core.numeric,
        "numpy._core.umath": np.core.umath,
    }

    def _load_from_bytes_cpu(b):
        return torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)

    torch.storage._load_from_bytes = _load_from_bytes_cpu
    try:
        for alias, module in numpy_aliases.items():
            if alias not in sys.modules:
                sys.modules[alias] = module
        with open(path, "rb") as f:
            obj = pickle.load(f)
    finally:
        torch.storage._load_from_bytes = original
    return obj


def _to_numpy_float32(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)
    return x.astype(np.float32, copy=False)


class BGEImageRetriever:
    def __init__(
        self,
        queryfeats_path: str,
        captionfeats_path: str,
    ):
        self.queryfeats = None
        self.imgids = []
        self.imgfeats = None

        self._load(queryfeats_path, captionfeats_path)

    def _load(self, queryfeats_path, captionfeats_path):
        print(f"Loading BGEImageRetriever from {captionfeats_path}...")

        raw_qfeats = _pickle_load_cpu(queryfeats_path)
        self.queryfeats = {
            qid: _to_numpy_float32(feat) for qid, feat in raw_qfeats.items()
        }

        raw_imgfeats = _pickle_load_cpu(captionfeats_path)
        img_vectors = []
        for img_id, feat in raw_imgfeats.items():
            self.imgids.append(img_id)
            img_vectors.append(_to_numpy_float32(feat))

        if not img_vectors:
            raise ValueError("No caption features loaded.")

        self.imgfeats = np.stack(img_vectors).astype(np.float32, copy=False)
        self.imgid_to_idx = {img_id: idx for idx, img_id in enumerate(self.imgids)}

    def retrieve(self, query_id, top_k: int = 5, candidate_ids=None):
        if query_id not in self.queryfeats:
            raise KeyError(f"Query id not found: {query_id}")

        query_feat = self.queryfeats[query_id]

        if candidate_ids:
            search_indices = np.asarray(
                [
                    self.imgid_to_idx[cid]
                    for cid in candidate_ids
                    if cid in self.imgid_to_idx
                ],
                dtype=np.int64,
            )
            if len(search_indices) == 0:
                return [], np.array([], dtype=np.float32)
            similarity = self.imgfeats[search_indices] @ query_feat
            search_imgids = [self.imgids[i] for i in search_indices]
        else:
            similarity = self.imgfeats @ query_feat
            search_imgids = self.imgids

        top_k = min(top_k, len(search_imgids))
        if top_k <= 0:
            return [], np.array([], dtype=np.float32)

        top_idx = np.argpartition(-similarity, top_k - 1)[:top_k]
        top_idx = top_idx[np.argsort(-similarity[top_idx])]
        top_k_img_ids = [search_imgids[i] for i in top_idx]
        top_k_scores = similarity[top_idx]
        return top_k_img_ids, top_k_scores

    def score_recall(self, query_ids, gt_ranking, k_values=None):
        if k_values is None:
            k_values = [1, 5, 10]
        results = {f"recall@{k}": 0.0 for k in k_values}
        total = len(query_ids)
        for query_id in query_ids:
            correct = gt_ranking.get(query_id, [])
            if not correct:
                continue
            retrieved, _ = self.retrieve(query_id, max(k_values))
            for k in k_values:
                if any(img in correct for img in retrieved[:k]):
                    results[f"recall@{k}"] += 1
        for k in k_values:
            results[f"recall@{k}"] /= max(total, 1)
        return results
