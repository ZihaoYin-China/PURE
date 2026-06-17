import io
import pickle
import sys

import numpy as np
import torch


def _pickle_load_cpu(path):
    """
    Load pickle files that may contain torch tensors originally saved on CUDA.
    Force all tensors to CPU during unpickling.
    """
    original = torch.storage._load_from_bytes
    original_numpy_aliases = {}

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
            original_numpy_aliases[alias] = sys.modules.get(alias)
            if alias not in sys.modules:
                sys.modules[alias] = module
        with open(path, "rb") as f:
            obj = pickle.load(f)
    finally:
        torch.storage._load_from_bytes = original
        for alias, original_module in original_numpy_aliases.items():
            if original_module is None:
                sys.modules.pop(alias, None)
            else:
                sys.modules[alias] = original_module

    return obj


def _to_numpy_float32(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)
    return x.astype(np.float32, copy=False)


class InternImgRetriever:
    def __init__(self, queryfeats_path: str, imgfeats_path, imgcapfeats_path=None, alpha: float = 0.2):
        self.queryfeats_path = queryfeats_path
        self.imgfeats_path = imgfeats_path
        self.imgcapfeats_path = imgcapfeats_path if alpha != 0 else None
        self.queryfeats = None
        self.imgids = []
        self.imgfeats = None

        assert 0 <= alpha <= 1, f"alpha should be in [0, 1], but got {alpha}"
        self.load_feats(queryfeats_path, imgfeats_path, self.imgcapfeats_path, alpha)

    def load_feats(self, queryfeats_path: str, imgfeats_path, imgcapfeats_path=None, alpha: float = 0.2):
        print(f"Loading InternImgRetriever from {imgfeats_path}...")
        raw_queryfeats = _pickle_load_cpu(queryfeats_path)
        self.queryfeats = {
            qid: _to_numpy_float32(feat) for qid, feat in raw_queryfeats.items()
        }

        imgfeats = {}
        if isinstance(imgfeats_path, list):
            for path in imgfeats_path:
                imgfeats.update(_pickle_load_cpu(path))
        else:
            imgfeats = _pickle_load_cpu(imgfeats_path)

        imgcapfeats = {}
        if imgcapfeats_path:
            if isinstance(imgcapfeats_path, list):
                for path in imgcapfeats_path:
                    imgcapfeats.update(_pickle_load_cpu(path))
            else:
                imgcapfeats = _pickle_load_cpu(imgcapfeats_path)

        img_vectors = []
        for img_id, img_feat in imgfeats.items():
            base_feat = _to_numpy_float32(img_feat)
            if imgcapfeats_path:
                imgcap_feat = _to_numpy_float32(imgcapfeats[img_id])
                img_vectors.append(alpha * imgcap_feat + (1 - alpha) * base_feat)
            else:
                img_vectors.append(base_feat)
            self.imgids.append(img_id)

        if not img_vectors:
            raise ValueError("No image features loaded.")
        self.imgfeats = np.stack(img_vectors).astype(np.float32, copy=False)
        self.imgid_to_idx = {img_id: idx for idx, img_id in enumerate(self.imgids)}

    def retrieve(self, query_id, top_k: int = 5, candidate_ids=None):
        if query_id not in self.queryfeats:
            raise KeyError(f"Query id not found in image query features: {query_id}")

        query_feat = self.queryfeats[query_id]

        search_indices = None
        if candidate_ids:
            search_indices = [
                self.imgid_to_idx[candidate_id]
                for candidate_id in candidate_ids
                if candidate_id in self.imgid_to_idx
            ]
            if not search_indices:
                search_indices = None

        if search_indices is None:
            similarity = self.imgfeats @ query_feat
            search_imgids = self.imgids
        else:
            search_indices = np.asarray(search_indices, dtype=np.int64)
            similarity = self.imgfeats[search_indices] @ query_feat
            search_imgids = [self.imgids[i] for i in search_indices]

        top_k = min(top_k, len(search_imgids))
        if top_k <= 0:
            return [], np.array([], dtype=np.float32)

        top_idx = np.argpartition(-similarity, top_k - 1)[:top_k]
        top_idx = top_idx[np.argsort(-similarity[top_idx])]
        top_k_img_ids = [search_imgids[i] for i in top_idx]
        top_k_scores = similarity[top_idx]
        return top_k_img_ids, top_k_scores
    
    def score_recall(self, text_query_ids, gt_ranking, k_values=[1, 5, 10]):
        results = {f"recall@{k}": 0.0 for k in k_values}
        total_queries = len(text_query_ids)
        for query_id in text_query_ids:
            correct_imgids = gt_ranking[query_id]
            retrieved_imgids, _ = self.retrieve(query_id, max(k_values))
            for k in k_values:
                if any(img in correct_imgids for img in retrieved_imgids[:k]):
                    results[f"recall@{k}"] += 1
        for k in k_values:
            results[f"recall@{k}"] /= total_queries
        return results


if __name__ == "__main__":

    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="webqa", choices=["webqa"])
    parser.add_argument("--alpha", type=float, default=0.2, help="Weight for image caption features (0 to 1)")
    args = parser.parse_args()

    queryfeats_path = f"eval/features/query/internvideo/{args.target}.pkl"
    imgfeats_path = [
        "eval/features/image/webqa.pkl"
    ]
    imgcapfeats_path = [
        "eval/features/image/webqa_imgcap.pkl"
    ]

    retriever = InternImgRetriever(
        queryfeats_path=queryfeats_path,
        imgfeats_path=imgfeats_path,
        imgcapfeats_path=imgcapfeats_path,
        alpha=args.alpha,
    )

    gt_ranking_path = f"dataset/query/{args.target}.json"
    with open(gt_ranking_path, 'r') as f:
        gt_ranking_data = json.load(f)
    gt_ranking = {qa['index']: qa['gt_images'] for qa in gt_ranking_data}

    print(retriever.score_recall(gt_ranking.keys(), gt_ranking))
