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

    # Some feature pickles were serialized with NumPy layouts that refer to
    # legacy/internal module paths like `numpy._core.*`. Newer environments may
    # only expose `numpy.core.*`, so add temporary aliases during unpickling.
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


class BGETextRetriever:
    def __init__(self, queryfeats_path: str, textfeats_path: str | list[str]):
        self.queryfeats_path = queryfeats_path
        self.textfeats_path = textfeats_path
        self.queryfeats = None
        self.textids = []
        self.textfeats = None

        self.load_feats(queryfeats_path, textfeats_path)

    def load_feats(self, queryfeats_path, textfeats_path):
        print(f"Loading BGETextRetriever from {textfeats_path}...")

        raw_queryfeats = _pickle_load_cpu(queryfeats_path)
        self.queryfeats = {
            qid: _to_numpy_float32(feat) for qid, feat in raw_queryfeats.items()
        }

        if isinstance(textfeats_path, list):
            textfeats = {}
            for path in textfeats_path:
                textfeats.update(_pickle_load_cpu(path))
        else:
            textfeats = _pickle_load_cpu(textfeats_path)

        text_vectors = []
        for text_id, text_feat in textfeats.items():
            self.textids.append(text_id)
            text_vectors.append(_to_numpy_float32(text_feat))

        if not text_vectors:
            raise ValueError("No text features loaded.")

        self.textfeats = np.stack(text_vectors).astype(np.float32, copy=False)

    def retrieve(self, query_id, top_k: int = 5):
        if query_id not in self.queryfeats:
            raise KeyError(f"Query id not found in query features: {query_id}")

        query_feat = self.queryfeats[query_id]
        similarity = self.textfeats @ query_feat

        top_k = min(top_k, len(self.textids))
        if top_k <= 0:
            return [], np.array([], dtype=np.float32)

        # Faster than full sort
        top_idx = np.argpartition(-similarity, top_k - 1)[:top_k]
        top_idx = top_idx[np.argsort(-similarity[top_idx])]

        top_k_text_ids = [self.textids[i].rsplit("_part", 1)[0] for i in top_idx]
        top_k_scores = similarity[top_idx]
        return top_k_text_ids, top_k_scores

    def score_recall(self, text_query_ids, gt_ranking, k_values=[1, 5, 10]):
        results = {f"recall@{k}": 0.0 for k in k_values}
        total_queries = len(text_query_ids)

        for query_id in text_query_ids:
            correct_textids = gt_ranking[query_id]
            retrieved_textids, _ = self.retrieve(query_id, max(k_values))

            for k in k_values:
                if any(text in correct_textids for text in retrieved_textids[:k]):
                    results[f"recall@{k}"] += 1

        for k in k_values:
            results[f"recall@{k}"] /= total_queries

        return results


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="squad", choices=["squad", "natural_questions", "hotpotqa"])
    args = parser.parse_args()

    queryfeats_path = f"eval/features/query/bge-large/{args.target}.pkl"
    textfeats_path = [
        "eval/features/text/squad.pkl",
        "eval/features/text/natural_questions.pkl",
        "eval/features/text/hotpotqa.pkl",
    ]

    retriever = BGETextRetriever(
        queryfeats_path=queryfeats_path,
        textfeats_path=textfeats_path,
    )

    gt_ranking_path = f"dataset/query/{args.target}.json"
    with open(gt_ranking_path, "r") as f:
        gt_ranking_data = json.load(f)
    gt_ranking = {qa["index"]: qa["gt_texts"] for qa in gt_ranking_data}

    print(retriever.score_recall(gt_ranking.keys(), gt_ranking))
