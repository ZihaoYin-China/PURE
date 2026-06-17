import json
import os
import pickle

import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


MODEL_NAME = "BAAI/bge-large-en-v1.5"
INSTRUCTION = "Represent this sentence for searching relevant passages: "


def _resolve_device(device: str | None) -> str:
    device = str(device or "").strip().lower()
    if device in {"", "auto"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _load_model(device: str) -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME, device=device)


def extract_query_feats_bge(input, output_path, model, batch_size=64):
    """
    Extract query features from the input JSON file and save them as a pickle file.
    Args:
        input (str): Path to the input JSON file.
        output_path (str): Path to save the pickle file.
    """
    with open(input, 'r') as f:
        data = json.load(f)

    id2feat = {}
    query_ids = []
    texts = []
    for row in data:
        query_ids.append(row['index'])
        texts.append(INSTRUCTION + row['question'])

    encoded_features = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    for query_id, feature in zip(query_ids, encoded_features):
        id2feat[query_id] = torch.tensor(feature, dtype=torch.float32)

    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, os.path.splitext(os.path.basename(input))[0] + '.pkl'), 'wb') as f:
        pickle.dump(id2feat, f)

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description="Extract query features using BGE and save them as a pickle file.")
    parser.add_argument("--input_path", type=str, default="dataset/query", help="Path to the input directory containing JSON files.")
    parser.add_argument("--output_path", type=str, default="eval/features/query/bge-large", help="Path to save the output pickle files.")
    parser.add_argument("--batch_size", type=int, default=64, help="Embedding batch size.")
    parser.add_argument("--device", type=str, default="auto", help="Embedding device: auto/cpu/cuda.")
    args = parser.parse_args()

    model = _load_model(_resolve_device(args.device))
    inputs = [
        os.path.join(args.input_path, input)
        for input in os.listdir(args.input_path)
        if input.endswith(".json")
    ]

    for input in tqdm(inputs):
        extract_query_feats_bge(
            input,
            args.output_path,
            model=model,
            batch_size=args.batch_size,
        )
