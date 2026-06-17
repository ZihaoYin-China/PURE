import argparse
import json
import pickle
from pathlib import Path

import torch


def _load_encoder(model_name, device):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: sentence_transformers. Install it with:\n"
            "  python -m pip install sentence-transformers\n"
            "Then rerun this script."
        ) from exc
    return SentenceTransformer(model_name, device=device)


def main():
    parser = argparse.ArgumentParser(description="Build BGE query feature pkl files for PURE retrieval.")
    parser.add_argument("--target", required=True)
    parser.add_argument("--route_dir", default="route/results_ood_vib_t5large_full")
    parser.add_argument("--router_model", default="t5-large")
    parser.add_argument("--output_dir", default="eval/features/query/bge-large")
    parser.add_argument("--model_name", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--query_prefix", default="")
    args = parser.parse_args()

    route_file = Path(args.route_dir) / args.router_model / f"{args.target}.json"
    if not route_file.is_file():
        raise FileNotFoundError(route_file)

    rows = json.loads(route_file.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected JSON list: {route_file}")

    ids = [str(row.get("index", i)) for i, row in enumerate(rows)]
    texts = [args.query_prefix + str(row.get("question", "")) for row in rows]
    if not all(text.strip() for text in texts):
        missing = [ids[i] for i, text in enumerate(texts) if not text.strip()][:5]
        raise ValueError(f"Missing question text for examples: {missing}")

    model = _load_encoder(args.model_name, args.device)
    emb = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True,
    ).detach().cpu().float()

    out = {qid: emb[i] for i, qid in enumerate(ids)}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{args.target}.pkl"
    with output_file.open("wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {len(out)} query features to {output_file}")


if __name__ == "__main__":
    main()
