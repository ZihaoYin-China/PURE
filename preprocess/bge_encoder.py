"""
BGE text encoder using the same SentenceTransformer stack as the existing
query/text feature builders.

The cached BAAI/bge-large-en-v1.5 SentenceTransformer uses CLS pooling plus
normalization. Using the raw HF model with mean pooling produces vectors in a
different space, which breaks dot-product retrieval against existing query
features.
"""

import torch
import numpy as np
from sentence_transformers import SentenceTransformer

# Use multiple CPU threads for faster inference on CPU (limit for 32GB container)
torch.set_num_threads(36)

MODEL_PATH = "/root/.cache/torch/sentence_transformers/BAAI_bge-large-en-v1.5"

_model = None
_device = None


def _get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_model():
    global _model, _device
    if _model is not None:
        return _model, _device

    _device = _get_device()
    _model = SentenceTransformer(MODEL_PATH, device=_device)
    return _model, _device


def encode(texts, batch_size=256, normalize=True, show_progress=True):
    """
    Encode a list of texts into BGE embeddings.

    Args:
        texts: list of strings
        batch_size: encoding batch size
        normalize: L2-normalize output embeddings (recommended for BGE)
        show_progress: print progress

    Returns:
        numpy array of shape (len(texts), 1024)
    """
    model, _ = _load_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def split_text_into_chunks(text, max_tokens=512):
    """Split a text into chunks of up to max_tokens tokens."""
    model, _ = _load_model()
    tokenizer = model.tokenizer
    token_ids = tokenizer.encode(text, truncation=False)
    if not token_ids:
        return [text]
    chunks = []
    for i in range(0, len(token_ids), max_tokens):
        chunk_ids = token_ids[i : i + max_tokens]
        chunks.append(tokenizer.decode(chunk_ids, skip_special_tokens=True))
    return chunks if chunks else [text]
