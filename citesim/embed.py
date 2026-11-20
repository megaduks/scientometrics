"""Sentence embeddings (unit-normalised, so dot product = cosine similarity)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"


def embed_texts(texts: list[str], device: str | None = None, batch_size: int = 512) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME, device=device)
    emb = model.encode(list(texts), batch_size=batch_size, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=True)
    return emb.astype(np.float16)


def cached_embeddings(texts: list[str], path: str | Path, device: str | None = None) -> np.ndarray:
    path = Path(path)
    if path.exists():
        emb = np.load(path)
        if len(emb) == len(texts):
            return emb
    emb = embed_texts(texts, device=device)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    return emb
