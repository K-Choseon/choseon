from __future__ import annotations

from typing import List, Iterable
import numpy as np
from openai import OpenAI


_EMBED_MODEL = "text-embedding-3-small"


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return (x / norm).astype("float32")


def embed_texts(texts: Iterable[str], model: str | None = None, batch_size: int = 128) -> np.ndarray:
    client = OpenAI()
    model_name = model or _EMBED_MODEL

    texts_list: List[str] = list(texts)
    vectors: List[List[float]] = []

    for i in range(0, len(texts_list), batch_size):
        batch = texts_list[i : i + batch_size]
        resp = client.embeddings.create(model=model_name, input=batch)
        vectors.extend([d.embedding for d in resp.data])

    arr = np.asarray(vectors, dtype="float32")
    return _normalize_rows(arr)


def embed_query(text: str, model: str | None = None) -> np.ndarray:
    return embed_texts([text], model=model)
