from __future__ import annotations

from typing import Tuple
import os
import faiss
import numpy as np


def build_faiss_ip_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def save_index(index: faiss.Index, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    faiss.write_index(index, path)


def load_index(path: str) -> faiss.Index:
    return faiss.read_index(path)


def search(index: faiss.Index, query_vec: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    if query_vec.ndim == 1:
        query_vec = query_vec[None, :]
    if query_vec.dtype != np.float32:
        query_vec = query_vec.astype("float32")
    scores, idx = index.search(query_vec, top_k)
    return scores[0], idx[0]
