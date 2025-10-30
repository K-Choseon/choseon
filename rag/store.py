from __future__ import annotations

import os
import json
import shutil
import time
import uuid
from typing import Dict, List, Any

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MANUALS_DIR = os.path.join(DATA_DIR, "manuals")
CATALOG_PATH = os.path.join(DATA_DIR, "catalog.json")


def _ensure_dirs() -> None:
    os.makedirs(MANUALS_DIR, exist_ok=True)


def load_catalog() -> Dict[str, Any]:
    _ensure_dirs()
    if not os.path.exists(CATALOG_PATH):
        return {"manuals": []}
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_catalog(catalog: Dict[str, Any]) -> None:
    _ensure_dirs()
    with open(CATALOG_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)


def list_manuals() -> List[Dict[str, Any]]:
    return load_catalog().get("manuals", [])


def _new_manual_id() -> str:
    return uuid.uuid4().hex[:12]


def _manual_dir(manual_id: str) -> str:
    return os.path.join(MANUALS_DIR, manual_id)


def manual_paths(manual_id: str) -> Dict[str, str]:
    base = _manual_dir(manual_id)
    return {
        "base": base,
        "pdf": os.path.join(base, "manual.pdf"),
        "meta": os.path.join(base, "meta.json"),
        "chunks": os.path.join(base, "chunks.json"),
        "emb": os.path.join(base, "emb.npy"),
        "index": os.path.join(base, "index.faiss"),
    }


def register_manual(title: str, pdf_src_path: str) -> Dict[str, Any]:
    _ensure_dirs()
    manual_id = _new_manual_id()
    paths = manual_paths(manual_id)
    os.makedirs(paths["base"], exist_ok=True)

    # store PDF
    shutil.copyfile(pdf_src_path, paths["pdf"])

    meta = {
        "id": manual_id,
        "title": title,
        "filename": os.path.basename(pdf_src_path),
        "created_at": int(time.time()),
        "pages": None,
        "chunk_count": 0,
    }
    with open(paths["meta"], "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    catalog = load_catalog()
    catalog.setdefault("manuals", []).append({"id": manual_id, "title": title})
    save_catalog(catalog)

    return meta


def update_meta_counts(manual_id: str, pages: int | None, chunk_count: int) -> None:
    paths = manual_paths(manual_id)
    with open(paths["meta"], "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta["pages"] = pages
    meta["chunk_count"] = chunk_count
    with open(paths["meta"], "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def save_chunks(manual_id: str, chunks: List[Dict[str, Any]]) -> None:
    paths = manual_paths(manual_id)
    with open(paths["chunks"], "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


def load_chunks(manual_id: str) -> List[Dict[str, Any]]:
    paths = manual_paths(manual_id)
    with open(paths["chunks"], "r", encoding="utf-8") as f:
        return json.load(f)


def save_embeddings(manual_id: str, emb: np.ndarray) -> None:
    paths = manual_paths(manual_id)
    np.save(paths["emb"], emb.astype("float32"))


def load_embeddings(manual_id: str) -> np.ndarray:
    paths = manual_paths(manual_id)
    return np.load(paths["emb"]).astype("float32")
