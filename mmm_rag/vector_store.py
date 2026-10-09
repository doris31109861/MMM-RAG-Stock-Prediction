"""
vector_store.py
================
輕量的 FAISS 向量資料庫包裝，對應論文的
「Semantic feature DB」與「Visual feature DB」（Fig.1 / Fig.2 中的 (A) 與圖示資料庫）。

用 cosine similarity（等價於：向量先做 L2 normalize，再用內積 IndexFlatIP）
對應論文 Eq.(9)(12) 的 cosine similarity top-K 檢索。
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np


class FaissStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.metadata: List[Dict[str, Any]] = []

    def add(self, vectors: np.ndarray, metadata_list: List[Dict[str, Any]]):
        vectors = np.asarray(vectors, dtype="float32")
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.metadata.extend(metadata_list)

    def search(self, query_vector: np.ndarray, top_k: int = 3) -> List[Tuple[float, Dict[str, Any]]]:
        if self.index.ntotal == 0:
            return []
        query_vector = np.asarray(query_vector, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(query_vector)
        top_k = min(top_k, self.index.ntotal)
        scores, idxs = self.index.search(query_vector, top_k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            results.append((float(score), self.metadata[idx]))
        return results

    def save(self, index_path: Path, meta_path: Path):
        faiss.write_index(self.index, str(index_path))
        with open(meta_path, "wb") as f:
            pickle.dump({"dim": self.dim, "metadata": self.metadata}, f)

    @classmethod
    def load(cls, index_path: Path, meta_path: Path) -> "FaissStore":
        with open(meta_path, "rb") as f:
            payload = pickle.load(f)
        store = cls(dim=payload["dim"])
        store.index = faiss.read_index(str(index_path))
        store.metadata = payload["metadata"]
        return store

    def __len__(self):
        return self.index.ntotal
