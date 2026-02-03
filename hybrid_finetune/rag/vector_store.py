"""
Offline vector store using FAISS.
"""

import faiss
import numpy as np
import pickle
from pathlib import Path


class OfflineVectorStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)
        self.documents = []

    def add(self, embeddings: np.ndarray, docs: list[str]):
        self.index.add(embeddings)
        self.documents.extend(docs)

    def search(self, query_embedding: np.ndarray, k: int = 5):
        distances, indices = self.index.search(query_embedding, k)
        return [self.documents[i] for i in indices[0]]

    def save(self, path: str):
        path = Path(path)
        faiss.write_index(self.index, str(path / "index.faiss"))
        with open(path / "docs.pkl", "wb") as f:
            pickle.dump(self.documents, f)

    @classmethod
    def load(cls, path: str, dim: int):
        path = Path(path)
        store = cls(dim)
        store.index = faiss.read_index(str(path / "index.faiss"))
        with open(path / "docs.pkl", "rb") as f:
            store.documents = pickle.load(f)
        return store
