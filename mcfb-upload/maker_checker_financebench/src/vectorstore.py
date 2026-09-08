"""A small, transparent vector store: a normalized matrix plus chunk metadata,
cosine similarity by dot product, persisted to disk.

At FinanceBench's scale (about 84 filings, a few thousand chunks per filing,
low hundreds of thousands of chunks for the shared store) this is the right
tool. It needs no external database, the retrieval logic is fully legible, and
building it this way is a deliberate not-over-engineering choice rather than
reaching for a heavyweight vector DB.

Per-document embeddings are cached to disk keyed by (doc, chunk params, embedder)
so the shared store is assembled from cached per-doc vectors instead of
re-embedding everything.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

from chunking import Chunk, chunk_document
from embeddings import Embedder

CACHE_DIR = os.environ.get(
    "MCFB_CACHE_DIR",
    os.path.join(os.path.dirname(__file__), "..", "vectorstores"),
)


@dataclass
class Retrieved:
    chunk: Chunk
    score: float


class VectorStore:
    def __init__(self, vectors: np.ndarray, chunks: list[Chunk]):
        assert vectors.shape[0] == len(chunks)
        self.vectors = vectors            # (N, dim), L2-normalized
        self.chunks = chunks

    def search(self, query_vec: np.ndarray, k: int = 5) -> list[Retrieved]:
        sims = self.vectors @ query_vec.reshape(-1)     # cosine, both normalized
        k = min(k, len(self.chunks))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [Retrieved(self.chunks[i], float(sims[i])) for i in top]

    @classmethod
    def concat(cls, stores: list["VectorStore"]) -> "VectorStore":
        vecs = np.vstack([s.vectors for s in stores])
        chunks: list[Chunk] = []
        for s in stores:
            chunks.extend(s.chunks)
        return cls(vecs, chunks)


def _cache_path(doc_name: str, embedder: Embedder, chunk_size: int, overlap: int) -> str:
    tag = f"{embedder.name}__cs{chunk_size}_ov{overlap}".replace("/", "-").replace(":", "-")
    return os.path.join(CACHE_DIR, tag, f"{doc_name}.npz")


def build_doc_store(
    doc_name: str,
    embedder: Embedder,
    chunk_size: int = 1200,
    overlap: int = 200,
    use_cache: bool = True,
) -> VectorStore:
    """Embed one filing's chunks, caching vectors + metadata to disk."""
    path = _cache_path(doc_name, embedder, chunk_size, overlap)
    if use_cache and os.path.exists(path):
        data = np.load(path, allow_pickle=True)
        meta = json.loads(str(data["meta"]))
        chunks = [Chunk(**m) for m in meta]
        return VectorStore(data["vectors"], chunks)

    chunks = chunk_document(doc_name, chunk_size, overlap)
    vectors = embedder.embed([c.text for c in chunks])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    meta = json.dumps([c.__dict__ for c in chunks])
    np.savez_compressed(path, vectors=vectors, meta=meta)
    return VectorStore(vectors, chunks)


def build_shared_store(
    doc_names: list[str],
    embedder: Embedder,
    chunk_size: int = 1200,
    overlap: int = 200,
) -> VectorStore:
    """Assemble the all-filings store from per-doc caches."""
    stores = [
        build_doc_store(d, embedder, chunk_size, overlap)
        for d in sorted(set(doc_names))
    ]
    return VectorStore.concat(stores)
