"""Pluggable text embedding backends.

Select with MCFB_EMBED_BACKEND (default 'local'):
  local   -- sentence-transformers, no API cost (downloads a small model once)
  gemini  -- google-genai text-embedding-004 (uses your Gemini key)
  openai  -- text-embedding-3-small
  voyage  -- voyage-3 (Anthropic's recommended embedding partner)
  mock    -- deterministic hashing embedding, no deps; for testing the plumbing

All backends return L2-normalized float32 vectors so the vector store can use a
plain dot product as cosine similarity.
"""
from __future__ import annotations

import hashlib
import os
from typing import Protocol

import numpy as np

BACKEND = os.environ.get("MCFB_EMBED_BACKEND", "local")
LOCAL_MODEL = os.environ.get("MCFB_LOCAL_EMBED_MODEL", "BAAI/bge-small-en-v1.5")


def _normalize(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class Embedder(Protocol):
    dim: int
    name: str
    def embed(self, texts: list[str]) -> np.ndarray: ...


class MockEmbedder:
    """Deterministic bag-of-hashed-tokens embedding. No external deps.

    Good enough to exercise chunking -> store -> retrieval -> metric end to end,
    and to unit-test that overlapping text retrieves itself. Not for real
    quality numbers; use a real backend for those.
    """
    def __init__(self, dim: int = 256):
        self.dim = dim
        self.name = f"mock-{dim}"

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in t.lower().split():
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
        return _normalize(out)


class LocalEmbedder:
    def __init__(self, model_name: str = LOCAL_MODEL):
        from sentence_transformers import SentenceTransformer
        self._m = SentenceTransformer(model_name)
        self.dim = self._m.get_sentence_embedding_dimension()
        self.name = f"local:{model_name}"

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self._m.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return _normalize(vecs)


class GeminiEmbedder:
    def __init__(self, model_name: str = "text-embedding-004"):
        from google import genai
        self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.model_name = model_name
        self.name = f"gemini:{model_name}"
        self.dim = 768

    def embed(self, texts: list[str]) -> np.ndarray:
        r = self._client.models.embed_content(model=self.model_name, contents=texts)
        return _normalize(np.array([e.values for e in r.embeddings], dtype=np.float32))


class OpenAIEmbedder:
    def __init__(self, model_name: str = "text-embedding-3-small"):
        from openai import OpenAI
        self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model_name = model_name
        self.name = f"openai:{model_name}"
        self.dim = 1536

    def embed(self, texts: list[str]) -> np.ndarray:
        r = self._client.embeddings.create(model=self.model_name, input=texts)
        return _normalize(np.array([d.embedding for d in r.data], dtype=np.float32))


class VoyageEmbedder:
    def __init__(self, model_name: str = "voyage-3"):
        import voyageai
        self._client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
        self.model_name = model_name
        self.name = f"voyage:{model_name}"
        self.dim = 1024

    def embed(self, texts: list[str]) -> np.ndarray:
        r = self._client.embed(texts, model=self.model_name, input_type="document")
        return _normalize(np.array(r.embeddings, dtype=np.float32))


def make_embedder(backend: str | None = None) -> Embedder:
    b = (backend or BACKEND).lower()
    if b == "local":
        return LocalEmbedder()
    if b == "gemini":
        return GeminiEmbedder()
    if b == "openai":
        return OpenAIEmbedder()
    if b == "voyage":
        return VoyageEmbedder()
    if b == "mock":
        return MockEmbedder()
    raise ValueError(f"unknown embed backend {b!r}")


if __name__ == "__main__":
    emb = MockEmbedder()
    v = emb.embed(["capital expenditures were 1577 million", "net income was 5363"])
    print("mock dim:", emb.dim, "shape:", v.shape, "norms:", np.round(np.linalg.norm(v, axis=1), 3))
