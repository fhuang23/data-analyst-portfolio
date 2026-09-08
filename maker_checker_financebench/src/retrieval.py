"""The FinanceBench eval-mode ladder as a retrieval-difficulty axis.

Modes (increasing difficulty):
  oracle        -- gold evidence page(s) supplied. Isolates reasoning.
  in_context    -- entire filing text supplied. Long-context, no retrieval.
  single_store  -- vector store over the one relevant filing. Doc known, span not.
  shared_store  -- vector store over all filings. Realistic end-to-end.

get_context(q, mode) returns the text to hand the reasoner.
retrieve(q, mode, ...) returns the retrieved chunks too, so retrieval quality
can be scored against gold evidence independently of the LLM.
"""
from __future__ import annotations

import functools
import os

from data_io import Question
from vectorstore import Retrieved, build_doc_store, build_shared_store

PDF_DIR = os.environ.get(
    "FB_PDF_DIR",
    os.path.join(os.path.dirname(__file__), "..", "pdfs"),
)


# --- context-only modes -----------------------------------------------------

def oracle_context(q: Question) -> str:
    return "\n\n".join(p for p in q.gold_pages if p)


def in_context(q: Question) -> str:
    import pymupdf  # modern import (replaces deprecated `fitz`)
    path = os.path.join(PDF_DIR, f"{q.doc_name}.pdf")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Fetch the FinanceBench pdfs/ folder for in_context mode."
        )
    doc = pymupdf.open(path)
    return "\n\n".join(page.get_text() for page in doc)


# --- retrieval modes --------------------------------------------------------

def _resolve_backend(embed_backend: str | None) -> str:
    return embed_backend or os.environ.get("MCFB_EMBED_BACKEND", "local")


@functools.lru_cache(maxsize=None)
def _shared_store(backend_sel: str, all_docs: tuple[str, ...]):
    """Cached by the backend *selector* (e.g. 'mock', 'local', 'gemini'), so the
    embedder is reconstructed from a valid selector, not from its display name.
    """
    from embeddings import make_embedder
    return build_shared_store(list(all_docs), make_embedder(backend_sel))


def retrieve(
    q: Question,
    mode: str,
    k: int = 5,
    embed_backend: str | None = None,
    all_docs: tuple[str, ...] | None = None,
) -> list[Retrieved]:
    """Return the top-k retrieved chunks for a question in a retrieval mode."""
    from embeddings import make_embedder
    backend_sel = _resolve_backend(embed_backend)
    embedder = make_embedder(backend_sel)
    qvec = embedder.embed([q.question])[0]

    if mode == "single_store":
        store = build_doc_store(q.doc_name, embedder)
    elif mode == "shared_store":
        if all_docs is None:
            raise ValueError("shared_store needs all_docs (the full doc list).")
        store = _shared_store(backend_sel, tuple(sorted(set(all_docs))))
    else:
        raise ValueError(f"{mode!r} is not a retrieval mode.")
    return store.search(qvec, k=k)


def get_context(
    q: Question,
    mode: str,
    k: int = 5,
    embed_backend: str | None = None,
    all_docs: tuple[str, ...] | None = None,
) -> str:
    if mode == "oracle":
        return oracle_context(q)
    if mode == "in_context":
        return in_context(q)
    if mode in ("single_store", "shared_store"):
        chunks = retrieve(q, mode, k=k, embed_backend=embed_backend, all_docs=all_docs)
        return "\n\n".join(r.chunk.text for r in chunks)
    raise ValueError(f"unknown mode {mode!r}")


MODES = ("oracle", "in_context", "single_store", "shared_store")
