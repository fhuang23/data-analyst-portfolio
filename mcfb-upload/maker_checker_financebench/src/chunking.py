"""PDF loading and chunking, with page numbers preserved on every chunk.

Page attribution matters here: FinanceBench gives each gold answer an
`evidence_page_num`, so if every chunk carries its source page we can measure
retrieval quality directly (did we retrieve the gold page?) without an LLM.

Chunks are cut within page boundaries so the page tag on each chunk is exact.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

PDF_DIR = os.environ.get(
    "FB_PDF_DIR",
    os.path.join(os.path.dirname(__file__), "..", "pdfs"),
)


@dataclass
class Chunk:
    doc_name: str
    page_num: int          # 0-indexed, aligns with FinanceBench evidence_page_num
    chunk_id: str
    text: str


def load_pdf_pages(doc_name: str) -> list[tuple[int, str]]:
    import pymupdf  # modern import (replaces deprecated `fitz`)
    path = os.path.join(PDF_DIR, f"{doc_name}.pdf")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Fetch the FinanceBench pdfs/ folder."
        )
    doc = pymupdf.open(path)
    return [(i, page.get_text()) for i, page in enumerate(doc)]


def chunk_document(
    doc_name: str,
    chunk_size: int = 1200,
    overlap: int = 200,
) -> list[Chunk]:
    """Character-based chunking within each page, with overlap.

    ~1200 chars is roughly 300 tokens, large enough to keep a financial table
    row and its header together, which naive small chunks tend to split.
    """
    chunks: list[Chunk] = []
    for page_num, text in load_pdf_pages(doc_name):
        text = text.strip()
        if not text:
            continue
        start = 0
        idx = 0
        step = max(1, chunk_size - overlap)
        while start < len(text):
            piece = text[start:start + chunk_size].strip()
            if piece:
                chunks.append(Chunk(
                    doc_name=doc_name,
                    page_num=page_num,
                    chunk_id=f"{doc_name}::p{page_num}::c{idx}",
                    text=piece,
                ))
                idx += 1
            start += step
    return chunks


if __name__ == "__main__":
    cs = chunk_document("3M_2018_10K")
    print(f"3M_2018_10K -> {len(cs)} chunks across "
          f"{len({c.page_num for c in cs})} pages")
    print("sample chunk:", cs[100].chunk_id)
    print(cs[100].text[:200].replace("\n", " "))
