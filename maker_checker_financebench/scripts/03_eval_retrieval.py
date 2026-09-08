#!/usr/bin/env python3
"""Step 4 eval: measure retrieval quality against gold evidence. No LLM needed.

Runs a retrieval mode (single_store or shared_store) over the questions and
reports recall@k using the gold evidence pages and spans as ground truth. This
is the retrieval-error axis, measured independently of the reasoner.

Usage:
    python scripts/03_eval_retrieval.py --mode single_store --k 5
    python scripts/03_eval_retrieval.py --mode shared_store --k 8 --backend gemini
    python scripts/03_eval_retrieval.py --mode single_store --backend mock --available-only

--available-only restricts to questions whose PDF is present locally, so you can
test on a subset before downloading the full pdfs/ folder.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_io import load_questions           # noqa: E402
from retrieval import retrieve, PDF_DIR       # noqa: E402
from retrieval_eval import score_retrieval, recall_at_k  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "retrieval_eval.csv")


def pdf_exists(doc_name: str) -> bool:
    return os.path.exists(os.path.join(PDF_DIR, f"{doc_name}.pdf"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["single_store", "shared_store"], default="single_store")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--backend", default=None, help="embed backend (default: env MCFB_EMBED_BACKEND)")
    ap.add_argument("--available-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    qs = load_questions()
    if args.available_only:
        qs = [q for q in qs if pdf_exists(q.doc_name)]
        print(f"restricting to {len(qs)} questions with local PDFs")
    if args.limit:
        qs = qs[:args.limit]
    if not qs:
        sys.exit("no questions to run (are the PDFs present?)")

    if args.available_only:
        all_docs = tuple(sorted({q.doc_name for q in qs if pdf_exists(q.doc_name)}))
        if args.mode == "shared_store":
            print(f"shared_store corpus restricted to {len(all_docs)} available "
                  f"filings (routing is easier than the full corpus)")
    else:
        all_docs = tuple(sorted({q.doc_name for q in load_questions()}))

    hits, rows = [], []
    for i, q in enumerate(qs, 1):
        chunks = retrieve(q, args.mode, k=args.k, embed_backend=args.backend,
                          all_docs=all_docs if args.mode == "shared_store" else None)
        hit = score_retrieval(chunks, q.evidence)
        hits.append(hit)
        # For shared_store, also record whether the top chunk came from the right doc.
        top_doc = chunks[0].chunk.doc_name if chunks else ""
        rows.append({
            "financebench_id": q.financebench_id,
            "doc_name": q.doc_name,
            "reasoning_bucket": q.reasoning_bucket,
            "page_hit": hit.page_hit,
            "text_hit": hit.text_hit,
            "best_containment": round(hit.best_containment, 3),
            "top_doc_correct": top_doc == q.doc_name,
        })
        if i % 20 == 0:
            print(f"  {i}/{len(qs)}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    r = recall_at_k(hits)
    doc_acc = sum(row["top_doc_correct"] for row in rows) / len(rows)
    print(f"\n=== retrieval eval: mode={args.mode} k={args.k} "
          f"backend={args.backend or os.environ.get('MCFB_EMBED_BACKEND', 'local')} ===")
    print(f"  questions:          {int(r['n'])}")
    print(f"  page recall@{args.k}:     {r['page_recall']:.3f}")
    print(f"  text recall@{args.k}:     {r['text_recall']:.3f}")
    print(f"  either recall@{args.k}:   {r['either_recall']:.3f}")
    print(f"  mean containment:   {r['mean_containment']:.3f}")
    if args.mode == "shared_store":
        print(f"  top-1 doc correct:  {doc_acc:.3f}  (right filing surfaced first)")
    print(f"\nwrote {os.path.relpath(OUT)}")


if __name__ == "__main__":
    main()
