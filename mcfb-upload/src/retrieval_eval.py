"""Retrieval-quality metrics, scored against FinanceBench gold evidence.

Two independent notions of a retrieval "hit", both requiring no LLM:

  * page hit  -- a retrieved chunk comes from the gold (doc_name, evidence_page).
  * text hit  -- a retrieved chunk contains most of the gold span's distinctive
                 tokens (robust to chunk boundaries and whitespace).

These give recall@k, which is the retrieval-error axis: how often the right
evidence even reaches the reasoner. The gap between reasoning accuracy in oracle
mode and in a retrieval mode is bounded by this.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from vectorstore import Retrieved

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if len(w) > 3}


def _distinctive(gold_span: str, max_tokens: int = 40) -> set[str]:
    toks = list(_tokens(gold_span))
    return set(toks[:max_tokens])


@dataclass
class RetrievalHit:
    page_hit: bool
    text_hit: bool
    best_containment: float


def score_retrieval(
    retrieved: list[Retrieved],
    gold_evidence: list[dict],
    text_threshold: float = 0.55,
) -> RetrievalHit:
    """Did the retrieved chunks cover any gold evidence block?"""
    gold_pages = {(e["doc_name"], e["evidence_page_num"]) for e in gold_evidence}
    page_hit = any((r.chunk.doc_name, r.chunk.page_num) in gold_pages for r in retrieved)

    best = 0.0
    for e in gold_evidence:
        want = _distinctive(e.get("evidence_text", ""))
        if not want:
            continue
        for r in retrieved:
            have = _tokens(r.chunk.text)
            containment = len(want & have) / len(want)
            best = max(best, containment)
    return RetrievalHit(
        page_hit=page_hit,
        text_hit=best >= text_threshold,
        best_containment=best,
    )


def recall_at_k(hits: list[RetrievalHit]) -> dict[str, float]:
    n = len(hits) or 1
    return {
        "page_recall": sum(h.page_hit for h in hits) / n,
        "text_recall": sum(h.text_hit for h in hits) / n,
        "either_recall": sum(h.page_hit or h.text_hit for h in hits) / n,
        "mean_containment": sum(h.best_containment for h in hits) / n,
        "n": float(len(hits)),
    }
