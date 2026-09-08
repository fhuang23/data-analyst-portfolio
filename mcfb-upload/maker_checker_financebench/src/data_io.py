"""Data loading, reasoning-category normalization, and stratified sampling.

No API keys needed. Everything here is deterministic and offline-testable.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Any

DATA_DIR = os.environ.get(
    "FB_DATA_DIR",
    os.path.join(os.path.dirname(__file__), "..", "data"),
)
QA_PATH = os.path.join(DATA_DIR, "financebench_open_source.jsonl")
DOC_PATH = os.path.join(DATA_DIR, "financebench_document_information.jsonl")


@dataclass
class Question:
    """One FinanceBench record, plus a normalized reasoning bucket."""
    financebench_id: str
    company: str
    doc_name: str
    question: str
    answer: str
    justification: str
    question_type: str
    question_reasoning: str | None
    evidence: list[dict[str, Any]]
    reasoning_bucket: str = field(default="")

    @property
    def gold_pages(self) -> list[str]:
        """Full-page evidence text blocks, as used by the oracle eval mode."""
        return [e.get("evidence_text_full_page", "") for e in self.evidence]

    @property
    def gold_spans(self) -> list[str]:
        """The exact cited spans, used to score grounding overlap."""
        return [e.get("evidence_text", "") for e in self.evidence]


def normalize_reasoning(raw: str | None) -> str:
    """Collapse FinanceBench's many reasoning labels into 3 stable buckets.

    FinanceBench mixes labels like 'Numerical reasoning OR Logical reasoning'.
    We bucket by the dominant skill so stratification and per-type reporting
    stay legible: extraction / numerical / logical.
    """
    if not raw:
        # 'None' reasoning is the domain-relevant qualitative set; treat as logical.
        return "logical"
    r = raw.lower()
    if "numerical" in r:
        return "numerical"
    if "logical" in r:
        return "logical"
    if "extraction" in r:
        return "extraction"
    return "logical"


def load_questions(qa_path: str = QA_PATH) -> list[Question]:
    out: list[Question] = []
    with open(qa_path) as fh:
        for line in fh:
            d = json.loads(line)
            q = Question(
                financebench_id=d["financebench_id"],
                company=d["company"],
                doc_name=d["doc_name"],
                question=d["question"],
                answer=d["answer"],
                justification=d.get("justification", ""),
                question_type=d["question_type"],
                question_reasoning=d.get("question_reasoning"),
                evidence=d.get("evidence", []),
            )
            q.reasoning_bucket = normalize_reasoning(q.question_reasoning)
            out.append(q)
    return out


def stratified_sample(
    questions: list[Question],
    n: int = 45,
    seed: int = 13,
) -> list[Question]:
    """Proportional stratified sample across (question_type x reasoning_bucket).

    Used to build the judge-validation subset so every skill and generation
    style is represented rather than over-sampling the easy extraction items.
    """
    rng = random.Random(seed)
    strata: dict[tuple[str, str], list[Question]] = {}
    for q in questions:
        key = (q.question_type, q.reasoning_bucket)
        strata.setdefault(key, []).append(q)

    total = len(questions)
    picked: list[Question] = []
    # Largest-remainder allocation so the sample sums to exactly n.
    quotas: list[tuple[tuple[str, str], int, float]] = []
    for key, items in strata.items():
        exact = n * len(items) / total
        quotas.append((key, int(exact), exact - int(exact)))
    allocated = sum(base for _, base, _ in quotas)
    remainder = n - allocated
    for key, base, _ in sorted(quotas, key=lambda t: t[2], reverse=True):
        take = base + (1 if remainder > 0 else 0)
        remainder -= 1 if remainder > 0 else 0
        pool = strata[key]
        take = min(take, len(pool))
        picked.extend(rng.sample(pool, take))
    rng.shuffle(picked)
    return picked


if __name__ == "__main__":
    qs = load_questions()
    from collections import Counter
    print(f"loaded {len(qs)} questions")
    print("reasoning buckets:", dict(Counter(q.reasoning_bucket for q in qs)))
    sample = stratified_sample(qs, n=45)
    print(f"stratified sample: {len(sample)}")
    print("sample buckets:", dict(Counter(q.reasoning_bucket for q in sample)))
    print("sample types:", dict(Counter(q.question_type for q in sample)))
