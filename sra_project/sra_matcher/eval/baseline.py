"""The baseline the reasoner must beat.

This mimics the coarse retrieval filter applied as a classifier: it checks only
the STRUCTURED fields — condition keyword, age range, sex — and never reads the
free-text inclusion/exclusion criteria. Everything the LLM adds (biomarker
polarity, prior-therapy logic, drug-class reasoning, exclusion handling) is
invisible here by design. The gap between this and the agent is the whole thesis.
"""
from __future__ import annotations

import re


def _age_bound(value) -> int | None:
    m = re.search(r"(\d+)", str(value or ""))
    return int(m.group(1)) if m else None


def predict(patient: dict, trial: dict) -> int:
    """Return 1 (surface) or 0 (skip) using structured fields only."""
    # Sex gate
    tsex = str(trial.get("sex", "ALL")).upper()
    psex = str(patient.get("sex", "ALL")).upper()
    if tsex in ("MALE", "FEMALE") and psex in ("MALE", "FEMALE") and tsex != psex:
        return 0

    # Age gate
    age = patient.get("age_years")
    if isinstance(age, int):
        lo = _age_bound(trial.get("min_age"))
        hi = _age_bound(trial.get("max_age"))
        if lo is not None and age < lo:
            return 0
        if hi is not None and age > hi:
            return 0

    # Condition keyword overlap (fuzzy, on purpose — a dumb filter)
    hay = (
        " ".join(trial.get("conditions", [])).lower()
        + " "
        + str(trial.get("eligibility_criteria", "")).lower()
    )
    conds = [c.lower() for c in patient.get("conditions", [])]
    if conds and not any(
        c in hay or any(word in hay for word in c.split()) for c in conds
    ):
        return 0

    return 1
