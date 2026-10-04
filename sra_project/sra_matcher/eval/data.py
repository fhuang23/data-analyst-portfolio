"""Labeled evaluation data.

A LabeledPair is one (patient, trial) example with a gold label, in exactly the
two-JSON-string form a Reasoner.judge() consumes — so the harness feeds live
reasoners the same shape the pipeline does, no adapter in between.

Bring your own labels. Two natural sources (README):
  - n2c2 2018 cohort-selection: per-criterion met/not-met; roll up to a trial
    label the same way the reasoner does (any hard exclusion -> ineligible, etc.).
  - TREC Clinical Trials: relevance 0/1/2. Map with GOLD_FROM_TREC below and
    decide up front whether TREC's "1 = would be eligible but excluded" counts
    as positive for your screening question (default here: it does not).

Fix the gold mapping once, in one place, before you tune anything — that
decision moves every number downstream.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterator

from ..schemas import TrialLabel


@dataclass
class LabeledPair:
    patient_profile_json: str  # a PatientProfile as JSON (the intake output)
    candidate_json: str        # one CT.gov candidate as JSON (a retrieval item)
    gold: TrialLabel
    patient_id: str = ""
    nct_id: str = ""


# TREC relevance -> our label. Tune to your positive-class definition.
GOLD_FROM_TREC = {
    2: TrialLabel.ELIGIBLE,
    1: TrialLabel.INELIGIBLE,  # "relevant but patient excluded" — not shortlisted
    0: TrialLabel.INELIGIBLE,
}


def load_pairs(path: str) -> list[LabeledPair]:
    """Read a JSONL file, one object per line with keys:
    patient_profile (obj), candidate (obj), gold (str: eligible|ineligible|uncertain),
    and optional patient_id / nct_id."""
    pairs: list[LabeledPair] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pairs.append(
                LabeledPair(
                    patient_profile_json=json.dumps(row["patient_profile"]),
                    candidate_json=json.dumps(row["candidate"]),
                    gold=TrialLabel(row["gold"]),
                    patient_id=row.get("patient_id", ""),
                    nct_id=row.get("nct_id", row["candidate"].get("nct_id", "")),
                )
            )
    return pairs


def iter_pairs(path: str) -> Iterator[LabeledPair]:
    yield from load_pairs(path)
