"""Synthetic labeled eval set.

Self-contained and offline: real trial text would come from ClinicalTrials.gov,
but fixtures must be deterministic and not depend on a live API, so these trials
are hand-authored with realistic `eligibility_criteria`. Swap in a licensed
benchmark (n2c2 2018 cohort selection, or a TREC Clinical Trials track) by
producing the same EvalCase shape.

gold_label: 1 = should surface to a screener, 0 = hard miss (ineligible).
The set is designed so a structured-only baseline gets the mechanical cases
(age, sex) right but MISSES the ones that need criteria reasoning (prior-therapy
exclusions, biomarker polarity, drug-class inference) — which is where the LLM
earns its lift.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvalCase:
    trial: dict
    gold_label: int
    note: str = ""
    patient: dict | None = None  # None -> harness falls back to the module PATIENT


# One patient, reused across trials (the running example).
PATIENT: dict = {
    "conditions": ["HER2-positive metastatic breast cancer", "metastatic breast cancer", "breast cancer"],
    "age_years": 62,
    "sex": "FEMALE",
    "biomarkers": ["HER2-positive"],
    "prior_therapies": ["trastuzumab", "pertuzumab", "docetaxel", "T-DM1"],
    "ecog_performance_status": 1,
    "key_comorbidities": [],
    "location": "Palo Alto, CA",
    "salient_history": (
        "62yo woman, HER2+ metastatic breast cancer, ECOG 1, measurable disease "
        "per RECIST 1.1, prior HER2-directed therapy through T-DM1 "
        "(1st line: trastuzumab+pertuzumab+docetaxel; 2nd line: T-DM1); "
        "no brain metastases."
    ),
}


def _trial(nct, title, criteria, *, min_age="18 Years", max_age="", sex="ALL",
           conditions=("Breast Cancer",)) -> dict:
    return {
        "nct_id": nct,
        "title": title,
        "min_age": min_age,
        "max_age": max_age,
        "sex": sex,
        "conditions": list(conditions),
        "eligibility_criteria": criteria,
    }


CASES: list[EvalCase] = [
    EvalCase(
        _trial("NCT-A01", "HER2+ MBC, T-DM1-naive cohort",
               "Inclusion Criteria:\n- HER2-positive metastatic breast cancer\n"
               "- ECOG performance status 0-1\n- Prior trastuzumab\n"
               "Exclusion Criteria:\n- Prior treatment with T-DM1 (ado-trastuzumab emtansine)"),
        gold_label=0,
        note="Exclusion: prior T-DM1. Baseline misses (structurally looks fine).",
    ),
    EvalCase(
        _trial("NCT-B02", "HER2+ MBC, later-line, T-DM1 allowed",
               "Inclusion Criteria:\n- HER2-positive metastatic breast cancer\n"
               "- ECOG 0-2\n- At least one prior HER2-directed therapy\n"
               "- Measurable disease per RECIST 1.1\nExclusion Criteria:\n- Uncontrolled infection"),
        gold_label=1,
        note="Clean eligible. Both should get this.",
    ),
    EvalCase(
        _trial("NCT-C03", "HER2-negative MBC study",
               "Inclusion Criteria:\n- HER2-negative metastatic breast cancer\n- ECOG 0-1"),
        gold_label=0,
        note="Biomarker polarity: patient is HER2+. Baseline misses (still 'breast cancer').",
    ),
    EvalCase(
        _trial("NCT-D04", "Breast cancer in older adults",
               "Inclusion Criteria:\n- Metastatic breast cancer\n- ECOG 0-2",
               min_age="65 Years"),
        gold_label=0,
        note="Mechanical age gate (patient 62 < 65). Baseline gets this right.",
    ),
    EvalCase(
        _trial("NCT-E05", "Male breast cancer study",
               "Inclusion Criteria:\n- Breast cancer\n- ECOG 0-1", sex="MALE"),
        gold_label=0,
        note="Sex gate. Baseline gets this right.",
    ),
    EvalCase(
        _trial("NCT-F06", "Post-T-DM1 progression cohort",
               "Inclusion Criteria:\n- HER2-positive metastatic breast cancer\n"
               "- Progression on or after T-DM1\n- ECOG 0-2\n- Measurable disease"),
        gold_label=1,
        note="Positive prior-therapy logic: requires prior T-DM1, patient has it.",
    ),
    EvalCase(
        _trial("NCT-G07", "HER2+ MBC with organ-function thresholds",
               "Inclusion Criteria:\n- HER2-positive metastatic breast cancer\n- ECOG 0-1\n"
               "- Adequate renal function: creatinine clearance >= 60 mL/min\n"
               "- Adequate hepatic function"),
        gold_label=1,
        note="Labs unstated -> reasoner should mark UNCERTAIN and still surface (gold=1).",
    ),
    EvalCase(
        _trial("NCT-H08", "HER2+ MBC, excludes active CNS disease",
               "Inclusion Criteria:\n- HER2-positive metastatic breast cancer\n- ECOG 0-1\n"
               "Exclusion Criteria:\n- Active or untreated CNS metastases"),
        gold_label=1,
        note="Negative finding resolves exclusion: patient has no brain mets.",
    ),
    EvalCase(
        _trial("NCT-I09", "HER2+ MBC, ADC-naive required",
               "Inclusion Criteria:\n- HER2-positive metastatic breast cancer\n- ECOG 0-1\n"
               "Exclusion Criteria:\n- Prior treatment with any antibody-drug conjugate"),
        gold_label=0,
        note="Drug-CLASS reasoning: T-DM1 is an ADC, so excluded. Baseline misses.",
    ),
    EvalCase(
        _trial("NCT-J10", "HER2+ MBC, <=2 prior lines",
               "Inclusion Criteria:\n- HER2-positive metastatic breast cancer\n- ECOG 0-1\n"
               "- No more than 2 prior lines of therapy in the metastatic setting"),
        gold_label=1,
        note="Line counting: patient has exactly 2 prior lines -> eligible.",
    ),
]
