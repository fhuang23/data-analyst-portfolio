"""Typed data contract. These pydantic models are what the LLM stages are
forced to emit (via LlmAgent.output_schema), so the wiring never parses free text.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Sex(str, Enum):
    ALL = "ALL"
    FEMALE = "FEMALE"
    MALE = "MALE"


class PatientProfile(BaseModel):
    """Structured patient representation from the intake stage.

    Hold only the features eligibility criteria actually turn on. Never put raw
    identifiers here (name, MRN, date of birth) — store an integer age, not a DOB.
    See the data-governance note in the README.
    """
    conditions: list[str] = Field(default_factory=list)
    age_years: Optional[int] = None
    sex: Sex = Sex.ALL
    biomarkers: list[str] = Field(default_factory=list)
    prior_therapies: list[str] = Field(default_factory=list)
    ecog_performance_status: Optional[int] = None
    key_comorbidities: list[str] = Field(default_factory=list)
    location: Optional[str] = None
    salient_history: str = ""


class Verdict(str, Enum):
    MET = "met"
    NOT_MET = "not_met"
    UNKNOWN = "unknown"  # needs-info: surfaced to a screener, never silently dropped


class CriterionVerdict(BaseModel):
    criterion: str
    kind: str = Field(description="'inclusion' or 'exclusion'")
    verdict: Verdict
    evidence: str = Field(
        default="", description="Verbatim snippet from the criteria text supporting the call"
    )
    rationale: str = ""


class TrialLabel(str, Enum):
    ELIGIBLE = "eligible"
    INELIGIBLE = "ineligible"
    UNCERTAIN = "uncertain"


class TrialVerdict(BaseModel):
    nct_id: str
    label: TrialLabel
    confidence: float = Field(ge=0.0, le=1.0)
    criteria: list[CriterionVerdict] = Field(default_factory=list)
    unknown_count: int = 0
    summary: str = ""
