"""ClinicalTrials.gov Data API v2 client, exposed as an ADK FunctionTool.

Base: https://clinicaltrials.gov/api/v2/studies  — no API key, public.
Pagination is cursor-based (nextPageToken); v1 (/api/query/) is retired.
The response wraps each study in a nested `protocolSection`.
"""
from __future__ import annotations

from typing import Any, Optional

import requests

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
TIMEOUT = 30


def _dig(d: dict, *path, default=None):
    """Safe nested .get() for the deeply-nested protocolSection."""
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return cur


def search_trials(
    condition: str,
    status: str = "RECRUITING",
    location: Optional[str] = None,
    geo: Optional[str] = None,
    max_results: int = 25,
) -> dict[str, Any]:
    """Search ClinicalTrials.gov for candidate studies (the coarse retrieval layer).

    This is deliberately the structured-filter BASELINE. No eligibility reasoning
    happens here — it just narrows ~500k studies to a scoreable candidate set.

    Args:
        condition: Disease/condition query -> query.cond.
        status: Recruitment status -> filter.overallStatus. Pipe-delimit to OR,
            e.g. "RECRUITING|NOT_YET_RECRUITING".
        location: Free-text location -> query.locn (e.g. "Palo Alto, California").
        geo: Optional radius filter, e.g. "distance(37.44,-122.14,50mi)".
        max_results: Page size (API max 1000; keep small for candidate sets).

    Returns:
        {"total": int, "candidates": [ {nct_id, title, status, phases, enrollment,
         sponsor, conditions, min_age, max_age, sex, healthy_volunteers,
         eligibility_criteria}, ... ]}
    """
    params: dict[str, Any] = {
        "query.cond": condition,
        "filter.overallStatus": status,
        "pageSize": min(max_results, 1000),
        "countTotal": "true",
        "sort": "LastUpdatePostDate:desc",
        "format": "json",
    }
    if location:
        params["query.locn"] = location
    if geo:
        params["filter.geo"] = geo

    resp = requests.get(
        BASE_URL, params=params, timeout=TIMEOUT,
        headers={"User-Agent": "sra-matcher/0.1"},
    )
    resp.raise_for_status()
    data = resp.json()

    candidates: list[dict[str, Any]] = []
    for study in data.get("studies", []):
        proto = study.get("protocolSection", {})
        elig = proto.get("eligibilityModule", {})
        candidates.append({
            "nct_id": _dig(proto, "identificationModule", "nctId", default="N/A"),
            "title": _dig(proto, "identificationModule", "briefTitle", default=""),
            "status": _dig(proto, "statusModule", "overallStatus", default=""),
            "phases": _dig(proto, "designModule", "phases", default=[]),
            "enrollment": _dig(proto, "designModule", "enrollmentInfo", "count"),
            "sponsor": _dig(proto, "sponsorCollaboratorsModule", "leadSponsor", "name", default=""),
            "conditions": _dig(proto, "conditionsModule", "conditions", default=[]),
            "min_age": elig.get("minimumAge", ""),
            "max_age": elig.get("maximumAge", ""),
            "sex": elig.get("sex", "ALL"),
            "healthy_volunteers": elig.get("healthyVolunteers"),
            # The free-text inclusion/exclusion narrative — the part keyword
            # search can't parse and where the LLM stage earns its slot.
            "eligibility_criteria": elig.get("eligibilityCriteria", ""),
        })
    return {"total": data.get("totalCount", len(candidates)), "candidates": candidates}


def get_trial_by_nct(nct_id: str) -> dict | None:
    """Fetch ONE trial's fields (incl. eligibility_criteria) by NCT id, via the
    single-study v2 endpoint. Used by the eval adapters to hydrate judged pairs
    without downloading the full corpus. Returns None if the id is unknown.
    """
    resp = requests.get(
        f"{BASE_URL}/{nct_id}", params={"format": "json"}, timeout=TIMEOUT,
        headers={"User-Agent": "sra-matcher/0.1"},
    )
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    study = resp.json()
    proto = study.get("protocolSection", {})
    elig = proto.get("eligibilityModule", {})
    return {
        "nct_id": _dig(proto, "identificationModule", "nctId", default=nct_id),
        "title": _dig(proto, "identificationModule", "briefTitle", default=""),
        "min_age": elig.get("minimumAge", ""),
        "max_age": elig.get("maximumAge", ""),
        "sex": elig.get("sex", "ALL"),
        "conditions": _dig(proto, "conditionsModule", "conditions", default=[]),
        "eligibility_criteria": elig.get("eligibilityCriteria", ""),
    }


# Same function exposed as a tool for any LlmAgent that should drive retrieval
# itself. The deterministic RetrievalAgent calls search_trials() directly instead.
try:
    from google.adk.tools import FunctionTool

    ctgov_search_tool = FunctionTool(func=search_trials)
except Exception:  # keep importable without ADK for unit-testing the client
    ctgov_search_tool = None
