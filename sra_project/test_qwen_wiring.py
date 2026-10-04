"""Wiring smoke test - stock Qwen via Ollama, no GPU, no ADK, no TREC data."""
from __future__ import annotations

import json
import sys
import time

import requests

from sra_matcher import prompts
from sra_matcher.schemas import TrialVerdict

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen3:4b"

PATIENT = {
    "conditions": ["non-small cell lung cancer"],
    "age_years": 67,
    "sex": "MALE",
    "biomarkers": ["EGFR exon 19 deletion"],
    "prior_therapies": ["osimertinib"],
    "ecog_performance_status": 1,
    "key_comorbidities": ["hypertension"],
    "location": "Palo Alto, California",
    "salient_history": "Progressed on first-line osimertinib after 14 months.",
}
CANDIDATE = {
    "nct_id": "NCT09999999",
    "title": "A study of a MET inhibitor in EGFR-mutant NSCLC after TKI progression",
    "status": "RECRUITING",
    "min_age": 18,
    "max_age": 75,
    "sex": "ALL",
    "healthy_volunteers": False,
    "eligibility_criteria": (
        "Inclusion: histologically confirmed NSCLC; documented EGFR activating "
        "mutation; progression on a prior EGFR TKI; ECOG 0-1. "
        "Exclusion: prior MET-directed therapy; symptomatic brain metastases; "
        "pregnant or breastfeeding."
    ),
}


def main() -> int:
    prompt = prompts.REASONER_INSTRUCTION.format(
        patient_profile_json=json.dumps(PATIENT),
        current_candidate_json=json.dumps(CANDIDATE),
    )
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "format": TrialVerdict.model_json_schema(),
        "options": {"temperature": 0},
    }

    print(f"Model: {MODEL}\nSending eligibility prompt to Ollama ...")
    t0 = time.perf_counter()
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=180)
        r.raise_for_status()
    except requests.exceptions.ConnectionError:
        print("\nCould not reach Ollama at localhost:11434. Is the Ollama app running?", file=sys.stderr)
        return 1
    except requests.exceptions.HTTPError as exc:
        print(f"\nOllama returned an error: {exc}\n{r.text}", file=sys.stderr)
        return 1
    wall = time.perf_counter() - t0

    content = r.json()["message"]["content"]
    try:
        verdict = TrialVerdict.model_validate_json(content)
    except Exception as exc:
        print(f"\nGot a response but it did not parse into TrialVerdict: {exc}")
        print("Raw content:\n", content, file=sys.stderr)
        return 1

    print(f"\n--- WIRING OK ({wall:.1f}s) ---")
    print(f"label:         {verdict.label.value}")
    print(f"confidence:    {verdict.confidence}")
    print(f"unknown_count: {verdict.unknown_count}")
    print(f"criteria:      {len(verdict.criteria)} judged")
    print(f"summary:       {verdict.summary}")
    print("\nFull verdict:")
    print(json.dumps(verdict.model_dump(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
