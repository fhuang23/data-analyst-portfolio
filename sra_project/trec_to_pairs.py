"""Convert trec_cases.json -> pairs.jsonl for the eval harness."""
from __future__ import annotations

import glob
import json
import re
import sys
from collections import Counter

from sra_matcher.eval.data import GOLD_FROM_TREC

_AGE_RE = re.compile(r"(\d+)")


def parse_age(value):
    if value is None:
        return None
    if isinstance(value, int):
        return value
    m = _AGE_RE.search(str(value))
    return int(m.group(1)) if m else None


def to_gold(raw):
    try:
        return GOLD_FROM_TREC[int(raw)].value
    except (ValueError, TypeError, KeyError):
        return str(raw)


def find_input():
    if len(sys.argv) > 1:
        return sys.argv[1]
    hits = glob.glob("trec_cases.json") or glob.glob("**/trec_cases.json", recursive=True)
    if not hits:
        sys.exit("Could not find trec_cases.json - pass its path as the first argument.")
    return hits[0]


def main():
    in_path = find_input()
    out_path = sys.argv[2] if len(sys.argv) > 2 else "pairs.jsonl"

    cases = json.load(open(in_path, encoding="utf-8"))
    balance = Counter()
    written = 0

    with open(out_path, "w", encoding="utf-8") as out:
        for i, case in enumerate(cases):
            patient = case["patient"]
            trial = case["trial"]

            profile = {
                "conditions": patient.get("conditions") or [],
                "salient_history": patient.get("salient_history", ""),
            }
            candidate = {
                "nct_id": trial.get("nct_id", ""),
                "title": trial.get("title", ""),
                "conditions": trial.get("conditions") or [],
                "min_age": parse_age(trial.get("min_age")),
                "max_age": parse_age(trial.get("max_age")),
                "sex": trial.get("sex") or "ALL",
                "eligibility_criteria": trial.get("eligibility_criteria", ""),
            }
            gold = to_gold(case.get("gold"))
            balance[gold] += 1

            out.write(json.dumps({
                "patient_id": str(i),
                "nct_id": candidate["nct_id"],
                "patient_profile": profile,
                "candidate": candidate,
                "gold": gold,
            }) + "\n")
            written += 1

    print(f"Wrote {written} pairs -> {out_path}")
    print("Gold class balance:", dict(balance))
    if balance.get("eligible", 0) == 0:
        print("WARNING: no 'eligible' pairs - check GOLD_FROM_TREC in sra_matcher/eval/data.py.")
    print("\nNote: pairs carry raw salient_history, not a structured PatientProfile. "
          "The rules baseline will look weak - that's expected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
