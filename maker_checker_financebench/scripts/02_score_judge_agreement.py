#!/usr/bin/env python3
"""Step 2b: validate the judge against your human labels.

Reads eval/validation_subset.csv (with human_label filled in), runs the judge on
each model_answer vs gold_answer, and reports judge-vs-human agreement and
Cohen's kappa. Writes eval/judge_agreement.md and a scored CSV.

Gate: only trust the automated judge for the full 150-question run if kappa
clears the bar you set (default 0.8). If it does not, read the disagreement log,
fix the rubric or tolerance, and re-run.

Usage:
    python scripts/02_score_judge_agreement.py --kappa-bar 0.8
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from judge import judge_answer  # noqa: E402
from metrics import raw_agreement, cohens_kappa, confusion  # noqa: E402

SUBSET = os.path.join(os.path.dirname(__file__), "..", "eval", "validation_subset.csv")
REPORT = os.path.join(os.path.dirname(__file__), "..", "eval", "judge_agreement.md")
SCORED = os.path.join(os.path.dirname(__file__), "..", "eval", "validation_scored.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kappa-bar", type=float, default=0.8)
    args = ap.parse_args()

    with open(SUBSET) as fh:
        rows = list(csv.DictReader(fh))

    labeled = [r for r in rows if r.get("human_label", "").strip()]
    if not labeled:
        sys.exit("No human_label values found. Fill that column in "
                 f"{os.path.relpath(SUBSET)} first.")
    if len(labeled) < len(rows):
        print(f"warning: {len(rows) - len(labeled)} rows unlabeled; using "
              f"{len(labeled)} labeled rows.")

    llm_call = None  # created lazily only if an item needs the LLM path

    human, judge_labels, disagreements = [], [], []
    for r in labeled:
        gold, cand = r["gold_answer"], r["model_answer"]
        try:
            jr = judge_answer(r["question"], gold, r.get("justification", ""), cand,
                              llm_call=_lazy_llm(llm_call))
        except RuntimeError:
            # First LLM-path item: build the call now and retry.
            from models import make_llm_call, DEFAULT_JUDGE_MODEL
            llm_call = make_llm_call(DEFAULT_JUDGE_MODEL)
            print(f"judge model: {DEFAULT_JUDGE_MODEL}")
            jr = judge_answer(r["question"], gold, r.get("justification", ""), cand,
                              llm_call=llm_call)
        hlabel = r["human_label"].strip().lower()
        human.append(hlabel)
        judge_labels.append(jr.verdict)
        r["judge_verdict"] = jr.verdict
        r["judge_path"] = jr.path
        if (hlabel == "correct") != (jr.verdict == "correct"):
            disagreements.append((r, jr))

    agr = raw_agreement(human, judge_labels)
    kappa = cohens_kappa(
        ["correct" if h == "correct" else "incorrect" for h in human],
        ["correct" if j == "correct" else "incorrect" for j in judge_labels],
    )
    cm = confusion(human, judge_labels)

    with open(SCORED, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(labeled[0].keys()))
        w.writeheader()
        w.writerows(labeled)

    lines = [
        "# Judge-vs-Human Agreement\n",
        f"- items labeled: **{len(labeled)}**",
        f"- raw agreement: **{agr:.3f}**",
        f"- Cohen's kappa (binary correct/not): **{kappa:.3f}**",
        f"- kappa bar: {args.kappa_bar} -> "
        f"**{'PASS' if kappa >= args.kappa_bar else 'FAIL, revise rubric'}**",
        "",
        "## Confusion (judge relative to human)",
        f"- true-correct: {cm['tp']}  true-incorrect: {cm['tn']}",
        f"- judge lenient (said correct, human said not): {cm['fp']}",
        f"- judge harsh (said not, human said correct): {cm['fn']}",
        "",
        "## Disagreements (audit these)",
    ]
    if not disagreements:
        lines.append("_none_")
    for r, jr in disagreements:
        lines.append(
            f"- `{r['financebench_id']}` [{r['predicted_judge_path']}] "
            f"human={r['human_label']} judge={jr.verdict} via {jr.path}\n"
            f"  - Q: {r['question'][:110]}\n"
            f"  - gold: {r['gold_answer'][:80]} | model: {r['model_answer'][:80]}\n"
            f"  - judge reason: {jr.reason}"
        )

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print("\n".join(lines[:6]))
    print(f"\nwrote {os.path.relpath(REPORT)} and {os.path.relpath(SCORED)}")


def _lazy_llm(call):
    """Return the call if built, else a sentinel that makes judge_answer raise
    so we can build it on first need."""
    return call


if __name__ == "__main__":
    main()
