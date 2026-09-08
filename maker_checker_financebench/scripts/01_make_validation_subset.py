#!/usr/bin/env python3
"""Step 1-2a: build the judge-validation subset.

Design note: the regenerable artifact and your hand-labeled artifact are kept in
SEPARATE files so a re-run can never destroy your labels.

  eval/validation_subset_template.csv  <- generated here; safe to overwrite
  eval/validation_subset.csv           <- your working copy; you label THIS one

On run:
  * always (re)writes the template.
  * creates the labeled working file from the template ONLY if it does not exist
    or has no labels yet.
  * if the working file already has labels, it is left untouched and the script
    says so. Delete it deliberately if you truly want to start over.

Usage:
    python scripts/01_make_validation_subset.py --n 45
    python scripts/01_make_validation_subset.py --n 45 --dry-run   # no API calls
    python scripts/01_make_validation_subset.py --n 45 --force     # overwrite labels
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_io import load_questions, stratified_sample  # noqa: E402
from judge import _looks_numeric_answer  # noqa: E402
from retrieval import oracle_context  # noqa: E402

EVAL_DIR = os.path.join(os.path.dirname(__file__), "..", "eval")
TEMPLATE = os.path.join(EVAL_DIR, "validation_subset_template.csv")
WORKING = os.path.join(EVAL_DIR, "validation_subset.csv")

FIELDS = [
    "financebench_id", "company", "doc_name", "question_type", "reasoning_bucket",
    "predicted_judge_path", "question", "gold_answer", "model_answer",
    "human_label", "notes",
]


def _has_labels(path: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                if str(row.get("human_label", "")).strip():
                    return True
    except (KeyError, csv.Error):
        pass
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=45)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="overwrite the labeled working file even if it has labels")
    args = ap.parse_args()

    qs = load_questions()
    sample = stratified_sample(qs, n=args.n, seed=args.seed)

    llm_call = None
    if not args.dry_run:
        from models import make_llm_call
        from maker import run_maker
        llm_call = make_llm_call("maker")
        print("generating answers with the maker model")

    rows = []
    for i, q in enumerate(sample, 1):
        model_answer = ""
        if not args.dry_run:
            out = run_maker(q.question, oracle_context(q), llm_call)
            model_answer = out.answer if out.can_answer else "[ABSTAIN]"
            print(f"  [{i}/{len(sample)}] {q.financebench_id}")
        rows.append({
            "financebench_id": q.financebench_id, "company": q.company,
            "doc_name": q.doc_name, "question_type": q.question_type,
            "reasoning_bucket": q.reasoning_bucket,
            "predicted_judge_path": "numeric" if _looks_numeric_answer(q.answer) else "llm",
            "question": q.question, "gold_answer": q.answer,
            "model_answer": model_answer, "human_label": "", "notes": "",
        })

    os.makedirs(EVAL_DIR, exist_ok=True)
    with open(TEMPLATE, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader(); w.writerows(rows)
    print(f"wrote template -> {os.path.relpath(TEMPLATE)}")

    # Protect the labeled working file.
    if _has_labels(WORKING) and not args.force:
        print(f"NOT touching {os.path.relpath(WORKING)} -- it already has labels. "
              f"Use --force to overwrite, or delete it first to start over.")
        return
    shutil.copyfile(TEMPLATE, WORKING)
    print(f"created working copy to label -> {os.path.relpath(WORKING)}")
    print("next: fill the 'human_label' column, then run 02_score_judge_agreement.py")


if __name__ == "__main__":
    main()
