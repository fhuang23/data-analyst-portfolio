#!/usr/bin/env python3
"""Steps 5-6: run the maker-checker pipeline and measure it against the
single-shot maker baseline, using the validated judge as the scorer.

The central claim this tests: does the checker earn its place? It should raise
accuracy on answered items (by catching wrong answers and abstaining or fixing
them), at some cost in coverage. Reports both, in oracle mode by default so the
comparison isolates reasoning quality from retrieval.

Usage:
    python scripts/05_run_pipeline.py --n 45 --mode oracle
    python scripts/05_run_pipeline.py --n 45 --mode oracle --backend mock   # free dry run
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_io import load_questions, stratified_sample          # noqa: E402
from retrieval import oracle_context, get_context               # noqa: E402
from maker import run_maker                                     # noqa: E402
from pipeline import run_pipeline                               # noqa: E402
from judge import judge_answer                                  # noqa: E402
from metrics import coverage_at_floor, bootstrap_ci            # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "pipeline_eval.csv")


def _context(q, mode, all_docs):
    if mode == "oracle":
        return oracle_context(q)
    return get_context(q, mode, all_docs=all_docs if mode == "shared_store" else None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=45)
    ap.add_argument("--mode", default="oracle",
                    choices=["oracle", "in_context", "single_store", "shared_store"])
    ap.add_argument("--backend", default=None, help="'mock' for a free dry run")
    ap.add_argument("--floor", type=float, default=0.95)
    args = ap.parse_args()

    qs = load_questions()
    sample = stratified_sample(qs, n=args.n, seed=13)
    all_docs = tuple(sorted({q.doc_name for q in qs}))

    from models import make_llm_call
    maker_model = args.backend or None
    # roles: maker + adjudicator use the maker model; checker + judge can be cheaper.
    maker_call = make_llm_call(args.backend or "maker")
    checker_call = make_llm_call(args.backend or "judge")
    adjudicator_call = make_llm_call(args.backend or "maker")
    judge_call = make_llm_call(args.backend or "judge")

    rows = []
    base_labels, pipe_labels, pipe_abstain = [], [], []
    for i, q in enumerate(sample, 1):
        ctx = _context(q, args.mode, all_docs)

        # Baseline: single-shot maker, judged.
        m = run_maker(q.question, ctx, maker_call)
        base_ans = m.answer if m.can_answer else "[ABSTAIN]"
        base_v = judge_answer(q.question, q.answer, q.justification, base_ans,
                              llm_call=judge_call).verdict
        base_labels.append(base_v)

        # Full pipeline.
        pr = run_pipeline(q.question, ctx, q.reasoning_bucket,
                          maker_call, checker_call, adjudicator_call)
        if pr.abstained:
            pipe_abstain.append(True)
            pipe_labels.append("abstained")
            pipe_v = "abstained"
        else:
            pipe_abstain.append(False)
            pipe_v = judge_answer(q.question, q.answer, q.justification, pr.answer,
                                  llm_call=judge_call).verdict
            pipe_labels.append(pipe_v)

        rows.append({
            "financebench_id": q.financebench_id, "reasoning_bucket": q.reasoning_bucket,
            "baseline_verdict": base_v,
            "pipeline_path": pr.path, "pipeline_abstained": pr.abstained,
            "pipeline_verdict": pipe_v,
            "checker_verdict": pr.checker.verdict if pr.checker else "",
            "grounded": pr.checker.grounded if pr.checker else "",
        })
        print(f"  {i}/{len(sample)}", end="\r")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # Baseline accuracy over all items ([ABSTAIN] counts as wrong).
    base_correct = sum(v == "correct" for v in base_labels)
    b_pt, b_lo, b_hi = bootstrap_ci(base_correct, len(base_labels))

    # Pipeline: coverage + accuracy on answered items at the floor.
    cov = coverage_at_floor(pipe_labels, pipe_abstain, floor=args.floor)
    ans_correct = sum(v == "correct" for v, ab in zip(pipe_labels, pipe_abstain) if not ab)
    n_ans = int(cov["n_answered"])
    a_pt, a_lo, a_hi = bootstrap_ci(ans_correct, n_ans) if n_ans else (0, 0, 0)

    print(f"\n\n=== maker-checker vs single-shot ({args.mode}, n={len(sample)}) ===")
    print(f"  single-shot maker accuracy:   {b_pt:.3f}  [{b_lo:.3f}, {b_hi:.3f}]")
    print(f"  pipeline coverage (answered): {cov['coverage']:.3f}  ({n_ans}/{len(sample)})")
    print(f"  pipeline answered-accuracy:   {a_pt:.3f}  [{a_lo:.3f}, {a_hi:.3f}]")
    print(f"  meets {args.floor:.0%} floor:            {'yes' if cov['meets_floor'] else 'no'}")
    print(f"\n  abstained via: "
          f"maker_declined={sum(r['pipeline_path']=='maker_declined' for r in rows)}, "
          f"adjudicator={sum(r['pipeline_path']=='adjudicator_abstained' for r in rows)}")
    print(f"  wrote {os.path.relpath(OUT)}")


if __name__ == "__main__":
    main()
