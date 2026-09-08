#!/usr/bin/env python3
"""Run the pipeline over N independent passes and report results with variance.

The maker model is not deterministic at temperature=0, so a single pass is not
reproducible. Rather than fight that, we run the whole evaluation R times and
report each metric as mean +/- standard deviation across runs, plus per-item
answer stability (how often each item lands on the same correctness verdict).

This turns run-to-run wobble from a hidden liability into a stated confidence
band -- the honest way to report a stochastic pipeline.

Usage:
    python scripts/06_run_pipeline_multi.py --n 45 --runs 3 --mode oracle
    python scripts/06_run_pipeline_multi.py --n 12 --runs 3 --backend mock
"""
from __future__ import annotations

import argparse
import os
import statistics as stats
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_io import load_questions, stratified_sample          # noqa: E402
from retrieval import oracle_context, get_context               # noqa: E402
from maker import run_maker                                     # noqa: E402
from pipeline import run_pipeline                               # noqa: E402
from judge import judge_answer                                  # noqa: E402
from models import make_llm_call                                # noqa: E402


def _context(q, mode, all_docs):
    if mode == "oracle":
        return oracle_context(q)
    return get_context(q, mode, all_docs=all_docs if mode == "shared_store" else None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=45)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--mode", default="oracle",
                    choices=["oracle", "in_context", "single_store", "shared_store"])
    ap.add_argument("--backend", default=None, help="'mock' for a free dry run")
    args = ap.parse_args()

    qs = load_questions()
    sample = stratified_sample(qs, n=args.n, seed=13)
    all_docs = tuple(sorted({q.doc_name for q in qs}))

    maker_call = make_llm_call(args.backend or "maker")
    checker_call = make_llm_call(args.backend or "judge")
    adjudicator_call = make_llm_call(args.backend or "maker")
    judge_call = make_llm_call(args.backend or "judge")

    base_acc, pipe_cov, pipe_acc = [], [], []
    # per-item verdict tally across runs, for stability reporting
    item_correct = defaultdict(int)   # times the pipeline answered AND was correct
    item_answered = defaultdict(int)  # times the pipeline answered at all

    for r in range(1, args.runs + 1):
        base_correct = pipe_answered = pipe_correct = 0
        for q in sample:
            ctx = _context(q, args.mode, all_docs)

            m = run_maker(q.question, ctx, maker_call)
            base_ans = m.answer if m.can_answer else "[ABSTAIN]"
            if judge_answer(q.question, q.answer, q.justification, base_ans,
                            llm_call=judge_call).verdict == "correct":
                base_correct += 1

            pr = run_pipeline(q.question, ctx, q.reasoning_bucket,
                              maker_call, checker_call, adjudicator_call)
            if not pr.abstained:
                pipe_answered += 1
                item_answered[q.financebench_id] += 1
                if judge_answer(q.question, q.answer, q.justification, pr.answer,
                                llm_call=judge_call).verdict == "correct":
                    pipe_correct += 1
                    item_correct[q.financebench_id] += 1

        base_acc.append(base_correct / len(sample))
        pipe_cov.append(pipe_answered / len(sample))
        pipe_acc.append(pipe_correct / pipe_answered if pipe_answered else 0.0)
        print(f"  run {r}/{args.runs}: baseline={base_acc[-1]:.3f} "
              f"coverage={pipe_cov[-1]:.3f} answered_acc={pipe_acc[-1]:.3f}")

    def ms(xs):
        return (stats.mean(xs), stats.pstdev(xs) if len(xs) > 1 else 0.0)

    b_m, b_s = ms(base_acc); c_m, c_s = ms(pipe_cov); a_m, a_s = ms(pipe_acc)
    print(f"\n=== {args.runs}-run summary ({args.mode}, n={len(sample)}) ===")
    print(f"  single-shot accuracy:      {b_m:.3f} +/- {b_s:.3f}")
    print(f"  pipeline coverage:         {c_m:.3f} +/- {c_s:.3f}")
    print(f"  pipeline answered-accuracy:{a_m:.3f} +/- {a_s:.3f}")

    # Per-item stability: items whose correctness flipped across runs are the
    # unstable ones worth flagging in the writeup.
    unstable = [fid for fid in item_answered
                if 0 < item_correct[fid] < item_answered[fid]]
    print(f"\n  items answered every run that flipped correctness: {len(unstable)}")
    for fid in unstable:
        print(f"    {fid}: correct {item_correct[fid]}/{item_answered[fid]} runs")


if __name__ == "__main__":
    main()
