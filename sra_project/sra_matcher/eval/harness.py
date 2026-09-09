"""Evaluation harness.

Scores the project's real eligibility reasoner against labeled patient-trial
pairs, compares to the structured baseline, and reports F1 / lift / cost matrix /
faithfulness / abstention. Works on any list[EvalCase] -- the synthetic fixtures
or an adapter-built set (see eval/trec.py).

    python -m sra_matcher.eval.harness            # synthetic set, full run
    python -m sra_matcher.eval.harness --dry-run  # synthetic set, baseline only
"""
from __future__ import annotations

import argparse
import asyncio
import json

from .. import config, state
from ..schemas import TrialLabel
from . import baseline, metrics
from .fixtures import CASES as SYNTHETIC_CASES, PATIENT as DEFAULT_PATIENT

SURFACE_LABELS = {TrialLabel.ELIGIBLE.value, TrialLabel.UNCERTAIN.value}
PRICE_IN_PER_M = 0.75
PRICE_OUT_PER_M = 3.75


def _patient_of(case):
    return case.patient or DEFAULT_PATIENT


def _label_to_binary(label: str | None) -> int:
    return 1 if label in SURFACE_LABELS else 0


def _fmt(c: metrics.Confusion) -> str:
    return (f"P={c.precision():.3f} R={c.recall():.3f} F1={c.f1():.3f}  "
            f"(TP={c.tp} FP={c.fp} FN={c.fn} TN={c.tn})")


async def _score_one(session_service, runner, case, i, attempt):
    from google.genai import types
    sid = f"case-{i}-a{attempt}"
    init_state = {
        state.PATIENT_PROFILE_JSON: json.dumps(_patient_of(case)),
        state.CURRENT_CANDIDATE_JSON: json.dumps(case.trial),
    }
    await session_service.create_session(
        app_name="sra_eval", user_id="eval", session_id=sid, state=init_state)
    msg = types.Content(role="user", parts=[types.Part(text="Evaluate eligibility.")])
    usage = {"prompt": 0, "output": 0}
    async for event in runner.run_async(user_id="eval", session_id=sid, new_message=msg):
        um = getattr(event, "usage_metadata", None)
        if um:
            usage["prompt"] += getattr(um, "prompt_token_count", 0) or 0
            usage["output"] += getattr(um, "candidates_token_count", 0) or 0
    sess = await session_service.get_session(app_name="sra_eval", user_id="eval", session_id=sid)
    v = sess.state.get(state.CURRENT_VERDICT)
    if hasattr(v, "model_dump"):
        v = v.model_dump()
    v = v or {}
    return _label_to_binary(v.get("label")), (v.get("criteria", []) or []), usage


_TRANSIENT = ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "500", "INTERNAL", "DEADLINE")


async def _run_agent(cases):
    """Run the reasoner over every case, with a live counter and retry/backoff."""
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from ..agents import _reasoner

    session_service = InMemorySessionService()
    runner = Runner(agent=_reasoner, app_name="sra_eval", session_service=session_service)

    preds, verdicts = [], []
    usage = {"prompt": 0, "output": 0}
    total = len(cases)
    failures = 0

    for i, case in enumerate(cases):
        nct = case.trial.get("nct_id", "?")
        pred, crits, u = 0, [], {"prompt": 0, "output": 0}
        for attempt in range(5):
            try:
                pred, crits, u = await _score_one(session_service, runner, case, i, attempt)
                break
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                transient = any(t in msg for t in _TRANSIENT)
                if transient and attempt < 4:
                    wait = 2.0 * (2 ** attempt)
                    print(f"\n  [retry {attempt+1}/5] {nct}: transient error, waiting {wait:.0f}s",
                          flush=True)
                    await asyncio.sleep(wait)
                    continue
                failures += 1
                print(f"\n  [skip] {nct}: {msg[:90]} -> counted as ineligible", flush=True)
                break
        preds.append(pred)
        verdicts.append(crits)
        usage["prompt"] += u["prompt"]
        usage["output"] += u["output"]
        done = i + 1
        if done % 5 == 0 or done == total:
            print(f"\r  scored {done}/{total}...", end="", flush=True)
    print()  # newline after the counter
    if failures:
        print(f"  ({failures} case(s) failed after retries, counted as ineligible)")
    return preds, verdicts, usage


def _report(cases, base_preds, agent_preds, verdicts, usage, w_fn, w_fp):
    golds = [c.gold_label for c in cases]
    base_c = metrics.confusion(base_preds, golds)
    print("\n=== Structured baseline (condition + age + sex only) ===")
    print("  " + _fmt(base_c))
    b_cost, b_pc = metrics.weighted_cost(base_c, w_fn, w_fp)
    print(f"  cost (w_fn={w_fn}, w_fp={w_fp}): total={b_cost:.1f}  per_case={b_pc:.2f}")

    if agent_preds is None:
        print("\n(dry run: agent not evaluated)")
        return

    agent_c = metrics.confusion(agent_preds, golds)
    print("\n=== Eligibility reasoner (LLM) ===")
    print("  " + _fmt(agent_c))
    a_cost, a_pc = metrics.weighted_cost(agent_c, w_fn, w_fp)
    print(f"  cost (w_fn={w_fn}, w_fp={w_fp}): total={a_cost:.1f}  per_case={a_pc:.2f}")

    print("\n=== Lift (reasoner - baseline) ===")
    print(f"  F1:   {agent_c.f1() - base_c.f1():+.3f}")
    print(f"  cost: {a_cost - b_cost:+.1f}  (negative = cheaper errors)")

    pairs, unknown, total = [], 0, 0
    for case, crits in zip(cases, verdicts):
        for cr in crits:
            pairs.append((cr.get("evidence", ""), case.trial.get("eligibility_criteria", "")))
            total += 1
            if cr.get("verdict") == "unknown":
                unknown += 1
    print("\n=== Reasoning quality ===")
    print(f"  evidence faithfulness (verbatim in criteria): {metrics.faithfulness_rate(pairs):.3f}")
    if total:
        print(f"  abstention rate (unknown verdicts): {unknown/total:.3f} ({unknown}/{total} criteria)")

    if usage and (usage["prompt"] or usage["output"]):
        dollars = usage["prompt"]/1e6*PRICE_IN_PER_M + usage["output"]/1e6*PRICE_OUT_PER_M
        print("\n=== Cost ===")
        print(f"  tokens: {usage['prompt']} in / {usage['output']} out")
        print(f"  est. ${dollars:.4f} total  |  ${dollars/max(len(cases),1):.4f} per pair")

    print("\n=== Per-case ===")
    print(f"  {'nct':12} {'gold':4} {'base':4} {'agent':5}  note")
    for case, bp, ap in zip(cases, base_preds, agent_preds):
        flag = "" if ap == case.gold_label else "  <-- agent wrong"
        print(f"  {case.trial.get('nct_id','?'):12} {case.gold_label:^4} {bp:^4} {ap:^5}  {case.note}{flag}")


async def evaluate(cases, *, dry_run=False, w_fn=10.0, w_fp=1.0):
    """Score a list of EvalCase. The single entry point for any dataset."""
    base_preds = [baseline.predict(_patient_of(c), c.trial) for c in cases]
    if dry_run:
        _report(cases, base_preds, None, None, None, w_fn, w_fp)
        return
    agent_preds, verdicts, usage = await _run_agent(cases)
    _report(cases, base_preds, agent_preds, verdicts, usage, w_fn, w_fp)


def main() -> None:
    ap = argparse.ArgumentParser(description="SRA eligibility eval harness (synthetic set)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--w-fn", type=float, default=10.0)
    ap.add_argument("--w-fp", type=float, default=1.0)
    args = ap.parse_args()
    asyncio.run(evaluate(SYNTHETIC_CASES, dry_run=args.dry_run, w_fn=args.w_fn, w_fp=args.w_fp))


if __name__ == "__main__":
    main()
