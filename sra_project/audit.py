"""Audit the flagged cases: re-run the same slice, dump full reasoning per case."""
from __future__ import annotations

import asyncio
import json
import sys

from sra_matcher import state
from sra_matcher.eval import baseline
from sra_matcher.eval.trec import parse_topics, parse_qrels, sample_pairs, build_cases


async def score_one(runner, session_service, case, i):
    from google.genai import types
    sid = f"audit-{i}"
    init = {
        state.PATIENT_PROFILE_JSON: json.dumps(case.patient or {}),
        state.CURRENT_CANDIDATE_JSON: json.dumps(case.trial),
    }
    await session_service.create_session(app_name="audit", user_id="a", session_id=sid, state=init)
    msg = types.Content(role="user", parts=[types.Part(text="Evaluate eligibility.")])
    try:
        async for _ in runner.run_async(user_id="a", session_id=sid, new_message=msg):
            pass
    except Exception as e:
        return {"error": str(e)[:200]}
    sess = await session_service.get_session(app_name="audit", user_id="a", session_id=sid)
    v = sess.state.get(state.CURRENT_VERDICT)
    if hasattr(v, "model_dump"):
        v = v.model_dump()
    return v or {}


async def main():
    topics = parse_topics("topics2021.xml")
    qrels = parse_qrels("qrels2021.txt")
    pairs = sample_pairs(qrels, n_topics=3, max_per_topic=10, seed=0)
    cases = await build_cases(pairs, topics, structure=False)  # --no-intake

    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from sra_matcher.agents import build_reasoner
    reasoner = build_reasoner("qwen")
    svc = InMemorySessionService()
    runner = Runner(agent=reasoner, app_name="audit", session_service=svc)

    SURFACE = {"eligible", "uncertain"}
    report = []
    for i, c in enumerate(cases):
        v = await score_one(runner, svc, c, i)
        label = v.get("label")
        agent_pred = 1 if label in SURFACE else 0
        base_pred = baseline.predict(c.patient or {}, c.trial)
        wrong = agent_pred != c.gold_label
        report.append({
            "nct": c.trial.get("nct_id"),
            "note": c.note,
            "gold": c.gold_label,
            "agent_pred": agent_pred,
            "base_pred": base_pred,
            "flagged_wrong": wrong,
            "verdict_label": label,
            "confidence": v.get("confidence"),
            "unknown_count": v.get("unknown_count"),
            "summary": v.get("summary"),
            "criteria": v.get("criteria", []),
            "error": v.get("error"),
            "patient_history": (c.patient or {}).get("salient_history", "")[:1500],
            "trial_criteria": c.trial.get("eligibility_criteria", "")[:2500],
        })
        mark = " <-- WRONG" if wrong else ""
        print(f"[{i:2}] {c.nct if hasattr(c,'nct') else c.trial.get('nct_id')} "
              f"gold={c.gold_label} agent={agent_pred} conf={v.get('confidence')}{mark}")

    with open("audit_dump.json", "w") as f:
        json.dump(report, f, indent=2)
    wrong = [r for r in report if r["flagged_wrong"]]
    print(f"\n{len(wrong)}/{len(report)} flagged wrong. Full detail in audit_dump.json")


if __name__ == "__main__":
    asyncio.run(main())
