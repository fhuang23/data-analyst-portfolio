"""Validate the judge against YOUR hand-audited buckets before trusting it.

This is the step that makes the judge defensible instead of circular. You already
bucketed some disagreements by hand (e.g. the 1.7B's 12: A:5 B:5 C:0 D:2). Record
which NCT went in which bucket in human_buckets below, run the judge on those exact
cases, and measure agreement. Report that agreement number; do NOT tune the judge
to hit your buckets (that would make the validation meaningless).

Layer-1 buckets (D artifact, A-by-inconsistency, A-by-ungrounding) are deterministic
and should match perfectly. The interesting agreement number is on the layer-2
contestable cases (A vs B vs C), where the LLM triage is doing the work.

Fill in HUMAN_BUCKETS from your own audit, point DUMP at your audit_dump.json (which
holds each case's agent verdict + criteria), set JUDGE_BACKEND to a model DIFFERENT
from the reasoner you audited, and run:
    python validate_judge.py
"""
from __future__ import annotations

import asyncio
import json
from collections import Counter

from sra_matcher.judge import Bucket, build_triage_agent, judge_case, TriageVerdict

# 1) YOUR hand-assigned buckets. nct -> Bucket. Fill from your own audit.
HUMAN_BUCKETS: dict[str, Bucket] = {
    # "NCT00002620": Bucket.MODEL_ERROR,
    # "NCT00003176": Bucket.LABEL_DEBATABLE,
    # ... the 12 you bucketed by hand ...
}

DUMP = "audit_dump.json"        # holds per-case agent verdict + patient + criteria
JUDGE_BACKEND = "qwen"           # MUST differ from the reasoner you audited


async def _triage(agent, svc, case: dict) -> TriageVerdict | None:
    from google.adk.runners import Runner
    from google.genai import types
    from sra_matcher import state
    sid = case["nct"]
    init = {
        state.PATIENT_PROFILE_JSON: json.dumps({"salient_history": case.get("patient_history", ""),
                                                "conditions": []}),
        state.CURRENT_CANDIDATE_JSON: json.dumps({"nct_id": case["nct"],
                                                  "eligibility_criteria": case.get("trial_criteria", "")}),
        "agent_verdict_json": json.dumps({"label": case.get("verdict_label"),
                                          "criteria": case.get("criteria", [])}),
        "gold_label": "eligible" if case["gold"] == 1 else "ineligible",
    }
    runner = Runner(agent=agent, app_name="judge", session_service=svc)
    await svc.create_session(app_name="judge", user_id="j", session_id=sid, state=init)
    msg = types.Content(role="user", parts=[types.Part(text="Audit this disagreement.")])
    try:
        async for _ in runner.run_async(user_id="j", session_id=sid, new_message=msg):
            pass
    except Exception as e:
        print(f"  ! triage failed on {sid}: {str(e)[:100]}")
        return None
    sess = await svc.get_session(app_name="judge", user_id="j", session_id=sid)
    v = sess.session.state.get("triage_verdict") if hasattr(sess, "session") else \
        sess.state.get("triage_verdict")
    if hasattr(v, "model_dump"):
        return v
    return TriageVerdict.model_validate(v) if v else None


async def main():
    dump = {c["nct"]: c for c in json.load(open(DUMP))}
    from google.adk.sessions import InMemorySessionService
    agent = build_triage_agent(JUDGE_BACKEND)
    svc = InMemorySessionService()

    agree = 0
    rows = []
    for nct, human in HUMAN_BUCKETS.items():
        case = dump.get(nct)
        if not case:
            print(f"  ? {nct} not in dump, skipping")
            continue
        vd = {"nct_id": nct, "label": case.get("verdict_label"),
              "criteria": case.get("criteria", []), "confidence": case.get("confidence", 0.5),
              "unknown_count": case.get("unknown_count", 0)}
        # run layer-2 only if layer-1 is clean and it disagrees (judge_case handles the rest)
        triage = await _triage(agent, svc, case)
        report = judge_case(vd, case.get("trial_criteria", ""), case["agent_pred"],
                            case["gold"], triage=triage)
        match = report.bucket == human
        agree += int(match)
        rows.append((nct, human.value, report.bucket.value, "OK" if match else "DIFF", report.note[:70]))

    print(f"\n{'nct':13}{'human':20}{'judge':20}{'':6}note")
    for nct, h, j, m, note in rows:
        print(f"{nct:13}{h:20}{j:20}{m:6}{note}")
    n = len(rows)
    print(f"\nJudge/human agreement: {agree}/{n} = {agree/n:.0%}" if n else "no cases")
    print("human buckets:", dict(Counter(h for _, h, _, _, _ in rows)))
    print("judge buckets:", dict(Counter(j for _, _, j, _, _ in rows)))
    print("\nReport this agreement number as-is. If low, the judge isn't ready OR "
          "some human buckets were generous — read the DIFF rows and decide honestly.")


if __name__ == "__main__":
    asyncio.run(main())
