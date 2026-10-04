import asyncio, json
from sra_matcher import state
from sra_matcher.eval.baseline import predict as baseline_predict
from sra_matcher.eval.trec import parse_topics, parse_qrels, sample_pairs, build_cases

SURFACE = {"eligible", "uncertain"}
OUT = "gemini_audit_dump.json"

async def score_one(runner, svc, case, i):
    from google.genai import types
    sid = f"g-{i}"
    init = {state.PATIENT_PROFILE_JSON: json.dumps(case.patient or {}),
            state.CURRENT_CANDIDATE_JSON: json.dumps(case.trial)}
    await svc.create_session(app_name="gdump", user_id="g", session_id=sid, state=init)
    msg = types.Content(role="user", parts=[types.Part(text="Evaluate eligibility.")])
    try:
        async for _ in runner.run_async(user_id="g", session_id=sid, new_message=msg): pass
    except Exception as e:
        return {"error": str(e)[:200]}
    sess = await svc.get_session(app_name="gdump", user_id="g", session_id=sid)
    v = sess.state.get(state.CURRENT_VERDICT)
    if hasattr(v, "model_dump"): v = v.model_dump()
    return v or {}

async def main():
    topics = parse_topics("topics2021.xml")
    qrels = parse_qrels("qrels2021.txt")
    pairs = sample_pairs(qrels, n_topics=20, max_per_topic=10, seed=0)
    cases = await build_cases(pairs, topics, structure=False)
    print(f"Built {len(cases)} cases. Scoring on Gemini...")
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from sra_matcher.agents import build_reasoner
    reasoner = build_reasoner("gemini"); svc = InMemorySessionService()
    runner = Runner(agent=reasoner, app_name="gdump", session_service=svc)
    rows, dis = [], 0
    for i, c in enumerate(cases):
        v = await score_one(runner, svc, c, i)
        label = v.get("label"); agent_pred = 1 if label in SURFACE else 0
        wrong = agent_pred != c.gold_label; dis += int(wrong)
        rows.append({"nct": c.trial.get("nct_id"), "note": c.note, "gold": c.gold_label,
            "agent_pred": agent_pred, "base_pred": baseline_predict(c.patient or {}, c.trial),
            "flagged_wrong": wrong, "verdict_label": label, "confidence": v.get("confidence"),
            "unknown_count": v.get("unknown_count"), "summary": v.get("summary",""),
            "criteria": v.get("criteria", []), "error": v.get("error"),
            "patient_history": (c.patient or {}).get("salient_history",""),
            "trial_criteria": c.trial.get("eligibility_criteria","")})
        if (i+1) % 20 == 0: print(f"  scored {i+1}/{len(cases)} ({dis} disagreements)")
    json.dump(rows, open(OUT,"w"), indent=1)
    print(f"\nWrote {OUT}: {len(rows)} cases, {dis} disagreements to audit.")

if __name__ == "__main__":
    asyncio.run(main())
