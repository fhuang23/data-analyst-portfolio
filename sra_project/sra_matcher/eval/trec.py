"""TREC Clinical Trials adapter.

Turns TREC CT (2021/2022) topics + qrels into the harness's EvalCase shape, so
the same scorer that runs on the synthetic fixtures runs on a third-party,
physician-judged benchmark.

WHAT THIS IS (and isn't):
  TREC CT is a *retrieval* task scored by NDCG over a ~400k-trial corpus. This
  adapter does NOT run retrieval. It repurposes the qrels as labeled patient-trial
  pairs to evaluate the per-pair eligibility CLASSIFIER. Report the classification
  F1 as such; it is not comparable to TREC NDCG leaderboard numbers. The judged
  pairs are a pooled (biased) sample, and criteria fetched live from CT.gov may
  drift from the original snapshot -- state both caveats.

INPUTS (public, from trec.nist.gov / trec-cds.org):
  topics: XML  <topics><topic number="1">patient case text...</topic>...</topics>
  qrels:  TREC 4-col whitespace:  <topic_id> 0 <nct_id> <relevance 0|1|2>

LABEL MAP (binary "surface?"):  qrel 2 -> 1 (eligible);  qrel 0/1 -> 0.

USAGE:
    python -m sra_matcher.eval.trec --topics topics.xml --qrels qrels.txt \
        --n-topics 15 --max-per-topic 40 --seed 0
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import xml.etree.ElementTree as ET

from .. import config, state
from .fixtures import EvalCase
from .harness import evaluate


# ---- parsing -----------------------------------------------------------------
def parse_topics(path: str) -> dict[str, str]:
    """topic_id -> patient case text.

    TREC topic files are not clean XML -- the case narratives contain bare '&'
    (e.g. "SOB & LE edema") and other stray tokens that break a strict parser.
    So: escape unescaped ampersands, try XML, and fall back to a regex extract.
    """
    import re
    raw = open(path, encoding="utf-8", errors="replace").read()
    fixed = re.sub(r"&(?!(amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);)", "&amp;", raw)

    topics: dict[str, str] = {}
    try:
        root = ET.fromstring(fixed)
        for t in root.iter("topic"):
            tid = t.get("number") or t.get("id")
            body = "".join(t.itertext()).strip()
            if tid and body:
                topics[str(tid)] = " ".join(body.split())
        if topics:
            return topics
    except ET.ParseError:
        pass

    # Regex fallback for any remaining malformed markup.
    for m in re.finditer(r"<topic[^>]*(?:number|id)=\"(\d+)\"[^>]*>(.*?)</topic>", raw, re.S):
        body = re.sub(r"<[^>]+>", " ", m.group(2))
        topics[m.group(1)] = " ".join(body.split())
    return topics


def parse_qrels(path: str) -> list[tuple[str, str, int]]:
    """List of (topic_id, nct_id, relevance). TREC 4-col whitespace format."""
    out = []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) != 4:
                continue
            topic_id, _iter, nct_id, rel = parts
            out.append((topic_id, nct_id, int(rel)))
    return out


# ---- sampling ----------------------------------------------------------------
def sample_pairs(qrels, *, n_topics=15, max_per_topic=40, neg_ratio=2.0, seed=0):
    """Keep cost bounded. Per selected topic: all eligible (2) and excluded (1)
    pairs, plus a capped sample of not-relevant (0) at `neg_ratio` x positives.
    """
    rng = random.Random(seed)
    by_topic: dict[str, list[tuple[str, int]]] = {}
    for tid, nct, rel in qrels:
        by_topic.setdefault(tid, []).append((nct, rel))

    chosen_topics = sorted(by_topic)[:n_topics]
    pairs = []
    for tid in chosen_topics:
        rows = by_topic[tid]
        pos = [(nct, rel) for nct, rel in rows if rel in (1, 2)]
        neg = [(nct, rel) for nct, rel in rows if rel == 0]
        rng.shuffle(neg)
        keep_neg = neg[: int(len(pos) * neg_ratio)]
        picked = (pos + keep_neg)[:max_per_topic]
        for nct, rel in picked:
            pairs.append((tid, nct, rel))
    return pairs


# ---- patient structuring -----------------------------------------------------
async def structure_patient(text: str) -> dict:
    """Run the project's intake agent to turn a TREC topic into a PatientProfile.
    Falls back to a raw-text profile if intake yields nothing.
    """
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types
    from ..agents import intake_agent

    svc = InMemorySessionService()
    runner = Runner(agent=intake_agent, app_name="sra_intake", session_service=svc)
    await svc.create_session(app_name="sra_intake", user_id="eval", session_id="s")
    msg = types.Content(role="user", parts=[types.Part(text=text)])
    async for _ in runner.run_async(user_id="eval", session_id="s", new_message=msg):
        pass
    sess = await svc.get_session(app_name="sra_intake", user_id="eval", session_id="s")
    profile = sess.state.get(state.PATIENT_PROFILE)
    if hasattr(profile, "model_dump"):
        profile = profile.model_dump()
    return profile or {"salient_history": text, "conditions": []}


# ---- build cases -------------------------------------------------------------
async def build_cases(pairs, topics, *, structure=True):
    from ..tools.ctgov import get_trial_by_nct

    profile_cache: dict[str, dict] = {}
    trial_cache: dict[str, dict] = {}
    cases: list[EvalCase] = []

    for tid, nct, rel in pairs:
        if tid not in topics:
            continue
        if tid not in profile_cache:
            text = topics[tid]
            profile_cache[tid] = (await structure_patient(text)) if structure \
                else {"salient_history": text, "conditions": []}
        if nct not in trial_cache:
            trial_cache[nct] = get_trial_by_nct(nct) or {}
        trial = trial_cache[nct]
        if not trial.get("eligibility_criteria"):
            continue  # skip unjudgeable pairs with no criteria text
        cases.append(EvalCase(
            trial=trial,
            gold_label=1 if rel == 2 else 0,
            note=f"topic {tid}, qrel {rel}",
            patient=profile_cache[tid],
        ))
    return cases


async def main_async(args):
    topics = parse_topics(args.topics)
    qrels = parse_qrels(args.qrels)
    pairs = sample_pairs(qrels, n_topics=args.n_topics,
                         max_per_topic=args.max_per_topic, seed=args.seed)
    print(f"Loaded {len(topics)} topics, {len(qrels)} qrels; sampled {len(pairs)} pairs.")
    cases = await build_cases(pairs, topics, structure=not args.no_intake)
    print(f"Built {len(cases)} scoreable cases "
          f"({sum(c.gold_label for c in cases)} eligible / "
          f"{sum(1 for c in cases if not c.gold_label)} not).")
    await evaluate(cases, dry_run=args.dry_run, w_fn=args.w_fn, w_fp=args.w_fp)


def main():
    ap = argparse.ArgumentParser(description="Evaluate on TREC Clinical Trials qrels")
    ap.add_argument("--topics", required=True, help="TREC topics XML")
    ap.add_argument("--qrels", required=True, help="TREC qrels file")
    ap.add_argument("--n-topics", type=int, default=15)
    ap.add_argument("--max-per-topic", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-intake", action="store_true",
                    help="skip intake structuring; pass raw topic text as the profile")
    ap.add_argument("--dry-run", action="store_true", help="baseline only, no model")
    ap.add_argument("--w-fn", type=float, default=10.0)
    ap.add_argument("--w-fp", type=float, default=1.0)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
