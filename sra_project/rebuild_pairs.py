"""Rebuild a BALANCED pairs.jsonl from the TREC 2021 source files."""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict

import requests

CT_URL = "https://clinicaltrials.gov/api/v2/studies/{}"
CACHE = "trial_cache.json"


def _find(name):
    hits = glob.glob(name) or glob.glob(f"**/{name}", recursive=True)
    if not hits:
        sys.exit(f"Could not find {name} - run from the project root.")
    return hits[0]


def load_topics():
    text = open(_find("topics2021.xml"), encoding="utf-8").read()
    out = {}
    for m in re.finditer(r'<topic\s+number="(\d+)"\s*>(.*?)</topic>', text, re.DOTALL):
        out[m.group(1)] = m.group(2).strip()
    return out


def load_qrels():
    rows = []
    for line in open(_find("qrels2021.txt"), encoding="utf-8"):
        parts = line.split()
        if len(parts) == 4:
            rows.append((parts[0], parts[2], int(parts[3])))
    return rows


def seed_cache():
    cache = {}
    if os.path.exists(CACHE):
        cache.update(json.load(open(CACHE, encoding="utf-8")))
    hits = glob.glob("trec_cases.json") or glob.glob("**/trec_cases.json", recursive=True)
    if hits:
        for case in json.load(open(hits[0], encoding="utf-8")):
            tr = case.get("trial", {})
            nct = tr.get("nct_id")
            if nct and nct not in cache:
                cache[nct] = tr
    return cache


def _dig(d, *path, default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


def fetch_trial(nct_id):
    try:
        r = requests.get(CT_URL.format(nct_id), params={"format": "json"}, timeout=30)
        if r.status_code == 404:
            return None
        r.raise_for_status()
    except requests.RequestException:
        return None
    study = r.json()
    if "studies" in study:
        studies = study["studies"]
        if not studies:
            return None
        study = studies[0]
    ps = study.get("protocolSection", study)
    elig = _dig(ps, "eligibilityModule", default={})
    return {
        "nct_id": nct_id,
        "title": _dig(ps, "identificationModule", "briefTitle", default=""),
        "conditions": _dig(ps, "conditionsModule", "conditions", default=[]),
        "min_age": elig.get("minimumAge", ""),
        "max_age": elig.get("maximumAge", ""),
        "sex": elig.get("sex", "ALL"),
        "eligibility_criteria": elig.get("eligibilityCriteria", ""),
    }


def _age(v):
    if v is None:
        return None
    m = re.search(r"(\d+)", str(v))
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", type=int, default=100)
    ap.add_argument("--neg1", type=int, default=50)
    ap.add_argument("--neg0", type=int, default=50)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default="pairs.jsonl")
    args = ap.parse_args()

    random.seed(args.seed)
    topics = load_topics()
    rows = load_qrels()

    buckets = defaultdict(list)
    for tid, nct, rel in rows:
        if tid in topics:
            buckets[rel].append((tid, nct))
    for rel in buckets:
        random.shuffle(buckets[rel])

    plan = [(2, args.pos), (1, args.neg1), (0, args.neg0)]
    sample = []
    for rel, n in plan:
        sample.extend((tid, nct, rel) for tid, nct in buckets[rel][:n])
    random.shuffle(sample)

    cache = seed_cache()
    need = sorted({nct for _, nct, _ in sample if nct not in cache})
    print(f"Sampled {len(sample)} pairs; fetching {len(need)} new trials ({len(cache)} cached) ...")
    for i, nct in enumerate(need, 1):
        got = fetch_trial(nct)
        if got:
            cache[nct] = got
        if i % 25 == 0:
            print(f"  fetched {i}/{len(need)}")
        time.sleep(0.12)
    json.dump(cache, open(CACHE, "w"), indent=0)

    gold_map = {2: "eligible", 1: "ineligible", 0: "ineligible"}
    balance = Counter()
    written = dropped = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for tid, nct, rel in sample:
            trial = cache.get(nct)
            if not trial or not trial.get("eligibility_criteria"):
                dropped += 1
                continue
            profile = {"conditions": [], "salient_history": topics[tid]}
            candidate = {
                "nct_id": nct,
                "title": trial.get("title", ""),
                "conditions": trial.get("conditions") or [],
                "min_age": _age(trial.get("min_age")),
                "max_age": _age(trial.get("max_age")),
                "sex": trial.get("sex") or "ALL",
                "eligibility_criteria": trial.get("eligibility_criteria", ""),
            }
            gold = gold_map[rel]
            balance[gold] += 1
            out.write(json.dumps({
                "patient_id": tid, "nct_id": nct,
                "patient_profile": profile, "candidate": candidate, "gold": gold,
            }) + "\n")
            written += 1

    print(f"\nWrote {written} pairs -> {args.out}  (dropped {dropped} with no trial text)")
    print("Gold balance:", dict(balance))
    pos = balance.get("eligible", 0)
    if pos:
        print(f"Positives: {pos}  ({pos / max(written,1):.0%} of the set)")
    else:
        print("WARNING: still no eligible pairs - check the qrels parse.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
