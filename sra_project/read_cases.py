import glob, json

NCTS = ["NCT00002620","NCT00003176","NCT00003465","NCT00001756","NCT00001760",
        "NCT00006413","NCT00047918","NCT00132015","NCT00233454","NCT00004062",
        "NCT00062348","NCT00084435"]

def _norm(s): return " ".join((s or "").split()).lower()
def load(name):
    hits = glob.glob(name) or glob.glob(f"**/{name}", recursive=True)
    return json.load(open(hits[0], encoding="utf-8")) if hits else None

def main():
    dump = {c["nct"]: c for c in (load("audit_dump.json") or [])}
    cache = load("trial_cache.json") or {}
    out = []
    for nct in NCTS:
        c = dump.get(nct)
        if not c:
            out.append(f"\n{'='*80}\n{nct}  -- NOT FOUND in audit_dump.json\n"); continue
        full = (cache.get(nct) or {}).get("eligibility_criteria") or c.get("trial_criteria","")
        hay = _norm(full)
        gold = "eligible" if c["gold"]==1 else "INELIGIBLE"
        agent = "surface/eligible" if c["agent_pred"]==1 else "skip/ineligible"
        out += [f"\n{'='*80}", f"{nct}   gold = {gold}   agent said = {agent}", "-"*80,
                "PATIENT HISTORY:", "  "+(c.get("patient_history","") or "(none)"),
                "\nFULL TRIAL CRITERIA:", "  "+(full or "(none)"),
                "\nAGENT'S PER-CRITERION VERDICTS (evidence checked vs criteria):"]
        crits = c.get("criteria") or []
        if not crits: out.append("  (no criteria - possible bucket D artifact)")
        for cr in crits:
            ev = _norm(cr.get("evidence",""))
            tag = "[grounded]" if (ev and ev in hay) else ("[no-evidence]" if not ev else "[FABRICATED]")
            out.append(f"  - [{cr.get('kind','?')}/{cr.get('verdict','?')}] {cr.get('criterion','')[:90]}")
            out.append(f"      {tag} evidence: {cr.get('evidence','')[:100]}")
        out.append(f"\nAGENT SUMMARY (reference only, do NOT audit from this): {c.get('summary','')[:200]}")
    report = "\n".join(out)
    open("audit_review.txt","w",encoding="utf-8").write(report)
    print(report)
    print(f"\n\nWrote audit_review.txt. Audit each from PATIENT + FULL CRITERIA, not the summary.")

if __name__ == "__main__":
    main()
