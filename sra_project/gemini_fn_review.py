import glob, json

def _n(s): return " ".join((s or "").split()).lower()
def load(name):
    hits = glob.glob(name) or glob.glob(f"**/{name}", recursive=True)
    return json.load(open(hits[0], encoding="utf-8")) if hits else None
def wrap(text, width=96, indent="    "):
    words=(text or "").split(); lines=[]; cur=indent
    for w in words:
        if len(cur)+len(w)+1>width: lines.append(cur); cur=indent+w
        else: cur+=(" " if cur.strip() else "")+w
    if cur.strip(): lines.append(cur)
    return "\n".join(lines) or (indent+"(none)")

def main():
    dump=load("gemini_audit_dump.json") or []
    cache=load("trial_cache.json") or {}
    fns=[r for r in dump if r.get("gold")==1 and r.get("agent_pred")==0]
    out=[f"GEMINI FALSE-NEGATIVE REVIEW  ({len(fns)} cases)",
         "Gold said ELIGIBLE; Gemini rejected. Expensive errors (w_fn=10).",
         "Read PATIENT + CRITERIA, decide your own verdict, then fill MY BUCKET.",
         "  A = Gemini wrong   B = gold generous   C = couldn't tell from raw history",""]
    for i,r in enumerate(fns,1):
        nct=r["nct"]
        full=(cache.get(nct) or {}).get("eligibility_criteria") or r.get("trial_criteria","")
        hay=_n(full)
        out.append("="*100)
        out.append(f"[{i}/{len(fns)}]  {nct}   Gemini said: {r.get('verdict_label')}   conf: {r.get('confidence')}   unknowns: {r.get('unknown_count')}")
        out.append("-"*100)
        out.append("WHY GEMINI REJECTED:")
        drivers=[]
        for c in r.get("criteria",[]):
            vd=c.get("verdict"); kind=(c.get("kind") or "")
            if (vd=="not_met") or (kind.startswith("excl") and vd=="met"):
                ev=_n(c.get("evidence","")); tag="[grounded]" if ev and ev in hay else ("[NO-EVIDENCE]" if not ev else "[FABRICATED]")
                drivers.append(f"    {tag}  [{kind}/{vd}]  {c.get('criterion','')[:88]}")
                if c.get("rationale"): drivers.append(f"        rationale: {c.get('rationale','')[:120]}")
        out.append("\n".join(drivers) if drivers else "    (no explicit driver found)")
        out.append("\nPATIENT HISTORY:"); out.append(wrap(r.get("patient_history","")))
        out.append("\nFULL TRIAL CRITERIA:"); out.append(wrap(full))
        out.append("\nMY BUCKET (A / B / C):  ______"); out.append("")
    open("gemini_fn_review.txt","w",encoding="utf-8").write("\n".join(out))
    print(f"Wrote gemini_fn_review.txt ({len(fns)} false negatives). Open: open -e gemini_fn_review.txt")

if __name__ == "__main__":
    main()
