# SRA eligibility matcher — ADK skeleton

A retrieve-then-reason pipeline that matches a patient to recruiting clinical
trials on ClinicalTrials.gov. The point of the architecture is to keep the LLM
off the ~500k-study haystack and spend tokens only where reasoning beats keyword
search: the free-text inclusion/exclusion criteria.

This is a **skeleton** — the wiring runs; the reasoning prompts are stubs you fill in.

## Pipeline

```
intake (LLM)  ->  retrieval (deterministic)  ->  eligibility fanout (LLM x N)  ->  ranking (deterministic)
```

- **intake** — structures free-text patient history into a `PatientProfile` (structured output).
- **retrieval** — coarse CT.gov filter (condition + status + location). This is the **baseline**.
- **eligibility fanout** — runs the reasoner once per candidate; parses criteria, judges each
  met / not_met / unknown with a cited snippet, rolls up to eligible / uncertain / ineligible.
- **ranking** — keeps eligible + uncertain, orders by confidence then fewest unknowns.

Retrieval and ranking are deliberately LLM-free — retrieval is the thing the reasoner must beat,
ranking is arithmetic.

## Layout

| File | Role |
|---|---|
| `pipeline.py` | root `SequentialAgent` |
| `agents.py` | the four stages (2 LLM, 2 custom `BaseAgent`) |
| `tools/ctgov.py` | ClinicalTrials.gov v2 client + `FunctionTool` |
| `schemas.py` | pydantic data contract (what the LLM stages emit) |
| `state.py` | the session-state keys passed between stages |
| `prompts.py` | instruction stubs — **this is where your real work goes** |
| `config.py` | model tiering + candidate-set size |
| `main.py` | Runner + sample patient |

## Run

```bash
pip install -r requirements.txt
cp .env.example .env    # add a GOOGLE_API_KEY, or point at Vertex
python -m sra_matcher.main          # one sample patient
adk web                             # interactive, from the project root
```

Retrieval needs no key (public API). The two LLM stages need model creds.

## Model tiering (the cost lever)

`config.py` puts intake on a cheap model and the eligibility judgment on a stronger one.
If you split the reasoner into parse (cheap) + adjudicate (strong), put the parse call on flash
and cache it per NCT id — criteria are static per record version. The fanout marks that cache hook.

## What to build next (not in this skeleton)

- **Eval harness.** Score the fanout output against labeled patient–trial pairs (n2c2 2018
  cohort-selection, or a TREC Clinical Trials track). Report precision/recall/F1 **and the lift
  over the retrieval-only baseline** — that lift is the whole thesis. Add a cost matrix weighting a
  missed eligible trial heavier than a wasted screening slot, and track $/patient across the run.
- **Abstention + faithfulness checks.** A small judged eval set for whether `unknown` verdicts were
  reasonable, and whether each cited `evidence` snippet is verbatim in the criteria text (cheap
  string check first; only escalate to an LLM judge for the semantic "does this support the call").
- **Data governance.** Don't log raw patient narratives; keep the identifiable profile scoped to the
  session and out of traces; demo on synthetic or de-identified patients. Trial text enters the
  reasoner as *data, never instructions* — it's delimited content in the prompt.
