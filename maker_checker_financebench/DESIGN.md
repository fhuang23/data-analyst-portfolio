# DESIGN.md — Maker-Checker Financial Disclosure Analyst

A maker-checker agent system for grounded question answering over public company
filings, evaluated on FinanceBench. One agent answers; a second agent audits the
answer against the filing before it is allowed to stand; unresolved cases abstain
to human review. The headline result is grounded accuracy and coverage at a fixed
precision floor, with every accepted answer tied to a cited primary source.

---

## 1. What this proves

This project is a proof-point carrying three claims to a hiring reader in risk,
analytics, BI, or the office of the CFO:

1. **Load-bearing multi-agent design.** The checker actively changes outputs by
   hunting the filing for disconfirming evidence. It is a maker-checker control,
   not a router tree with a buzzword attached.
2. **Domain judgment.** Correctness on financial disclosures depends on knowing
   what a right answer is (which line item, which period, which sign convention).
   That is the differentiator, and it is exactly where off-the-shelf retrieval
   systems fail by hallucinating financial specifics.
3. **Evaluation rigor.** The scorer is built and validated against human labels
   before any accuracy number is trusted. Answers are reported at a precision
   floor with an explicit abstention budget, not as a single vanity accuracy.

"Maker-checker" is a literal internal-controls term, so the design reads at face
value to an accountant, a controller, or a risk lead without translation.

---

## 2. Dataset (verified against the repo)

Source: `patronus-ai/financebench`, open-source split.

- **150 questions** over **32 large-cap companies** (3M, AMD, Amazon, JPMorgan,
  Netflix, Pfizer, Walmart, and similar).
- Document mix: **112 10-K, 15 10-Q, 14 earnings releases, 9 8-K**.
- Every record ships gold fields: `question`, `answer`, `justification` (CFA
  rationale), and `evidence` (a list of spans, each with `evidence_text`,
  `doc_name`, `evidence_page_num`, and `evidence_text_full_page`).
- **Every question is single-document.** Zero require cross-filing synthesis. The
  hard part is finding the right span inside one long filing and reasoning over it
  correctly, which is FinanceBench's documented failure mode.
- Evidence structure: 115 single-span, 31 two-span, 4 three-span. Multi-span
  questions are where numerical reasoning combines line items.
- Skill mix (from `question_reasoning`): roughly one third information
  extraction, one third numerical reasoning, one third logical or judgment
  (reasoned yes/no), plus combined categories.

**Critical fact: FinanceBench ships no scorer.** The provided
`evaluation_playground.ipynb` is retrieval-and-generation only. It runs each model
and dumps `gold_answer` next to `model_answer` in a CSV, then stops. The published
failure rate was produced by manual human review, not by any code in the repo.
Building and validating the scorer is therefore part of this project, not a
given.

Access note: the full 10,231-question set is gated behind a request to Patronus.
Plan around the 150 open-source items. That is enough for an evaluation harness,
not for training or fine-tuning. All results must report uncertainty (see 7).

---

## 3. Architecture

A mostly deterministic pipeline with one genuinely agentic control loop. Flow:

```
question + company/doc
        |
        v
   [ Retriever ]  --- returns candidate spans from the target filing
        |
        v
   [ Maker ]  --- proposes answer + cites the spans it used
        |
        v
   [ Checker ]  --- audits the answer against the filing; hunts disconfirming
        |            evidence; challenges extraction / recomputes math /
        |            steelmans the opposite conclusion
        v
   [ Adjudicator ]  --- reconciles maker vs checker
        |
        +--> accept (answer + grounded citation)
        +--> abstain (route to human review)  <-- precision-floor safety valve
```

### Retriever
Reuses the FinanceBench eval-mode ladder rather than inventing retrieval. See 6.
Candidate generation is deterministic (embedding retrieval over the filing), so it
is fast and reproducible. No agent here.

### Maker agent
Given the question and retrieved spans, produce an answer and, mandatorily, the
exact span(s) it relied on. An answer with no citation is treated as an abstention,
not a guess.

### Checker agent (the load-bearing part)
The checker attacks a different failure mode per reasoning type:

- **Extraction questions:** verify the cited span literally exists in the filing
  text and that the extracted value matches the span. Catches fabricated line
  items and wrong-period pulls.
- **Numerical questions:** independently recompute the metric from the retrieved
  line items. Flag any disagreement with the maker's arithmetic. Catches sign
  errors, wrong denominators, and stale figures.
- **Logical / judgment questions:** search the filing for disconfirming evidence
  and construct the opposite conclusion. Force the maker to reconcile. Catches
  confident but one-sided reads.

### Adjudicator + abstention gate
If maker and checker reconcile, accept with the grounded citation. If they cannot
reconcile within N rounds, abstain and route to human review. Abstention is the
mechanism that buys the precision floor: declining a hard case is cheaper than
answering it wrong. This is the same human-in-the-loop gate pattern used in prior
work.

### Why not more agents
Retrieval, extraction verification, and arithmetic are deterministic or
single-competency steps that do not benefit from autonomy. The only place genuine
agentic behavior earns its cost is the maker-checker disagreement loop. A larger
swarm would add latency and nondeterminism to a task whose whole point is
reproducibility at a precision floor. The design deliberately keeps agent count
minimal and load-bearing.

---

## 4. The judge (scorer) and its validation

Because no scorer ships with the dataset, the judge is a first-class deliverable
and its credibility must be established before any system number is reported.

### Scoring rubric
- **Numeric answers:** tolerance-based match against the gold answer (exact after
  normalizing units and rounding, or within a small stated relative tolerance).
  Deterministic, no LLM needed.
- **Qualitative and yes/no answers:** LLM-judge given the question, the gold
  answer, and the gold `justification` as reference, returning correct / incorrect
  / partially-correct with a reason.

### Validation protocol (do this first)
1. Hand-label a stratified subset of the 150 (across the three `question_type`
   buckets and the reasoning categories), on the order of 40 to 50 items.
2. Run the judge on that subset.
3. Report judge-vs-human agreement (raw agreement and Cohen's kappa).
4. Only trust the automated judge for the full run if agreement clears a
   pre-registered bar (for example kappa >= 0.8). Document every disagreement and
   the rubric change it motivated.

The writeup line this produces: "FinanceBench ships no automated grader, so I
built an LLM-judge and validated it against a hand-labeled subset, reporting
judge-vs-human agreement before trusting a single accuracy number." That
discipline is the credibility centerpiece.

---

## 5. Metrics

Defined precisely so results are unambiguous:

- **Answer accuracy:** fraction of non-abstained answers judged correct,
  regardless of citation.
- **Grounded accuracy:** fraction judged correct AND whose cited span overlaps the
  gold evidence span (right answer for the right reason). This is the headline.
- **Grounding precision:** of the spans the system cites, fraction that overlap
  gold. Measures whether citations are real or decorative.
- **Abstention rate / coverage:** abstention is the fraction routed to human
  review; coverage is one minus abstention.
- **Precision floor result:** tune the abstention threshold so that accuracy among
  non-abstained answers meets a fixed target (for example 95%), then report the
  coverage achieved at that floor. This is the operationally meaningful number.
- **Reasoning-isolated error:** error in `oracle` mode (gold page supplied). A
  failure here is a pure reasoning failure.
- **Retrieval error budget:** `sharedStore` error minus `oracle` error. Separates
  the retrieval-error budget from the reasoning-error budget.

Report all rates with bootstrap confidence intervals given n = 150.

---

## 6. Eval harness and modes

Reuse the FinanceBench eval-mode ladder as a difficulty axis:

- **`oracle`** — gold evidence page supplied. Isolates reasoning from retrieval.
- **`inContext`** — entire filing in context. Long-context reasoning, no retrieval.
- **`singleStore`** — vector store over the one relevant filing. Doc known, span
  unknown.
- **`sharedStore`** — vector store over all filings. Realistic end-to-end: find
  the doc, then the span, then reason.

Primary reporting: run the maker-checker system in **`oracle`** (reasoning ceiling)
and **`sharedStore`** (realistic). The gap between the two is the retrieval error
budget.

---

## 7. Baselines

1. **Single-shot LLM, no checker, oracle mode.** The reasoning ceiling and the
   thing the checker must beat. Ablating the checker on/off is the core evidence
   that the multi-agent structure earns its place.
2. **Single-shot LLM, sharedStore.** The realistic no-checker baseline.
3. **Published reference (context only).** The paper's human-eval failure rate is
   a directional reference, not a directly comparable number, because it predates
   current models and used human grading. Frame your own re-run as the baseline;
   cite the published number only for context.

The central ablation is checker-off vs checker-on at equal retrieval. The claim to
defend: the checker raises grounded accuracy and/or lets you hold a higher
precision floor at acceptable coverage.

---

## 8. Build order

Sequenced so nothing is measured before there is a trusted way to measure it:

1. Harness skeleton: load data, wire the eval modes, dump results.
2. Judge plus judge-validation on the hand-labeled subset. Do not proceed until
   agreement clears the bar.
3. Single-shot baseline in oracle mode (reasoning ceiling).
4. Add retrieval (sharedStore); measure the retrieval gap.
5. Add the maker-checker loop.
6. Add the abstention gate; tune to the precision floor; report coverage.
7. Ablations: checker on/off, per-reasoning-type breakdown, mode-by-mode table.

---

## 9. Repo structure

```
maker-checker-financebench/
  data/                 financebench_open_source.jsonl, document_information.jsonl
  pdfs/                  source filings
  src/
    retrieval.py        eval-mode ladder (oracle/inContext/singleStore/sharedStore)
    maker.py            answer + mandatory citation
    checker.py          per-reasoning-type audit
    adjudicator.py      reconcile + abstain
    judge.py            numeric tolerance + LLM-judge
    harness.py          run modes, collect results
    metrics.py          grounded accuracy, precision floor, CIs
  eval/
    human_labels.csv    hand-labeled validation subset
    judge_agreement.md  kappa + disagreement log
  results/              per-mode CSVs and summary tables
  DESIGN.md
  README.md
```

---

## 10. Limitations and threats to validity

Stated plainly, because naming them is part of the rigor signal:

- **Small n.** 150 questions. All headline numbers carry bootstrap confidence
  intervals; no over-precise claims.
- **Judge is itself a model.** Validated against human labels, not assumed
  correct. Residual judge error is disclosed alongside results.
- **Scope.** Large-cap US filings only. No generalization claim beyond that
  population.
- **Single-document by construction.** This dataset does not test cross-filing
  synthesis, so the project makes no claim there.
- **Comparability.** The published failure rate is not a like-for-like baseline;
  the honest comparison is checker-off vs checker-on within this harness.
