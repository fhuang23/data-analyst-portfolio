# Maker-Checker Financial Disclosure Analyst

Grounded question answering over public company filings (FinanceBench), with a
maker-checker control: one agent answers, a second audits the answer against the
filing, and unresolved cases abstain to human review. See `DESIGN.md` for the
full rationale and architecture.

This repository implements the **measurement layer**: data loading, the scorer
FinanceBench does not ship, the judge-validation protocol (steps 1-2), and the
**retrieval layer** (step 4) with a retrieval-quality eval scored against gold
evidence and needing no LLM. The maker-checker agent loop is built on top once
the judge is trusted.

## Why measurement first

FinanceBench provides questions, gold answers, and gold evidence spans, but **no
automated scorer** (its published numbers came from manual human review). So the
first deliverable is a judge, and the first thing to establish is that the judge
agrees with a human. Nothing downstream is meaningful until it does.

## Layout

```
src/
  data_io.py         load 150 questions, reasoning buckets, stratified sample
  judge.py           deterministic numeric matching + LLM-judge fallback
  metrics.py         agreement, Cohen's kappa, bootstrap CIs, precision floor
  models.py          Anthropic call wrapper (lazy import, env-driven model ids)
  maker.py           single-shot answerer (baseline + candidate generator)
  chunking.py        PDF -> page-tagged chunks (pymupdf)
  embeddings.py      local / gemini / openai / voyage / mock backends
  vectorstore.py     numpy cosine store + per-doc embedding cache
  retrieval.py       eval-mode ladder: oracle, in_context, single/shared store
  retrieval_eval.py  recall@k vs gold evidence pages and spans (no LLM)
scripts/
  01_make_validation_subset.py   stratified sample -> answers -> label template
  02_score_judge_agreement.py    human labels -> judge -> agreement + kappa
  03_eval_retrieval.py           retrieval quality vs gold evidence, no LLM
eval/           validation subset, scored output, agreement report
results/        retrieval eval output
vectorstores/   per-doc embedding cache (created on first run)
data/           financebench jsonl files
pdfs/           source filings (fetch from the FinanceBench repo)
```

## Run it in a notebook

`run_measurement_layer.ipynb` wires steps 1-2 and 4 together for interactive use: it imports the `src/` functions directly (no CLI), pauses for you to hand-label in section 2, and shows disagreements and retrieval hits inline. Set keys with `os.environ[...]` in the first cell (the kernel won't see shell exports). Use `mock` model/embedder backends for a free dry run.

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...            # required for maker + LLM-judge
export MCFB_MAKER_MODEL=claude-sonnet-4-5   # override to a model you can call
export MCFB_JUDGE_MODEL=claude-haiku-4-5    # override to a model you can call
```

The deterministic parts (`judge.py` numeric path, `metrics.py`, `data_io.py`)
run and self-test with **no key**:

```bash
python src/judge.py      # numeric matcher self-test
python src/metrics.py    # kappa / bootstrap / coverage self-test
python src/data_io.py    # data load + stratification check
```

## Run steps 1-2

```bash
# 1. Build the stratified validation subset and generate candidate answers.
python scripts/01_make_validation_subset.py --n 45
#    (add --dry-run to write the template with no API calls)

# 2. Hand-label eval/validation_subset.csv: fill 'human_label' with
#    'correct' or 'incorrect' for each model_answer vs gold_answer.

# 3. Validate the judge against your labels.
python scripts/02_score_judge_agreement.py --kappa-bar 0.8
#    -> eval/judge_agreement.md  (agreement, kappa, disagreement log)
```

If kappa clears the bar, the judge is trusted and you proceed to the single-shot
baseline and then the maker-checker loop. If not, the disagreement log tells you
which rubric or tolerance to fix before any full run.

## Run step 4 (retrieval quality)

Needs the source PDFs in `pdfs/` and an embedding backend. This eval uses no
LLM: it scores whether retrieval surfaces the gold evidence.

```bash
export MCFB_EMBED_BACKEND=local        # or: gemini | openai | voyage | mock
# single_store: doc known, find the passage
python scripts/03_eval_retrieval.py --mode single_store --k 5
# shared_store: realistic, find the right filing then the passage
python scripts/03_eval_retrieval.py --mode shared_store --k 8
# test the plumbing with no key or model download:
python scripts/03_eval_retrieval.py --mode single_store --backend mock --available-only
```

Reads out page recall@k, text recall@k, mean containment, and (for
`shared_store`) how often the correct filing surfaces first. The gap between
oracle-mode reasoning accuracy and retrieval-mode accuracy is bounded by this
recall: it is the retrieval-error budget, separated from the reasoning-error
budget.

## Judge design (summary)

- Gold answers that are essentially a number (about a third of the set) are
  scored deterministically, with scale-invariant matching so `$1577.00`,
  `$1,577 million`, and `$1.577 billion` all agree, and percentages compared at
  face value.
- Everything else (yes/no with justification, qualitative claims) goes to an
  LLM-judge given the gold answer and the CFA justification as reference.
- If a numeric answer will not parse cleanly on both sides, it defers to the
  LLM-judge rather than guess.

## Notes

- n = 150 is small; every reported rate carries a bootstrap CI (`metrics.py`).
- Vector-store retrieval modes are explicit stubs (build-order step 4).
- Source filing PDFs are not vendored here; fetch the `pdfs/` folder from the
  FinanceBench repo for `in_context` mode.
