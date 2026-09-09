# SRA Matcher: Clinical Trial Eligibility Matching Agent

An LLM agent that matches patients to clinical trials by reasoning over free-text eligibility criteria, built on Google's Agent Development Kit (ADK) with [ClinicalTrials.gov](https://clinicaltrials.gov) as the data source.

The interesting part of this project is not the agent. It is the evaluation. I benchmarked the LLM reasoner against two baselines (a hand-written rules engine and an XGBoost classifier) on the TREC 2021 Clinical Trials track, scored on two axes at once (accuracy and a cost model that penalizes false negatives 10x more than false positives), to answer a specific question: does LLM reasoning earn its keep on this task, or is a cheaper model good enough?

## Results

Evaluated on 200 physician-judged patient-trial pairs from TREC 2021, scored on F1 (higher is better) and a total misclassification cost under a 10:1 false-negative-to-false-positive matrix (lower is better):

| Approach | F1 | Cost | Notes |
|---|---|---|---|
| Rules baseline | 0.519 | 130 | Condition, age, and sex filtering |
| XGBoost (5-fold CV) | 0.519 | 387 | TF-IDF and structured features |
| LLM reasoner (zero-shot) | **0.605** | 276 | Highest F1, but not the cheapest |

Three findings, and the tension between the first two is the real story:

1. **The LLM reasoner had the highest F1 by a clear margin (0.605 vs 0.519 for both baselines).** Eligibility matching is a reasoning task, not a lookup task, and the gap showed up exactly where you would expect: criteria phrased as negations, comparative thresholds, and implicit clinical logic.
2. **But the LLM was not the cheapest under the cost model.** The rules baseline came in at cost 130 against the LLM's 276. So the honest question this benchmark surfaces is not "which model wins" but "is the LLM's accuracy gain worth roughly double the misclassification cost for a given deployment?" That tradeoff, not a single leaderboard number, is what a real screening system would have to decide.
3. **XGBoost only matched the rules baseline on F1, and it was the most expensive.** Not because the model was weak, but because the signal was not there: every engineered feature came back near-zero importance, with the single most important feature being patient-description length (importance 0.065). Eligibility reasoning has almost no lexical surface signal for a bag-of-features model to exploit; two trials with opposite inclusion logic can share nearly identical vocabulary. This is easy to miss if you only report your best model, so I kept it front and center. It is a statement about the problem, not the classifier.

The 10:1 cost weighting reflects that the error types are not symmetric: missing an eligible patient (a false negative) is treated as far more costly than surfacing an ineligible one for review, so the cost column penalizes the two error types accordingly rather than treating plain F1 as the whole picture.

## How it works

A four-stage retrieve-then-reason pipeline:

1. **Retrieve.** Pull candidate trials from ClinicalTrials.gov for a given patient profile, narrowing the search space before any expensive reasoning happens.
2. **Parse.** Extract and structure the free-text eligibility criteria (inclusion and exclusion) from each candidate trial.
3. **Reason.** The LLM agent evaluates the patient against each criterion and produces a per-trial eligibility judgment with its reasoning.
4. **Rank.** Aggregate the per-criterion judgments into a final match decision and ordering.

Separating retrieval from reasoning keeps the expensive LLM step focused on the small set of trials that actually warrant it, which is both a cost and a latency decision.

## Tech stack

- **Agent framework:** Google Agent Development Kit (ADK)
- **LLM:** Gemini (via Google AI Studio API key)
- **Baselines:** scikit-learn / XGBoost, plus a hand-written rules engine
- **Data source:** ClinicalTrials.gov API
- **Benchmark:** TREC 2021 Clinical Trials track
- **Language:** Python

## Evaluation methodology

The benchmark uses 200 physician-judged patient-trial pairs from the TREC 2021 Clinical Trials track. Each approach produces an eligibility decision for every pair, scored against the gold labels on two axes: standard F1, and a total misclassification cost under a 10:1 false-negative-to-false-positive weighting. Reporting both, rather than collapsing them into one figure, is deliberate: it keeps the accuracy-versus-cost tradeoff visible instead of hiding it inside a single metric. The XGBoost baseline was 5-fold cross-validated on the same pairs, so it had access to training labels the zero-shot LLM never saw, which makes its inability to beat the rules baseline a finding about the task rather than about tuning.

## Getting started

```bash
# Clone the portfolio repo and enter this project
git clone https://github.com/fhuang23/data-analyst-portfolio.git
cd data-analyst-portfolio/sra_project

# Environment
pip install -r requirements.txt

# Configure your Gemini API key
export GOOGLE_API_KEY="your-key-here"

# Launch the agent (opens the ADK web UI to run patient profiles)
adk web

# Reproduce the evaluation (all three approaches on TREC 2021)
python -m sra_matcher.eval.harness
```

## Project structure

```
sra_project/
  sra_matcher/            # ADK agent package
    pipeline.py           # four-stage retrieve-then-reason orchestration
    agent.py / agents.py  # agent definitions
    schemas.py            # Pydantic contracts passed between stages
    prompts.py            # reasoning prompts
    state.py / config.py  # session state and configuration
    main.py               # package entry point
    tools/
      ctgov.py            # ClinicalTrials.gov API client
    eval/                 # evaluation harness (the core deliverable)
      harness.py          # runs and scores all three approaches
      baseline.py         # non-LLM baseline(s)
      metrics.py          # F1 and cost-weighted scoring
      trec.py             # loads TREC 2021 topics and qrels
  trec_cases.json         # 200 fetched TREC 2021 patient-trial cases
  qrels2021.txt           # TREC 2021 gold relevance judgments
  topics2021.xml          # TREC 2021 patient topics
  requirements.txt
  README.md
```

## Limitations and future work

- Zero-shot reasoning was the strongest baseline; few-shot and structured prompting were not fully explored and are a clear next step.
- The benchmark is TREC 2021 only. Broader validation across trial types and patient populations would strengthen the eligibility claims.
- The retrieval stage is a candidate for improvement: better first-stage recall would raise the ceiling on everything downstream.

## About this project

Built as the flagship piece of my agentic AI portfolio. My background is roughly fifteen years as a professional poker player, a career built on quantitative decision-making under uncertainty and cost-weighted expected value, which is also how I approached the evaluation design here. I am currently transitioning into data analytics, AI/ML, and research roles.

Portfolio: [github.com/fhuang23/data-analyst-portfolio](https://github.com/fhuang23/data-analyst-portfolio)
