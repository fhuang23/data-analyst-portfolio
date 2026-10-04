# Clinical Trial Eligibility Matching: An Agentic Pipeline and Its Evaluation

## Summary

I built an LLM agent that matches patients to clinical trials by reasoning over free-text
eligibility criteria, and a rigorous harness to evaluate it. The project's primary output is not
a deployable system but a **controlled evaluation with an honest negative result**: model
capability manifests clearly as evidence-grounding (a frontier model is far more faithful than a
small local one), but grounding alone does not make the system trustworthy for clinical
screening. The evaluation itself surfaced the key limitation — assessing a clinical-reasoning
agent's correctness requires clinical expertise, which bounds both what the system can do
autonomously and what a non-clinical evaluator can certify.

## What I built

A four-stage pipeline (Google ADK):

1. **Intake** (LLM) — structures free-text patient history into a typed profile.
2. **Retrieval** (deterministic) — coarse filter against ClinicalTrials.gov; serves as the baseline.
3. **Eligibility reasoning** (pluggable LLM) — per-criterion met / not-met / unknown judgments
   with cited evidence, rolled up to eligible / ineligible / uncertain. The reasoner is a
   swappable slot (local Qwen 1.7B via Ollama, or Gemini via API) so models run through an
   identical path.
4. **Ranking** (deterministic) — shortlists eligible + uncertain for human review.

An evaluation harness scores any reasoner against physician-judged labels, reporting
precision/recall, a cost-weighted error (false negatives weighted 10x false positives, since
missing an eligible patient is the expensive error), **evidence faithfulness** (fraction of cited
evidence found verbatim in the criteria), and **abstention rate**.

## Data

Patients are physician-authored synthetic vignettes from the TREC 2021 Clinical Trials track;
relevance labels (0 = not relevant, 1 = excluded, 2 = eligible) are the track's expert assessor
judgments; trial eligibility text is fetched from ClinicalTrials.gov. I map relevance-2 to
"eligible" and 0/1 to "ineligible," treating topically-relevant-but-excluded trials as
non-surfaced. Labels are pooled and imperfect — a limitation that proved central.

> Note: the TREC topics/qrels and fetched trial text are not redistributed in this repo (usage
> terms / size). The code expects them locally; see the data section of the scripts for the
> expected file names.

## Findings

**1. Capability shows up as grounding.** On matched conditions, the local 1.7B had evidence
faithfulness of **0.41** (it fabricated the majority of its citations), while Gemini scored
**0.99** (nearly all citations verbatim). A manual audit of the 1.7B's disagreements independently
confirmed this: roughly 10 of 12 were genuine errors, consistent with the measured hallucination
rate. Two independent methods agreeing is the strongest result here.

**2. The two models fail in opposite directions.** The 1.7B errs by *reckless over-surfacing*
(hallucinated eligibility -> false positives). Gemini, on 200 pairs across 20 patients, errs by
*cautious over-rejection*: 63/200 disagreements, skewed toward false negatives (36). It does not
fabricate; it over-applies real criteria.

**3. Grounded over-rejection has distinct causes.** Auditing the false negatives from stated
evidence only (no clinical inference), two separable causes emerged. *Input under-specification*:
fed raw history rather than structured profiles, the model correctly could not verify criteria
whose facts were absent — an engineering-fixable limitation. *Interpretive rigidity*: in other
cases the model had the information but applied criteria too literally or inferred unstated facts
(e.g. assuming a treatment typical for a diagnosis) — a genuine judgment gap that structured input
would not fix. A subset also rejected while flagging an inclusion as unverifiable, violating the
specified abstention policy — a calibration weakness provable by logic alone.

**4. The central limitation: most disagreements cannot be adjudicated without clinical expertise.**
As a non-clinical evaluator, I could confidently classify only a minority of Gemini's
disagreements — clear errors (unstated-fact inference, policy-violating rejection) and clear label
errors (e.g. a dialysis patient failing a hard creatinine-clearance cutoff, definitionally
impossible to meet). The majority were genuine clinical-interpretation questions beyond a
layperson's ability to certify. This is itself the finding.

## Conclusion

A grounded frontier model is a plausible **assistive triage layer** — it surfaces candidates and
flags uncertainty — but this evaluation does not support autonomous deployment or clinician-
workload *replacement*. Its errors are the expensive kind (dropping eligible patients), its
disagreements with expert labels are mostly unadjudicable without clinicians, and the benchmark
itself is too label-noisy to cleanly separate model quality from label quality beyond a point.
Validating this class of system requires clinical reviewers in the loop — precisely the oversight
the system was meant to assist, not remove.

## Cost and infrastructure notes

The local model is free but RAM-bound (a 16GB machine barely runs a 4B model). The frontier model
costs ~$0.003/pair (~$0.70 for 200 pairs) but free-tier API quotas (~20 requests/day) make batch
evaluation impractical without billing. This local-vs-hosted tradeoff — free-but-constrained vs
capable-but-metered — is a practical finding for anyone choosing where to run eval-scale LLM
workloads.

## Honest limitations

Small samples (30-200 pairs, few topics) mean accuracy figures are illustrative, not population
estimates; the grounding and faithfulness results, on larger denominators, carry more weight. A
single benchmark. Non-clinical auditing has a hard ceiling. An initial audit was too lenient and
was corrected after measuring faithfulness — the stricter count is reported.

## Stack

Python, Google ADK, google-genai, Ollama (local Qwen), pydantic (typed verdict schema),
ClinicalTrials.gov API, TREC 2021 Clinical Trials benchmark.
