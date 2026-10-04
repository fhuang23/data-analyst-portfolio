"""Judge: a reasoning-auditor for the eligibility agent's verdicts.

It does NOT re-decide eligibility (that would just be a second reasoner, and it
can't overrule a physician's gold label). Instead it audits the AGENT'S WORK on
axes that are checkable without deciding eligibility, then uses an LLM only for
the one genuinely judgment-y question, which it always escalates to a human.

Two layers:

  LAYER 1 - deterministic checks (no LLM, cannot be circular). Runs on every
  verdict. Catches the failures that are about the agent's output itself:
    - grounding:     is each cited `evidence` a verbatim substring of the
                     criteria text? (paraphrased/invented evidence = not grounded)
    - consistency:   does the agent's final `label` follow from its own
                     per-criterion verdicts? (an exclusion marked MET but a label
                     of ELIGIBLE is a self-contradiction, catchable by logic)
    - artifact:      empty/malformed output -> bucket D

  LAYER 2 - LLM triage (only for agent-vs-gold DISAGREEMENTS that passed layer 1
  clean). This is the contestable part: is the disagreement a real model error
  (A), or is the gold label arguable and the agent defensible (B), or is the
  deciding fact simply missing (C)? The LLM forms its OWN read of the evidence
  first, then classifies. Every layer-2 verdict is flagged needs_human=True,
  because this is exactly where an automated judge is least trustworthy.

Use a DIFFERENT model for layer 2 than the reasoner under audit; a judge from
the same family shares its blind spots. Validate the whole judge against a
hand-audited sample (see validate_judge.py) before trusting its buckets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from .schemas import CriterionVerdict, TrialLabel, TrialVerdict, Verdict


# --- the audit vocabulary ----------------------------------------------------

class Bucket(str, Enum):
    MODEL_ERROR = "A_model_error"        # agent mishandled a plainly-stated criterion
    LABEL_DEBATABLE = "B_label_debatable"  # agent defensible, gold arguable
    MISSING_INFO = "C_missing_info"      # deciding fact absent from patient text
    ARTIFACT = "D_artifact"              # empty/malformed agent output
    CLEAN = "clean"                      # no disagreement / nothing to flag


@dataclass
class DeterministicFindings:
    """Layer-1 results. All computed by code, no model involved."""
    is_artifact: bool = False
    grounding_rate: float = 1.0          # fraction of evidence strings found verbatim
    ungrounded_criteria: list[str] = field(default_factory=list)
    label_consistent: bool = True        # does label follow from criteria verdicts?
    consistency_note: str = ""

    @property
    def clean(self) -> bool:
        return (not self.is_artifact) and self.label_consistent and self.grounding_rate >= 0.999


# --- LAYER 1: deterministic checks (cannot be circular) ----------------------

def _norm(s: str) -> str:
    return " ".join((s or "").split()).lower()


def check_grounding(verdict: TrialVerdict, criteria_text: str) -> tuple[float, list[str]]:
    """Fraction of non-empty evidence strings that appear verbatim in the criteria
    text. This is your faithfulness metric, applied per-verdict. Pure string check."""
    hay = _norm(criteria_text)
    checked = ok = 0
    ungrounded = []
    for c in verdict.criteria or []:
        ev = _norm(c.evidence)
        if not ev:
            continue
        checked += 1
        if ev in hay:
            ok += 1
        else:
            ungrounded.append(c.criterion[:80])
    rate = ok / checked if checked else 1.0
    return rate, ungrounded


def check_consistency(verdict: TrialVerdict) -> tuple[bool, str]:
    """Does the agent's final label follow from its OWN per-criterion verdicts?
    Uses the rollup rule stated in the reasoner prompt. This catches internal
    contradictions (a real, bucket-A error) by pure logic, without any eligibility
    judgment of our own."""
    crits = verdict.criteria or []
    if not crits:
        return True, "no criteria to check"

    def is_excl(c: CriterionVerdict) -> bool:
        return (c.kind or "").strip().lower().startswith("excl")

    def is_incl(c: CriterionVerdict) -> bool:
        return (c.kind or "").strip().lower().startswith("incl")

    hard_fail = any(
        (is_excl(c) and c.verdict == Verdict.MET) or (is_incl(c) and c.verdict == Verdict.NOT_MET)
        for c in crits
    )
    any_unknown = any(c.verdict == Verdict.UNKNOWN for c in crits)

    if hard_fail:
        implied = TrialLabel.INELIGIBLE
    elif any_unknown:
        implied = TrialLabel.UNCERTAIN
    else:
        implied = TrialLabel.ELIGIBLE

    if verdict.label == implied:
        return True, f"label {verdict.label.value} matches implied {implied.value}"
    return False, (f"label is {verdict.label.value} but the agent's own criteria imply "
                   f"{implied.value} (hard_fail={hard_fail}, any_unknown={any_unknown})")


def run_layer1(verdict_dict: dict, criteria_text: str) -> DeterministicFindings:
    """All deterministic checks on one agent verdict."""
    if not verdict_dict or not verdict_dict.get("label"):
        return DeterministicFindings(is_artifact=True, consistency_note="empty/missing verdict")
    try:
        verdict = TrialVerdict.model_validate(verdict_dict)
    except Exception as exc:  # malformed output that didn't parse
        return DeterministicFindings(is_artifact=True, consistency_note=f"unparseable: {exc}")

    grounding_rate, ungrounded = check_grounding(verdict, criteria_text)
    consistent, note = check_consistency(verdict)
    return DeterministicFindings(
        is_artifact=False,
        grounding_rate=grounding_rate,
        ungrounded_criteria=ungrounded,
        label_consistent=consistent,
        consistency_note=note,
    )


# --- LAYER 2: LLM triage for the contestable disagreements -------------------

class TriageVerdict(BaseModel):
    """Layer-2 output. Only produced for agent-vs-gold disagreements that were
    clean under layer 1. Always treated as needs_human."""
    own_read: str = Field(description="The judge's OWN verdict on the disputed criterion, "
                                      "formed from patient+criteria BEFORE comparing to agent/gold")
    bucket: Bucket
    justification: str = Field(description="One sentence quoting the specific criterion in dispute")


TRIAGE_INSTRUCTION = """\
You are auditing a DISAGREEMENT between an eligibility AGENT and a physician GOLD label.
Do NOT assume GOLD is automatically correct; physician labels are the reference but are
pooled human judgments that can be arguable. Reason from the primary evidence yourself.

PATIENT (JSON):
{patient_profile_json}

TRIAL CRITERIA (JSON; the eligibility_criteria field is free text):
{current_candidate_json}

AGENT_VERDICT (JSON; the agent's label + its per-criterion reasoning):
{agent_verdict_json}

GOLD_LABEL: {gold_label}   (eligible = should surface; ineligible = should not)

STEP 1 - Form your OWN verdict first. Read the PATIENT and CRITERIA and decide, for the
criterion that actually drives this case: does the patient clearly MEET it, clearly FAIL
it, or is it UNDETERMINABLE from the given patient text? Base this ONLY on the evidence.
Put this in `own_read`.

STEP 2 - Classify the disagreement (pick the FIRST that applies):
  C_missing_info:    you could NOT reach a confident verdict in Step 1 because the deciding
                     fact is absent from the PATIENT text. Neither agent nor gold could know.
  B_label_debatable: your Step-1 verdict AGREES WITH THE AGENT, and the GOLD label is not
                     supported by the criteria text. Reading the evidence, you side with the agent.
  A_model_error:     your Step-1 verdict AGREES WITH GOLD, the criterion is plainly stated,
                     and the agent clearly mishandled it.

Return ONLY the schema fields. `justification` must quote the specific criterion in dispute.
Do not flatter the agent; if it plainly erred, say A even if that is the less charitable read.
"""


def build_triage_agent(backend: str):
    """Layer-2 LLM. Pass a backend DIFFERENT from the reasoner under audit.
    Reuses the same model selection as build_reasoner so 'gemini'/'qwen' resolve
    identically."""
    from google.adk.agents import LlmAgent
    from google.adk.models.lite_llm import LiteLlm
    from . import config

    if backend == "qwen":
        model = LiteLlm(model=f"ollama_chat/{config.QWEN_MODEL}")
    else:
        model = config.REASONER_MODEL

    return LlmAgent(
        name="audit_triage",
        model=model,
        instruction=TRIAGE_INSTRUCTION,
        output_schema=TriageVerdict,
        output_key="triage_verdict",
    )


# --- combined report ---------------------------------------------------------

@dataclass
class JudgeReport:
    nct_id: str
    agent_pred: int          # 1 = surfaced, 0 = skipped
    gold: int
    disagrees: bool
    findings: DeterministicFindings
    bucket: Bucket
    needs_human: bool
    note: str = ""


def judge_case(verdict_dict: dict, criteria_text: str, agent_pred: int, gold: int,
               triage: Optional[TriageVerdict] = None) -> JudgeReport:
    """Combine layer 1 (always) with layer 2 (only when supplied) into one report.

    Decision order:
      1. layer-1 artifact           -> D, no human needed (clearly broken output)
      2. layer-1 inconsistency/ungrounded -> A (real reasoning defect), needs_human
         (a self-contradiction or invented evidence IS a model error, provable by code)
      3. no disagreement with gold  -> clean
      4. clean layer 1 + disagreement -> use layer-2 triage bucket, needs_human=True
    """
    f = run_layer1(verdict_dict, criteria_text)
    nct = (verdict_dict or {}).get("nct_id", "")
    disagrees = agent_pred != gold

    if f.is_artifact:
        return JudgeReport(nct, agent_pred, gold, disagrees, f, Bucket.ARTIFACT,
                           needs_human=False, note=f.consistency_note)

    if not f.label_consistent or f.grounding_rate < 0.999:
        note = f.consistency_note if not f.label_consistent else \
            f"ungrounded evidence: {f.ungrounded_criteria}"
        # a provable reasoning defect is a model error regardless of gold
        return JudgeReport(nct, agent_pred, gold, disagrees, f, Bucket.MODEL_ERROR,
                           needs_human=True, note=note)

    if not disagrees:
        return JudgeReport(nct, agent_pred, gold, disagrees, f, Bucket.CLEAN,
                           needs_human=False, note="agrees with gold, reasoning clean")

    # clean reasoning but disagrees with gold -> the contestable case
    if triage is None:
        return JudgeReport(nct, agent_pred, gold, disagrees, f, Bucket.LABEL_DEBATABLE,
                           needs_human=True, note="disagreement needs layer-2 triage (not run)")
    return JudgeReport(nct, agent_pred, gold, disagrees, f, triage.bucket,
                       needs_human=True,
                       note=f"own_read: {triage.own_read} | {triage.justification}")
