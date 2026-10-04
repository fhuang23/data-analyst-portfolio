"""Pluggable eligibility reasoners.

The reasoner is the one swappable unit in the pipeline. Given the patient
profile and ONE candidate trial (both as JSON strings — exactly the state seam
the fanout already uses: patient_profile_json + current_candidate_json), it
returns a TrialVerdict plus a Usage record.

Every contender implements the same `judge()`, so the fanout (agents.py) and the
eval harness (eval/harness.py) treat them interchangeably. The only thing that
varies between runs is which reasoner sits in the slot — which is what makes the
cost comparison fair.

Contenders (XGBoost dropped — it never consumed the eligibility prompt the way
these do, so it doesn't belong in this slot):

    GeminiReasoner    frontier model via google-genai, billed per token
    VllmQwenReasoner  self-hosted fine-tuned Qwen via a local vLLM endpoint
    RulesReasoner     deterministic keyword/threshold baseline, zero marginal cost

Cost lives in cost.py, keyed to the Usage each reasoner emits: a reasoner
measures what it consumed; a CostModel prices it. Keeping those apart means
swapping a cloud vendor or a GPU rate never touches reasoning code.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

import requests

from . import config, prompts
from .schemas import CriterionVerdict, Sex, TrialLabel, TrialVerdict, Verdict


# --- the seam ----------------------------------------------------------------

@dataclass
class Usage:
    """What one judge() call consumed. Exactly one cost family is populated:
    an API reasoner fills tokens, a self-hosted one leans on wall_seconds.
    wall_seconds is measured for every reasoner so latency is always comparable.
    """
    input_tokens: int = 0
    output_tokens: int = 0
    wall_seconds: float = 0.0
    served_locally: bool = False  # True -> price on GPU time, not tokens


@dataclass
class ReasonerResult:
    verdict: TrialVerdict
    usage: Usage


@runtime_checkable
class Reasoner(Protocol):
    """Every contender is this. name is used in harness tables and trace events."""
    name: str

    def judge(self, patient_profile_json: str, candidate_json: str) -> ReasonerResult:
        ...


def _prompt(patient_profile_json: str, candidate_json: str) -> str:
    """The exact instruction the ADK LlmAgent used, rendered outside ADK so the
    harness can call a reasoner without spinning a Runner. Same template, same
    {..} keys the fanout already writes to state."""
    return prompts.REASONER_INSTRUCTION.format(
        patient_profile_json=patient_profile_json,
        current_candidate_json=candidate_json,
    )


# --- contender 1: frontier API (Gemini) --------------------------------------

class GeminiReasoner:
    """Frontier reasoner via google-genai structured output.

    Same instruction and same TrialVerdict schema the ADK LlmAgent enforced with
    output_schema, so behaviour matches the original pipeline. We call genai
    directly (rather than through an ADK sub-agent) for two reasons: the harness
    can run it without a Runner, and usage_metadata is trivial to read off the
    response for the cost accounting.
    """

    def __init__(self, model: Optional[str] = None, name: str = "gemini"):
        from google import genai
        from google.genai import types

        self._types = types
        self._client = genai.Client()  # reads GOOGLE_API_KEY, or Vertex env
        self.model = model or config.REASONER_MODEL
        self.name = name

    def judge(self, patient_profile_json: str, candidate_json: str) -> ReasonerResult:
        t0 = time.perf_counter()
        resp = self._client.models.generate_content(
            model=self.model,
            contents=_prompt(patient_profile_json, candidate_json),
            config=self._types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=TrialVerdict,
                temperature=0.0,
            ),
        )
        wall = time.perf_counter() - t0
        verdict = TrialVerdict.model_validate_json(resp.text)
        um = getattr(resp, "usage_metadata", None)
        usage = Usage(
            input_tokens=getattr(um, "prompt_token_count", 0) or 0,
            output_tokens=getattr(um, "candidates_token_count", 0) or 0,
            wall_seconds=wall,
        )
        return ReasonerResult(verdict, usage)


# --- contender 2: self-hosted fine-tuned Qwen --------------------------------

class VllmQwenReasoner:
    """Fine-tuned Qwen served by a local vLLM OpenAI-compatible endpoint.

    Same prompt, same schema — enforced with vLLM guided decoding (guided_json)
    so a small model can't drift off-schema. We time the call and price it on
    GPU-seconds in cost.py, because a self-hosted model has no per-token bill:
    the cost is the box it runs on.

    Point base_url at your served adapter, e.g.:
        vllm serve Qwen/Qwen3-8B-Instruct --enable-lora \\
            --lora-modules sra=/path/to/your/lora-adapter
        # then model="sra" selects the adapter
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        name: str = "qwen-lora",
        timeout: int = 120,
    ):
        self.base_url = (base_url or config.VLLM_BASE_URL).rstrip("/")
        self.model = model or config.QWEN_MODEL
        self.name = name
        self.timeout = timeout
        self._schema = TrialVerdict.model_json_schema()

    def judge(self, patient_profile_json: str, candidate_json: str) -> ReasonerResult:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": _prompt(patient_profile_json, candidate_json)}],
            "temperature": 0.0,
            # vLLM extension: constrain output to the TrialVerdict schema. Top-level
            # (not under extra_body) because this is a raw HTTP POST, not the SDK.
            "guided_json": self._schema,
        }
        t0 = time.perf_counter()
        r = requests.post(
            f"{self.base_url}/v1/chat/completions", json=payload, timeout=self.timeout
        )
        r.raise_for_status()
        wall = time.perf_counter() - t0
        content = r.json()["choices"][0]["message"]["content"]
        verdict = TrialVerdict.model_validate_json(content)
        return ReasonerResult(verdict, Usage(wall_seconds=wall, served_locally=True))


# --- contender 3: rules baseline (no LLM) ------------------------------------

class RulesReasoner:
    """Deterministic floor: check the structured gates CT.gov already exposes
    (age window, sex) plus a keyword scan for obvious exclusion terms. Everything
    it can't resolve from structured fields becomes `unknown`, not a guess.

    This is the reasoner-slot analogue of your retrieval baseline: the cheap,
    LLM-free call that the LLMs have to beat. Zero marginal cost.
    """

    name = "rules"

    # crude, on purpose — a baseline, not a competitor.
    _EXCLUSION_HINTS = ("pregnan", "prior ", "history of", "must not", "exclusion")

    def judge(self, patient_profile_json: str, candidate_json: str) -> ReasonerResult:
        t0 = time.perf_counter()
        profile = json.loads(patient_profile_json or "{}")
        cand = json.loads(candidate_json or "{}")

        criteria: list[CriterionVerdict] = []
        unknown = 0
        hard_fail = False

        age = profile.get("age_years")
        lo, hi = cand.get("min_age"), cand.get("max_age")
        if age is not None and (lo is not None or hi is not None):
            ok = (lo is None or age >= lo) and (hi is None or age <= hi)
            criteria.append(CriterionVerdict(
                criterion="age window", kind="inclusion",
                verdict=Verdict.MET if ok else Verdict.NOT_MET,
                evidence=f"min_age={lo}, max_age={hi}",
            ))
            hard_fail = hard_fail or not ok
        else:
            unknown += 1

        want_sex = (cand.get("sex") or Sex.ALL.value).upper()
        pt_sex = (profile.get("sex") or Sex.ALL.value).upper()
        if want_sex != Sex.ALL.value and pt_sex != Sex.ALL.value:
            ok = want_sex == pt_sex
            criteria.append(CriterionVerdict(
                criterion="sex", kind="inclusion",
                verdict=Verdict.MET if ok else Verdict.NOT_MET, evidence=f"sex={want_sex}",
            ))
            hard_fail = hard_fail or not ok

        crit_text = (cand.get("eligibility_criteria") or "").lower()
        if any(h in crit_text for h in self._EXCLUSION_HINTS):
            unknown += 1  # there ARE exclusions here the rules can't adjudicate

        if hard_fail:
            label, conf = TrialLabel.INELIGIBLE, 0.6
        elif unknown:
            label, conf = TrialLabel.UNCERTAIN, 0.3
        else:
            label, conf = TrialLabel.ELIGIBLE, 0.4  # deliberately unconfident

        verdict = TrialVerdict(
            nct_id=cand.get("nct_id", ""),
            label=label,
            confidence=conf,
            criteria=criteria,
            unknown_count=unknown,
            summary="rules baseline: structured gates only",
        )
        return ReasonerResult(verdict, Usage(wall_seconds=time.perf_counter() - t0))
