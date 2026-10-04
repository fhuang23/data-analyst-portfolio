"""The four pipeline stages.

Intake stays an LLM agent (structuring). Retrieval and ranking stay
deterministic. The eligibility fanout no longer owns an LLM — it owns the loop
and delegates each per-candidate judgment to a pluggable Reasoner (reasoners.py).
Swapping Gemini for the fine-tuned Qwen or the rules baseline is a one-line
change in pipeline.py and touches nothing here, which is exactly the property the
eval harness relies on.
"""
from __future__ import annotations

import json
from typing import AsyncGenerator

from google.adk.agents import BaseAgent, LlmAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types

from . import config, prompts, state
from .reasoners import Reasoner
from .schemas import PatientProfile, TrialLabel
from .tools.ctgov import search_trials


def _say(author: str, text: str) -> Event:
    """A minimal progress event so runs are traceable in `adk web` / logs."""
    return Event(author=author, content=types.Content(role="model", parts=[types.Part(text=text)]))


# --- Stage 1: intake (LLM, structured output) --------------------------------
intake_agent = LlmAgent(
    name="intake",
    model=config.INTAKE_MODEL,
    instruction=prompts.INTAKE_INSTRUCTION,
    output_schema=PatientProfile,
    output_key=state.PATIENT_PROFILE,
)


# --- Stage 2: retrieval (deterministic tool call, NOT an LLM) ----------------
class RetrievalAgent(BaseAgent):
    """Coarse structured filter against CT.gov: condition + status (+ location).
    This IS the baseline everything downstream is measured against."""

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        profile = ctx.session.state.get(state.PATIENT_PROFILE) or {}
        if isinstance(profile, str):
            profile = json.loads(profile)

        conditions = profile.get("conditions") or ["condition"]
        condition = conditions[0]
        location = profile.get("location")

        result = search_trials(
            condition=condition, location=location, max_results=config.MAX_CANDIDATES
        )
        ctx.session.state[state.CANDIDATES] = result["candidates"]
        ctx.session.state[state.PATIENT_PROFILE_JSON] = json.dumps(profile)
        yield _say(
            self.name,
            f"Retrieved {len(result['candidates'])} candidates "
            f"(of {result['total']} total) for '{condition}'.",
        )


# --- Stage 3: per-candidate eligibility reasoning (fan-out over the list) -----
class EligibilityFanout(BaseAgent):
    """Maps a pluggable Reasoner over the candidate list.

    The reasoner is a plain object (reasoners.Reasoner), not an ADK sub-agent, so
    it's held outside the pydantic model fields via object.__setattr__. That's the
    one ADK-specific wart here — if your google-adk version rejects it, declare a
    typed field with arbitrary_types_allowed instead. Everything else is
    contender-agnostic: same state seam in (patient_profile_json,
    current_candidate_json), same TrialVerdict out.
    """

    def __init__(self, reasoner: Reasoner, name: str = "eligibility_fanout"):
        super().__init__(name=name)
        object.__setattr__(self, "_reasoner", reasoner)

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        candidates = ctx.session.state.get(state.CANDIDATES, [])
        profile_json = ctx.session.state.get(state.PATIENT_PROFILE_JSON, "{}")
        verdicts: list[dict] = []
        elapsed = 0.0

        # Criteria are static per NCT id + record version, so this stays the
        # natural place to cache parsed criteria and skip re-reasoning. TODO.
        for cand in candidates:
            cand_json = json.dumps(cand)
            ctx.session.state[state.CURRENT_CANDIDATE_JSON] = cand_json
            result = self._reasoner.judge(profile_json, cand_json)
            ctx.session.state[state.CURRENT_VERDICT] = result.verdict.model_dump()
            verdicts.append(result.verdict.model_dump())
            elapsed += result.usage.wall_seconds

        ctx.session.state[state.VERDICTS] = verdicts
        yield _say(
            self.name,
            f"Scored {len(verdicts)} candidates via {self._reasoner.name} "
            f"in {elapsed:.1f}s.",
        )


# --- Stage 4: ranking / triage (deterministic) -------------------------------
class RankingAgent(BaseAgent):
    """Keep eligible + uncertain; order by confidence, then fewest unknowns.
    The 'uncertain' bucket is surfaced to a human screener, never dropped."""

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        verdicts = ctx.session.state.get(state.VERDICTS, [])
        keep = [
            v for v in verdicts
            if v.get("label") in (TrialLabel.ELIGIBLE.value, TrialLabel.UNCERTAIN.value)
        ]
        keep.sort(key=lambda v: (-float(v.get("confidence", 0.0)), int(v.get("unknown_count", 0))))
        ctx.session.state[state.SHORTLIST] = keep
        yield _say(self.name, f"Shortlist: {len(keep)} trials for screener review.")


# --- Module-level reasoner agents for the eval harness ----------------------
# The harness drives a standalone LlmAgent through an ADK Runner. Restore that
# entry point, and make the backend switchable so Gemini and Qwen run through
# the identical harness path (true apples-to-apples).
from google.adk.models.lite_llm import LiteLlm
from .schemas import TrialVerdict


def build_reasoner(backend: str = "gemini"):
    """Return an eligibility-reasoner LlmAgent for the given backend.
    'gemini' uses config.REASONER_MODEL; 'qwen' uses local Ollama via LiteLLM.
    """
    if backend == "qwen":
        model = LiteLlm(model=f"ollama_chat/{config.QWEN_MODEL}")
    else:
        model = config.REASONER_MODEL
    return LlmAgent(
        name="eligibility_reasoner",
        model=model,
        instruction=prompts.REASONER_INSTRUCTION,
        output_schema=TrialVerdict,
        output_key=state.CURRENT_VERDICT,
    )


# Backward-compatible default the harness imports.
_reasoner = build_reasoner("gemini")
