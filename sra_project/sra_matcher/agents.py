"""The four pipeline stages.

Two are LLM agents (intake structuring, per-candidate reasoning); two are
deterministic custom agents (retrieval, ranking). Keeping retrieval and ranking
LLM-free is deliberate: retrieval is the baseline the reasoner must beat, and
ranking is arithmetic, not judgment.
"""
from __future__ import annotations

import json
from typing import AsyncGenerator

from google.adk.agents import BaseAgent, LlmAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types

from . import config, prompts, state
from .schemas import PatientProfile, TrialLabel, TrialVerdict
from .tools.ctgov import search_trials


def _say(author: str, text: str) -> Event:
    """A minimal progress event so runs are traceable in `adk web` / logs."""
    return Event(author=author, content=types.Content(role="model", parts=[types.Part(text=text)]))


# --- Stage 1: intake (LLM, structured output) --------------------------------
# Receives the user's free-text history; emits a PatientProfile into state.
# output_schema forces valid JSON; output_key writes it to session.state.
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

    This IS the baseline. Everything the reasoning stage adds is measured against
    what this returns.
    """

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
# The inner reasoner reads one candidate + the profile via {..} templating and
# writes a structured TrialVerdict back to state.
_reasoner = LlmAgent(
    name="eligibility_reasoner",
    model=config.REASONER_MODEL,
    instruction=prompts.REASONER_INSTRUCTION,
    output_schema=TrialVerdict,
    output_key=state.CURRENT_VERDICT,
)


class EligibilityFanout(BaseAgent):
    """Maps the reasoner over the candidate list.

    ADK has no native 'map a sub-agent over a dynamic collection' primitive
    (LoopAgent repeats for refinement; ParallelAgent needs a fixed sub-agent set),
    so a custom BaseAgent owns the loop and runs the sub-agent once per candidate.
    """

    reasoner: LlmAgent

    def __init__(self, reasoner: LlmAgent):
        # Pass the sub-agent as a typed field AND register it via sub_agents so
        # the framework knows the hierarchy (tracing, lifecycle).
        super().__init__(name="eligibility_fanout", reasoner=reasoner, sub_agents=[reasoner])

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        candidates = ctx.session.state.get(state.CANDIDATES, [])
        verdicts: list[dict] = []

        # Criteria are static per NCT id + record version, so this is the natural
        # place to cache parsed criteria and skip re-reasoning unchanged trials.
        # cache: dict[str, dict] = load_criteria_cache()   # TODO

        for cand in candidates:
            ctx.session.state[state.CURRENT_CANDIDATE_JSON] = json.dumps(cand)
            async for event in self.reasoner.run_async(ctx):
                yield event  # surface the reasoner's events for tracing
            verdict = ctx.session.state.get(state.CURRENT_VERDICT)
            if verdict is not None:
                verdicts.append(verdict if isinstance(verdict, dict) else verdict.model_dump())

        ctx.session.state[state.VERDICTS] = verdicts
        yield _say(self.name, f"Scored {len(verdicts)} candidates.")


# --- Stage 4: ranking / triage (deterministic) -------------------------------
class RankingAgent(BaseAgent):
    """Keep eligible + uncertain; order by confidence, then fewest unknowns.

    The 'uncertain' bucket is intentionally surfaced, not dropped — in a research
    setting a human screener adjudicates it.
    """

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        verdicts = ctx.session.state.get(state.VERDICTS, [])
        keep = [
            v for v in verdicts
            if v.get("label") in (TrialLabel.ELIGIBLE.value, TrialLabel.UNCERTAIN.value)
        ]
        keep.sort(key=lambda v: (-float(v.get("confidence", 0.0)), int(v.get("unknown_count", 0))))
        ctx.session.state[state.SHORTLIST] = keep
        yield _say(self.name, f"Shortlist: {len(keep)} trials for screener review.")
