"""Root pipeline: intake -> retrieve -> reason (fan-out) -> rank.

`root_agent` is the name ADK's `adk web` / `adk run` looks for. The one line that
now matters for the study is which reasoner fills the slot below — the rest of
the pipeline is identical whichever contender you pick.
"""
from __future__ import annotations

import os

from google.adk.agents import SequentialAgent

from .agents import EligibilityFanout, RankingAgent, RetrievalAgent, intake_agent
from .reasoners import GeminiReasoner, RulesReasoner, VllmQwenReasoner

# The slot. Set SRA_REASONER=gemini|qwen|rules to switch without editing code.
_REASONERS = {
    "gemini": GeminiReasoner,
    "qwen": VllmQwenReasoner,
    "rules": RulesReasoner,
}
_choice = os.getenv("SRA_REASONER", "gemini").lower()
reasoner = _REASONERS.get(_choice, GeminiReasoner)()

retrieval_agent = RetrievalAgent(name="retrieval")
fanout_agent = EligibilityFanout(reasoner=reasoner)
ranking_agent = RankingAgent(name="ranking")

root_agent = SequentialAgent(
    name="sra_matcher",
    sub_agents=[intake_agent, retrieval_agent, fanout_agent, ranking_agent],
)
