"""Root pipeline: intake -> retrieve -> reason (fan-out) -> rank.

`root_agent` is the name ADK's `adk web` / `adk run` looks for, so you can launch
the whole thing with `adk web` from the project root once your model creds are set.
"""
from __future__ import annotations

from google.adk.agents import SequentialAgent

from .agents import EligibilityFanout, RankingAgent, RetrievalAgent, _reasoner, intake_agent

retrieval_agent = RetrievalAgent(name="retrieval")
fanout_agent = EligibilityFanout(reasoner=_reasoner)
ranking_agent = RankingAgent(name="ranking")

root_agent = SequentialAgent(
    name="sra_matcher",
    sub_agents=[intake_agent, retrieval_agent, fanout_agent, ranking_agent],
)
