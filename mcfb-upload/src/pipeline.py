"""The maker-checker pipeline.

Flow:
    maker answers (with a mandatory citation)
      -> maker declined?                     -> ABSTAIN
      -> checker audits (per reasoning type + deterministic grounding)
           -> checker agrees AND grounded    -> ACCEPT maker's answer
           -> otherwise                      -> adjudicator decides
                -> adjudicator abstains       -> ABSTAIN
                -> else                       -> ACCEPT adjudicator's answer

The abstention outcomes are what buy the precision floor: the system answers
only when maker and checker align or the adjudicator can decide on the evidence.
Every result carries the full trace so accept/abstain decisions are auditable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from maker import MakerOutput, run_maker
from checker import CheckerOutput, run_checker
from adjudicator import AdjudicationResult, run_adjudicator


@dataclass
class PipelineResult:
    answer: str
    abstained: bool
    path: str                       # 'maker_declined' | 'agreed' | 'adjudicated' | 'adjudicator_abstained'
    maker: MakerOutput
    checker: Optional[CheckerOutput] = None
    adjudicator: Optional[AdjudicationResult] = None
    notes: dict = field(default_factory=dict)


def run_pipeline(
    question: str,
    context: str,
    reasoning_bucket: str,
    maker_call,
    checker_call,
    adjudicator_call,
    ground_threshold: float = 0.6,
) -> PipelineResult:
    maker = run_maker(question, context, maker_call)
    if not maker.can_answer:
        return PipelineResult("", True, "maker_declined", maker)

    checker = run_checker(question, context, maker, reasoning_bucket,
                          checker_call, ground_threshold)

    # Clean agreement AND a real citation -> accept as-is.
    if checker.verdict == "agree" and checker.grounded:
        return PipelineResult(maker.answer, False, "agreed", maker, checker)

    # Disagreement, uncertainty, or an ungrounded citation -> adjudicate.
    adj = run_adjudicator(question, context, maker, checker, adjudicator_call)
    if adj.abstain:
        return PipelineResult("", True, "adjudicator_abstained", maker, checker, adj)
    return PipelineResult(adj.final_answer, False, "adjudicated", maker, checker, adj)
