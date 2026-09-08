"""The checker: audits the maker's answer against the filing before it stands.

Two layers:
  1. A deterministic grounding check (no LLM): does the maker's cited span
     actually appear in the supplied context? A hallucinated citation is caught
     here for free, regardless of reasoning type.
  2. A reasoning-type-specific LLM audit that attacks the failure mode that type
     is prone to:
       extraction -- verify the cited span supports the extracted value.
       numerical  -- independently recompute and compare.
       logical    -- hunt the filing for disconfirming evidence; steelman the
                     opposite conclusion; is the answer still defensible?

The checker is the load-bearing agent: its verdict can flip the answer or send
it to abstention. It is not a rubber stamp.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from maker import MakerOutput

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if len(w) > 3}


def is_grounded(citation: str, context: str, threshold: float = 0.6) -> float:
    """Fraction of the citation's distinctive tokens present in the context.

    Returned as a score in [0,1]; callers threshold it. An empty citation
    scores 0 (an answer with no citation is treated as ungrounded).
    """
    cit = _tokens(citation)
    if not cit:
        return 0.0
    ctx = _tokens(context)
    return len(cit & ctx) / len(cit)


# --- reasoning-type-specific audit instructions ---------------------------

_AUDIT = {
    "extraction": (
        "This is an EXTRACTION question. Verify that the candidate's cited span "
        "actually appears in the excerpt and that the extracted value matches "
        "what the span states. If the span does not support the value, or the "
        "citation is not in the excerpt, the answer is wrong."
    ),
    "numerical": (
        "This is a NUMERICAL question. Independently locate the relevant figures "
        "in the excerpt and recompute the answer yourself, step by step. Then "
        "compare your recomputed value to the candidate. If they differ by more "
        "than rounding, the candidate is wrong -- put your value in proposed_answer."
    ),
    "logical": (
        "This is a JUDGMENT question. Actively search the excerpt for evidence "
        "that would CONTRADICT the candidate's conclusion, and construct the "
        "strongest opposing case. If disconfirming evidence makes the conclusion "
        "indefensible, the answer is wrong. If the conclusion still holds after "
        "that scrutiny, it is right."
    ),
}

CHECKER_SYSTEM = (
    "You are an auditor checking another analyst's answer to a question about a "
    "company filing. You are adversarial in the useful sense: your job is to find "
    "errors, not to agree. {audit}\n\n"
    "Reply with STRICT JSON only: "
    '{{"verdict": "agree" | "disagree" | "uncertain", '
    '"reason": "<one or two sentences>", '
    '"proposed_answer": "<your answer if you disagree, else empty>"}}'
)


@dataclass
class CheckerOutput:
    verdict: str            # 'agree' | 'disagree' | 'uncertain'
    reason: str
    proposed_answer: str
    grounding: float        # [0,1] deterministic citation-grounding score
    grounded: bool
    raw: str


def build_checker_prompt(question: str, context: str, maker: MakerOutput,
                         max_context_chars: int = 60000) -> str:
    return (
        f"FILING EXCERPT:\n{context[:max_context_chars]}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"CANDIDATE ANSWER:\n{maker.answer}\n\n"
        f"CANDIDATE'S CITED SPAN:\n{maker.citation}\n\n"
        "Audit the candidate. JSON only."
    )


def run_checker(
    question: str,
    context: str,
    maker: MakerOutput,
    reasoning_bucket: str,
    llm_call,
    ground_threshold: float = 0.6,
) -> CheckerOutput:
    grounding = is_grounded(maker.citation, context, ground_threshold)
    grounded = grounding >= ground_threshold

    audit = _AUDIT.get(reasoning_bucket, _AUDIT["logical"])
    system = CHECKER_SYSTEM.format(audit=audit)
    raw = llm_call(system, build_checker_prompt(question, context, maker)).strip()
    clean = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        obj = json.loads(clean)
        verdict = str(obj.get("verdict", "uncertain")).lower()
        if verdict not in {"agree", "disagree", "uncertain"}:
            verdict = "uncertain"
        return CheckerOutput(
            verdict=verdict,
            reason=str(obj.get("reason", "")).strip(),
            proposed_answer=str(obj.get("proposed_answer", "")).strip(),
            grounding=grounding,
            grounded=grounded,
            raw=raw,
        )
    except json.JSONDecodeError:
        # Unparseable audit is treated as 'uncertain' -> pushes to adjudication.
        return CheckerOutput("uncertain", f"unparseable checker output: {raw[:80]}",
                             "", grounding, grounded, raw)
