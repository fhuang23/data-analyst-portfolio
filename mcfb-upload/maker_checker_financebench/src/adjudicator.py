"""The adjudicator: final call when the checker does not cleanly agree.

Sees the question, the excerpt, the maker's answer, and the checker's critique
(and proposed answer), then decides. Its job is to RESOLVE the disagreement
against the evidence -- verify each candidate answer against the excerpt and
adopt the one the filing actually supports -- not to treat disagreement itself
as a reason to abstain. It abstains only when neither candidate is verifiable.

(Earlier version framed abstention as the default for any unclear conflict,
which caused it to abstain even when one side was clearly correct -- e.g. it
threw away the auditor's correct recomputation on a numerical item, and the
analyst's correct answer on another. The instruction below makes it verify and
adopt, and reserve abstention for genuinely unresolvable cases.)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from checker import CheckerOutput
from maker import MakerOutput

ADJUDICATOR_SYSTEM = (
    "You are the final adjudicator for a question about a company filing. An "
    "analyst gave an answer; an auditor challenged it and may have proposed a "
    "different answer. You are deciding between two candidate answers.\n\n"
    "Your task is to RESOLVE the disagreement using ONLY the excerpt:\n"
    "1. Independently verify the ANALYST'S answer against the excerpt -- locate "
    "the supporting figures or text and check the reasoning.\n"
    "2. Independently verify the AUDITOR'S proposed answer the same way.\n"
    "3. Adopt whichever candidate the excerpt actually supports, and set "
    "abstain=false. It is equally fine to side with the analyst or the auditor -- "
    "pick the one the evidence backs.\n"
    "4. Abstain (abstain=true) ONLY if NEITHER candidate is supported by the "
    "excerpt, or the excerpt genuinely lacks the information to decide. Do NOT "
    "abstain merely because the two disagree -- disagreement is expected and "
    "resolving it is your job.\n\n"
    "Reply with STRICT JSON only: "
    '{"final_answer": "<the answer you endorse, or empty if abstaining>", '
    '"abstain": true | false, "reason": "<one sentence naming which side the '
    'evidence supported and why>"}'
)


@dataclass
class AdjudicationResult:
    final_answer: str
    abstain: bool
    reason: str
    raw: str


def build_adjudicator_prompt(question: str, context: str, maker: MakerOutput,
                             checker: CheckerOutput, max_context_chars: int = 60000) -> str:
    return (
        f"FILING EXCERPT:\n{context[:max_context_chars]}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"CANDIDATE A -- ANALYST'S ANSWER:\n{maker.answer}\n\n"
        f"AUDITOR'S VERDICT ON A: {checker.verdict}\n"
        f"AUDITOR'S REASONING: {checker.reason}\n\n"
        f"CANDIDATE B -- AUDITOR'S PROPOSED ANSWER:\n{checker.proposed_answer or '(none proposed)'}\n\n"
        "Verify each candidate against the excerpt and adopt the supported one. "
        "Abstain only if neither is supported. JSON only."
    )


def run_adjudicator(
    question: str,
    context: str,
    maker: MakerOutput,
    checker: CheckerOutput,
    llm_call,
) -> AdjudicationResult:
    raw = llm_call(ADJUDICATOR_SYSTEM,
                   build_adjudicator_prompt(question, context, maker, checker)).strip()
    clean = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        obj = json.loads(clean)
        abstain = bool(obj.get("abstain", True))
        final = str(obj.get("final_answer", "")).strip()
        # No answer but not marked abstain -> treat as abstain (fail safe).
        if not final and not abstain:
            abstain = True
        return AdjudicationResult(final, abstain, str(obj.get("reason", "")).strip(), raw)
    except json.JSONDecodeError:
        # Unparseable adjudication fails safe to abstention.
        return AdjudicationResult("", True, f"unparseable adjudicator output: {raw[:80]}", raw)
