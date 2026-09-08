"""Single-shot answerer.

This is both the no-checker baseline and the answer generator used for judge
validation. It answers the question from the supplied context and cites the
exact text it relied on. An answer with no citation is an abstention, not a
guess, which is the same discipline the checker enforces later.

Numerical questions make the model want to show its work, which breaks a strict
"JSON only" instruction. Two things address that: the prompt now asks the model
to reason first and emit the JSON object on the FINAL line, and the parser
extracts the last valid JSON object anywhere in the response rather than
requiring the whole reply to be clean JSON.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

MAKER_SYSTEM = (
    "You are a financial analyst answering a question strictly from the supplied "
    "excerpt of a company filing. Use only the excerpt. If the excerpt does not "
    "contain enough to answer, say so rather than guessing.\n"
    "You may reason and show your calculation first. Then, on the FINAL line of "
    "your reply, output ONLY this JSON object and nothing after it:\n"
    '{"answer": "<concise answer, include the figure and unit>", '
    '"citation": "<the exact sentence or line from the excerpt you relied on>", '
    '"can_answer": true or false}'
)


@dataclass
class MakerOutput:
    answer: str
    citation: str
    can_answer: bool
    raw: str


def build_maker_prompt(question: str, context: str, max_context_chars: int = 60000) -> str:
    ctx = context[:max_context_chars]
    return (
        f"FILING EXCERPT:\n{ctx}\n\nQUESTION:\n{question}\n\n"
        "Reason if you need to, then give the JSON object on the final line."
    )


def _extract_last_json(text: str) -> dict | None:
    """Return the last JSON object in text that has an 'answer' key.

    Handles: fenced ```json blocks, JSON followed by trailing prose, and JSON
    embedded after a reasoning preamble. Scans candidate {...} blocks from the
    end so the model's final structured answer wins over any earlier draft.
    """
    # strip code fences first so their braces don't confuse the scan
    stripped = re.sub(r"```(?:json)?|```", "", text)
    # non-greedy brace blocks, including nested-free objects (sufficient here)
    candidates = re.findall(r"\{[^{}]*\}", stripped, re.DOTALL)
    for block in reversed(candidates):
        try:
            obj = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "answer" in obj:
            return obj
    return None


def run_maker(question: str, context: str, llm_call) -> MakerOutput:
    raw = llm_call(MAKER_SYSTEM, build_maker_prompt(question, context)).strip()

    # First try: whole reply is clean JSON (fast path).
    clean = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    obj = None
    try:
        parsed = json.loads(clean)
        if isinstance(parsed, dict) and "answer" in parsed:
            obj = parsed
    except json.JSONDecodeError:
        pass

    # Fallback: pull the last JSON object out of a reasoned reply.
    if obj is None:
        obj = _extract_last_json(raw)

    if obj is None:
        # Genuinely no structured answer -> abstain rather than invent one.
        return MakerOutput(answer="", citation="", can_answer=False, raw=raw)

    return MakerOutput(
        answer=str(obj.get("answer", "")).strip(),
        citation=str(obj.get("citation", "")).strip(),
        can_answer=bool(obj.get("can_answer", True)),
        raw=raw,
    )
