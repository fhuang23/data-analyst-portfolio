"""The scorer FinanceBench does not ship.

Two paths:
  1. Deterministic numeric matching when the gold answer is essentially a number
     (a dollar figure or a percentage). No model call, fully reproducible.
  2. LLM-judge for prose / yes-no-with-justification answers, given the gold
     answer and the CFA justification as reference.

Design choice: if either side will not parse cleanly as a number, we fall back
to the LLM-judge rather than guess. Confident numeric match only when both sides
parse. This keeps the deterministic path high-precision.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Numeric parsing
# ---------------------------------------------------------------------------

_SCALE = {
    "thousand": 1e3, "thousands": 1e3, "k": 1e3,
    "million": 1e6, "millions": 1e6, "mm": 1e6, "mn": 1e6, "m": 1e6,
    "billion": 1e9, "billions": 1e9, "bn": 1e9, "b": 1e9,
    "trillion": 1e12, "trillions": 1e12,
}

# A number, optionally with $, commas, parentheses (negative), and a scale word.
_NUM = re.compile(
    r"""(?P<paren>\()?\s*
        (?P<sign>[-+])?\s*\$?\s*
        (?P<num>\d[\d,]*(?:\.\d+)?)\s*
        (?P<scale>thousand|thousands|million|millions|billion|billions|
                  trillion|trillions|mm|mn|bn|k|m|b)?\s*
        (?P<pct>%)?\s*
        (?P<parenclose>\))?
    """,
    re.IGNORECASE | re.VERBOSE,
)


@dataclass
class ParsedNumber:
    value: float          # normalized to base units (or face value for pct)
    is_pct: bool


def _looks_numeric_answer(text: str) -> bool:
    """True when the whole answer is basically a number, not a sentence.

    A short answer dominated by a single numeric token with little prose.
    Sentences with numbers ('margin fell 1.7% because...') return False and
    are sent to the LLM-judge.
    """
    stripped = text.strip()
    # Count word-like tokens that are not numbers/units/currency.
    tokens = re.findall(r"[A-Za-z]+", stripped)
    unit_words = set(_SCALE) | {"usd", "approximately", "approx", "about", "per",
                                "share", "shares", "ratio", "x"}
    prose = [t for t in tokens if t.lower() not in unit_words]
    has_number = bool(re.search(r"\d", stripped))
    return has_number and len(prose) <= 2


def parse_number(text: str) -> ParsedNumber | None:
    """Extract the first clean number from text, normalized to base units."""
    m = _NUM.search(text.replace("\u2212", "-"))  # unicode minus
    if not m or not m.group("num"):
        return None
    raw = m.group("num").replace(",", "")
    try:
        val = float(raw)
    except ValueError:
        return None
    if m.group("sign") == "-" or (m.group("paren") and m.group("parenclose")):
        val = -val
    scale = m.group("scale")
    if scale:
        val *= _SCALE[scale.lower()]
    # Recognize the words 'percent'/'pct' as percent markers, not just '%'.
    low = text.lower()
    is_pct = bool(m.group("pct")) or ("percent" in low) or bool(re.search(r"\bpct\b", low))
    return ParsedNumber(value=val, is_pct=is_pct)


def numeric_match(
    gold: str,
    cand: str,
    rel_tol: float = 0.01,
    abs_tol: float = 0.005,
) -> bool | None:
    """Compare two numeric answers within tolerance.

    Returns True/False if both parse and are comparable, else None (meaning:
    defer to the LLM-judge).

    Two financial-domain quirks are handled deliberately:

    * Unit expression. FinanceBench gold answers are bare numbers in the unit
      the question names (usually millions), while a model may answer
      '$1,577 million' or '$1.577 billion'. We treat magnitudes as equal up to a
      clean 1000^k scale shift, so 1577, 1577 million, and 1.577 billion all
      match. The only thing this over-accepts is an identical mantissa exactly
      1000x off, which effectively never occurs for real financial figures.
    * Percentages are NOT scale-shifted (1.7% must not match 17%); they are
      compared at face value, and a percentage never matches a raw magnitude.
    """
    g, c = parse_number(gold), parse_number(cand)
    if g is None or c is None:
        return None
    if g.is_pct != c.is_pct:
        return None  # percentage vs magnitude: not comparable, defer

    if g.is_pct:
        diff = abs(g.value - c.value)
        # Small abs floor so 1.7 vs 1.71 (rounding) counts as equal.
        return diff <= max(abs_tol, rel_tol * abs(g.value), 0.02)

    # Magnitude: accept a match at any 1000^p unit shift of the candidate.
    a = g.value
    for p in range(-4, 5):
        b = c.value * (1000.0 ** p)
        if abs(a - b) <= max(abs_tol, rel_tol * abs(a)):
            return True
    return False


# ---------------------------------------------------------------------------
# LLM-judge
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = (
    "You are a strict grader for financial-disclosure question answering. "
    "You are given a QUESTION, the GOLD answer written by a chartered financial "
    "analyst, the analyst's JUSTIFICATION, and a CANDIDATE answer to grade. "
    "Decide whether the candidate is factually correct with respect to the gold "
    "answer. Ignore differences in wording, formatting, rounding within about 1 "
    "percent, and extra correct context. Mark 'incorrect' if the core figure, "
    "direction, or conclusion disagrees with the gold answer, or if the candidate "
    "declines to answer. Reply with STRICT JSON only: "
    '{"verdict": "correct" | "incorrect" | "partially_correct", '
    '"reason": "<one sentence>"}'
)


def build_judge_prompt(question: str, gold: str, justification: str, cand: str) -> str:
    return (
        f"QUESTION:\n{question}\n\n"
        f"GOLD ANSWER:\n{gold}\n\n"
        f"JUSTIFICATION:\n{justification}\n\n"
        f"CANDIDATE ANSWER:\n{cand}\n\n"
        "Grade the candidate. JSON only."
    )


@dataclass
class JudgeResult:
    verdict: str          # 'correct' | 'incorrect' | 'partially_correct'
    reason: str
    path: str             # 'numeric' | 'llm'


def judge_answer(
    question: str,
    gold: str,
    justification: str,
    cand: str,
    llm_call=None,
    rel_tol: float = 0.01,
) -> JudgeResult:
    """Grade one candidate answer.

    llm_call: a function(system:str, prompt:str) -> str returning the model's
    text. Injected so this module has no hard API dependency and stays testable.
    Only invoked when the deterministic numeric path cannot decide.
    """
    if _looks_numeric_answer(gold):
        m = numeric_match(gold, cand, rel_tol=rel_tol)
        if m is not None:
            return JudgeResult(
                verdict="correct" if m else "incorrect",
                reason=f"numeric tolerance {rel_tol:.0%}",
                path="numeric",
            )
    if llm_call is None:
        raise RuntimeError(
            "LLM-judge needed for this item but no llm_call was provided. "
            "Pass models.make_llm_call() or handle the numeric path upstream."
        )
    import json
    raw = llm_call(JUDGE_SYSTEM, build_judge_prompt(question, gold, justification, cand))
    raw = raw.strip()
    # Strip accidental code fences.
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        obj = json.loads(raw)
        verdict = str(obj.get("verdict", "")).lower()
        if verdict not in {"correct", "incorrect", "partially_correct"}:
            verdict = "incorrect"
        return JudgeResult(verdict=verdict, reason=str(obj.get("reason", "")), path="llm")
    except json.JSONDecodeError:
        return JudgeResult(verdict="incorrect", reason=f"unparseable judge output: {raw[:80]}", path="llm")


# ---------------------------------------------------------------------------
# Offline self-test for the deterministic numeric path
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cases = [
        # (gold, candidate, expected)
        ("$1577.00", "$1,577 million", True),   # unit expression, same value
        ("$1577.00", "1577", True),
        ("$1577.00", "$1.577 billion", True),   # 1000x unit shift, same value
        ("$1577.00", "$1600", False),
        ("$8.70", "$8.71", True),
        ("$8.70", "$9.10", False),
        ("1.7%", "1.7 percent", True),          # 'percent' word recognized
        ("1.7%", "1.71%", True),
        ("1.7%", "2.4%", False),
        ("1.7%", "17%", False),                 # percentages not scale-shifted
        ("$(200)", "-200", True),               # parentheses = negative
        ("0.96", "0.95", False),                # different 2-dp ratio value
        ("0.96", "1.20", False),
    ]
    passed = 0
    for gold, cand, exp in cases:
        got = numeric_match(gold, cand)
        ok = (got is exp)
        passed += ok
        print(f"{'ok ' if ok else 'FAIL'}  gold={gold!r:14} cand={cand!r:18} -> {got} (exp {exp})")
    print(f"\n{passed}/{len(cases)} numeric cases passed")

    prose = "No, the company is managing its CAPEX and Fixed Assets efficiently."
    print(f"\n'looks numeric' on prose answer -> {_looks_numeric_answer(prose)} (want False)")
    print(f"'looks numeric' on '$1577.00'   -> {_looks_numeric_answer('$1577.00')} (want True)")
