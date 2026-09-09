"""Pure metric functions. No I/O, no model — unit-testable in isolation.

Task framing: binary "surface to a screener?" classification.
  gold  1 = the patient should be shown this trial (eligible, or uncertain-and-worth-review)
  gold  0 = the trial is a hard miss (ineligible)
A false negative (missed an eligible trial) is the costly error; a false
positive (a wasted screening slot) is cheap. The cost matrix encodes that.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Confusion:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    def f1(self) -> float:
        p, r = self.precision(), self.recall()
        d = p + r
        return 2 * p * r / d if d else 0.0


def confusion(preds: list[int], golds: list[int]) -> Confusion:
    """Build a confusion matrix from aligned 0/1 prediction and gold lists."""
    if len(preds) != len(golds):
        raise ValueError("preds and golds must be the same length")
    c = Confusion()
    for p, g in zip(preds, golds):
        if p == 1 and g == 1:
            c.tp += 1
        elif p == 1 and g == 0:
            c.fp += 1
        elif p == 0 and g == 1:
            c.fn += 1
        else:
            c.tn += 1
    return c


def weighted_cost(c: Confusion, w_fn: float = 10.0, w_fp: float = 1.0) -> tuple[float, float]:
    """Cost of errors under an asymmetric matrix. Returns (total, per_case)."""
    total = w_fn * c.fn + w_fp * c.fp
    return total, (total / c.n if c.n else 0.0)


def faithfulness_rate(evidence_and_criteria: list[tuple[str, str]]) -> float:
    """Fraction of cited evidence snippets that appear verbatim in the trial's
    criteria text. A cheap, deterministic check that the reasoner isn't
    confabulating its citations. Only non-empty evidence strings are scored.
    """
    checked = ok = 0
    for evidence, criteria in evidence_and_criteria:
        ev = (evidence or "").strip().lower()
        if not ev:
            continue
        checked += 1
        if ev in (criteria or "").lower():
            ok += 1
    return ok / checked if checked else 0.0
