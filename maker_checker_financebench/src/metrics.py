"""Agreement, Cohen's kappa, and bootstrap confidence intervals.

Pure Python / no heavy deps so it runs and tests offline. Used to validate the
judge (judge-vs-human agreement) and later to report system accuracy with
uncertainty given the small n=150.
"""
from __future__ import annotations

import random
from collections import Counter


def to_binary(label: str) -> int:
    """Map a verdict to correct(1)/not-correct(0).

    'partially_correct' is treated as not-correct under the strict grading the
    precision floor demands. Change here if you want a lenient variant.
    """
    return 1 if str(label).strip().lower() == "correct" else 0


def raw_agreement(a: list[str], b: list[str]) -> float:
    assert len(a) == len(b) and a, "need equal, non-empty label lists"
    return sum(x == y for x, y in zip(a, b)) / len(a)


def cohens_kappa(a: list[str], b: list[str]) -> float:
    """Cohen's kappa on two raters' labels (any label set)."""
    assert len(a) == len(b) and a, "need equal, non-empty label lists"
    n = len(a)
    po = raw_agreement(a, b)
    ca, cb = Counter(a), Counter(b)
    labels = set(ca) | set(cb)
    pe = sum((ca.get(l, 0) / n) * (cb.get(l, 0) / n) for l in labels)
    if pe == 1.0:
        return 1.0  # perfect and degenerate
    return (po - pe) / (1 - pe)


def confusion(human: list[str], judge: list[str]) -> dict[str, int]:
    """Confusion counts using binary correct/not-correct."""
    h = [to_binary(x) for x in human]
    j = [to_binary(x) for x in judge]
    out = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for hh, jj in zip(h, j):
        if jj == 1 and hh == 1:
            out["tp"] += 1
        elif jj == 0 and hh == 0:
            out["tn"] += 1
        elif jj == 1 and hh == 0:
            out["fp"] += 1  # judge said correct, human said not
        else:
            out["fn"] += 1  # judge said not, human said correct
    return out


def bootstrap_ci(
    successes: int,
    n: int,
    iters: int = 10000,
    alpha: float = 0.05,
    seed: int = 7,
) -> tuple[float, float, float]:
    """Bootstrap CI for a proportion. Returns (point, lo, hi).

    Reported alongside every accuracy number because n=150 (and validation
    subsets are smaller) make point estimates noisy.
    """
    if n == 0:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    data = [1] * successes + [0] * (n - successes)
    means = []
    for _ in range(iters):
        s = sum(data[rng.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    lo = means[int((alpha / 2) * iters)]
    hi = means[int((1 - alpha / 2) * iters)]
    return (successes / n, lo, hi)


def coverage_at_floor(
    labels: list[str],
    abstained: list[bool],
    floor: float = 0.95,
) -> dict[str, float]:
    """Given per-item correctness labels and which items abstained, report
    accuracy on answered items and the coverage achieved.

    This is the operational readout: how much can we auto-answer while keeping
    accuracy on answered items at or above the floor.
    """
    answered = [to_binary(l) for l, ab in zip(labels, abstained) if not ab]
    n_ans = len(answered)
    acc = (sum(answered) / n_ans) if n_ans else 0.0
    return {
        "coverage": n_ans / len(labels) if labels else 0.0,
        "answered_accuracy": acc,
        "meets_floor": float(acc >= floor),
        "n_answered": float(n_ans),
        "n_total": float(len(labels)),
    }


if __name__ == "__main__":
    # Synthetic judge-vs-human check.
    human = ["correct", "correct", "incorrect", "correct", "incorrect",
             "incorrect", "correct", "correct", "incorrect", "correct"]
    judge = ["correct", "correct", "incorrect", "incorrect", "incorrect",
             "incorrect", "correct", "correct", "correct", "correct"]
    print("raw agreement:", round(raw_agreement(human, judge), 3))
    print("cohen's kappa:", round(cohens_kappa(human, judge), 3))
    print("confusion:", confusion(human, judge))
    pt, lo, hi = bootstrap_ci(successes=42, n=45)
    print(f"bootstrap CI for 42/45: {pt:.3f} [{lo:.3f}, {hi:.3f}]")
    print("coverage@floor:", coverage_at_floor(
        labels=["correct", "correct", "incorrect", "correct"],
        abstained=[False, False, True, False],
        floor=0.95,
    ))
