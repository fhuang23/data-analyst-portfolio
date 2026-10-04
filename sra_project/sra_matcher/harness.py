"""The harness: run any Reasoner over labeled pairs, score it, price it.

One loop, contender-agnostic. Because every reasoner implements the same
judge(), adding your fine-tuned Qwen alongside Gemini and the rules baseline is
a list entry, not a code change here — the property the whole design was for.

Two things get measured, held apart:

  Quality  — the operational question is "does the shortlist catch eligible
             trials without flooding screeners?", so:
               predicted positive = shortlisted = label in {eligible, uncertain}
                                    (matches RankingAgent, which keeps both)
               gold positive      = truly eligible
             A miss (truly eligible, dropped) is the expensive error; a false
             positive is a wasted screening slot. We report precision/recall/F1
             and F-beta (default beta=2, recall-weighted) to encode that the two
             errors don't cost the same — the "cost-weighted F1" from the README.

  Cost     — fixed (training) + marginal (tokens or GPU-seconds), summed over
             the run, plus the breakeven volume between any two contenders.

Run:  python -m sra_matcher.eval.harness path/to/pairs.jsonl
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field

from .. import config, cost as cost_mod
from ..reasoners import (
    GeminiReasoner,
    Reasoner,
    RulesReasoner,
    VllmQwenReasoner,
)
from ..schemas import TrialLabel
from .data import LabeledPair, load_pairs

# how a predicted TrialLabel maps to the shortlist decision
SHORTLISTED = {TrialLabel.ELIGIBLE, TrialLabel.UNCERTAIN}


@dataclass
class Scores:
    name: str
    n: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    errors: int = 0            # judge() calls that raised (counted as a miss)
    fixed_usd: float = 0.0
    marginal_usd: float = 0.0  # summed over the run
    wall_seconds: float = 0.0

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    def fbeta(self, beta: float = 2.0) -> float:
        p, r = self.precision, self.recall
        b2 = beta * beta
        d = b2 * p + r
        return (1 + b2) * p * r / d if d else 0.0

    @property
    def total_usd(self) -> float:
        return self.fixed_usd + self.marginal_usd

    @property
    def mean_marginal_usd(self) -> float:
        return self.marginal_usd / self.n if self.n else 0.0


def run_contender(reasoner: Reasoner, cost_model: cost_mod.CostModel,
                  pairs: list[LabeledPair]) -> Scores:
    s = Scores(name=reasoner.name, fixed_usd=cost_model.fixed_cost_usd)
    for pair in pairs:
        s.n += 1
        try:
            result = reasoner.judge(pair.patient_profile_json, pair.candidate_json)
        except Exception as exc:  # a failed judgment is a dropped trial, not a skip
            s.errors += 1
            if pair.gold == TrialLabel.ELIGIBLE:
                s.fn += 1
            else:
                s.tn += 1
            print(f"  ! {reasoner.name} errored on {pair.nct_id}: {exc}", file=sys.stderr)
            continue

        pred_pos = result.verdict.label in SHORTLISTED
        gold_pos = pair.gold == TrialLabel.ELIGIBLE
        if pred_pos and gold_pos:
            s.tp += 1
        elif pred_pos and not gold_pos:
            s.fp += 1
        elif not pred_pos and gold_pos:
            s.fn += 1
        else:
            s.tn += 1

        s.marginal_usd += cost_model.marginal_cost(result.usage)
        s.wall_seconds += result.usage.wall_seconds
    return s


def breakeven_volume(a: Scores, b: Scores) -> float | None:
    """Prediction volume at which a and b cost the same:
        fixed_a + V*m_a = fixed_b + V*m_b
    Returns the crossover volume, or None if their marginal costs are equal
    (parallel lines — whoever has the lower fixed cost always wins)."""
    dm = a.mean_marginal_usd - b.mean_marginal_usd
    if abs(dm) < 1e-12:
        return None
    v = (b.fixed_usd - a.fixed_usd) / dm
    return v if v > 0 else None


def print_table(results: list[Scores], beta: float = 2.0) -> None:
    baseline = next((r for r in results if r.name == "rules"), None)
    print(f"\n{'contender':<12}{'P':>7}{'R':>7}{'F1':>7}{f'F{beta:g}':>7}"
          f"{'lift':>7}{'$/1k':>9}{'fixed$':>9}{'total$':>10}")
    print("-" * 85)
    for r in results:
        f1, fb = r.fbeta(1.0), r.fbeta(beta)
        lift = "" if baseline is None else f"{r.recall - baseline.recall:+.2f}"
        print(f"{r.name:<12}{r.precision:>7.2f}{r.recall:>7.2f}{f1:>7.2f}{fb:>7.2f}"
              f"{lift:>7}{r.mean_marginal_usd * 1000:>9.3f}"
              f"{r.fixed_usd:>9.2f}{r.total_usd:>10.2f}")
    if baseline is not None:
        print("\nlift = recall improvement over the rules baseline "
              "(the whole thesis is that this is positive and worth its cost).")

    # breakeven between the two LLM contenders, if both ran
    api = next((r for r in results if r.name == "gemini"), None)
    qwen = next((r for r in results if r.name == "qwen-lora"), None)
    if api and qwen:
        v = breakeven_volume(api, qwen)
        if v is None:
            cheaper = min(api, qwen, key=lambda r: r.fixed_usd)
            print(f"\nBreakeven: marginal costs equal; {cheaper.name} always cheaper "
                  f"(lower fixed cost).")
        else:
            print(f"\nBreakeven: ~{v:,.0f} predictions. Below this, {api.name} is cheaper; "
                  f"above it, {qwen.name} amortizes its training cost.")


def build_contenders() -> list[tuple[Reasoner, cost_mod.CostModel]]:
    """Wire the slate. Prices/paths are placeholders — set them from your own
    rate card, GPU rental, and served endpoint before trusting the $ columns."""
    return [
        (RulesReasoner(), cost_mod.ZeroCost()),
        (
            GeminiReasoner(),
            # TODO: current rate card for config.REASONER_MODEL, USD per 1M tokens
            cost_mod.TokenCost(in_price_per_mtok=1.25, out_price_per_mtok=5.00),
        ),
        (
            VllmQwenReasoner(),
            # TODO: fixed = LoRA training hours * GPU $/hr; rate = your GPU $/hr.
            cost_mod.GpuCost(gpu_hourly_usd=2.00, fixed_cost_usd=8.00),
        ),
    ]


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: python -m sra_matcher.eval.harness <pairs.jsonl>", file=sys.stderr)
        return 2
    pairs = load_pairs(argv[1])
    print(f"Loaded {len(pairs)} labeled pairs.")
    results: list[Scores] = []
    for reasoner, cm in build_contenders():
        print(f"Running {reasoner.name} ...")
        results.append(run_contender(reasoner, cm, pairs))
    print_table(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
