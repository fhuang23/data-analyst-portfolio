"""Cost models — price a Usage record in dollars.

Kept separate from reasoners on purpose: a reasoner measures what it consumed, a
CostModel decides what that costs. Swapping cloud vendors or GPU rates never
touches reasoning code.

Two shapes carry the whole study:
    TokenCost  fixed = 0,            marginal = tokens * price      (external API)
    GpuCost    fixed = training run, marginal = gpu_seconds * rate  (self-hosted)

Breakeven falls straight out of the two: total(V) = fixed + V * marginal. Below
the crossover volume the API is cheaper; above it, the fine-tune amortizes. That
crossover is the single chart the cost-efficiency thesis rests on.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .reasoners import Usage


@runtime_checkable
class CostModel(Protocol):
    name: str
    fixed_cost_usd: float  # one-time, amortized over all predictions

    def marginal_cost(self, usage: Usage) -> float:
        """Dollar cost of a single prediction."""
        ...


@dataclass
class ZeroCost:
    """The rules baseline: no training, no per-call spend."""
    name: str = "free"
    fixed_cost_usd: float = 0.0

    def marginal_cost(self, usage: Usage) -> float:
        return 0.0


@dataclass
class TokenCost:
    """Per-token API pricing. Prices are USD per 1M tokens — read them off the
    provider's current rate card and pass them in; don't hardcode a number that
    goes stale."""
    in_price_per_mtok: float
    out_price_per_mtok: float
    name: str = "api-tokens"
    fixed_cost_usd: float = 0.0

    def marginal_cost(self, usage: Usage) -> float:
        return (
            usage.input_tokens / 1e6 * self.in_price_per_mtok
            + usage.output_tokens / 1e6 * self.out_price_per_mtok
        )


@dataclass
class GpuCost:
    """Self-hosted pricing: a one-time training cost plus GPU rent per call.

        fixed_cost_usd = training_hours * gpu_hourly   (log the LoRA run once)
        marginal       = wall_seconds  * gpu_hourly / 3600

    Honest caveat: pricing per-call on wall_seconds assumes one request owns the
    GPU for its full wall time. Under batching/concurrency the effective cost is
    gpu_hourly / (requests_per_hour at your batch size), which is much lower and
    is what actually shifts the breakeven point. When you have real throughput
    numbers from your serving setup, swap this for ThroughputGpuCost below.
    """
    gpu_hourly_usd: float
    fixed_cost_usd: float = 0.0
    name: str = "gpu-seconds"

    def marginal_cost(self, usage: Usage) -> float:
        return usage.wall_seconds * self.gpu_hourly_usd / 3600.0


@dataclass
class ThroughputGpuCost:
    """Amortized self-hosted cost once you can measure sustained throughput.
    marginal = gpu_hourly / requests_per_hour. Prefer this over GpuCost for the
    final breakeven chart — serial wall-time overstates real per-call cost."""
    gpu_hourly_usd: float
    requests_per_hour: float
    fixed_cost_usd: float = 0.0
    name: str = "gpu-throughput"

    def marginal_cost(self, usage: Usage) -> float:
        return self.gpu_hourly_usd / max(self.requests_per_hour, 1e-9)
