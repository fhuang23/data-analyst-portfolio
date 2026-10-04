"""Run the eval harness locally: rules baseline vs stock Qwen via Ollama."""
from __future__ import annotations

import sys
import time

import requests

from sra_matcher import cost as cost_mod
from sra_matcher import prompts
from sra_matcher.reasoners import ReasonerResult, RulesReasoner, Usage
from sra_matcher.schemas import TrialVerdict
from sra_matcher.eval.data import load_pairs
from sra_matcher.eval.harness import print_table, run_contender

OLLAMA_URL = "http://localhost:11434/api/chat"


class OllamaQwenReasoner:
    def __init__(self, model="qwen3:4b", name=None):
        self.model = model
        self.name = name or "qwen-lora"
        self._schema = TrialVerdict.model_json_schema()

    def judge(self, patient_profile_json, candidate_json):
        prompt = prompts.REASONER_INSTRUCTION.format(
            patient_profile_json=patient_profile_json,
            current_candidate_json=candidate_json,
        )
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "format": self._schema,
            "options": {"temperature": 0},
        }
        t0 = time.perf_counter()
        r = requests.post(OLLAMA_URL, json=payload, timeout=180)
        r.raise_for_status()
        wall = time.perf_counter() - t0
        content = r.json()["message"]["content"]
        verdict = TrialVerdict.model_validate_json(content)
        return ReasonerResult(verdict, Usage(wall_seconds=wall, served_locally=True))


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen3:4b"
    pairs = load_pairs("pairs.jsonl")
    print(f"Loaded {len(pairs)} pairs. Running rules baseline + {model} ...")
    print("(local Qwen is ~5-8s/pair, so this takes a while)\n")

    slate = [
        (RulesReasoner(), cost_mod.ZeroCost()),
        (OllamaQwenReasoner(model=model),
         cost_mod.GpuCost(gpu_hourly_usd=2.00, fixed_cost_usd=8.00)),
    ]
    results = []
    for reasoner, cm in slate:
        print(f"Running {reasoner.name} ...")
        results.append(run_contender(reasoner, cm, pairs))
    print_table(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
