"""Thin LLM-call wrapper so the rest of the code has no hard SDK dependency.

make_llm_call() returns a function(system: str, prompt: str) -> str.
Default provider is Anthropic. Model ids are read from env so you can point
each role at whatever model you have access to without editing code:

    MCFB_MAKER_MODEL   (default: claude-sonnet-4-5)
    MCFB_JUDGE_MODEL   (default: claude-haiku-4-5)

Set MCFB_MODEL to override both at once. Override the defaults to any model id
your API key can call.
"""
from __future__ import annotations

import os

DEFAULT_MAKER_MODEL = os.environ.get("MCFB_MODEL") or os.environ.get(
    "MCFB_MAKER_MODEL", "claude-sonnet-4-5")
DEFAULT_JUDGE_MODEL = os.environ.get("MCFB_MODEL") or os.environ.get(
    "MCFB_JUDGE_MODEL", "claude-haiku-4-5")


def _mock_call(system: str, prompt: str) -> str:
    """Deterministic canned responses, no API. Detects role from the system
    prompt and returns validly-shaped JSON so the whole pipeline (maker ->
    checker -> adjudicator -> judge) runs end to end without a key. Values are
    NOT meaningful; this only proves the plumbing. Behavior varies by prompt
    length so every branch (agree / disagree / abstain) gets exercised.
    """
    s = system.lower()
    n = len(prompt)
    if "adjudicator" in s:
        abstain = (n % 3 == 0)
        ans = "" if abstain else "mock adjudicated answer"
        return f'{{"final_answer": "{ans}", "abstain": {str(abstain).lower()}, "reason": "mock adj"}}'
    if "auditor" in s:
        verdict = "agree" if (n % 2 == 0) else "disagree"
        prop = "" if verdict == "agree" else "mock proposed answer"
        return f'{{"verdict": "{verdict}", "reason": "mock audit", "proposed_answer": "{prop}"}}'
    if "grader" in s:
        verdict = "correct" if (n % 2 == 0) else "incorrect"
        return f'{{"verdict": "{verdict}", "reason": "mock judge"}}'
    # maker
    return '{"answer": "mock answer 123", "citation": "mock cited line", "can_answer": true}'


def _resolve(role: str) -> str:
    """Read the model id from env at CALL time (not frozen at import), so
    setting env vars in a notebook cell always takes effect on the next call
    regardless of import order."""
    override = os.environ.get("MCFB_MODEL")
    if override:
        return override
    if role == "judge":
        return os.environ.get("MCFB_JUDGE_MODEL", "claude-haiku-4-5")
    return os.environ.get("MCFB_MAKER_MODEL", "claude-sonnet-4-5")


def make_llm_call(model: str | None = None, temperature: float = 0.0, max_tokens: int = 1024):
    """Return a callable(system, prompt) -> text.

    model=None resolves from env at call time. Pass model='mock' for a keyless
    deterministic backend. Pass 'maker'/'judge' to resolve that role from env.
    """
    if model in (None, "maker", "judge"):
        model = _resolve(model or "maker")

    if model == "mock":
        return _mock_call

    try:
        import anthropic  # lazy
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "The 'anthropic' package is needed for a real (non-mock) run. "
            "Install it with:  pip install anthropic   "
            "(or set MOCK=True / model='mock' for a free dry run)."
        ) from e

    # Optional workspace scoping: set MCFB_WORKSPACE_ID if your API key is an
    # org-level key that the account requires to name a workspace.
    default_headers = {}
    wsid = os.environ.get("MCFB_WORKSPACE_ID")
    if wsid:
        default_headers["anthropic-workspace-id"] = wsid
    client = anthropic.Anthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        default_headers=default_headers or None,
    )

    state = {"temp_kwarg": True}  # remember once whether the SDK takes temperature

    def _extract(resp) -> str:
        return "".join(
            b.text for b in resp.content if getattr(b, "type", "") == "text"
        )

    def _call(system: str, prompt: str) -> str:
        base = dict(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        # Newer SDKs drop the temperature kwarg; pass it through extra_body so
        # determinism (temperature=0) is preserved either way.
        if state["temp_kwarg"]:
            try:
                return _extract(client.messages.create(temperature=temperature, **base))
            except TypeError as e:
                if "temperature" not in str(e):
                    raise
                state["temp_kwarg"] = False  # stop trying the kwarg on later calls
        return _extract(client.messages.create(extra_body={"temperature": temperature}, **base))

    return _call
