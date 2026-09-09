"""Run the pipeline once against a sample patient and print the shortlist.

    python -m sra_matcher.main

Requires model credentials (see .env.example). Retrieval hits the public
ClinicalTrials.gov API and needs no key.
"""
from __future__ import annotations

import asyncio
import json

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from . import config, state
from .pipeline import root_agent

SAMPLE_PATIENT = """\
62-year-old woman with HER2-positive metastatic breast cancer, ECOG 1.
Prior treatment: trastuzumab + pertuzumab + docetaxel, then T-DM1 (progressed).
No brain metastases. Adequate organ function. Lives in Palo Alto, California.
"""

SESSION_ID = "s1"


async def run(patient_text: str = SAMPLE_PATIENT) -> list[dict]:
    session_service = InMemorySessionService()
    runner = Runner(
        agent=root_agent, app_name=config.APP_NAME, session_service=session_service
    )
    await session_service.create_session(
        app_name=config.APP_NAME, user_id=config.DEFAULT_USER_ID, session_id=SESSION_ID
    )

    message = types.Content(role="user", parts=[types.Part(text=patient_text)])
    async for event in runner.run_async(
        user_id=config.DEFAULT_USER_ID, session_id=SESSION_ID, new_message=message
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if getattr(part, "text", None):
                    print(f"[{event.author}] {part.text}")

    session = await session_service.get_session(
        app_name=config.APP_NAME, user_id=config.DEFAULT_USER_ID, session_id=SESSION_ID
    )
    shortlist = session.state.get(state.SHORTLIST, [])
    print("\n=== SHORTLIST ===")
    print(json.dumps(shortlist, indent=2)[:4000])
    return shortlist


if __name__ == "__main__":
    asyncio.run(run())
