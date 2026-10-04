"""Per-stage timing and per-run logging to Supabase.

Timing
------
Call `start_timing()` at the start of a request, then wrap each pipeline stage:

    with stage("plan"):
        sub_questions = await planner.plan(question)

Timings are kept in a context variable, so concurrent requests never mix, and
tasks spawned with asyncio.gather share the request's collector. Repeated
stages (e.g. a second critique pass) add up under the same name. Wrap a
parallel fan-out as ONE stage around the gather, not each item inside it,
or the parallel times get summed.

Logging
-------
`schedule_log(row)` writes a row to the `research_runs` table in the
background; it never slows down or breaks the response. Logging is OFF
unless SUPABASE_URL and SUPABASE_SERVICE_KEY are set.
"""

import asyncio
import contextvars
import json
import logging
import os
import time
import urllib.error
import urllib.request
from contextlib import contextmanager

logger = logging.getLogger(__name__)

TABLE = "research_runs"
INSERT_TIMEOUT_SECONDS = 10


# ---- stage timing ----

class StageTimings:
    def __init__(self) -> None:
        self.ms: dict[str, float] = {}

    def add(self, name: str, ms: float) -> None:
        self.ms[name] = round(self.ms.get(name, 0.0) + ms, 1)


_current: contextvars.ContextVar[StageTimings | None] = contextvars.ContextVar("stage_timings", default=None)


def start_timing() -> StageTimings:
    timings = StageTimings()
    _current.set(timings)
    return timings


@contextmanager
def stage(name: str):
    """Time a block (sync or async body) and record it under `name`."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        timings = _current.get()
        if timings is not None:
            timings.add(name, (time.perf_counter() - t0) * 1000)


# ---- Supabase logging ----

def _config() -> tuple[str, str] | None:
    from app.auth import supabase_base_url

    url = supabase_base_url(os.getenv("SUPABASE_URL"))
    key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_SECRET_KEY")
    return (url, key) if url and key else None


def _insert(url: str, key: str, row: dict) -> None:
    headers = {
        "apikey": key,
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }
    # Legacy service_role keys are JWTs and also go in Authorization; the newer
    # sb_secret_ keys must only be sent as `apikey`.
    if key.startswith("eyJ"):
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(
        f"{url}/rest/v1/{TABLE}",
        data=json.dumps(row, default=str).encode(),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=INSERT_TIMEOUT_SECONDS):
        pass


async def log_run(row: dict) -> None:
    config = _config()
    if not config:
        return
    try:
        await asyncio.to_thread(_insert, *config, row)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:300]
        logger.warning("Run log insert failed (%s): %s", e.code, body)
    except Exception as e:  # logging must never break a request
        logger.warning("Run log insert failed: %s", e)


_pending: set[asyncio.Task] = set()


def schedule_log(row: dict) -> None:
    """Fire-and-forget insert; keeps a reference so the task isn't garbage-collected."""
    task = asyncio.create_task(log_run(row))
    _pending.add(task)
    task.add_done_callback(_pending.discard)


def build_row(
    *,
    user: dict | None,
    pipeline: str,
    question: str,
    total_ms: float,
    timings: StageTimings,
    result: dict | None = None,
    error: str | None = None,
) -> dict:
    result = result or {}
    return {
        "user_id": (user or {}).get("id"),
        "user_email": (user or {}).get("email"),
        "pipeline": pipeline,
        "question": question,
        "status": "error" if error else "ok",
        "error": error,
        "sub_questions": result.get("sub_questions", []),
        "report": result.get("report"),
        "sources": result.get("sources", []),
        "approved": result.get("approved"),
        "iterations": result.get("iterations"),
        "remaining_issues": result.get("remaining_issues", []),
        "total_ms": round(total_ms),
        "stage_timings_ms": dict(timings.ms),
    }