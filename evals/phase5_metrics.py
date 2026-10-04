"""Phase 5 metrics: hand-rolled orchestrator vs LangGraph, side by side.

Runs the same questions through both full-report endpoints over HTTP, so both
pipelines pay identical serialization/transport overhead and the comparison
is apples to apples.

Requires the API to be running:
    uvicorn app.main:app

Then, from the project root:
    python -m evals.phase5_metrics
"""

import asyncio
import json
import statistics
import time
import urllib.error
import urllib.request

from evals.dataset import EVAL_QUESTIONS

BASE_URL = "http://127.0.0.1:8000"

PIPELINES = {
    "orchestrator": "/research/report",
    "langgraph": "/research/graph",
}

# Same 3 questions as Phases 3-4 so results are comparable across phases.
TEST_QUESTIONS = [q.question for q in EVAL_QUESTIONS[:3]]

# Each report generation is ~7 LLM calls; pause between EVERY run so each one
# lands in a fresh per-minute quota window (free tier: 15/min).
PAUSE_BETWEEN_RUNS = 45
REQUEST_TIMEOUT = 300

# Field names differ slightly between response models; try common spellings.
REPORT_KEYS = ("report", "final_report", "answer")
SOURCES_KEYS = ("sources",)
ITERATION_KEYS = ("iterations", "iteration", "revision_count")
APPROVED_KEYS = ("approved", "is_approved")

_warned_keys: set[str] = set()


def _pick(data: dict, keys: tuple[str, ...]):
    for k in keys:
        if k in data:
            return data[k]
    return None


def _post(path: str, question: str) -> dict:
    body = json.dumps({"question": question}).encode()
    req = urllib.request.Request(
        BASE_URL + path,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        return json.loads(resp.read())


async def run_once(pipeline: str, question: str) -> dict:
    path = PIPELINES[pipeline]
    t0 = time.perf_counter()
    data = await asyncio.to_thread(_post, path, question)
    latency = time.perf_counter() - t0

    iterations = _pick(data, ITERATION_KEYS)
    approved = _pick(data, APPROVED_KEYS)
    sources = _pick(data, SOURCES_KEYS)
    report = _pick(data, REPORT_KEYS)

    # If a field can't be found, show the actual response keys once so the
    # key tuples above can be adjusted.
    if None in (iterations, approved, sources, report) and pipeline not in _warned_keys:
        _warned_keys.add(pipeline)
        print(f"    [note] {pipeline} response keys: {sorted(data.keys())}")

    return {
        "pipeline": pipeline,
        "question": question,
        "latency": latency,
        "iterations": iterations,
        "approved": approved,
        "num_sources": len(sources) if isinstance(sources, list) else None,
        "report_chars": len(report) if isinstance(report, str) else None,
    }


def _fmt(value, spec: str = "") -> str:
    return "?" if value is None else format(value, spec)


def _summary(rows: list[dict]) -> dict:
    latencies = [r["latency"] for r in rows]
    iters = [r["iterations"] for r in rows if isinstance(r["iterations"], int)]
    sources = [r["num_sources"] for r in rows if r["num_sources"] is not None]
    chars = [r["report_chars"] for r in rows if r["report_chars"] is not None]
    approved = [r for r in rows if r["approved"] is True]
    revised = [i for i in iters if i > 1]
    return {
        "runs": len(rows),
        "median_latency": f"{statistics.median(latencies):.1f}s" if latencies else "?",
        "latency_range": f"{min(latencies):.1f}-{max(latencies):.1f}s" if latencies else "?",
        "avg_iterations": f"{statistics.mean(iters):.2f}" if iters else "?",
        "revised": f"{len(revised)}/{len(iters)}" if iters else "?",
        "approved": f"{len(approved)}/{len(rows)}",
        "avg_sources": f"{statistics.mean(sources):.1f}" if sources else "?",
        "avg_report_chars": f"{statistics.mean(chars):,.0f}" if chars else "?",
    }


async def main() -> None:
    # Interleave pipelines per question so both see similar network/quota
    # conditions, rather than running all of one then all of the other.
    plan = [(p, q) for q in TEST_QUESTIONS for p in PIPELINES]
    total = len(plan)
    print(
        f"Running {len(TEST_QUESTIONS)} questions x {len(PIPELINES)} pipelines "
        f"= {total} report generations\n"
        f"(pausing {PAUSE_BETWEEN_RUNS}s between runs to respect the 15/min quota; "
        f"~{total * PAUSE_BETWEEN_RUNS // 60 + 2} min total)\n"
    )

    rows: dict[str, list[dict]] = {p: [] for p in PIPELINES}

    for i, (pipeline, q) in enumerate(plan, 1):
        try:
            r = await run_once(pipeline, q)
            rows[pipeline].append(r)
            approved = {True: "approved", False: "NOT approved"}.get(r["approved"], "approved=?")
            print(
                f"  [{pipeline:<12}] {q[:40]:<40} "
                f"{_fmt(r['iterations'])} iter, {approved}, "
                f"{r['latency']:.1f}s, {_fmt(r['num_sources'])} sources"
            )
        except urllib.error.URLError as e:
            print(f"  [{pipeline:<12}] {q[:40]:<40} FAILED: {e}")
            if isinstance(e.reason, ConnectionRefusedError):
                print("\nCan't reach the API. Start it first: uvicorn app.main:app")
                return
        except Exception as e:
            print(f"  [{pipeline:<12}] {q[:40]:<40} FAILED: {str(e)[:60]}")

        if i < total:
            print(f"    ...pausing {PAUSE_BETWEEN_RUNS}s for quota window...")
            await asyncio.sleep(PAUSE_BETWEEN_RUNS)

    if not any(rows.values()):
        print("\nNo successful runs.")
        return

    summaries = {p: _summary(r) for p, r in rows.items() if r}
    labels = [
        ("runs", "Successful runs"),
        ("median_latency", "Median latency"),
        ("latency_range", "Latency range"),
        ("avg_iterations", "Avg iterations"),
        ("revised", "Triggered a revision"),
        ("approved", "Approved (within cap)"),
        ("avg_sources", "Avg sources / report"),
        ("avg_report_chars", "Avg report length (chars)"),
    ]

    print("\n" + "=" * 66)
    print("PHASE 5 ORCHESTRATOR vs LANGGRAPH SUMMARY")
    print("=" * 66)
    header = f"{'Metric':<28}" + "".join(f"{p:>19}" for p in summaries)
    print(header)
    print("-" * 66)
    for key, label in labels:
        print(f"{label:<28}" + "".join(f"{str(s[key]):>19}" for s in summaries.values()))
    print("=" * 66)
    print("Both pipelines measured over HTTP on the same questions, interleaved.")
    print("Small sample (n=3 per pipeline) due to free-tier quota; treat as directional.")


if __name__ == "__main__":
    asyncio.run(main())