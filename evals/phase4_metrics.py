import asyncio
import time
import statistics

from app.agents.orchestrator import orchestrator


TEST_QUESTIONS = [
    "Compare REST and GraphQL for a high-traffic API",
    "What are the tradeoffs of microservices vs a monolith?",
    "How should a team choose between SQL and NoSQL databases?",
]

# One run per question. Long pause between runs so each ~7-call report
# generation lands in a fresh per-minute quota window (free tier: 15/min).
PAUSE_BETWEEN_RUNS = 45


async def run_once(question: str) -> dict:
    t0 = time.perf_counter()
    result = await orchestrator.run_report(question)
    latency = time.perf_counter() - t0

    return {
        "question": question,
        "approved": result.approved,
        "iterations": result.iterations,
        "revised": result.iterations > 1,
        "remaining_issues": len(result.remaining_issues),
        "latency": latency,
        "report_chars": len(result.report),
        "num_sources": len(result.sources),
    }


async def main() -> None:
    rows = []
    n = len(TEST_QUESTIONS)
    print(f"Running {n} report generations "
          f"(pausing {PAUSE_BETWEEN_RUNS}s between to respect the 15/min quota)...\n")

    for i, q in enumerate(TEST_QUESTIONS, 1):
        try:
            r = await run_once(q)
            rows.append(r)
            tag = "revised" if r["revised"] else "clean 1st pass"
            approved = "approved" if r["approved"] else f"NOT approved ({r['remaining_issues']} left)"
            print(
                f"  {q[:42]:<42} "
                f"{r['iterations']} iter ({tag}), {approved}, "
                f"{r['latency']:.1f}s, {r['num_sources']} sources"
            )
        except Exception as e:
            print(f"  {q[:42]:<42} FAILED: {str(e)[:60]}")

        # Pause before the next question (skip after the last one)
        if i < n:
            print(f"    ...pausing {PAUSE_BETWEEN_RUNS}s for quota window...")
            await asyncio.sleep(PAUSE_BETWEEN_RUNS)

    if not rows:
        print("\nNo successful runs.")
        return

    revised = sum(1 for r in rows if r["revised"])
    approved = sum(1 for r in rows if r["approved"])
    iterations = [r["iterations"] for r in rows]
    latencies = [r["latency"] for r in rows]

    print("\n" + "=" * 60)
    print("PHASE 4 SELF-CORRECTION METRICS SUMMARY")
    print("=" * 60)
    print(f"Report generations:       {len(rows)}")
    print(f"Triggered a revision:     {revised}/{len(rows)}")
    print(f"Approved (within cap):    {approved}/{len(rows)}")
    print(f"Avg iterations:           {statistics.mean(iterations):.2f}")
    print(f"Iteration range:          {min(iterations)}-{max(iterations)}")
    print(f"Median latency:           {statistics.median(latencies):.1f}s")
    print("=" * 60)
    print("Small sample (n=3) due to free-tier quota; revision behavior is")
    print("non-deterministic, so treat rates as directional, not precise.")


if __name__ == "__main__":
    asyncio.run(main())