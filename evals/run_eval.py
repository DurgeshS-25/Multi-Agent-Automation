"""Full-pipeline evaluation harness.

Runs the report pipeline over the eval dataset and scores each result on
measurable dimensions:
  - citation_coverage: fraction of report sentences carrying a [n] citation
  - groundedness: did the critic approve (every claim supported)?
  - iterations: how many synthesis passes (revision behavior)
  - latency, fact/source counts

Prints per-question rows and an aggregate summary, plus a per-category
breakdown. Paces requests to respect the free-tier per-minute quota.
"""

import asyncio
import re
import statistics

from app.agents.orchestrator import orchestrator
from evals.dataset import EVAL_QUESTIONS


# Pace between questions to stay under the free-tier per-minute quota.
PAUSE_BETWEEN = 45


def citation_coverage(report: str) -> float:
    """Fraction of sentences that contain at least one [n] citation marker.

    A rough but objective proxy for how well-grounded the prose is: a report
    where most sentences cite a source is more traceable than one where few do.
    """
    # Split into sentences on ., !, ? followed by space/newline
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", report) if s.strip()]
    if not sentences:
        return 0.0
    cited = sum(1 for s in sentences if re.search(r"\[\d+", s))
    return cited / len(sentences)


async def eval_one(question: str) -> dict:
    result = await orchestrator.run_report(question)
    return {
        "approved": result.approved,          # critic groundedness sign-off
        "iterations": result.iterations,
        "citation_coverage": citation_coverage(result.report),
        "num_sources": len(result.sources),
        "report_chars": len(result.report),
        "remaining_issues": len(result.remaining_issues),
    }


async def main() -> None:
    import time
    rows = []
    n = len(EVAL_QUESTIONS)
    print(f"Evaluating {n} questions (pacing {PAUSE_BETWEEN}s between)...\n")

    for i, item in enumerate(EVAL_QUESTIONS, 1):
        t0 = time.perf_counter()
        try:
            r = await eval_one(item.question)
            r["latency"] = time.perf_counter() - t0
            r["category"] = item.category
            r["question"] = item.question
            rows.append(r)
            print(
                f"[{i:2}/{n}] {item.category:<15} "
                f"cov={r['citation_coverage']:.0%} "
                f"{'approved' if r['approved'] else 'NOT-appr'} "
                f"iter={r['iterations']} "
                f"{r['latency']:.0f}s  {item.question[:38]}"
            )
        except Exception as e:
            print(f"[{i:2}/{n}] {item.category:<15} FAILED: {str(e)[:50]}")

        if i < n:
            await asyncio.sleep(PAUSE_BETWEEN)

    if not rows:
        print("\nNo successful runs.")
        return

    # --- Aggregate ---
    cov = [r["citation_coverage"] for r in rows]
    lat = [r["latency"] for r in rows]
    approved = sum(1 for r in rows if r["approved"])
    revised = sum(1 for r in rows if r["iterations"] > 1)

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Questions evaluated:     {len(rows)}/{n}")
    print(f"Mean citation coverage:  {statistics.mean(cov):.0%}")
    print(f"Groundedness (approved): {approved}/{len(rows)} ({approved/len(rows):.0%})")
    print(f"Triggered a revision:    {revised}/{len(rows)}")
    print(f"Median latency:          {statistics.median(lat):.0f}s")
    print(f"Avg sources/report:      {statistics.mean([r['num_sources'] for r in rows]):.1f}")

    # --- Per-category breakdown ---
    print("\nBy category:")
    cats = sorted(set(r["category"] for r in rows))
    for c in cats:
        crows = [r for r in rows if r["category"] == c]
        ccov = statistics.mean([r["citation_coverage"] for r in crows])
        cappr = sum(1 for r in crows if r["approved"])
        print(f"  {c:<15} n={len(crows)}  cov={ccov:.0%}  approved={cappr}/{len(crows)}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())