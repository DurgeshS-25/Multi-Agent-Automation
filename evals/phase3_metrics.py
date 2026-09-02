import asyncio
import time
import statistics

from app.agents.planner import planner_agent
from app.agents.researcher import researcher
from app.agents.extractor import extractor
from app.search.tavily_client import tavily_search


# Broad, multi-angle questions — the kind deep research is meant for.
TEST_QUESTIONS = [
    "Compare REST and GraphQL for a high-traffic API",
    "What are the tradeoffs of microservices vs a monolith?",
    "How should a team choose between SQL and NoSQL databases?",
]

# Gemini Flash-Lite pricing (USD per 1M tokens). Update if the model changes.
PRICE_INPUT_PER_1M = 0.10
PRICE_OUTPUT_PER_1M = 0.40


async def measure_search_speedup(sub_questions: list[str]) -> float:
    """Cache-fair speedup: warm the cache, then time parallel BEFORE
    sequential so neither benefits from the other's warm-up."""
    # Warm cache
    await researcher.search_all(sub_questions)

    # Parallel first
    t = time.perf_counter()
    await researcher.search_all(sub_questions)
    par = time.perf_counter() - t

    # Sequential second (same warm state)
    t = time.perf_counter()
    for sq in sub_questions:
        tavily_search.search(sq)
    seq = time.perf_counter() - t

    return seq / par if par > 0 else 0.0


async def run_question(question: str) -> dict:
    # Full pipeline latency (plan -> search -> extract)
    t0 = time.perf_counter()
    subs = planner_agent.plan(question)
    sq_results = await researcher.search_all(subs)
    findings = await extractor.extract_all(sq_results)
    total_latency = time.perf_counter() - t0

    # Search speedup (measured separately, cache-fair)
    speedup = await measure_search_speedup(subs)

    # Fact yield
    fact_counts = [len(f.facts) for f in findings if not f.error]
    total_facts = sum(fact_counts)
    failed = sum(1 for f in findings if f.error)

    return {
        "question": question,
        "num_sub_questions": len(subs),
        "total_latency": total_latency,
        "search_speedup": speedup,
        "total_facts": total_facts,
        "avg_facts_per_subq": statistics.mean(fact_counts) if fact_counts else 0,
        "failed_subq": failed,
    }


async def main() -> None:
    rows = []
    print(f"Running {len(TEST_QUESTIONS)} deep-research queries...\n")

    for q in TEST_QUESTIONS:
        try:
            r = await run_question(q)
            rows.append(r)
            print(
                f"✓ {q[:42]:<42} "
                f"{r['total_latency']:.1f}s  "
                f"{r['num_sub_questions']} sub-qs  "
                f"{r['total_facts']} facts  "
                f"{r['search_speedup']:.1f}x search"
                + (f"  ({r['failed_subq']} failed)" if r['failed_subq'] else "")
            )
        except Exception as e:
            print(f"✗ {q[:42]:<42} FAILED: {e}")

    if not rows:
        print("\nNo successful runs.")
        return

    latencies = [r["total_latency"] for r in rows]
    speedups = [r["search_speedup"] for r in rows]
    facts = [r["total_facts"] for r in rows]
    subqs = [r["num_sub_questions"] for r in rows]

    print("\n" + "=" * 58)
    print("PHASE 3 DEEP-RESEARCH METRICS SUMMARY")
    print("=" * 58)
    print(f"Queries run:              {len(rows)}")
    print(f"Median pipeline latency:  {statistics.median(latencies):.1f}s")
    print(f"Avg sub-questions:        {statistics.mean(subqs):.1f}")
    print(f"Avg facts/question:       {statistics.mean(facts):.1f}")
    print(f"Median search speedup:    {statistics.median(speedups):.1f}x "
          f"(parallel vs sequential)")
    print(f"Search speedup range:     {min(speedups):.1f}x - {max(speedups):.1f}x")
    print("=" * 58)
    print("Note: search speedup is bounded by sub-question count (~N-way).")


if __name__ == "__main__":
    asyncio.run(main())