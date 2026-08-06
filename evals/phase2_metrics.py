import time
import statistics

from app.agents.planner import planner_agent


# Broad questions worth decomposing (planner is meant for multi-angle topics).
TEST_QUESTIONS = [
    "Compare REST and GraphQL for a high-traffic API",
    "What are the tradeoffs of microservices vs a monolith?",
    "How should a startup choose between AWS, GCP, and Azure?",
    "What makes a vector database different from a relational database?",
    "How do you design a scalable rate limiter?",
]

# Gemini 3.6 Flash pricing (USD per 1M tokens), verified Aug 2026.
# gemini-flash-latest currently resolves to gemini-3.6-flash.
PRICE_INPUT_PER_1M = 1.50
PRICE_OUTPUT_PER_1M = 7.50


def run_query(question: str) -> dict:
    t0 = time.perf_counter()
    meta = planner_agent.plan_with_meta(question)
    latency = time.perf_counter() - t0

    cost = (
        meta.in_tokens / 1_000_000 * PRICE_INPUT_PER_1M
        + meta.out_tokens / 1_000_000 * PRICE_OUTPUT_PER_1M
    )

    return {
        "question": question,
        "latency": latency,
        "raw_count": meta.raw_count,
        "valid_count": meta.valid_count,
        "dropped": meta.raw_count - meta.valid_count,
        "in_tokens": meta.in_tokens,
        "out_tokens": meta.out_tokens,
        "cost": cost,
    }


def main() -> None:
    rows = []
    print(f"Running {len(TEST_QUESTIONS)} planning queries...\n")

    for q in TEST_QUESTIONS:
        try:
            r = run_query(q)
            rows.append(r)
            print(
                f"✓ {q[:45]:<45} "
                f"{r['latency']:.2f}s  "
                f"{r['valid_count']} sub-qs "
                f"(raw {r['raw_count']}, dropped {r['dropped']})  "
                f"${r['cost']:.6f}"
            )
        except Exception as e:
            print(f"✗ {q[:45]:<45} FAILED: {e}")

    if not rows:
        print("\nNo successful runs.")
        return

    latencies = [r["latency"] for r in rows]
    valid_counts = [r["valid_count"] for r in rows]
    costs = [r["cost"] for r in rows]
    total_dropped = sum(r["dropped"] for r in rows)
    total_raw = sum(r["raw_count"] for r in rows)

    print("\n" + "=" * 55)
    print("PHASE 2 PLANNER METRICS SUMMARY")
    print("=" * 55)
    print(f"Queries run:            {len(rows)}")
    print(f"Avg planner latency:    {statistics.mean(latencies):.2f}s")
    print(f"Median latency:         {statistics.median(latencies):.2f}s")
    print(f"Avg sub-questions:      {statistics.mean(valid_counts):.1f}")
    print(f"Sub-question range:     {min(valid_counts)}-{max(valid_counts)}")
    print(f"Validation dropped:     {total_dropped}/{total_raw} raw sub-questions")
    print(f"Avg cost/plan:          ${statistics.mean(costs):.6f}")
    print(f"Avg tokens/plan:        "
          f"{statistics.mean([r['in_tokens'] for r in rows]):.0f} in / "
          f"{statistics.mean([r['out_tokens'] for r in rows]):.0f} out")
    print("=" * 55)


if __name__ == "__main__":
    main()