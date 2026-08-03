import time
import re
import statistics

from app.search.tavily_client import tavily_search
from app.llm.gemini_client import gemini_summarizer


# A small, varied benchmark set. Fixed list = repeatable measurements.
TEST_QUESTIONS = [
    "What is retrieval augmented generation?",
    "How does a vector database work?",
    "What are the main differences between REST and GraphQL?",
    "What is the CAP theorem in distributed systems?",
    "How does OAuth2 authorization code flow work?",
]

# Gemini Flash pricing (USD per 1M tokens). Update if rates change.
# These are placeholders — we'll verify actual rates before writing bullets.
PRICE_INPUT_PER_1M = 1.50
PRICE_OUTPUT_PER_1M = 7.50


def count_citations(answer: str) -> int:
    """Count inline [n] or [n, m] style citation markers in the answer."""
    # Matches [1], [2, 3], [1,2,3] etc.
    matches = re.findall(r"\[\d+(?:\s*,\s*\d+)*\]", answer)
    return len(matches)


def run_query(question: str) -> dict:
    """Run one query, timing each stage. Tokens come from the answer itself."""
    # --- Search stage ---
    t0 = time.perf_counter()
    results = tavily_search.search(question)
    t_search = time.perf_counter() - t0

    # --- LLM stage ---
    t1 = time.perf_counter()
    answer = gemini_summarizer.summarize(question, results)
    t_llm = time.perf_counter() - t1

    # Token usage travels with the answer (captured in the summarizer)
    in_tokens = answer.in_tokens
    out_tokens = answer.out_tokens
    cost = (
        in_tokens / 1_000_000 * PRICE_INPUT_PER_1M
        + out_tokens / 1_000_000 * PRICE_OUTPUT_PER_1M
    )

    return {
        "question": question,
        "search_time": t_search,
        "llm_time": t_llm,
        "total_time": t_search + t_llm,
        "num_sources": len(results),
        "num_citations": count_citations(answer.answer),
        "in_tokens": in_tokens,
        "out_tokens": out_tokens,
        "cost": cost,
    }


def main() -> None:
    rows = []
    print(f"Running {len(TEST_QUESTIONS)} queries...\n")

    for q in TEST_QUESTIONS:
        try:
            r = run_query(q)
            rows.append(r)
            print(
                f"✓ {q[:45]:<45} "
                f"{r['total_time']:.2f}s  "
                f"{r['num_citations']} cites  "
                f"${r['cost']:.6f}"
            )
        except Exception as e:
            print(f"✗ {q[:45]:<45} FAILED: {e}")

    if not rows:
        print("\nNo successful runs.")
        return

    # --- Aggregate metrics ---
    total_times = [r["total_time"] for r in rows]
    search_times = [r["search_time"] for r in rows]
    llm_times = [r["llm_time"] for r in rows]
    costs = [r["cost"] for r in rows]
    cited = sum(1 for r in rows if r["num_citations"] > 0)

    print("\n" + "=" * 55)
    print("PHASE 1 METRICS SUMMARY")
    print("=" * 55)
    print(f"Queries run:            {len(rows)}")
    print(f"Avg total latency:      {statistics.mean(total_times):.2f}s")
    print(f"  - avg search:         {statistics.mean(search_times):.2f}s")
    print(f"  - avg LLM:            {statistics.mean(llm_times):.2f}s")
    print(f"Median latency:         {statistics.median(total_times):.2f}s")
    print(f"Avg cost/query:         ${statistics.mean(costs):.6f}")
    print(f"Citation coverage:      {cited}/{len(rows)} "
          f"({cited/len(rows)*100:.0f}% of answers cited sources)")
    print(f"Avg tokens/query:       "
          f"{statistics.mean([r['in_tokens'] for r in rows]):.0f} in / "
          f"{statistics.mean([r['out_tokens'] for r in rows]):.0f} out")
    print("=" * 55)


if __name__ == "__main__":
    main()