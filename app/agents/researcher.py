import asyncio
from dataclasses import dataclass

from app.search.tavily_client import tavily_search, SearchResult, SearchError


@dataclass
class SubQuestionResult:
    """One sub-question paired with its search results."""
    sub_question: str
    results: list[SearchResult]
    error: str | None = None  # set if this sub-question's search failed


class Researcher:
    """Runs searches for many sub-questions concurrently."""

    async def search_all(self, sub_questions: list[str]) -> list[SubQuestionResult]:
        """Fire one search per sub-question, all in parallel.

        Uses asyncio.gather to overlap the N network calls instead of
        running them sequentially. A failure in one sub-question does not
        sink the others — it's captured per-item.
        """
        async def _one(sq: str) -> SubQuestionResult:
            try:
                results = await tavily_search.search_async(sq)
                return SubQuestionResult(sub_question=sq, results=results)
            except SearchError as e:
                # Isolate the failure to this sub-question
                return SubQuestionResult(sub_question=sq, results=[], error=str(e))

        # Launch all searches concurrently, wait for all to finish
        return await asyncio.gather(*(_one(sq) for sq in sub_questions))


researcher = Researcher()