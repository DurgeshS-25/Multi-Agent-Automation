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
        """Fire searches with bounded concurrency (rate-limit safety)."""
        from app.llm.retry import gather_bounded
        from app.config import settings

        async def _one(sq: str) -> SubQuestionResult:
            try:
                results = await tavily_search.search_async(sq)
                return SubQuestionResult(sub_question=sq, results=results)
            except SearchError as e:
                return SubQuestionResult(sub_question=sq, results=[], error=str(e))

        return await gather_bounded(
            [lambda q=sq: _one(q) for sq in sub_questions],
            limit=settings.max_concurrency,
        )


researcher = Researcher()