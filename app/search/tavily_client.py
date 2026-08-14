import asyncio
from dataclasses import dataclass

from tavily import TavilyClient

from app.config import settings


@dataclass
class SearchResult:
    title: str
    url: str
    content: str


class SearchError(Exception):
    pass


def _parse_results(response: dict) -> list["SearchResult"]:
    """Normalize a raw Tavily response into typed SearchResult objects.

    NOTE: Tavily returns the top result with a direct URL, but gates
    lower-ranked results behind authenticated /goto redirect stubs whose
    tokens are opaque (encrypted protobuf, not client-resolvable). The
    title + content fields are always clean, so grounding/summarization
    is unaffected; only the raw citation link may be a redirect stub.
    """
    results = []
    for item in response.get("results", []):
        results.append(
            SearchResult(
                title=item.get("title", "").strip(),
                url=item.get("url", "").strip(),
                content=item.get("content", "").strip(),
            )
        )
    return results


class TavilySearch:
    def __init__(self) -> None:
        self._client = TavilyClient(api_key=settings.tavily_api_key)

    def search(self, query: str) -> list[SearchResult]:
        """Synchronous search."""
        if not query or not query.strip():
            raise SearchError("Query cannot be empty.")

        try:
            response = self._client.search(
                query=query,
                search_depth=settings.search_depth,
                max_results=settings.max_search_results,
            )
        except Exception as e:
            raise SearchError(f"Tavily search failed: {e}") from e

        return _parse_results(response)

    async def search_async(self, query: str) -> list[SearchResult]:
        """Async variant of search(). Runs the blocking Tavily SDK call in a
        thread so it's awaitable and won't block the event loop. Enables
        concurrent searches (see Phase 3 parallel fan-out)."""
        if not query or not query.strip():
            raise SearchError("Query cannot be empty.")

        try:
            # Tavily's SDK is synchronous; run it in a thread to avoid
            # blocking the event loop while the network call is in flight.
            response = await asyncio.to_thread(
                self._client.search,
                query=query,
                search_depth=settings.search_depth,
                max_results=settings.max_search_results,
            )
        except Exception as e:
            raise SearchError(f"Tavily search failed: {e}") from e

        return _parse_results(response)


tavily_search = TavilySearch()