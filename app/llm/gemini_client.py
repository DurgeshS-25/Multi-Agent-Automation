from dataclasses import dataclass
from google import genai

from app.config import settings
from app.search.tavily_client import SearchResult


@dataclass
class ResearchAnswer:
    """Structured output of the summarizer."""
    answer: str
    sources: list[str]


class SummarizerError(Exception):
    """Raised when the LLM call fails."""
    pass


_PROMPT_TEMPLATE = """You are a research assistant. Answer the user's question \
using ONLY the numbered search results provided below. Do not use any prior \
knowledge. If the search results do not contain enough information to answer, \
say so honestly.

When you use information from a source, cite it inline using its number in \
square brackets, like [1] or [2].

Question:
{question}

Search results:
{context}

Write a clear, concise answer (3-6 sentences) grounded in the sources above."""


class GeminiSummarizer:
    def __init__(self) -> None:
        # New SDK: create a client instead of configuring a global + model object
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._model = settings.gemini_model

    def _format_context(self, results: list[SearchResult]) -> str:
        blocks = []
        for i, r in enumerate(results, start=1):
            blocks.append(f"[{i}] {r.title}\nURL: {r.url}\n{r.content}")
        return "\n\n".join(blocks)

    def summarize(self, question: str, results: list[SearchResult]) -> ResearchAnswer:
        if not results:
            raise SummarizerError("No search results to summarize.")

        prompt = _PROMPT_TEMPLATE.format(
            question=question,
            context=self._format_context(results),
        )

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
            )
            answer_text = response.text.strip()
        except Exception as e:
            raise SummarizerError(f"Gemini summarization failed: {e}") from e

        return ResearchAnswer(
            answer=answer_text,
            sources=[r.url for r in results],
        )


gemini_summarizer = GeminiSummarizer()