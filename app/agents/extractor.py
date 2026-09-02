import asyncio
from dataclasses import dataclass

from google import genai
from pydantic import BaseModel, Field

from app.config import settings
from app.llm.retry import with_retry_async
from app.agents.researcher import SubQuestionResult


class ExtractedFacts(BaseModel):
    """Schema-constrained extractor output."""
    facts: list[str] = Field(
        ...,
        description="Concise, standalone facts relevant to the sub-question, "
                    "each drawn from the provided sources.",
    )


@dataclass
class SubQuestionFacts:
    """A sub-question paired with the facts extracted for it."""
    sub_question: str
    facts: list[str]
    sources: list[str]
    error: str | None = None


class ExtractorError(Exception):
    pass


_EXTRACT_PROMPT = """You are a research extractor. From the search results below, \
extract only the concise, factual statements that help answer the sub-question. \

Rules:
- Each fact must be a standalone statement (understandable on its own).
- Use ONLY information present in the search results — no prior knowledge.
- Omit marketing fluff, opinions, and anything not relevant to the sub-question.
- Prefer specific facts (numbers, mechanisms, tradeoffs) over vague claims.

Sub-question: {sub_question}

Search results:
{context}"""


class Extractor:
    def __init__(self) -> None:
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._model = settings.gemini_model

    def _format_context(self, sqr: SubQuestionResult) -> str:
        blocks = []
        for i, r in enumerate(sqr.results, start=1):
            blocks.append(f"[{i}] {r.title}\n{r.content}")
        return "\n\n".join(blocks)

    async def _extract_one(self, sqr: SubQuestionResult) -> SubQuestionFacts:
        # If the upstream search failed or returned nothing, skip cleanly
        if sqr.error or not sqr.results:
            return SubQuestionFacts(
                sub_question=sqr.sub_question,
                facts=[],
                sources=[],
                error=sqr.error or "No search results to extract from.",
            )

        prompt = _EXTRACT_PROMPT.format(
            sub_question=sqr.sub_question,
            context=self._format_context(sqr),
        )

        try:
            response = await with_retry_async(
                lambda: self._client.aio.models.generate_content(
                    model=self._model,
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": ExtractedFacts,
                    },
                )
            )
            parsed: ExtractedFacts = response.parsed
        except Exception as e:
            return SubQuestionFacts(
                sub_question=sqr.sub_question,
                facts=[],
                sources=[r.url for r in sqr.results],
                error=str(e),
            )

        return SubQuestionFacts(
            sub_question=sqr.sub_question,
            facts=parsed.facts if parsed else [],
            sources=[r.url for r in sqr.results],
        )

    async def extract_all(
        self, sq_results: list[SubQuestionResult]
    ) -> list[SubQuestionFacts]:
        """Extract facts for every sub-question, with bounded concurrency."""
        from app.llm.retry import gather_bounded
        from app.config import settings

        return await gather_bounded(
            [lambda s=s: self._extract_one(s) for s in sq_results],
            limit=settings.max_concurrency,
        )


extractor = Extractor()