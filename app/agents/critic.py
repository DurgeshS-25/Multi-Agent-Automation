from dataclasses import dataclass

from google import genai
from pydantic import BaseModel, Field

from app.config import settings
from app.llm.retry import with_retry_async
from app.agents.extractor import SubQuestionFacts


class CritiqueResult(BaseModel):
    """Schema-constrained critic verdict."""
    approved: bool = Field(
        ...,
        description="True if every claim in the report is supported by the "
                    "provided facts; False if any unsupported claims exist.",
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Specific claims from the report not backed by the facts. "
                    "Empty if approved.",
    )


@dataclass
class Critique:
    approved: bool
    unsupported_claims: list[str]
    in_tokens: int = 0
    out_tokens: int = 0


class CriticError(Exception):
    pass


_CRITIC_PROMPT = """You are a fact-checking critic. Your ONLY job is to verify \
that every claim in the REPORT is supported by the provided FACTS.

A claim is "unsupported" if it states something the facts do not establish — \
a number, comparison, or assertion that is not backed by any provided fact. \
Do NOT flag a claim just because it is reworded; flag it only if the facts do \
not support its substance. General connective or framing sentences that make \
no factual assertion are fine.

Be strict but fair:
- If every factual claim is supported, set approved = true and return an empty list.
- If any claim is unsupported, set approved = false and list the specific \
offending claims (quote or closely paraphrase each).

FACTS:
{facts_block}

REPORT:
{report}"""


class Critic:
    def __init__(self) -> None:
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._model = settings.gemini_model

    def _format_facts(self, findings: list[SubQuestionFacts]) -> str:
        blocks = []
        for finding in findings:
            if finding.error or not finding.facts:
                continue
            lines = [f"Sub-question: {finding.sub_question}"]
            for fact in finding.facts:
                lines.append(f"  - {fact}")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    async def critique(
        self, report: str, findings: list[SubQuestionFacts]
    ) -> Critique:
        facts_block = self._format_facts(findings)
        if not facts_block:
            raise CriticError("No facts available to critique against.")

        prompt = _CRITIC_PROMPT.format(facts_block=facts_block, report=report)

        try:
            response = await with_retry_async(
                lambda: self._client.aio.models.generate_content(
                    model=self._model,
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": CritiqueResult,
                    },
                )
            )
            parsed: CritiqueResult = response.parsed
            usage = response.usage_metadata
        except Exception as e:
            raise CriticError(f"Critique failed: {e}") from e

        return Critique(
            approved=parsed.approved,
            unsupported_claims=parsed.unsupported_claims,
            in_tokens=usage.prompt_token_count,
            out_tokens=usage.candidates_token_count,
        )


critic = Critic()