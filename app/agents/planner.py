import re

from google import genai
from pydantic import BaseModel, Field

from app.config import settings
from app.llm.retry import with_retry


MIN_SUB_QUESTIONS = 2
MAX_SUB_QUESTIONS = 6


class PlanResult(BaseModel):
    """Structured plan returned by the planner. Gemini is constrained to
    this schema, so output is always valid — no manual JSON parsing."""
    sub_questions: list[str] = Field(
        ...,
        description="3-5 focused sub-questions the main question decomposes into.",
    )


class PlannerError(Exception):
    pass


_PLANNER_PROMPT = """You are a research planner. Break the user's question into \
3 to 5 focused, non-overlapping sub-questions that together fully cover what \
must be researched to answer it well.

Rules:
- Each sub-question must be independently searchable on its own.
- Do not repeat the original question verbatim.
- Cover distinct angles (e.g. definitions, comparisons, tradeoffs, use cases), \
not slight rewordings of the same thing.

Question: {question}"""


class PlannerAgent:
    def __init__(self) -> None:
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._model = settings.gemini_model

    def _validate(self, sub_questions: list[str]) -> list[str]:
        """Clean and clamp planner output.

        Schema guarantees structure (a list of strings); this guarantees
        quality: no empties, no near-duplicates, sane count. Cleans rather
        than rejects, erroring only if nothing usable survives.
        """
        cleaned: list[str] = []
        seen: set[str] = set()

        for q in sub_questions:
            # Strip whitespace and any stray leading numbering like "1. " or "2) "
            q = q.strip()
            q = re.sub(r"^\s*\d+[\.\)]\s*", "", q)

            if not q:
                continue  # drop empties

            # Dedupe on a normalized key (lowercased, punctuation-insensitive)
            key = re.sub(r"[^\w\s]", "", q.lower()).strip()
            if key in seen:
                continue  # drop near-duplicates
            seen.add(key)

            cleaned.append(q)

        if len(cleaned) < MIN_SUB_QUESTIONS:
            raise PlannerError(
                f"Planner produced only {len(cleaned)} usable sub-question(s); "
                f"need at least {MIN_SUB_QUESTIONS}."
            )

        # Clamp the upper bound to control downstream search cost
        return cleaned[:MAX_SUB_QUESTIONS]

    def plan(self, question: str) -> list[str]:
        """Decompose a question into validated sub-questions."""
        if not question or not question.strip():
            raise PlannerError("Question cannot be empty.")

        prompt = _PLANNER_PROMPT.format(question=question)

        try:
            response = with_retry(
                lambda: self._client.models.generate_content(
                    model=self._model,
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": PlanResult,
                    },
                )
            )
            # The SDK parses the schema-constrained JSON for us
            plan: PlanResult = response.parsed
        except Exception as e:
            raise PlannerError(f"Planning failed: {e}") from e

        if not plan or not plan.sub_questions:
            raise PlannerError("Planner returned no sub-questions.")

        return self._validate(plan.sub_questions)


planner_agent = PlannerAgent()