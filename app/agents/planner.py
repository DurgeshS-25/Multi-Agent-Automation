import re
from dataclasses import dataclass

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


@dataclass
class PlanMeta:
    """Planner output plus measurement metadata (for evals)."""
    sub_questions: list[str]
    raw_count: int       # how many the model returned before validation
    valid_count: int     # how many survived validation
    in_tokens: int
    out_tokens: int


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

    def _generate(self, question: str):
        """Shared Gemini call returning the raw response (with retry)."""
        prompt = _PLANNER_PROMPT.format(question=question)
        return with_retry(
            lambda: self._client.models.generate_content(
                model=self._model,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": PlanResult,
                },
            )
        )

    def plan(self, question: str) -> list[str]:
        """Decompose a question into validated sub-questions."""
        if not question or not question.strip():
            raise PlannerError("Question cannot be empty.")

        try:
            response = self._generate(question)
            plan: PlanResult = response.parsed
        except Exception as e:
            raise PlannerError(f"Planning failed: {e}") from e

        if not plan or not plan.sub_questions:
            raise PlannerError("Planner returned no sub-questions.")

        return self._validate(plan.sub_questions)

    def plan_with_meta(self, question: str) -> PlanMeta:
        """Like plan(), but also returns measurement metadata for evals."""
        if not question or not question.strip():
            raise PlannerError("Question cannot be empty.")

        try:
            response = self._generate(question)
            plan: PlanResult = response.parsed
            usage = response.usage_metadata
        except Exception as e:
            raise PlannerError(f"Planning failed: {e}") from e

        if not plan or not plan.sub_questions:
            raise PlannerError("Planner returned no sub-questions.")

        raw = plan.sub_questions
        validated = self._validate(raw)

        return PlanMeta(
            sub_questions=validated,
            raw_count=len(raw),
            valid_count=len(validated),
            in_tokens=usage.prompt_token_count,
            out_tokens=usage.candidates_token_count,
        )


planner_agent = PlannerAgent()