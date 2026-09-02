
from dataclasses import dataclass, field

from app.agents.planner import planner_agent, PlannerError
from app.agents.researcher import researcher
from app.agents.extractor import extractor, SubQuestionFacts


@dataclass
class DeepResearchResult:
    """Full output of the deep-research pipeline."""
    question: str
    sub_questions: list[str]
    findings: list[SubQuestionFacts] = field(default_factory=list)


class OrchestratorError(Exception):
    pass


class Orchestrator:
    """Runs the full deep-research pipeline as a single call:
    plan -> parallel search -> parallel extract.
    """

    async def run(self, question: str) -> DeepResearchResult:
        if not question or not question.strip():
            raise OrchestratorError("Question cannot be empty.")

        # 1. Plan — decompose into sub-questions (sync; it's one quick call)
        try:
            sub_questions = planner_agent.plan(question)
        except PlannerError as e:
            raise OrchestratorError(f"Planning failed: {e}") from e

        # 2. Search — one search per sub-question, bounded-concurrent
        sq_results = await researcher.search_all(sub_questions)

        # 3. Extract — distill each sub-question's results, bounded-concurrent
        findings = await extractor.extract_all(sq_results)

        return DeepResearchResult(
            question=question,
            sub_questions=sub_questions,
            findings=findings,
        )


orchestrator = Orchestrator()