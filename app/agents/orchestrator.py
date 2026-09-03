from dataclasses import dataclass, field

from app.agents.planner import planner_agent, PlannerError
from app.agents.researcher import researcher
from app.agents.extractor import extractor, SubQuestionFacts
from app.agents.report_writer import report_writer, FinalReport


@dataclass
class DeepResearchResult:
    """Output of the deep-research pipeline (facts grouped by sub-question)."""
    question: str
    sub_questions: list[str]
    findings: list[SubQuestionFacts] = field(default_factory=list)


@dataclass
class ReportResult:
    """Output of the full report pipeline (synthesized + self-critiqued)."""
    question: str
    sub_questions: list[str]
    report: str
    sources: list[str]
    approved: bool
    iterations: int
    remaining_issues: list[str] = field(default_factory=list)


class OrchestratorError(Exception):
    pass


class Orchestrator:
    """Runs the research pipeline. Two entry points:
    - run(): plan -> parallel search -> parallel extract (facts by sub-question)
    - run_report(): the above + synthesize -> critique -> revise (final report)
    """

    async def _research(self, question: str):
        """Shared front half: plan -> search -> extract."""
        if not question or not question.strip():
            raise OrchestratorError("Question cannot be empty.")

        try:
            sub_questions = planner_agent.plan(question)
        except PlannerError as e:
            raise OrchestratorError(f"Planning failed: {e}") from e

        sq_results = await researcher.search_all(sub_questions)
        findings = await extractor.extract_all(sq_results)
        return sub_questions, findings

    async def run(self, question: str) -> DeepResearchResult:
        sub_questions, findings = await self._research(question)
        return DeepResearchResult(
            question=question,
            sub_questions=sub_questions,
            findings=findings,
        )

    async def run_report(self, question: str) -> ReportResult:
        sub_questions, findings = await self._research(question)

        final: FinalReport = await report_writer.write(question, findings)

        return ReportResult(
            question=question,
            sub_questions=sub_questions,
            report=final.report,
            sources=final.sources,
            approved=final.approved,
            iterations=final.iterations,
            remaining_issues=final.remaining_issues,
        )


orchestrator = Orchestrator()