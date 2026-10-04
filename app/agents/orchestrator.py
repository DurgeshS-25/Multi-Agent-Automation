import time
from dataclasses import dataclass, field

from app.agents.planner import planner_agent, PlannerError
from app.agents.researcher import researcher
from app.agents.extractor import extractor, SubQuestionFacts
from app.agents.report_writer import report_writer, FinalReport
from app.observability.logging_config import get_logger, kv
from app.run_log import stage

log = get_logger("orchestrator")


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

    Stage timings use the same names as the LangGraph nodes (planner,
    researcher, extractor, ...) so the two pipelines compare directly.
    """

    async def _research(self, question: str):
        """Shared front half: plan -> search -> extract."""
        if not question or not question.strip():
            raise OrchestratorError("Question cannot be empty.")

        with stage("planner"):
            try:
                sub_questions = planner_agent.plan(question)
            except PlannerError as e:
                raise OrchestratorError(f"Planning failed: {e}") from e
        log.info(kv(event="planned", sub_questions=len(sub_questions)))

        # Timed around the whole fan-out, so parallel searches count once.
        with stage("researcher"):
            sq_results = await researcher.search_all(sub_questions)
        total_results = sum(len(r.results) for r in sq_results)
        search_errors = sum(1 for r in sq_results if r.error)
        log.info(kv(event="searched", results=total_results, errors=search_errors))

        with stage("extractor"):
            findings = await extractor.extract_all(sq_results)
        total_facts = sum(len(f.facts) for f in findings)
        extract_errors = sum(1 for f in findings if f.error)
        log.info(kv(event="extracted", facts=total_facts, errors=extract_errors))

        return sub_questions, findings

    async def run(self, question: str) -> DeepResearchResult:
        log.info(kv(event="deep_start", question=question))
        t0 = time.perf_counter()
        sub_questions, findings = await self._research(question)
        log.info(kv(event="deep_done", seconds=round(time.perf_counter() - t0, 1)))
        return DeepResearchResult(
            question=question,
            sub_questions=sub_questions,
            findings=findings,
        )

    async def run_report(self, question: str) -> ReportResult:
        log.info(kv(event="report_start", question=question))
        t0 = time.perf_counter()

        sub_questions, findings = await self._research(question)

        # Synthesis + critique + any revisions; report_writer times its own
        # "synthesizer" and "critic" stages.
        final: FinalReport = await report_writer.write(question, findings)
        log.info(kv(
            event="report_done",
            approved=final.approved,
            iterations=final.iterations,
            remaining_issues=len(final.remaining_issues),
            seconds=round(time.perf_counter() - t0, 1),
        ))

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