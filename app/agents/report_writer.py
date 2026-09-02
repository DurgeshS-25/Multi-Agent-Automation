from dataclasses import dataclass

from app.agents.extractor import SubQuestionFacts
from app.agents.synthesizer import synthesizer, SynthesizedReport
from app.agents.critic import critic


@dataclass
class FinalReport:
    """The end product of the synthesize -> critique -> revise loop."""
    report: str
    sources: list[str]
    approved: bool              # did the critic approve the final version?
    iterations: int             # how many synthesis passes it took
    remaining_issues: list[str] # unsupported claims still flagged at the end (if any)


class ReportWriter:
    """Runs the self-correction loop: synthesize, critique, and revise until
    the critic approves or a revision cap is hit."""

    def __init__(self, max_revisions: int = 2) -> None:
        self._max_revisions = max_revisions

    async def write(
        self, question: str, findings: list[SubQuestionFacts]
    ) -> FinalReport:
        # Initial synthesis
        draft: SynthesizedReport = await synthesizer.synthesize(question, findings)
        iterations = 1

        verdict = await critic.critique(draft.report, findings)

        # Revision loop: while not approved and under the cap, re-synthesize
        revisions = 0
        while not verdict.approved and revisions < self._max_revisions:
            revisions += 1
            iterations += 1
            draft = await synthesizer.synthesize(
                question,
                findings,
                previous_report=draft.report,
                unsupported_claims=verdict.unsupported_claims,
            )
            verdict = await critic.critique(draft.report, findings)

        return FinalReport(
            report=draft.report,
            sources=draft.sources,
            approved=verdict.approved,
            iterations=iterations,
            remaining_issues=verdict.unsupported_claims,
        )


report_writer = ReportWriter()