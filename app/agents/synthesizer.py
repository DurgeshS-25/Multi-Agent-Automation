from dataclasses import dataclass

from google import genai

from app.config import settings
from app.llm.retry import with_retry_async
from app.agents.extractor import SubQuestionFacts


@dataclass
class SynthesizedReport:
    """The synthesizer's output: a coherent cited report."""
    report: str
    sources: list[str]
    in_tokens: int = 0
    out_tokens: int = 0


class SynthesizerError(Exception):
    pass


_SYNTH_PROMPT = """You are a research synthesizer. Using ONLY the facts provided \
below, write a clear, coherent report that answers the main question. The facts \
are grouped by the sub-question they came from.

Rules:
- Use ONLY the provided facts. Do not add outside knowledge.
- Weave the facts into flowing prose — do NOT just list them or copy them \
verbatim. Organize by theme, not by sub-question.
- Where facts conflict or show tradeoffs, present both sides fairly.
- Cite facts inline using their [n] source markers where relevant.
- Aim for a well-structured answer of a few short paragraphs.

Main question: {question}

Facts (grouped by sub-question):
{facts_block}{revision_block}"""


_REVISION_TEMPLATE = """

IMPORTANT — This is a REVISION. A previous draft contained claims NOT supported \
by the facts above. Rewrite the report so every claim is fully supported. \
Remove or correct these specific unsupported claims:
{claims}

Previous draft (for reference — fix its problems, keep what was well-supported):
{previous}"""


class Synthesizer:
    def __init__(self) -> None:
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._model = settings.gemini_model

    def _format_facts(self, findings: list[SubQuestionFacts]) -> tuple[str, list[str]]:
        """Build the facts block for the prompt and collect a flat, numbered
        source list. Returns (facts_block, sources)."""
        blocks = []
        sources: list[str] = []
        source_index: dict[str, int] = {}

        for finding in findings:
            if finding.error or not finding.facts:
                continue
            local_markers = []
            for url in finding.sources:
                if url not in source_index:
                    source_index[url] = len(sources) + 1
                    sources.append(url)
                local_markers.append(f"[{source_index[url]}]")
            marker_str = "".join(local_markers)

            lines = [f"Sub-question: {finding.sub_question} {marker_str}"]
            for fact in finding.facts:
                lines.append(f"  - {fact}")
            blocks.append("\n".join(lines))

        return "\n\n".join(blocks), sources

    async def synthesize(
        self,
        question: str,
        findings: list[SubQuestionFacts],
        previous_report: str | None = None,
        unsupported_claims: list[str] | None = None,
    ) -> SynthesizedReport:
        """Synthesize a report. If previous_report + unsupported_claims are
        given, this is a revision pass that fixes the flagged claims."""
        facts_block, sources = self._format_facts(findings)

        if not facts_block:
            raise SynthesizerError("No facts available to synthesize.")

        # Build the optional revision block
        revision_block = ""
        if previous_report and unsupported_claims:
            claims_str = "\n".join(f"  - {c}" for c in unsupported_claims)
            revision_block = _REVISION_TEMPLATE.format(
                claims=claims_str, previous=previous_report
            )

        prompt = _SYNTH_PROMPT.format(
            question=question,
            facts_block=facts_block,
            revision_block=revision_block,
        )

        try:
            response = await with_retry_async(
                lambda: self._client.aio.models.generate_content(
                    model=self._model,
                    contents=prompt,
                )
            )
            report_text = response.text.strip()
            usage = response.usage_metadata
        except Exception as e:
            raise SynthesizerError(f"Synthesis failed: {e}") from e

        return SynthesizedReport(
            report=report_text,
            sources=sources,
            in_tokens=usage.prompt_token_count,
            out_tokens=usage.candidates_token_count,
        )


synthesizer = Synthesizer()