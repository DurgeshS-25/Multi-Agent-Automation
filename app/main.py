from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.search.tavily_client import tavily_search, SearchError
from app.llm.gemini_client import gemini_summarizer, SummarizerError
from app.agents.planner import planner_agent, PlannerError
from app.agents.orchestrator import orchestrator, OrchestratorError

app = FastAPI(
    title="Multi-Agent Automation",
    description="Research assistant API — plan, search the web, and return grounded, cited answers.",
    version="0.3.0",
)


# ---- Request / Response schemas ----

class PlanRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        description="The broad question to decompose into sub-questions.",
        examples=["Compare REST and GraphQL for a high-traffic API"],
    )


class PlanResponse(BaseModel):
    question: str
    sub_questions: list[str]
    num_sub_questions: int


class ResearchRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        description="The research question to answer.",
        examples=["What is retrieval augmented generation?"],
    )
    include_plan: bool = Field(
        default=False,
        description="If true, also decompose the question into sub-questions "
                    "(planning only; full per-sub-question search is in /research/deep).",
    )


class ResearchResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    sub_questions: list[str] = []


class DeepResearchRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        description="A broad question to research across multiple sub-questions.",
        examples=["Compare REST and GraphQL for a high-traffic API"],
    )


class SubQuestionFinding(BaseModel):
    sub_question: str
    facts: list[str]
    sources: list[str]
    error: str | None = None


class DeepResearchResponse(BaseModel):
    question: str
    sub_questions: list[str]
    findings: list[SubQuestionFinding]


# ---- Routes ----

@app.get("/health")
def health() -> dict:
    """Simple liveness check."""
    return {"status": "ok"}


@app.post("/plan", response_model=PlanResponse)
def plan(request: PlanRequest) -> PlanResponse:
    """Decompose a question into focused sub-questions."""
    try:
        sub_questions = planner_agent.plan(request.question)
    except PlannerError as e:
        raise HTTPException(status_code=502, detail=f"Planning failed: {e}")

    return PlanResponse(
        question=request.question,
        sub_questions=sub_questions,
        num_sub_questions=len(sub_questions),
    )


@app.post("/research", response_model=ResearchResponse)
def research(request: ResearchRequest) -> ResearchResponse:
    """Single-search pipeline: (optional plan) -> search -> summarize."""
    sub_questions: list[str] = []
    if request.include_plan:
        try:
            sub_questions = planner_agent.plan(request.question)
        except PlannerError as e:
            raise HTTPException(status_code=502, detail=f"Planning failed: {e}")

    try:
        results = tavily_search.search(request.question)
    except SearchError as e:
        raise HTTPException(status_code=502, detail=f"Search failed: {e}")

    try:
        answer = gemini_summarizer.summarize(request.question, results)
    except SummarizerError as e:
        raise HTTPException(status_code=502, detail=f"Summarization failed: {e}")

    return ResearchResponse(
        question=request.question,
        answer=answer.answer,
        sources=answer.sources,
        sub_questions=sub_questions,
    )


@app.post("/research/deep", response_model=DeepResearchResponse)
async def research_deep(request: DeepResearchRequest) -> DeepResearchResponse:
    """Deep multi-agent pipeline: plan -> parallel search -> parallel extract.

    Returns structured facts grouped by sub-question. NOTE: synthesis of
    these findings into a single cited prose report is Phase 4 — this
    endpoint returns the organized research, not yet a final essay.
    """
    try:
        result = await orchestrator.run(request.question)
    except OrchestratorError as e:
        raise HTTPException(status_code=502, detail=f"Deep research failed: {e}")

    return DeepResearchResponse(
        question=result.question,
        sub_questions=result.sub_questions,
        findings=[
            SubQuestionFinding(
                sub_question=f.sub_question,
                facts=f.facts,
                sources=f.sources,
                error=f.error,
            )
            for f in result.findings
        ],
    )