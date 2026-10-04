from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from app.search.tavily_client import tavily_search, SearchError
from app.llm.gemini_client import gemini_summarizer, SummarizerError
from app.agents.planner import planner_agent, PlannerError
from app.agents.orchestrator import orchestrator, OrchestratorError
from app.graph.research_graph import research_graph

app = FastAPI(
    title="Multi-Agent Automation",
    description="Research assistant API — plan, search, synthesize, and self-critique cited reports.",
    version="0.5.0",
)

@app.get("/",include_in_schema=False)
def root():
    return RedirectResponse(url="/ui/")


# ---- Request / Response schemas ----

class PlanRequest(BaseModel):
    question: str = Field(..., min_length=3,
        description="The broad question to decompose into sub-questions.",
        examples=["Compare REST and GraphQL for a high-traffic API"])


class PlanResponse(BaseModel):
    question: str
    sub_questions: list[str]
    num_sub_questions: int


class ResearchRequest(BaseModel):
    question: str = Field(..., min_length=3,
        description="The research question to answer.",
        examples=["What is retrieval augmented generation?"])
    include_plan: bool = Field(default=False,
        description="If true, also decompose into sub-questions (planning only; "
                    "full per-sub-question search is in /research/deep).")


class ResearchResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    sub_questions: list[str] = []


class DeepResearchRequest(BaseModel):
    question: str = Field(..., min_length=3,
        description="A broad question to research across multiple sub-questions.",
        examples=["Compare REST and GraphQL for a high-traffic API"])


class SubQuestionFinding(BaseModel):
    sub_question: str
    facts: list[str]
    sources: list[str]
    error: str | None = None


class DeepResearchResponse(BaseModel):
    question: str
    sub_questions: list[str]
    findings: list[SubQuestionFinding]


class ReportRequest(BaseModel):
    question: str = Field(..., min_length=3,
        description="A broad question to research and synthesize into a report.",
        examples=["Compare REST and GraphQL for a high-traffic API"])


class ReportResponse(BaseModel):
    question: str
    sub_questions: list[str]
    report: str
    sources: list[str]
    approved: bool
    iterations: int
    remaining_issues: list[str] = []


class GraphReportResponse(BaseModel):
    question: str
    sub_questions: list[str]
    report: str
    sources: list[str]
    approved: bool
    iterations: int


# ---- Routes ----

@app.get("/health")
def health() -> dict:
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
    """Deep pipeline: plan -> parallel search -> parallel extract.
    Returns structured facts grouped by sub-question (no synthesis)."""
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


@app.post("/research/report", response_model=ReportResponse)
async def research_report(request: ReportRequest) -> ReportResponse:
    """Full pipeline (hand-rolled orchestrator): plan -> search -> extract ->
    synthesize -> critique -> revise. Returns a cited report with approval
    status and revision count."""
    try:
        result = await orchestrator.run_report(request.question)
    except OrchestratorError as e:
        raise HTTPException(status_code=502, detail=f"Report generation failed: {e}")

    return ReportResponse(
        question=result.question,
        sub_questions=result.sub_questions,
        report=result.report,
        sources=result.sources,
        approved=result.approved,
        iterations=result.iterations,
        remaining_issues=result.remaining_issues,
    )


@app.post("/research/graph", response_model=GraphReportResponse)
async def research_graph_endpoint(request: ReportRequest) -> GraphReportResponse:
    """Same full pipeline as /research/report, but orchestrated by LangGraph
    (a compiled StateGraph) instead of the hand-rolled orchestrator. Provided
    alongside /research/report to compare the two implementations."""
    try:
        final_state = await research_graph.ainvoke({
            "question": request.question,
            "iterations": 0,
            "max_revisions": 2,
        })
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Graph research failed: {e}")

    return GraphReportResponse(
        question=final_state.get("question", request.question),
        sub_questions=final_state.get("sub_questions", []),
        report=final_state.get("report", ""),
        sources=final_state.get("sources", []),
        approved=final_state.get("approved", False),
        iterations=final_state.get("iterations", 0),
    )
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="ui")   