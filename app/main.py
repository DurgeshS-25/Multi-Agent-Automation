import time
from pathlib import Path

from dotenv import load_dotenv

# Load .env into the process environment so auth and run logging see the same
# values locally as they do on the host.
load_dotenv()

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.search.tavily_client import tavily_search, SearchError
from app.llm.gemini_client import gemini_summarizer, SummarizerError
from app.agents.planner import planner_agent, PlannerError
from app.agents.orchestrator import orchestrator, OrchestratorError
from app.graph.research_graph import research_graph
from app.auth import setup_auth
from app.run_log import build_row, schedule_log, start_timing

app = FastAPI(
    title="Multi-Agent Automation",
    description="Research assistant API — plan, search, synthesize, and self-critique cited reports.",
    version="0.6.0",
)

setup_auth(app)


@app.get("/", include_in_schema=False)
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
    total_ms: int | None = None
    stage_timings_ms: dict[str, float] = {}


class GraphReportResponse(BaseModel):
    question: str
    sub_questions: list[str]
    report: str
    sources: list[str]
    approved: bool
    iterations: int
    total_ms: int | None = None
    stage_timings_ms: dict[str, float] = {}


def _user(http: Request) -> dict | None:
    return getattr(http.state, "user", None)


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
async def research_report(request: ReportRequest, http: Request) -> ReportResponse:
    """Full pipeline (hand-rolled orchestrator): plan -> search -> extract ->
    synthesize -> critique -> revise. Returns a cited report with approval
    status, revision count and timings. Every run is logged."""
    timings = start_timing()
    t0 = time.perf_counter()
    try:
        result = await orchestrator.run_report(request.question)
    except OrchestratorError as e:
        schedule_log(build_row(
            user=_user(http), pipeline="orchestrator", question=request.question,
            total_ms=(time.perf_counter() - t0) * 1000, timings=timings, error=str(e),
        ))
        raise HTTPException(status_code=502, detail=f"Report generation failed: {e}")

    total_ms = (time.perf_counter() - t0) * 1000
    response = ReportResponse(
        question=result.question,
        sub_questions=result.sub_questions,
        report=result.report,
        sources=result.sources,
        approved=result.approved,
        iterations=result.iterations,
        remaining_issues=result.remaining_issues,
        total_ms=round(total_ms),
        stage_timings_ms=dict(timings.ms),
    )
    schedule_log(build_row(
        user=_user(http), pipeline="orchestrator", question=request.question,
        total_ms=total_ms, timings=timings, result=response.model_dump(),
    ))
    return response


@app.post("/research/graph", response_model=GraphReportResponse)
async def research_graph_endpoint(request: ReportRequest, http: Request) -> GraphReportResponse:
    """Same full pipeline as /research/report, but orchestrated by LangGraph
    (a compiled StateGraph) instead of the hand-rolled orchestrator. Each graph
    node is timed from the stream of updates. Every run is logged."""
    timings = start_timing()
    t0 = time.perf_counter()
    inputs = {"question": request.question, "iterations": 0, "max_revisions": 2}

    final_state: dict = {}
    try:
        step_started = time.perf_counter()
        async for mode, chunk in research_graph.astream(inputs, stream_mode=["updates", "values"]):
            if mode == "values":
                final_state = chunk
                continue
            now = time.perf_counter()
            nodes = [n for n in chunk if not n.startswith("__")]
            if nodes:
                # Nodes in the same step ran together; record the step under all their names.
                timings.add("+".join(nodes), (now - step_started) * 1000)
            step_started = now
    except Exception as e:
        schedule_log(build_row(
            user=_user(http), pipeline="langgraph", question=request.question,
            total_ms=(time.perf_counter() - t0) * 1000, timings=timings, error=str(e),
        ))
        raise HTTPException(status_code=502, detail=f"Graph research failed: {e}")

    total_ms = (time.perf_counter() - t0) * 1000
    response = GraphReportResponse(
        question=final_state.get("question", request.question),
        sub_questions=final_state.get("sub_questions", []),
        report=final_state.get("report", ""),
        sources=final_state.get("sources", []),
        approved=final_state.get("approved", False),
        iterations=final_state.get("iterations", 0),
        total_ms=round(total_ms),
        stage_timings_ms=dict(timings.ms),
    )
    schedule_log(build_row(
        user=_user(http), pipeline="langgraph", question=request.question,
        total_ms=total_ms, timings=timings, result=response.model_dump(),
    ))
    return response


# ---- Web UI ----
# Mounted last so it can't shadow any API route.
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="ui")