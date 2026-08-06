from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.search.tavily_client import tavily_search, SearchError
from app.llm.gemini_client import gemini_summarizer, SummarizerError
from app.agents.planner import planner_agent, PlannerError

app = FastAPI(
    title="Multi-Agent Automation",
    description="Research assistant API — plan, search the web, and return grounded, cited answers.",
    version="0.2.0",
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
                    "(planning only; full per-sub-question search arrives in Phase 3).",
    )


class ResearchResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    sub_questions: list[str] = []  # populated only when include_plan=True


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
    """Run the research pipeline: (optional plan) -> search -> summarize.

    NOTE: When include_plan=True, the plan is returned for inspection but
    each sub-question is not yet searched independently — that wiring lands
    in Phase 3. For now the answer still comes from a single search.
    """
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