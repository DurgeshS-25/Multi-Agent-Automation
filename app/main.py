from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.search.tavily_client import tavily_search, SearchError
from app.llm.gemini_client import gemini_summarizer, SummarizerError

app = FastAPI(
    title="Multi-Agent Automation",
    description="Research assistant API — search the web and return grounded, cited answers.",
    version="0.1.0",
)


# ---- Request / Response schemas ----

class ResearchRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        description="The research question to answer.",
        examples=["What is retrieval augmented generation?"],
    )


class ResearchResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]


# ---- Routes ----

@app.get("/health")
def health() -> dict:
    """Simple liveness check."""
    return {"status": "ok"}


@app.post("/research", response_model=ResearchResponse)
def research(request: ResearchRequest) -> ResearchResponse:
    """Run the single-agent research pipeline: search -> summarize."""
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
    )