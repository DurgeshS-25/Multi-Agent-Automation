from typing import TypedDict

from app.agents.researcher import SubQuestionResult
from app.agents.extractor import SubQuestionFacts


class ResearchState(TypedDict, total=False):
    """Shared state that flows through every node in the research graph.

    Each node reads the fields it needs and returns a partial update, which
    LangGraph merges into this state. `total=False` means nodes don't have
    to populate every field up front — they fill in their piece as the graph
    progresses.
    """
    # --- Input ---
    question: str

    # --- Filled by the planner node ---
    sub_questions: list[str]

    # --- Filled by the researcher node ---
    search_results: list[SubQuestionResult]

    # --- Filled by the extractor node ---
    findings: list[SubQuestionFacts]

    # --- Filled by the synthesizer node ---
    report: str
    sources: list[str]

    # --- Filled by the critic node ---
    approved: bool
    unsupported_claims: list[str]

    # --- Loop control ---
    iterations: int          # how many synthesis passes have run
    max_revisions: int       # cap, set at graph invocation