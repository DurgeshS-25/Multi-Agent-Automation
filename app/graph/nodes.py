"""Node functions for the research graph.

Each node has the same contract: receive the shared state, do one step of
work (by calling an existing agent), and return a partial state update that
LangGraph merges back into the state. The agents themselves are unchanged —
these are thin adapters.
"""

from app.graph.state import ResearchState

from app.agents.planner import planner_agent
from app.agents.researcher import researcher
from app.agents.extractor import extractor
from app.agents.synthesizer import synthesizer
from app.agents.critic import critic


async def planner_node(state: ResearchState) -> dict:
    """question -> sub_questions"""
    sub_questions = planner_agent.plan(state["question"])
    return {"sub_questions": sub_questions}


async def researcher_node(state: ResearchState) -> dict:
    """sub_questions -> search_results (parallel)"""
    results = await researcher.search_all(state["sub_questions"])
    return {"search_results": results}


async def extractor_node(state: ResearchState) -> dict:
    """search_results -> findings (parallel)"""
    findings = await extractor.extract_all(state["search_results"])
    return {"findings": findings}


async def synthesizer_node(state: ResearchState) -> dict:
    """findings -> report (+ increments iteration count).

    On a revision pass (iterations >= 1 and claims were flagged), it feeds
    the previous report and unsupported claims back in so the synthesizer
    fixes them.
    """
    iterations = state.get("iterations", 0)
    previous_report = state.get("report") if iterations > 0 else None
    unsupported = state.get("unsupported_claims") if iterations > 0 else None

    report = await synthesizer.synthesize(
        state["question"],
        state["findings"],
        previous_report=previous_report,
        unsupported_claims=unsupported,
    )
    return {
        "report": report.report,
        "sources": report.sources,
        "iterations": iterations + 1,
    }


async def critic_node(state: ResearchState) -> dict:
    """report -> verdict (approved + unsupported_claims)"""
    verdict = await critic.critique(state["report"], state["findings"])
    return {
        "approved": verdict.approved,
        "unsupported_claims": verdict.unsupported_claims,
    }