"""The research pipeline as a LangGraph state machine.

Linear flow: planner -> researcher -> extractor -> synthesizer -> critic.
Then a CONDITIONAL edge after the critic expresses the revision loop:
  - approved, or revision cap hit  -> END
  - otherwise                      -> back to synthesizer (revise)
"""

from typing import Literal

from langgraph.graph import StateGraph, START, END

from app.graph.state import ResearchState
from app.graph.nodes import (
    planner_node,
    researcher_node,
    extractor_node,
    synthesizer_node,
    critic_node,
)


def _route_after_critic(state: ResearchState) -> Literal["revise", "done"]:
    """Router function: reads state, returns the name of the next step.

    This IS the revision loop, expressed declaratively. Per LangGraph
    convention the router only reads state and returns a string — no side
    effects, no LLM calls.
    """
    approved = state.get("approved", False)
    iterations = state.get("iterations", 0)
    max_revisions = state.get("max_revisions", 2)

    # Stop if the critic approved, or we've used up our revision budget.
    # iterations counts synthesis passes; allow (1 initial + max_revisions).
    if approved or iterations > max_revisions:
        return "done"
    return "revise"


def build_research_graph():
    """Construct and compile the research graph."""
    graph = StateGraph(ResearchState)

    # Register nodes
    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("extractor", extractor_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("critic", critic_node)

    # Linear edges: the straight part of the pipeline
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "extractor")
    graph.add_edge("extractor", "synthesizer")
    graph.add_edge("synthesizer", "critic")

    # Conditional edge after the critic — THE REVISION LOOP.
    # The router returns "revise" or "done"; the path map sends each to a node.
    graph.add_conditional_edges(
        "critic",
        _route_after_critic,
        {
            "revise": "synthesizer",  # loop back to re-synthesize
            "done": END,
        },
    )

    return graph.compile()


# Compile once at import; reuse for every request.
research_graph = build_research_graph()