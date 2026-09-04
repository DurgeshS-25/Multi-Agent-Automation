"""Evaluation dataset for the research pipeline.

Questions are kept as structured data (not hardcoded in the harness) so the
set can grow and be categorized without touching eval logic. Categories let
us slice results (e.g. does the pipeline do worse on open-ended questions?).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalQuestion:
    question: str
    category: str


# ~15 diverse questions across categories. Deliberately varied so the
# benchmark reflects real-world spread, not one narrow type.
EVAL_QUESTIONS: list[EvalQuestion] = [
    # --- Technical comparisons ---
    EvalQuestion("Compare REST and GraphQL for a high-traffic API", "tech-comparison"),
    EvalQuestion("What are the tradeoffs of microservices vs a monolith?", "tech-comparison"),
    EvalQuestion("How should a team choose between SQL and NoSQL databases?", "tech-comparison"),
    EvalQuestion("Compare gRPC and REST for internal service communication", "tech-comparison"),

    # --- Concept explanations ---
    EvalQuestion("How does retrieval-augmented generation work?", "concept"),
    EvalQuestion("What is the CAP theorem and why does it matter?", "concept"),
    EvalQuestion("How does OAuth2 authorization code flow work?", "concept"),
    EvalQuestion("What is eventual consistency in distributed systems?", "concept"),

    # --- Decision / how-to ---
    EvalQuestion("How do you design a scalable rate limiter?", "how-to"),
    EvalQuestion("What are best practices for caching in a web application?", "how-to"),
    EvalQuestion("How should you approach database indexing for read-heavy workloads?", "how-to"),

    # --- Open-ended / tradeoff-heavy ---
    EvalQuestion("What are the main challenges in scaling a real-time chat system?", "open-ended"),
    EvalQuestion("What factors matter when choosing a message queue?", "open-ended"),
    EvalQuestion("What are the security considerations for a public-facing API?", "open-ended"),
    EvalQuestion("How do you decide between server-side and client-side rendering?", "open-ended"),
]