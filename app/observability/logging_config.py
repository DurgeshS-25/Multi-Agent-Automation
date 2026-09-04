"""Structured logging setup.

Emits one structured (key=value) log line per pipeline event, so a request's
full journey — sub-questions, searches, extractions, critic verdict,
revisions — is visible and greppable in the logs. Uses the stdlib `logging`
module (no extra dependency). For heavier needs (distributed tracing, a UI
over traces) LangSmith or OpenTelemetry would be the next step.
"""

import logging
import sys


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging once, at app startup."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root = logging.getLogger()
    # Avoid duplicate handlers if called more than once (e.g. reload)
    if not root.handlers:
        root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Get a named logger for a component (e.g. 'planner', 'critic')."""
    return logging.getLogger(name)


def kv(**fields) -> str:
    """Format key=value pairs into one structured log message.

    Example: kv(event='plan', sub_questions=4) -> "event=plan sub_questions=4"
    Keeps logs greppable and machine-parseable without a JSON dependency.
    """
    parts = []
    for k, v in fields.items():
        # Quote strings containing spaces so each pair stays atomic
        if isinstance(v, str) and " " in v:
            v = f'"{v}"'
        parts.append(f"{k}={v}")
    return " ".join(parts)