# Multi-Agent Automation

A multi-agent research assistant that decomposes broad questions into
sub-questions, searches the web in parallel, synthesizes findings, and
self-critiques the result before returning a cited report.

## Stack

- **Orchestration:** LangGraph (added Phase 5)
- **Search:** Tavily
- **LLM:** Gemini
- **API:** FastAPI

## Status

✅ Phase 1: Single-agent baseline (search → summarize) — complete
🚧 Phase 2: Planner agent (query decomposition) — next

## Setup

1. `python -m venv venv && source venv/bin/activate`
2. `pip install -r requirements.txt`
3. `cp .env.example .env` and add your API keys
4. `uvicorn app.main:app --reload` (available from Segment 4)