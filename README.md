# Multi-Agent Automation

A multi-agent research assistant that decomposes a broad question into
focused sub-questions, searches the web for each, and returns a grounded,
cited answer. Built as an incremental, phase-by-phase system with real
benchmarked metrics at each stage.

> **Status:** Phases 1–2 complete (single-agent baseline + query planner).
> Parallel retrieval, synthesis, self-critique, and orchestration are in
> progress. See [PROGRESS.md](./PROGRESS.md) for a detailed build log with
> per-phase metrics.

---

## What it does

Given a research question, the system:

1. **Plans** — decomposes the question into 3–5 focused, non-overlapping
   sub-questions (Gemini structured output).
2. **Searches** — retrieves live web results (Tavily).
3. **Grounds** — answers strictly from retrieved sources, with inline `[n]`
   citations, refusing to rely on the model's own memory.

Parallel per-sub-question search, multi-source synthesis, and a self-critique
revision loop are the next phases.

## Architecture

```
                    ┌──────────────┐
   question  ───▶   │  Planner     │  decompose into sub-questions
                    │  (Gemini)    │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  Search      │  live web results
                    │  (Tavily)    │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  Summarizer  │  grounded, cited answer
                    │  (Gemini)    │
                    └──────┬───────┘
                           │
                        answer + sources
```

Design principles carried throughout:

- **Typed contracts** between layers (dataclasses / Pydantic models) so a
  change in one component (e.g. swapping the search provider) doesn't ripple
  outward.
- **Config-driven** model and search settings — the model name lives in
  `.env`, not in code, so a provider deprecation is a one-line change.
- **Shared resilience** — a single retry helper (exponential backoff on
  transient 503/429) is reused by every agent.

## Tech stack

| Layer | Choice |
|---|---|
| API | FastAPI |
| LLM | Google Gemini (`gemini-flash-latest`) via `google-genai` |
| Web search | Tavily |
| Validation / schemas | Pydantic, `pydantic-settings` |

## Project structure

```
app/
  config.py            # centralized settings (keys, model) from .env
  main.py              # FastAPI app + endpoints
  search/
    tavily_client.py   # search wrapper -> typed SearchResult
  llm/
    gemini_client.py   # grounded summarizer -> ResearchAnswer
    retry.py           # shared exponential-backoff retry helper
  agents/
    planner.py         # question -> validated sub-questions
evals/
  phase1_metrics.py    # latency / cost / citation-coverage harness
  phase2_metrics.py    # planner latency / decomposition / cost harness
```

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API keys
cp .env.example .env
# then edit .env and fill in:
#   TAVILY_API_KEY   (free tier: https://tavily.com)
#   GEMINI_API_KEY   (free tier: https://aistudio.google.com/apikey)
```

## Running

```bash
uvicorn app.main:app --reload
```

Then open the interactive API docs at `http://127.0.0.1:8000/docs`.

### API

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness check |
| `/plan` | POST | Decompose a question into sub-questions |
| `/research` | POST | Search + grounded, cited answer (optional `include_plan`) |

Example:

```bash
curl -X POST http://127.0.0.1:8000/research \
  -H "Content-Type: application/json" \
  -d '{"question": "What is retrieval augmented generation?"}'
```

## Benchmarks

Measured on small fixed benchmark sets (see `evals/`). Medians are reported
where a single retry-delayed outlier would otherwise skew the mean.

**Phase 1 — search + summarize**

| Metric | Value |
|---|---|
| Median end-to-end latency | ~7.6s |
| Citation coverage | 100% of answers grounded in sources |
| Avg tokens / query | ~1,400 in / ~150 out |

**Phase 2 — planner**

| Metric | Value |
|---|---|
| Median planner latency | ~5.8s |
| Sub-questions / query | 4.0 avg (consistent) |
| Avg tokens / plan | ~110 in / ~100 out |

## Roadmap

- ✅ Phase 1 — Single-agent baseline (search → summarize)
- ✅ Phase 2 — Planner agent (query decomposition)
- ⬜ Phase 3 — Parallel search + extraction per sub-question
- ⬜ Phase 4 — Synthesis + self-critique revision loop
- ⬜ Phase 5 — LangGraph orchestration
- ⬜ Phase 6 — Expanded evaluation + productionization