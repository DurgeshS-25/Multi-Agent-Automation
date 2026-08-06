# Project Progress

A multi-agent research assistant that decomposes broad questions into
sub-questions, searches the web, and returns grounded, cited answers.
This document tracks what has been built, phase by phase.

**Stack:** FastAPI · Gemini (`gemini-flash-latest`) · Tavily · Pydantic

---

## Phase 1 — Single-Agent Baseline ✅

The core retrieval-and-summarize loop: take a question, search the web,
return a grounded answer with citations, exposed as a real HTTP service.

### What was built

- **Config layer** (`app/config.py`) — centralized `pydantic-settings`
  config so API keys and the model name live in one place, loaded from
  `.env`. Model name is config-driven, not hardcoded.
- **Search wrapper** (`app/search/tavily_client.py`) — Tavily client that
  normalizes results into a typed `SearchResult` dataclass with error
  handling. Isolates all search-provider details behind one interface.
- **Summarizer** (`app/llm/gemini_client.py`) — Gemini call that answers
  strictly from retrieved sources (grounding enforced in-prompt) with
  inline `[n]` citations. Returns a typed `ResearchAnswer` carrying token
  usage for cost measurement.
- **API** (`app/main.py`) — `POST /research` (question in, cited answer
  out) and `GET /health` for liveness checks. Pydantic request/response
  schemas give input validation and auto-generated OpenAPI docs.
- **Evaluation harness** (`evals/phase1_metrics.py`) — runs a fixed
  benchmark set and reports latency (split by stage), token cost, and
  citation coverage.

### Measured metrics (5-query benchmark)

| Metric | Value |
|---|---|
| Median end-to-end latency | 7.6s (search ~0.1s / LLM ~7.5s) |
| Citation coverage | 100% (every answer grounded in sources) |
| Citations per answer | 5–13 (avg ~7.5) |
| Avg tokens/query | ~1,400 in / ~150 out |
| Avg cost/query | ~$0.003 (at published `gemini-3.6-flash` rates) |

### Engineering notes

- The LLM call dominates latency (~98%); web search is negligible.
- Model name is config-driven — when `gemini-2.0-flash` was deprecated
  mid-build, recovery was a one-line `.env` change.
- Migrated from the legacy `google-generativeai` SDK to `google-genai`
  after the former masked a model-availability error behind a misleading
  quota message.
- Tavily returns direct URLs for top results but occasionally gates
  lower-ranked ones behind opaque redirect stubs; documented as a known
  limitation since content/title fields (used for grounding) are
  unaffected.

---

## Phase 2 — Planner Agent (Query Decomposition) ✅

Adds the first piece of agentic reasoning: a planner that breaks a broad
question into focused, independently-searchable sub-questions before any
retrieval happens.

### What was built

- **Planner agent** (`app/agents/planner.py`) — decomposes a question into
  3–5 sub-questions using Gemini's **structured output** (a Pydantic
  `response_schema`), so output is guaranteed valid JSON with no manual
  parsing. Exposes `plan()` (clean list, for the API) and
  `plan_with_meta()` (adds token/validation metadata, for evals).
- **Plan validation** — cleans and clamps planner output: strips stray
  numbering, drops empty and near-duplicate sub-questions (normalized,
  punctuation-insensitive dedup), and caps the count to bound downstream
  search cost. Errors only if too few usable sub-questions survive.
- **Shared retry helper** (`app/llm/retry.py`) — exponential-backoff retry
  (1s/2s/4s) for transient API failures (503/429), retrying only transient
  errors and failing fast on real ones. Extracted into one helper used by
  every agent, so resilience is consistent and defined once.
- **API integration** (`app/main.py`) — standalone `POST /plan`
  (decomposition only) plus an `include_plan` flag on `POST /research`
  that returns the plan alongside the answer. Version bumped to 0.2.0.
- **Evaluation harness** (`evals/phase2_metrics.py`) — measures planner
  latency, decomposition consistency, validation effect, and cost.

### Measured metrics (5-query benchmark)

| Metric | Value |
|---|---|
| Median planner latency | 5.8s (typical range 3.6–6.6s) |
| Sub-questions per query | 4.0 avg (consistent, range 4–4) |
| Validation dropped | 0 of 20 raw sub-questions (clean output) |
| Avg tokens/plan | ~110 in / ~100 out |
| Avg cost/plan | ~$0.0009 |

### Engineering notes

- Structured output via Pydantic schema eliminated the whole class of
  malformed-JSON failures that ad-hoc prompt-and-parse approaches hit.
- One benchmark query took ~95s due to retry backoff after a transient
  rate-limit; the median (5.8s) is the honest latency figure, not the
  mean (which that single outlier inflated ~4x).
- Planning is a short single LLM call with no search context, so it is
  both faster and far cheaper per call than the Phase 1 pipeline.

---

## Roadmap

- ✅ **Phase 1** — Single-agent baseline (search → summarize)
- ✅ **Phase 2** — Planner agent (query decomposition)
- ⬜ **Phase 3** — Parallel search + extraction per sub-question
- ⬜ **Phase 4** — Synthesis + self-critique revision loop
- ⬜ **Phase 5** — LangGraph orchestration
- ⬜ **Phase 6** — Evaluation harness + productionization

### API surface (current)

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness check |
| `/plan` | POST | Decompose a question into sub-questions |
| `/research` | POST | Search + grounded answer (optional `include_plan`) |