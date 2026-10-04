# Multi-Agent Automation

A multi-agent research assistant that decomposes a broad question into
focused sub-questions, searches the web for each in parallel, synthesizes the
findings into a cited report, and critiques and revises its own draft before
returning it. Built as an incremental, phase-by-phase system with benchmarked
metrics at each stage.

> **Status:** Phases 1–5 complete: single-agent baseline, query planner,
> parallel retrieval, synthesis with a self-critique revision loop, and
> LangGraph orchestration. Phase 6 (expanded evaluation + productionization)
> is in progress, starting with structured logging. See
> [PROGRESS.md](./PROGRESS.md) for a detailed build log with per-phase metrics.

---

## What it does

Given a research question, the system:

1. **Plans**: decomposes the question into 3–5 focused, non-overlapping
   sub-questions (Gemini structured output).
2. **Searches**: retrieves live web results for every sub-question in
   parallel (Tavily).
3. **Synthesizes**: combines evidence across all sub-questions into a single
   report, answering strictly from retrieved sources with inline `[n]`
   citations rather than the model's own memory.
4. **Critiques & revises**: a critic pass reviews the draft for gaps,
   unsupported claims, and citation problems, and the report is revised
   before it is returned.

The full pipeline runs under two interchangeable orchestrators: a hand-rolled
orchestrator and a LangGraph state machine. Both are exposed so their output
and performance can be compared directly.

## Architecture

```
                    ┌──────────────┐
   question  ───▶   │  Planner     │  decompose into sub-questions
                    │  (Gemini)    │
                    └──────┬───────┘
                           │  sub-questions
          ┌────────────────┼────────────────┐
   ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐
   │ Search +    │  │ Search +    │  │ Search +    │   parallel, one
   │ extract     │  │ extract     │  │ extract     │   per sub-question
   │ (Tavily)    │  │ (Tavily)    │  │ (Tavily)    │
   └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
          └────────────────┼────────────────┘
                    ┌──────▼───────┐
                    │ Synthesizer  │  multi-source, cited draft
                    │  (Gemini)    │
                    └──────┬───────┘
                    ┌──────▼───────┐
                    │   Critic     │  review draft ──┐
                    │  (Gemini)    │                 │ revise
                    └──────┬───────┘  ◀──────────────┘
                           │
                  final report + sources

   Orchestrated by: app/agents/orchestrator.py  (hand-rolled)
                or: app/graph/research_graph.py (LangGraph)
```

Design principles carried throughout:

- **Typed contracts** between layers (dataclasses / Pydantic models), so a
  change in one component (e.g. swapping the search provider) doesn't ripple
  outward.
- **Config-driven** model, concurrency, and search settings: the model name
  lives in `.env`, not in code, so a provider deprecation is a one-line change.
- **Shared resilience**: a single retry helper (exponential backoff on
  transient 503/429) is reused by every agent.
- **Observability**: structured logging across the pipeline, so each stage's
  timing and outcome can be traced per request.

## Tech stack

| Layer | Choice |
|---|---|
| API | FastAPI |
| LLM | Google Gemini (`gemini-flash-latest`) via `google-genai` |
| Web search | Tavily |
| Orchestration | Hand-rolled orchestrator + LangGraph |
| Validation / schemas | Pydantic, `pydantic-settings` |

## Project structure

```
app/
  config.py            # centralized settings (keys, model, concurrency) from .env
  main.py              # FastAPI app + endpoints
  search/
    tavily_client.py   # search wrapper -> typed SearchResult
  llm/
    gemini_client.py   # grounded summarizer -> ResearchAnswer
    retry.py           # shared exponential-backoff retry helper
  agents/
    planner.py         # question -> validated sub-questions
    orchestrator.py    # hand-rolled deep-research + full report pipeline
  graph/
    research_graph.py  # same pipeline as a LangGraph state machine
  observability/       # structured logging across pipeline stages
  static/
    index.html         # browser UI for the research assistant
evals/
  dataset.py           # fixed benchmark question set
  run_eval.py          # eval runner
  phase1_metrics.py    # latency / cost / citation-coverage harness
  phase2_metrics.py    # planner latency / decomposition / cost harness
  phase3_metrics.py    # parallel retrieval metrics harness
  phase4_metrics.py    # synthesis + self-critique metrics harness
  phase5_metrics.py    # orchestrator vs LangGraph side-by-side harness
Dockerfile             # container image for the API
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

### Docker

```bash
docker build -t multi-agent-automation .
docker run --env-file .env -p 8000:8000 multi-agent-automation
```

### API

| Endpoint | Method | Purpose | Phase |
|---|---|---|---|
| `/health` | GET | Liveness check | — |
| `/plan` | POST | Decompose a question into sub-questions | 2 |
| `/research` | POST | Single search + grounded, cited answer (optional `include_plan`) | 1 |
| `/research/deep` | POST | Plan + parallel search and extraction per sub-question | 3 |
| `/research/report` | POST | Full pipeline: plan → parallel search → synthesis → critique → revision (hand-rolled orchestrator) | 4 |
| `/research/graph` | POST | Same full pipeline, orchestrated as a LangGraph state machine | 5 |

Example:

```bash
curl -X POST http://127.0.0.1:8000/research/report \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the tradeoffs of small modular nuclear reactors?"}'
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

**Phase 3 — parallel search + extraction** (`/research/deep`, n=3)

| Metric | Value |
|---|---|
| Median pipeline latency | ~8.7s |
| Sub-questions / query | 4.0 avg |
| Facts extracted / query | ~22 avg |
| Search speedup (parallel vs sequential) | 2.2x median (2.2–2.4x range) |

Speedup is bounded by sub-question count (roughly N-way for N sub-questions).

**Phase 4 — synthesis + self-critique** (`/research/report`, n=3)

| Metric | Value |
|---|---|
| Median end-to-end latency | ~13.3s |
| Sources per report | 13–19 |
| Approved within iteration cap | 3/3 |
| Revisions triggered | 0/3 (all drafts approved on first pass) |

Small sample due to free-tier quota (15 requests/min); revision behavior is
non-deterministic, so treat rates as directional.

**Phase 5 — orchestrator vs LangGraph** (`/research/report` vs `/research/graph`, n=3 each)

| Metric | Hand-rolled orchestrator | LangGraph |
|---|---|---|
| Median end-to-end latency | ~15.6s | ~15.3s |
| Latency range | 12.0–28.9s | 11.6–24.2s |
| Avg sources / report | 17.0 | 16.7 |
| Avg report length | ~2,770 chars | ~3,176 chars |
| Approved within iteration cap | 3/3 | 3/3 |
| Revisions triggered | 0/3 | 0/3 |

Both pipelines were measured over HTTP on the same questions, interleaved so
they saw the same network and quota conditions. LangGraph reached parity with
the hand-rolled orchestrator, with no measurable orchestration overhead.

## Roadmap

- ✅ Phase 1 — Single-agent baseline (search → summarize)
- ✅ Phase 2 — Planner agent (query decomposition)
- ✅ Phase 3 — Parallel search + extraction per sub-question
- ✅ Phase 4 — Synthesis + self-critique revision loop
- ✅ Phase 5 — LangGraph orchestration
- 🔄 Phase 6 — Expanded evaluation + productionization (structured logging done)