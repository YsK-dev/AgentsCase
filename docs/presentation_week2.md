# Slide 1 - Scenario and Goal

## Multi-Agent Local LLM Assistant

- Goal: answer user requests from start to finish using multiple specialized agents.
- Value: better reliability than a single call by separating routing, research, analysis, and final synthesis.
- Constraints: local/open-source models only (Ollama), tool-enabled reasoning, deterministic fallbacks.

Example user task:
- "Latest NVIDIA earnings and what changed vs last quarter?"

System outcome:
- gathers web evidence, runs calculations when needed, and returns one final report with sources.

---

# Slide 2 - Agents and Roles

## Architecture (LangGraph)

- `supervisor`: classifies query and selects route.
- `researcher`: executes web/news search tools.
- `analyst`: executes safe Python for calculations/verification.
- `editor`: composes final output from collected context.

Flow:
- `supervisor -> researcher -> editor` for factual/news
- `supervisor -> researcher -> analyst -> editor` for data/comparison
- `supervisor -> editor` for general/creative

Why this design:
- clear separation of concerns,
- easy to test each stage,
- easy to add/replace agents later.

---

# Slide 3 - Model Selection Strategy

## Multi-Model Routing

Models:
- Fast model: low latency (`MODEL_FAST`)
- Accurate model: higher quality (`MODEL_ACCURATE`)

Modes:
- `auto` (default): complexity-aware routing by task length, keywords, query type, role
- `fast`: latency-first
- `accurate`: quality-first

Fallback:
- if primary model fails/returns empty, system tries fallback candidates in order.

Rationale:
- simple requests should not pay high latency,
- high-complexity and data-heavy tasks need stronger reasoning quality.

---

# Slide 4 - Tools and Safety

## Required Tools

1. Web Search:
- SearXNG primary
- DuckDuckGo fallback
- news-specific search path for current events

2. Code Execution:
- Python tool with AST safety validation
- restricted modules/calls
- isolated subprocess + timeout

Safety controls:
- blocked imports (`os`, `subprocess`, `socket`, etc.)
- blocked calls (`eval`, `exec`, `open`, etc.)
- output truncation and timeout handling

---

# Slide 5 - Quality, Testing, and Ops

## Engineering Quality

- Unit tests for normal flow and predictable errors:
  - invalid model mode
  - missing topic input
  - unsafe code execution attempts
  - graph routing behavior
- CI on push/PR (`pytest`)
- structured JSON logs (`logs/agent_system.log`)

Potential bonus next steps:
- MCP integration for additional tools/context
- LangFuse or similar tracing dashboard for latency/error metrics
- richer benchmark set for model-tier calibration

---

# Slide 6 - References

- LangGraph docs: https://docs.langchain.com/oss/python/langgraph/
- Ollama docs: https://docs.ollama.com/
- SearXNG docs/API: https://docs.searxng.org/
- Python AST docs: https://docs.python.org/3/library/ast.html
- FrugalGPT (LLM cascades): https://arxiv.org/abs/2305.05176
- RouteLLM (LLM routing): https://arxiv.org/abs/2406.18665
