# Search + Runtime Benchmark (Post-Fix)

**Date:** February 13, 2026  
**Environment:** local Ollama + SearXNG endpoint (`http://13.60.59.170/search`)

## What Was Fixed

1. Added **demo timing tracking** (per-stage + total runtime).
2. Replaced fragile supervisor LLM routing with a **deterministic heuristic router** (`SUPERVISOR_MODE=heuristic` default).
3. Switched editor for `news/factual/data` to **deterministic synthesis** from collected search results (`EDITOR_MODE=deterministic` default).
4. Improved search relevance with query-aware **result reranking** in `src/tools.py`.

## Demo Timing Evidence

### Query: `latest NVIDIA earnings results`
- `supervisor`: `0.4 ms`
- `researcher`: `3836.2 ms`
- `editor`: `34.4 ms`
- `total`: `3876.1 ms`

### Query: `how to optimize python list comprehension performance`
- `supervisor`: `0.1 ms`
- `researcher`: `3349.3 ms`
- `editor`: `35.3 ms`
- `total`: `3388.5 ms`

These runs confirm the main latency is now search/network, not LLM routing/editor generation stalls.

## Search Benchmark (Latest Run)

**Command**
```bash
venv/bin/python benchmark_search.py --repeats 2 --max-results 5
```

### Summary

| Engine | Avg Latency (ms) | Success-Only Latency (ms) | Avg Relevance | Query Success Rate |
|---|---:|---:|---:|---:|
| SearXNG | 1617.1 | 1617.1 | 0.854 | 100% |
| DuckDuckGo | 4038.4 | 4038.4 | 0.752 | 100% |

### Winner
- Speed: **SearXNG**
- Accuracy proxy: **SearXNG**
- Reliability: **SearXNG**

## Notes

- DuckDuckGo path is now stable but still slower in this environment due anti-bot handling/fallback chain.
- Deterministic editor mode avoids empty/incorrect model-generated summaries and makes output timing predictable.
