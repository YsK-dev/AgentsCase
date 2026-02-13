# Search Benchmark Report

**Date:** February 13, 2026  
**Command:** `venv/bin/python benchmark_search.py --repeats 3 --max-results 5`

## Scope

Comparison between:
- `SearXNG` (`http://13.60.59.170/search`)
- `DuckDuckGo` tool path after reliability fixes
We saw Searxng was way better most of time issue before was related to ddg not responding or just respond 202, 400 errors which at first cause ddg to be faster but worse content

DuckDuckGo execution order in code:
1. Direct DuckDuckGo HTML/Lite endpoints
2. SearXNG with `engines=duckduckgo`
3. Mirror fallback via `r.jina.ai` when direct DDG is blocked jina acts like a layerbetween url and llm 

This fallback chain was added because direct DDG frequently returns anti-bot responses (`HTTP 202/403`) from this runtime.

## Results Summary

| Engine | Avg Latency (ms) | Success-Only Latency (ms) | Avg Relevance | Query Success Rate |
|---|---:|---:|---:|---:|
| SearXNG | 1307.8 | 1440.4 | 0.803 | 100% |
| DuckDuckGo | 4062.0 | 4213.1 | 0.803 | 100% |

**Winner (this run):**
- Speed: **SearXNG**
- Accuracy proxy: **SearXNG**
- Reliability: **SearXNG**

## Per-Query Breakdown

| Query | SearXNG Median ms | SearXNG Results | SearXNG Rel | DuckDuckGo Median ms | DuckDuckGo Results | DuckDuckGo Rel |
|---|---:|---:|---:|---:|---:|---:|
| latest NVIDIA earnings results | 1108.4 | 5.00 | 0.67 | 2744.0 | 5.00 | 0.67 |
| LangChain tool calling tutorial | 703.4 | 5.00 | 0.75 | 5649.1 | 5.00 | 0.75 |
| how to optimize python list comprehension performance | 1024.6 | 5.00 | 0.60 | 3372.8 | 5.00 | 0.60 |
| AWS EC2 pricing calculator | 3148.6 | 5.00 | 1.00 | 5696.9 | 5.00 | 1.00 |
| best practices for PostgreSQL indexing | 554.0 | 5.00 | 1.00 | 2847.4 | 5.00 | 1.00 |

## Notes

- DuckDuckGo now returns the requested result depth (`5`) in this run, with anti-bot handling and fallback logic enabled.
- Relevance is a keyword-overlap proxy from `benchmark_search.py`; it is useful for relative comparison but not a full semantic quality metric.
