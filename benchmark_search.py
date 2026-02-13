#!/usr/bin/env python3
import argparse
import re
import statistics
import time
from typing import Callable, Dict, List, Tuple

from src.tools import run_duckduckgo_search, run_searxng_search

DEFAULT_QUERIES = [
    "latest NVIDIA earnings results",
    "latest news in Türkiye",
    "BIST related suggestıons",
    "LangChain tool calling tutorial",
    "how to optimize python list comprehension performance",
    "AWS EC2 pricing calculator",
    "best practices for PostgreSQL indexing",
]

STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "what", "when", "where",
    "who", "how", "why", "are", "was", "were", "is", "in", "on", "to", "of", "a",
    "an", "latest", "best",
}


def _tokenize(text: str) -> set:
    terms = set(re.findall(r"[a-z0-9]{3,}", text.lower()))
    return {term for term in terms if term not in STOPWORDS}


def _relevance_score(query: str, results: List[Dict[str, str]]) -> float:
    query_terms = _tokenize(query)
    if not query_terms or not results:
        return 0.0

    best = 0.0
    for item in results:
        text = " ".join(
            [item.get("title", ""), item.get("snippet", ""), item.get("url", "")]
        )
        terms = _tokenize(text)
        if not terms:
            continue
        overlap = len(query_terms & terms) / len(query_terms)
        best = max(best, overlap)
    return best


def _run_one(
    search_fn: Callable[[str, int], List[Dict[str, str]]],
    query: str,
    max_results: int,
) -> Tuple[float, int, float, str]:
    start = time.perf_counter()
    try:
        results = search_fn(query, max_results=max_results)
        latency_ms = (time.perf_counter() - start) * 1000
        score = _relevance_score(query, results)
        return latency_ms, len(results), score, ""
    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        return latency_ms, 0, 0.0, str(exc)


def benchmark_engine(
    name: str,
    search_fn: Callable[[str, int], List[Dict[str, str]]],
    queries: List[str],
    max_results: int,
    repeats: int,
) -> Dict:
    rows = []
    successful_latencies = []
    for query in queries:
        latencies = []
        success_latencies = []
        counts = []
        scores = []
        errors = []

        for _ in range(repeats):
            latency_ms, result_count, score, error = _run_one(search_fn, query, max_results)
            latencies.append(latency_ms)
            counts.append(result_count)
            scores.append(score)
            if error:
                errors.append(error)
            elif result_count > 0:
                success_latencies.append(latency_ms)

        ok_runs = sum(1 for c in counts if c > 0)
        successful_latencies.extend(success_latencies)
        rows.append(
            {
                "query": query,
                "latency_ms": statistics.median(latencies),
                "success_latency_ms": (
                    statistics.median(success_latencies) if success_latencies else None
                ),
                "result_count": statistics.mean(counts),
                "relevance": statistics.mean(scores),
                "ok_runs": ok_runs,
                "errors": errors,
            }
        )

    avg_latency = statistics.mean(row["latency_ms"] for row in rows)
    avg_success_latency = (
        statistics.mean(successful_latencies) if successful_latencies else None
    )
    avg_relevance = statistics.mean(row["relevance"] for row in rows)
    query_success_rate = sum(1 for row in rows if row["ok_runs"] > 0) / len(rows)

    return {
        "name": name,
        "rows": rows,
        "avg_latency": avg_latency,
        "avg_success_latency": avg_success_latency,
        "avg_relevance": avg_relevance,
        "query_success_rate": query_success_rate,
        "repeats": repeats,
    }


def _shorten(text: str, limit: int = 48) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def print_engine_report(report: Dict) -> None:
    print(f"\n=== {report['name']} ===")
    print(
        f"{'Query':48} | {'Median ms':>9} | {'Succ ms':>8} | {'Results':>7} | {'Rel':>5} | {'OK runs':>7}"
    )
    print("-" * 90)
    for row in report["rows"]:
        success_latency = (
            f"{row['success_latency_ms']:.1f}" if row["success_latency_ms"] is not None else "n/a"
        )
        print(
            f"{_shorten(row['query']):48} | "
            f"{row['latency_ms']:9.1f} | "
            f"{success_latency:>8} | "
            f"{row['result_count']:7.2f} | "
            f"{row['relevance']:5.2f} | "
            f"{row['ok_runs']:>2}/{report['repeats']}"
        )
        if row["errors"]:
            print(f"{'':48} | error: {row['errors'][0]}")

    print("-" * 90)
    print(f"Average latency (ms): {report['avg_latency']:.1f}")
    if report["avg_success_latency"] is not None:
        print(f"Success-only latency: {report['avg_success_latency']:.1f}")
    else:
        print("Success-only latency: n/a")
    print(f"Average relevance:    {report['avg_relevance']:.3f}")
    print(f"Query success rate:   {report['query_success_rate']:.0%}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare SearXNG vs DuckDuckGo on speed and relevance."
    )
    parser.add_argument(
        "--queries",
        nargs="*",
        default=DEFAULT_QUERIES,
        help="List of benchmark queries.",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=5,
        help="Maximum results per query.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=2,
        help="How many times to run each query per engine.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queries = [q.strip() for q in args.queries if q.strip()]
    if not queries:
        raise SystemExit("No queries provided.")

    searx_report = benchmark_engine(
        "SearXNG",
        run_searxng_search,
        queries,
        max_results=args.max_results,
        repeats=args.repeats,
    )
    ddg_report = benchmark_engine(
        "DuckDuckGo",
        run_duckduckgo_search,
        queries,
        max_results=args.max_results,
        repeats=args.repeats,
    )

    print_engine_report(searx_report)
    print_engine_report(ddg_report)

    comparable_reports = [r for r in [searx_report, ddg_report] if r["avg_success_latency"] is not None]
    if comparable_reports:
        speed_winner = min(comparable_reports, key=lambda r: r["avg_success_latency"])["name"]
    else:
        speed_winner = min([searx_report, ddg_report], key=lambda r: r["avg_latency"])["name"]
    relevance_winner = max(
        [searx_report, ddg_report], key=lambda r: r["avg_relevance"]
    )["name"]
    reliability_winner = max(
        [searx_report, ddg_report], key=lambda r: r["query_success_rate"]
    )["name"]

    print("\n=== Summary ===")
    print(f"Speed winner:       {speed_winner}")
    print(f"Accuracy winner*:   {relevance_winner}")
    print(f"Reliability winner: {reliability_winner}")
    print("*Accuracy is measured with a keyword-overlap relevance proxy.")


if __name__ == "__main__":
    main()
