# src/tools.py
import ast
import base64
import html
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Any, Dict, List

from langchain_core.tools import tool

SEARXNG_URL = os.getenv("SEARXNG_URL", "http://13.60.59.170/search")
SEARXNG_TIMEOUT = float(os.getenv("SEARXNG_TIMEOUT", "5"))
SEARCH_MAX_RESULTS = int(os.getenv("SEARCH_MAX_RESULTS", "10"))
DUCKDUCKGO_HTML_URL = os.getenv("DUCKDUCKGO_HTML_URL", "https://html.duckduckgo.com/html/")
DUCKDUCKGO_LITE_URL = os.getenv("DUCKDUCKGO_LITE_URL", "https://lite.duckduckgo.com/lite/")
DUCKDUCKGO_PROXY_PREFIX = os.getenv(
    "DUCKDUCKGO_PROXY_PREFIX",
    "https://r.jina.ai/http://duckduckgo.com/html/",
)
DUCKDUCKGO_TIMEOUT = float(os.getenv("DUCKDUCKGO_TIMEOUT", "4"))
DUCKDUCKGO_MIN_INTERVAL = float(os.getenv("DUCKDUCKGO_MIN_INTERVAL", "0.2"))
DUCKDUCKGO_RETRIES = int(os.getenv("DUCKDUCKGO_RETRIES", "1"))
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)
_DDG_LOCK = threading.Lock()
_DDG_LAST_REQUEST_TS = 0.0
_RERANK_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "what", "when", "where",
    "who", "how", "why", "are", "was", "were", "is", "in", "on", "to", "of", "a",
    "an", "latest", "best", "news", "today",
}
_FOCUS_TERMS = {
    "python", "langchain", "postgresql", "nvidia", "aws", "docker", "kubernetes",
    "javascript", "typescript", "java", "golang", "rust", "sql", "linux",
}
PYTHON_CODE_MAX_CHARS = int(os.getenv("PYTHON_CODE_MAX_CHARS", "4000"))
PYTHON_EXEC_TIMEOUT = float(os.getenv("PYTHON_EXEC_TIMEOUT", "4.0"))
PYTHON_OUTPUT_MAX_CHARS = int(os.getenv("PYTHON_OUTPUT_MAX_CHARS", "4000"))
SAFE_PYTHON_MODULES = {
    "collections",
    "datetime",
    "decimal",
    "fractions",
    "functools",
    "itertools",
    "json",
    "math",
    "random",
    "re",
    "statistics",
}
BLOCKED_PYTHON_CALLS = {
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "help",
    "input",
    "open",
}
BLOCKED_PYTHON_ROOTS = {
    "asyncio",
    "builtins",
    "ctypes",
    "importlib",
    "multiprocessing",
    "os",
    "pathlib",
    "resource",
    "shutil",
    "signal",
    "socket",
    "subprocess",
    "sys",
    "threading",
}
SAFE_PYTHON_RUNNER = r"""
import base64
import contextlib
import io
import json
import sys

SAFE_MODULES = {
    "collections",
    "datetime",
    "decimal",
    "fractions",
    "functools",
    "itertools",
    "json",
    "math",
    "random",
    "re",
    "statistics",
}

def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    top_level = name.split(".")[0]
    if top_level not in SAFE_MODULES:
        raise ImportError(f"Import '{name}' is not allowed")
    return __import__(name, globals, locals, fromlist, level)

SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "pow": pow,
    "print": print,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    "__import__": safe_import,
}

payload = {"stdout": "", "stderr": "", "error": ""}
code = base64.b64decode(sys.argv[1]).decode("utf-8")
stdout_buffer = io.StringIO()
stderr_buffer = io.StringIO()

try:
    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
        exec(compile(code, "<user_code>", "exec"), {"__builtins__": SAFE_BUILTINS}, {})
except Exception as exc:
    payload["error"] = str(exc)

payload["stdout"] = stdout_buffer.getvalue()
payload["stderr"] = stderr_buffer.getvalue()
print(json.dumps(payload))
"""


class DuckDuckGoRateLimitError(RuntimeError):
    """Raised when DuckDuckGo returns a temporary anti-bot/rate-limit response."""


class UnsafePythonCodeError(ValueError):
    """Raised when user code violates safety constraints."""


class ToolSelector:
    """Returns the appropriate tools for each agent role."""

    @staticmethod
    def get_tools(agent_role: str):
        if agent_role == "supervisor":
            return []
        elif agent_role == "researcher":
            return [web_search, searxng_search, duckduckgo_search, news_search]
        elif agent_role == "analyst":
            return [python_repl]
        else:
            return [web_search, searxng_search, duckduckgo_search, news_search, python_repl]


def _normalize_text(text: Any) -> str:
    if text is None:
        return ""
    return str(text).strip()


def _clean_text(text: Any) -> str:
    value = html.unescape(_normalize_text(text))
    return re.sub(r"\s+", " ", value).strip()


def _dedupe_results(results: List[Dict[str, str]]) -> List[Dict[str, str]]:
    deduped = []
    seen_urls = set()
    for item in results:
        url = _normalize_text(item.get("url"))
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        deduped.append(item)
    return deduped


def _format_search_results(results: List[Dict[str, str]]) -> str:
    if not results:
        return "No results found."

    formatted = []
    for i, item in enumerate(results, 1):
        title = _clean_text(item.get("title")) or "No title"
        url = _normalize_text(item.get("url")) or "N/A"
        snippet = _clean_text(item.get("snippet")) or "No snippet"
        formatted.append(
            f"[{i}] {title}\n"
            f"    URL: {url}\n"
            f"    {snippet}"
        )
    return "\n\n".join(formatted)


def _tokenize_for_rank(text: str) -> set:
    terms = set(re.findall(r"[a-z0-9]{3,}", text.lower()))
    return {term for term in terms if term not in _RERANK_STOPWORDS}


def _rerank_by_query(query: str, results: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if not results:
        return results

    query_terms = _tokenize_for_rank(query)
    if not query_terms:
        return results
    focus_terms = query_terms & _FOCUS_TERMS

    scored = []
    for idx, item in enumerate(results):
        haystack = " ".join(
            [
                _normalize_text(item.get("title")),
                _normalize_text(item.get("snippet")),
                _normalize_text(item.get("url")),
            ]
        )
        terms = _tokenize_for_rank(haystack)
        overlap = len(query_terms & terms) / len(query_terms) if terms else 0.0
        focus_overlap = len(focus_terms & terms)
        score = overlap + (0.4 * focus_overlap)
        if focus_terms and focus_overlap == 0:
            score *= 0.5
        scored.append((score, idx, item))

    # Keep only relevant items when possible; otherwise preserve original list.
    positive = [row for row in scored if row[0] > 0]
    if positive:
        scored = positive

    scored.sort(key=lambda row: (-row[0], row[1]))
    return [row[2] for row in scored]


def _attribute_root_name(node: ast.AST) -> str:
    current = node
    while isinstance(current, ast.Attribute):
        current = current.value
    if isinstance(current, ast.Name):
        return current.id
    return ""


def _validate_python_code(code: str) -> None:
    if not isinstance(code, str) or not code.strip():
        raise UnsafePythonCodeError("Code input is empty.")
    if len(code) > PYTHON_CODE_MAX_CHARS:
        raise UnsafePythonCodeError(
            f"Code too long ({len(code)} chars). Max allowed is {PYTHON_CODE_MAX_CHARS}."
        )

    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise UnsafePythonCodeError(f"Syntax error: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name.split(".")[0]
                if module not in SAFE_PYTHON_MODULES:
                    raise UnsafePythonCodeError(
                        f"Import '{module}' is not allowed."
                    )

        if isinstance(node, ast.ImportFrom):
            module = (node.module or "").split(".")[0]
            if module not in SAFE_PYTHON_MODULES:
                raise UnsafePythonCodeError(
                    f"Import from '{module or '[relative]'}' is not allowed."
                )

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_PYTHON_CALLS:
                raise UnsafePythonCodeError(
                    f"Call to '{node.func.id}' is blocked."
                )
            root = _attribute_root_name(node.func)
            if root in BLOCKED_PYTHON_ROOTS:
                raise UnsafePythonCodeError(
                    f"Calls through '{root}' are blocked."
                )

        if isinstance(node, ast.Attribute):
            root = _attribute_root_name(node)
            if root in BLOCKED_PYTHON_ROOTS:
                raise UnsafePythonCodeError(
                    f"Access to '{root}' is blocked."
                )

        if isinstance(node, ast.Name) and node.id in {
            "__builtins__",
            "__loader__",
            "__spec__",
            "__import__",
        }:
            raise UnsafePythonCodeError(
                f"Name '{node.id}' is blocked."
            )


def _run_safe_python_code(code: str) -> str:
    encoded_code = base64.b64encode(code.encode("utf-8")).decode("ascii")
    try:
        process = subprocess.run(
            [sys.executable, "-I", "-c", SAFE_PYTHON_RUNNER, encoded_code],
            capture_output=True,
            text=True,
            timeout=PYTHON_EXEC_TIMEOUT,
            env={},
        )
    except subprocess.TimeoutExpired:
        return f"Error: Code execution timed out after {PYTHON_EXEC_TIMEOUT:.1f}s."

    if process.returncode != 0 and not process.stdout.strip():
        stderr = (process.stderr or "").strip()
        return f"Error: Python runner failed ({process.returncode}). {stderr}"

    stdout = (process.stdout or "").strip()
    if not stdout:
        stderr = (process.stderr or "").strip()
        if stderr:
            return f"Error: Empty response. {stderr}"
        return "Code executed successfully (no output)."

    try:
        payload = json.loads(stdout.splitlines()[-1])
    except json.JSONDecodeError:
        return f"Error: Failed to parse execution output. Raw output: {stdout[:500]}"

    error = str(payload.get("error", "")).strip()
    out = str(payload.get("stdout", "")).strip()
    err = str(payload.get("stderr", "")).strip()

    if error:
        result = f"Error: {error}"
    elif out:
        result = out
    elif err:
        result = err
    else:
        result = "Code executed successfully (no output)."

    if len(result) > PYTHON_OUTPUT_MAX_CHARS:
        return result[:PYTHON_OUTPUT_MAX_CHARS] + "\n... (output truncated)"
    return result


def _fetch_json(url: str, params: Dict[str, str]) -> Dict[str, Any]:
    encoded = urllib.parse.urlencode(params)
    request = urllib.request.Request(
        f"{url}?{encoded}",
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=SEARXNG_TIMEOUT) as response:
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}")
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise RuntimeError(reason) from exc


def _wait_for_ddg_slot() -> None:
    global _DDG_LAST_REQUEST_TS
    if DUCKDUCKGO_MIN_INTERVAL <= 0:
        return

    with _DDG_LOCK:
        now = time.monotonic()
        wait_s = DUCKDUCKGO_MIN_INTERVAL - (now - _DDG_LAST_REQUEST_TS)
        if wait_s > 0:
            time.sleep(wait_s)
        _DDG_LAST_REQUEST_TS = time.monotonic()


def _build_http_headers(accept_json: bool = False) -> Dict[str, str]:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }
    if accept_json:
        headers["Accept"] = "application/json"
    else:
        headers["Accept"] = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
    return headers


def _fetch_html(url: str, timeout: float) -> str:
    request = urllib.request.Request(url, headers=_build_http_headers(accept_json=False))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = int(getattr(response, "status", 0))
            body = response.read().decode("utf-8", errors="ignore")
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="ignore")
        except Exception:
            pass
        status = int(getattr(exc, "code", 0))
        if status in (202, 403, 429):
            raise DuckDuckGoRateLimitError(f"HTTP {status}") from exc
        raise RuntimeError(f"HTTP {status}") from exc
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise RuntimeError(reason) from exc

    if status in (202, 403, 429):
        raise DuckDuckGoRateLimitError(f"HTTP {status}")
    if status != 200:
        raise RuntimeError(f"HTTP {status}")
    return body


def _fetch_text(url: str, timeout: float) -> str:
    request = urllib.request.Request(url, headers=_build_http_headers(accept_json=False))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = int(getattr(response, "status", 0))
            body = response.read().decode("utf-8", errors="ignore")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {int(getattr(exc, 'code', 0))}") from exc
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise RuntimeError(reason) from exc

    if status != 200:
        raise RuntimeError(f"HTTP {status}")
    return body


def _parse_searx_results(data: Dict[str, Any], max_results: int) -> List[Dict[str, str]]:
    parsed = []
    for item in data.get("results", []):
        parsed.append(
            {
                "title": _clean_text(item.get("title")) or "No title",
                "url": _normalize_text(item.get("url")),
                "snippet": _clean_text(item.get("content") or item.get("snippet")) or "No snippet",
            }
        )
    return _dedupe_results(parsed)[:max_results]


def _decode_ddg_redirect_url(href: str) -> str:
    href = _normalize_text(href)
    if not href:
        return ""

    parsed = urllib.parse.urlparse(href)
    if parsed.path.startswith("/l/"):
        qs = urllib.parse.parse_qs(parsed.query)
        uddg = qs.get("uddg", [])
        if uddg:
            return urllib.parse.unquote(uddg[0])

    if href.startswith("//"):
        return f"https:{href}"
    if href.startswith("/"):
        return f"https://duckduckgo.com{href}"
    return href


class _DuckDuckGoHTMLParser(HTMLParser):
    def __init__(self, max_results: int):
        super().__init__()
        self.max_results = max_results
        self.results: List[Dict[str, str]] = []
        self._capture_title = False
        self._capture_snippet = False
        self._snippet_tag = ""
        self._title_parts: List[str] = []
        self._snippet_parts: List[str] = []
        self._current_href = ""

    def handle_starttag(self, tag: str, attrs: List[tuple]):
        if len(self.results) >= self.max_results:
            return

        attr_map = dict(attrs)
        css_classes = _normalize_text(attr_map.get("class"))

        if tag == "a" and ("result__a" in css_classes or "result-link" in css_classes):
            self._capture_title = True
            self._title_parts = []
            self._current_href = _decode_ddg_redirect_url(attr_map.get("href", ""))
            return

        if tag in {"a", "div", "td"} and ("result__snippet" in css_classes or "result-snippet" in css_classes):
            self._capture_snippet = True
            self._snippet_tag = tag
            self._snippet_parts = []

    def handle_data(self, data: str):
        if self._capture_title:
            self._title_parts.append(data)
        if self._capture_snippet:
            self._snippet_parts.append(data)

    def handle_endtag(self, tag: str):
        if self._capture_title and tag == "a":
            title = _clean_text("".join(self._title_parts))
            if title and self._current_href:
                self.results.append(
                    {"title": title, "url": self._current_href, "snippet": "No snippet"}
                )
            self._capture_title = False
            self._title_parts = []
            self._current_href = ""

        if self._capture_snippet and tag == self._snippet_tag:
            snippet = _clean_text("".join(self._snippet_parts))
            if snippet:
                for item in reversed(self.results):
                    if item.get("snippet") in ("", "No snippet"):
                        item["snippet"] = snippet
                        break
            self._capture_snippet = False
            self._snippet_tag = ""
            self._snippet_parts = []


def _parse_jina_ddg_markdown(markdown: str, max_results: int) -> List[Dict[str, str]]:
    results = []
    seen_urls = set()
    pattern = re.compile(r"\[([^\]]+)\]\((http://duckduckgo\.com/l/\?uddg=[^)]+)\)")

    for match in pattern.finditer(markdown):
        title = _clean_text(match.group(1))
        raw_url = match.group(2)
        if not title or title.startswith("!"):
            continue

        decoded_url = _decode_ddg_redirect_url(raw_url)
        if not decoded_url:
            continue
        if decoded_url in seen_urls:
            continue
        seen_urls.add(decoded_url)
        results.append({"title": title, "url": decoded_url, "snippet": "No snippet"})
        if len(results) >= max_results:
            break

    return results[:max_results]


def run_searxng_search(query: str, max_results: int = SEARCH_MAX_RESULTS) -> List[Dict[str, str]]:
    data = _fetch_json(SEARXNG_URL, {"q": query, "format": "json"})
    ranked = _rerank_by_query(query, _parse_searx_results(data, max_results * 2))
    return ranked[:max_results]


def run_searxng_news_search(query: str, max_results: int = SEARCH_MAX_RESULTS) -> List[Dict[str, str]]:
    all_results = []
    query_variants = [query]

    query_lower = query.lower()
    if "turkey" not in query_lower and "türkiye" not in query_lower:
        turkish_cities = ["istanbul", "ankara", "izmir", "antalya", "bursa"]
        if any(city in query_lower for city in turkish_cities):
            query_variants.append(f"{query} Turkey")

    for q in query_variants:
        for time_range in ["day", "month", "year", ""]:
            params = {"q": q, "format": "json", "categories": "news"}
            if time_range:
                params["time_range"] = time_range

            data = _fetch_json(SEARXNG_URL, params)
            for item in data.get("results", []):
                url = _normalize_text(item.get("url"))
                source = _normalize_text(item.get("engine"))
                if not source and url:
                    source = urllib.parse.urlparse(url).netloc
                all_results.append(
                    {
                        "title": _clean_text(item.get("title")) or "No title",
                        "url": url,
                        "snippet": _clean_text(item.get("content") or item.get("snippet")) or "No snippet",
                        "date": _clean_text(item.get("publishedDate") or item.get("date")) or "Unknown date",
                        "source": source or "Unknown source",
                    }
                )

            all_results = _dedupe_results(all_results)
            if len(all_results) >= max_results:
                return all_results[:max_results]

    return all_results[:max_results]


def run_duckduckgo_search(query: str, max_results: int = SEARCH_MAX_RESULTS) -> List[Dict[str, str]]:
    endpoints = [
        f"{DUCKDUCKGO_HTML_URL}?{urllib.parse.urlencode({'q': query, 'kl': 'us-en'})}",
        f"{DUCKDUCKGO_LITE_URL}?{urllib.parse.urlencode({'q': query, 'kl': 'us-en'})}",
    ]
    last_error: Exception | None = None
    best_results: List[Dict[str, str]] = []

    for attempt in range(DUCKDUCKGO_RETRIES):
        for endpoint in endpoints:
            _wait_for_ddg_slot()
            try:
                html_doc = _fetch_html(endpoint, timeout=DUCKDUCKGO_TIMEOUT)
                parser = _DuckDuckGoHTMLParser(max_results=max_results)
                parser.feed(html_doc)
                results = _rerank_by_query(query, _dedupe_results(parser.results))[:max_results]
                if len(results) > len(best_results):
                    best_results = results
                if len(results) >= max_results:
                    return results
            except DuckDuckGoRateLimitError as exc:
                last_error = exc
                continue
            except Exception as exc:
                last_error = exc
                continue

        # Exponential backoff after each full endpoint cycle.
        backoff = min(5.0, 0.6 * (2 ** attempt))
        time.sleep(backoff)

    # Fallback 1: route through SearXNG restricted to DuckDuckGo engine.
    try:
        data = _fetch_json(
            SEARXNG_URL,
            {"q": query, "format": "json", "engines": "duckduckgo"},
        )
        searx_ddg_results = _rerank_by_query(query, _parse_searx_results(data, max_results * 2))[:max_results]
        if len(searx_ddg_results) > len(best_results):
            best_results = searx_ddg_results
        if len(searx_ddg_results) >= max_results:
            return searx_ddg_results
    except Exception as exc:
        last_error = exc

    # Fallback 2: fetch via text mirror and parse DDG result links.
    try:
        proxy_url = f"{DUCKDUCKGO_PROXY_PREFIX}?{urllib.parse.urlencode({'q': query})}"
        proxy_markdown = _fetch_text(proxy_url, timeout=max(8.0, DUCKDUCKGO_TIMEOUT))
        proxy_results = _rerank_by_query(query, _parse_jina_ddg_markdown(proxy_markdown, max_results * 2))[:max_results]
        if len(proxy_results) > len(best_results):
            best_results = proxy_results
        if len(proxy_results) >= max_results:
            return proxy_results
    except Exception as exc:
        last_error = exc

    if best_results:
        return best_results
    if last_error is not None:
        raise RuntimeError(f"DuckDuckGo search failed after retries and fallback: {last_error}")
    return []


@tool
def searxng_search(query: str) -> str:
    """Search the web using the configured SearXNG endpoint."""
    try:
        results = run_searxng_search(query)
        if not results:
            return "No results found from SearXNG."
        return _format_search_results(results)
    except Exception as e:
        return f"SearXNG search error: {e}"


@tool
def duckduckgo_search(query: str) -> str:
    """Search the web using DuckDuckGo."""
    try:
        results = run_duckduckgo_search(query)
        if not results:
            return "No results found from DuckDuckGo."
        return _format_search_results(results)
    except Exception as e:
        return f"DuckDuckGo search error: {e}"


@tool
def web_search(query: str) -> str:
    """Search the web for general information. Uses SearXNG first, then DuckDuckGo as fallback."""
    errors = []

    try:
        searx_results = run_searxng_search(query)
        if searx_results:
            return _format_search_results(searx_results)
        errors.append("SearXNG returned no results.")
    except Exception as e:
        errors.append(f"SearXNG error: {e}")

    try:
        ddg_results = run_duckduckgo_search(query)
        if ddg_results:
            return _format_search_results(ddg_results)
        errors.append("DuckDuckGo returned no results.")
    except Exception as e:
        errors.append(f"DuckDuckGo error: {e}")

    return "Search error: " + " | ".join(errors)


@tool
def news_search(query: str) -> str:
    """Search for recent news articles. Returns headlines with dates, sources, and summaries."""
    try:
        results = run_searxng_news_search(query)
        if not results:
            return "No news results found."

        formatted = []
        for i, item in enumerate(results, 1):
            formatted.append(
                f"[{i}] {_clean_text(item.get('title')) or 'No title'}\n"
                f"    Date: {_clean_text(item.get('date')) or 'Unknown date'} | Source: {_clean_text(item.get('source')) or 'Unknown source'}\n"
                f"    URL: {_normalize_text(item.get('url')) or 'N/A'}\n"
                f"    {_clean_text(item.get('snippet')) or 'No snippet'}"
            )
        return "\n\n".join(formatted)
    except Exception as e:
        return f"News search error: {e}"


@tool
def python_repl(code: str) -> str:
    """Safely execute restricted Python code for calculations and small data checks."""
    try:
        _validate_python_code(code)
    except UnsafePythonCodeError as e:
        return f"Error: Unsafe code blocked. {e}"
    except Exception as e:
        return f"Error: Validation failed. {e}"

    return _run_safe_python_code(code)
