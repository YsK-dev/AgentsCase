# tests/test_tools.py — Unit tests for code execution safety & tool helpers
import pytest
import src.tools as tools_module

from src.tools import (
    _validate_python_code,
    _run_safe_python_code,
    _clean_text,
    _normalize_text,
    _dedupe_results,
    _format_search_results,
    UnsafePythonCodeError,
)


# ─── _validate_python_code: normal cases ────────────────────────────────────

class TestValidatePythonCodeNormal:
    """Tests that safe code passes validation."""

    def test_simple_math(self):
        _validate_python_code("print(1 + 2)")

    def test_safe_import_math(self):
        _validate_python_code("import math\nprint(math.sqrt(16))")

    def test_safe_import_json(self):
        _validate_python_code("import json\nprint(json.dumps({'a': 1}))")

    def test_safe_import_statistics(self):
        _validate_python_code("import statistics\nprint(statistics.mean([1,2,3]))")

    def test_multiline_safe_code(self):
        code = "x = [1, 2, 3]\ny = sum(x)\nprint(y)"
        _validate_python_code(code)


# ─── _validate_python_code: error / blocked cases ───────────────────────────

class TestValidatePythonCodeBlocked:
    """Tests that unsafe code is rejected."""

    def test_empty_code_rejected(self):
        with pytest.raises(UnsafePythonCodeError, match="empty"):
            _validate_python_code("")

    def test_whitespace_only_rejected(self):
        with pytest.raises(UnsafePythonCodeError, match="empty"):
            _validate_python_code("   \n  ")

    def test_import_os_blocked(self):
        with pytest.raises(UnsafePythonCodeError, match="not allowed"):
            _validate_python_code("import os")

    def test_import_subprocess_blocked(self):
        with pytest.raises(UnsafePythonCodeError, match="not allowed"):
            _validate_python_code("import subprocess")

    def test_import_sys_blocked(self):
        with pytest.raises(UnsafePythonCodeError, match="not allowed"):
            _validate_python_code("import sys")

    def test_from_os_blocked(self):
        with pytest.raises(UnsafePythonCodeError, match="not allowed"):
            _validate_python_code("from os import path")

    def test_eval_blocked(self):
        with pytest.raises(UnsafePythonCodeError, match="blocked"):
            _validate_python_code("eval('1+1')")

    def test_exec_blocked(self):
        with pytest.raises(UnsafePythonCodeError, match="blocked"):
            _validate_python_code("exec('print(1)')")

    def test_open_blocked(self):
        with pytest.raises(UnsafePythonCodeError, match="blocked"):
            _validate_python_code("open('/etc/passwd')")

    def test_dunder_builtins_blocked(self):
        with pytest.raises(UnsafePythonCodeError, match="blocked"):
            _validate_python_code("x = __builtins__")

    def test_dunder_import_blocked(self):
        with pytest.raises(UnsafePythonCodeError, match="blocked"):
            _validate_python_code("x = __import__('os')")

    def test_syntax_error_rejected(self):
        with pytest.raises(UnsafePythonCodeError, match="Syntax error"):
            _validate_python_code("def foo(")

    def test_code_too_long(self):
        long_code = "x = 1\n" * 5000
        with pytest.raises(UnsafePythonCodeError, match="too long"):
            _validate_python_code(long_code)


# ─── _run_safe_python_code: execution tests ─────────────────────────────────

class TestRunSafePythonCode:
    """Tests that the sandboxed runner produces correct output."""

    def test_simple_print(self):
        result = _run_safe_python_code("print('hello')")
        assert "hello" in result

    def test_math_calculation(self):
        result = _run_safe_python_code("print(2 ** 10)")
        assert "1024" in result

    def test_import_math_in_runner(self):
        result = _run_safe_python_code("import math\nprint(math.pi)")
        assert "3.14" in result

    def test_runtime_error_captured(self):
        result = _run_safe_python_code("print(1 / 0)")
        assert "Error" in result or "ZeroDivision" in result

    def test_blocked_import_in_runner(self):
        result = _run_safe_python_code("import os\nprint(os.getcwd())")
        assert "Error" in result or "not allowed" in result


# ─── Helper function tests ──────────────────────────────────────────────────

class TestSearchHelpers:
    """Tests for text cleaning and result formatting utilities."""

    def test_normalize_text_none(self):
        assert _normalize_text(None) == ""

    def test_normalize_text_strips(self):
        assert _normalize_text("  hello  ") == "hello"

    def test_clean_text_html_entities(self):
        assert "hello & world" in _clean_text("hello &amp; world")

    def test_dedupe_results(self):
        items = [
            {"url": "http://a.com", "title": "A"},
            {"url": "http://a.com", "title": "A dup"},
            {"url": "http://b.com", "title": "B"},
        ]
        deduped = _dedupe_results(items)
        assert len(deduped) == 2
        assert deduped[0]["title"] == "A"
        assert deduped[1]["title"] == "B"

    def test_format_search_results_empty(self):
        assert _format_search_results([]) == "No results found."

    def test_format_search_results_single(self):
        items = [{"title": "Test", "url": "http://t.com", "snippet": "A snippet"}]
        result = _format_search_results(items)
        assert "[1]" in result
        assert "Test" in result
        assert "http://t.com" in result


class TestNewsSearchFallback:
    """Tests fallback behavior for news search when SearXNG is unavailable."""

    def test_news_search_falls_back_to_duckduckgo(self, monkeypatch):
        def _searx_fail(*args, **kwargs):
            raise RuntimeError("[Errno 61] Connection refused")

        def _ddg_ok(*args, **kwargs):
            return [
                {
                    "title": "Istanbul update",
                    "url": "https://example.com/news/istanbul",
                    "snippet": "Latest local developments.",
                }
            ]

        monkeypatch.setattr(tools_module, "run_searxng_news_search", _searx_fail)
        monkeypatch.setattr(tools_module, "run_duckduckgo_search", _ddg_ok)

        result = tools_module.news_search.invoke({"query": "Istanbul news"})

        assert "[1] Istanbul update" in result
        assert "Source: example.com" in result
        assert "URL: https://example.com/news/istanbul" in result
