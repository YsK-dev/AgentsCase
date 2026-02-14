# tests/test_agents.py — Unit tests for agent logic (heuristic classifier & fallback report)
from src.agents import _classify_query_heuristic, _build_fallback_report, _parse_research_items


# ─── Heuristic query classifier tests ───────────────────────────────────────

class TestClassifyQueryHeuristic:
    """Verify the deterministic keyword-based query router."""

    def test_news_keyword(self):
        agents, qtype, _ = _classify_query_heuristic("latest news on AI")
        assert qtype == "news"
        assert "researcher" in agents
        assert "editor" in agents

    def test_data_keyword_compare(self):
        agents, qtype, _ = _classify_query_heuristic("compare Python vs Java performance")
        assert qtype == "data"
        assert "analyst" in agents

    def test_data_keyword_analysis(self):
        agents, qtype, _ = _classify_query_heuristic("statistical analysis of housing prices")
        assert qtype == "data"

    def test_general_keyword_poem(self):
        agents, qtype, _ = _classify_query_heuristic("write a poem about nature")
        assert qtype == "general"
        assert agents == ["editor"]

    def test_general_keyword_joke(self):
        agents, qtype, _ = _classify_query_heuristic("tell me a joke")
        assert qtype == "general"

    def test_default_is_factual(self):
        agents, qtype, _ = _classify_query_heuristic("how does photosynthesis work")
        assert qtype == "factual"
        assert "researcher" in agents
        assert "editor" in agents

    def test_editor_always_last(self):
        """Editor must always be the last agent in the route."""
        for query in [
            "latest news", "compare data", "write a poem", "how does X work",
        ]:
            agents, _, _ = _classify_query_heuristic(query)
            assert agents[-1] == "editor", f"Editor not last for: {query}"

    def test_returns_three_elements(self):
        result = _classify_query_heuristic("anything")
        assert len(result) == 3
        agents, qtype, reason = result
        assert isinstance(agents, list)
        assert isinstance(qtype, str)
        assert isinstance(reason, str)

    def test_substring_not_matched_as_keyword(self):
        """'databases' should NOT match 'data' — word-boundary matching."""
        _, qtype, _ = _classify_query_heuristic("tell me about databases")
        assert qtype == "factual", "Substring 'data' inside 'databases' should not trigger data route"

    def test_word_boundary_data_match(self):
        """The standalone word 'data' should still match."""
        _, qtype, _ = _classify_query_heuristic("show me the data for sales")
        assert qtype == "data"


# ─── _parse_research_items tests ────────────────────────────────────────────

class TestParseResearchItems:

    def test_parses_numbered_items(self):
        research = (
            "[1] First Title\n"
            "    URL: http://example.com/1\n"
            "    Date: 2026-01-01 | Source: Example\n"
            "    Some snippet text.\n\n"
            "[2] Second Title\n"
            "    URL: http://example.com/2\n"
            "    Another snippet.\n"
        )
        items = _parse_research_items(research)
        assert len(items) == 2
        assert items[0]["title"] == "First Title"
        assert items[0]["url"] == "http://example.com/1"
        assert items[1]["title"] == "Second Title"

    def test_empty_input(self):
        assert _parse_research_items("") == []

    def test_max_items_limit(self):
        lines = ""
        for i in range(1, 20):
            lines += f"[{i}] Title {i}\n    URL: http://example.com/{i}\n\n"
        items = _parse_research_items(lines, max_items=3)
        assert len(items) == 3


# ─── _build_fallback_report tests ───────────────────────────────────────────

class TestBuildFallbackReport:

    def test_news_report_has_headlines_header(self):
        research = "[1] Breaking News\n    URL: http://n.com\n    A snippet.\n"
        report = _build_fallback_report("AI news", "news", research, "")
        assert "Key Headlines" in report

    def test_factual_report_has_findings_header(self):
        research = "[1] Some Fact\n    URL: http://f.com\n    Detail.\n"
        report = _build_fallback_report("what is X", "factual", research, "")
        assert "Findings" in report

    def test_includes_analysis_when_provided(self):
        report = _build_fallback_report("topic", "data", "", "Analysis content here")
        assert "Analysis" in report

    def test_handles_empty_research(self):
        report = _build_fallback_report("topic", "general", "", "")
        assert "No structured results" in report

    def test_preserves_urls(self):
        research = "[1] Title\n    URL: http://example.com/test\n    Snip.\n"
        report = _build_fallback_report("topic", "factual", research, "")
        assert "http://example.com/test" in report
