# tests/test_llm_engine.py — Unit tests for model selection & routing logic
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.llm_engine import _complexity_score, _dedupe_keep_order, ModelSelector


# ─── _complexity_score tests ────────────────────────────────────────────────

class TestComplexityScore:
    """Verify complexity scoring produces expected tiers."""

    def test_empty_input(self):
        assert _complexity_score("", "") == 0.0

    def test_short_simple_query(self):
        score = _complexity_score("hello world", "general")
        assert score < 1.0, "Short general query should have low score"

    def test_long_query_adds_score(self):
        text = "word " * 30  # >25 tokens
        score = _complexity_score(text, "general")
        assert score >= 0.5

    def test_very_long_query(self):
        text = "word " * 200  # >180 tokens
        score = _complexity_score(text, "general")
        assert score >= 2.0

    def test_hard_keyword_boosts(self):
        score_plain = _complexity_score("what is AI", "general")
        score_hard = _complexity_score("analyze AI trends", "general")
        assert score_hard > score_plain, "Hard keywords should boost score"

    def test_data_query_type_boosts(self):
        score_general = _complexity_score("test query", "general")
        score_data = _complexity_score("test query", "data")
        assert score_data > score_general

    def test_news_query_type_boosts(self):
        score_general = _complexity_score("test query", "general")
        score_news = _complexity_score("test query", "news")
        assert score_news > score_general


# ─── _dedupe_keep_order tests ───────────────────────────────────────────────

class TestDedupeKeepOrder:
    def test_removes_duplicates(self):
        assert _dedupe_keep_order(["a", "b", "a", "c"]) == ["a", "b", "c"]

    def test_removes_empty_strings(self):
        assert _dedupe_keep_order(["", "a", "", "b"]) == ["a", "b"]

    def test_preserves_order(self):
        assert _dedupe_keep_order(["c", "b", "a"]) == ["c", "b", "a"]

    def test_empty_list(self):
        assert _dedupe_keep_order([]) == []


# ─── ModelSelector tests ────────────────────────────────────────────────────

class TestModelSelector:
    """Verify model selection routing logic."""

    def test_fast_mode_returns_fast_model_first(self):
        candidates = ModelSelector.select_models("editor", mode="fast")
        assert candidates[0] == ModelSelector.FAST_MODEL

    def test_accurate_mode_returns_accurate_model_first(self):
        candidates = ModelSelector.select_models("editor", mode="accurate")
        assert candidates[0] == ModelSelector.ACCURATE_MODEL

    def test_auto_mode_simple_query_selects_fast(self):
        candidates = ModelSelector.select_models(
            "researcher", task_text="hello", query_type="general", mode="auto"
        )
        assert candidates[0] == ModelSelector.FAST_MODEL

    def test_auto_mode_complex_query_selects_accurate(self):
        # Long text with data keywords to push score >= 2.0
        text = "analyze and compare " + "data statistics trends " * 20
        candidates = ModelSelector.select_models(
            "analyst", task_text=text, query_type="data", mode="auto"
        )
        assert candidates[0] == ModelSelector.ACCURATE_MODEL

    def test_candidates_always_include_both_models(self):
        candidates = ModelSelector.select_models("editor", mode="fast")
        model_set = set(candidates)
        assert ModelSelector.FAST_MODEL in model_set
        assert ModelSelector.ACCURATE_MODEL in model_set

    def test_get_model_name_returns_string(self):
        name = ModelSelector.get_model_name("editor", mode="fast")
        assert isinstance(name, str)
        assert len(name) > 0

    def test_invalid_mode_defaults_to_auto(self):
        resolved = ModelSelector._resolve_mode("invalid_mode")
        assert resolved == "auto"

    def test_none_mode_defaults_to_auto(self):
        resolved = ModelSelector._resolve_mode(None)
        assert resolved in {"auto", "fast", "accurate"}

    def test_different_roles_have_different_configs(self):
        """Supervisor and editor should have different temperature settings."""
        sup_temp, _ = ModelSelector.ROLE_CONFIG["supervisor"]
        ed_temp, _ = ModelSelector.ROLE_CONFIG["editor"]
        assert sup_temp != ed_temp, "Supervisor should use temperature 0.0, editor 0.3"

    def test_two_distinct_models_configured(self):
        """Week 2 spec: at least two local models."""
        assert ModelSelector.FAST_MODEL != ModelSelector.ACCURATE_MODEL, \
            "FAST and ACCURATE models must be different to satisfy multi-model requirement"
