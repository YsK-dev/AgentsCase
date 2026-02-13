import os
import re
from typing import Any, Dict, List, Optional

from langchain_ollama import ChatOllama
from src.logger import get_logger, log_event

_log = get_logger("llm_engine")


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _complexity_score(task_text: str, query_type: str) -> float:
    text = (task_text or "").strip().lower()
    if not text:
        return 0.0

    score = 0.0
    token_estimate = len(text.split())
    if token_estimate > 25:
        score += 0.5
    if token_estimate > 80:
        score += 0.8
    if token_estimate > 180:
        score += 1.0

    hard_keywords = [
        "compare",
        "analysis",
        "analyze",
        "calculate",
        "forecast",
        "trend",
        "statistics",
        "debug",
        "optimize",
        "benchmark",
        "verify",
    ]
    if any(k in text for k in hard_keywords):
        score += 1.0

    if query_type in {"data", "news"}:
        score += 0.8

    return score


class RoutedChatModel:
    """
    Lightweight wrapper that tries multiple local models in order.
    If the primary model errors or returns empty content, fallback models are tried.
    """

    def __init__(
        self,
        candidate_models: List[str],
        temperature: float,
        extra_kwargs: Dict[str, Any],
    ):
        self.candidate_models = _dedupe_keep_order(candidate_models)
        self.temperature = temperature
        self.extra_kwargs = dict(extra_kwargs)
        self._tools = None
        self.last_used_model: Optional[str] = None

    def bind_tools(self, tools):
        clone = RoutedChatModel(
            candidate_models=self.candidate_models,
            temperature=self.temperature,
            extra_kwargs=self.extra_kwargs,
        )
        clone._tools = tools
        return clone

    def _build_model(self, model_name: str):
        llm = ChatOllama(model=model_name, temperature=self.temperature, **self.extra_kwargs)
        if self._tools is not None:
            llm = llm.bind_tools(self._tools)
        return llm

    def invoke(self, messages):
        errors = []
        for model_name in self.candidate_models:
            try:
                llm = self._build_model(model_name)
                response = llm.invoke(messages)

                content = getattr(response, "content", "")
                tool_calls = getattr(response, "tool_calls", None)
                has_tool_calls = bool(tool_calls)
                has_text = isinstance(content, str) and bool(content.strip())

                if has_tool_calls or has_text:
                    self.last_used_model = model_name
                    return response

                errors.append(f"{model_name}: empty response")
            except Exception as exc:
                _log.warning("Model %s failed: %s", model_name, exc)
                errors.append(f"{model_name}: {exc}")

        _log.error("All candidate models failed: %s", errors)
        raise RuntimeError("All candidate local models failed. " + " | ".join(errors))


class ModelSelector:
    """
    Dynamic local model selection based on role + task complexity.

    Modes:
    - fast: prefer smaller/faster model
    - accurate: prefer larger/more capable model
    - auto: choose based on task/query complexity
    """

    FAST_MODEL = os.getenv("MODEL_FAST", "deepseek-coder:1.3b")
    ACCURATE_MODEL = os.getenv("MODEL_ACCURATE", "qwen3:1.7b")
    BALANCED_MODEL = os.getenv("MODEL_BALANCED", FAST_MODEL)
    DEFAULT_MODE = os.getenv("MODEL_MODE", "auto").lower()

    # role -> (temperature, kwargs)
    ROLE_CONFIG = {
        "supervisor": (0.0, {"num_predict": 256}),
        "researcher": (0.3, {"num_predict": 512}),
        "analyst": (0.3, {"num_predict": 512}),
        "editor": (0.3, {"num_predict": 512}),
    }

    @staticmethod
    def _resolve_mode(mode: Optional[str]) -> str:
        selected = (mode or ModelSelector.DEFAULT_MODE or "auto").lower()
        if selected not in {"auto", "fast", "accurate"}:
            return "auto"
        return selected

    @staticmethod
    def _select_tier(agent_role: str, task_text: str, query_type: str, mode: str) -> str:
        if mode in {"fast", "accurate"}:
            return mode

        score = _complexity_score(task_text, query_type)
        role = agent_role.lower()
        if role in {"analyst", "editor"}:
            score += 0.4
        if role == "supervisor" and len((task_text or "").split()) > 25:
            score += 0.3

        if score >= 2.0:
            return "accurate"
        return "fast"

    @staticmethod
    def _candidate_models_for_tier(tier: str) -> List[str]:
        if tier == "accurate":
            return [
                ModelSelector.ACCURATE_MODEL,
                ModelSelector.BALANCED_MODEL,
                ModelSelector.FAST_MODEL,
            ]
        return [
            ModelSelector.FAST_MODEL,
            ModelSelector.BALANCED_MODEL,
            ModelSelector.ACCURATE_MODEL,
        ]

    @staticmethod
    def select_models(
        agent_role: str,
        task_text: str = "",
        query_type: str = "",
        mode: Optional[str] = None,
    ) -> List[str]:
        resolved_mode = ModelSelector._resolve_mode(mode)
        tier = ModelSelector._select_tier(agent_role, task_text, query_type, resolved_mode)
        return _dedupe_keep_order(ModelSelector._candidate_models_for_tier(tier))

    @staticmethod
    def get_model(
        agent_role: str,
        task_text: str = "",
        query_type: str = "",
        mode: Optional[str] = None,
    ) -> RoutedChatModel:
        role = agent_role.lower()
        temperature, kwargs = ModelSelector.ROLE_CONFIG.get(
            role,
            ModelSelector.ROLE_CONFIG["editor"],
        )
        candidates = ModelSelector.select_models(
            role,
            task_text=task_text,
            query_type=query_type,
            mode=mode,
        )
        log_event(_log, "model_selected", role=role, candidates=candidates,
                  query_type=query_type, temperature=temperature)
        return RoutedChatModel(candidates, temperature=temperature, extra_kwargs=kwargs)

    @staticmethod
    def get_model_name(
        agent_role: str,
        task_text: str = "",
        query_type: str = "",
        mode: Optional[str] = None,
    ) -> str:
        role = agent_role.lower()
        candidates = ModelSelector.select_models(
            role,
            task_text=task_text,
            query_type=query_type,
            mode=mode,
        )
        if not candidates:
            return ModelSelector.FAST_MODEL
        if len(candidates) == 1:
            return candidates[0]
        return f"{candidates[0]} (fallback: {candidates[1]})"
