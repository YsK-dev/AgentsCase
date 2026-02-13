# src/llm_engine.py
import os

from langchain_ollama import ChatOllama


class ModelSelector:
    """
    Selects the best model for each agent role.

    Strategy:
    - supervisor : Structured routing → deepseek-coder:1.3b
    - researcher : Fast tool orchestration → deepseek-coder:1.3b
    - analyst   : Code generation  → deepseek-coder:1.3b
    - editor    : Stable text generation → deepseek-coder:1.3b
    """

    # (model_name, temperature, extra_kwargs)
    SUPERVISOR_MODEL = os.getenv("MODEL_SUPERVISOR", "deepseek-coder:1.3b")
    RESEARCHER_MODEL = os.getenv("MODEL_RESEARCHER", "deepseek-coder:1.3b")
    ANALYST_MODEL = os.getenv("MODEL_ANALYST", "deepseek-coder:1.3b")
    EDITOR_MODEL = os.getenv(
        "MODEL_EDITOR",
        "deepseek-coder:1.3b",
    )

    MODELS = {
        "supervisor": (SUPERVISOR_MODEL, 0.0, {"num_predict": 256}),
        "researcher": (RESEARCHER_MODEL, 0.3, {"num_predict": 512}),
        "analyst":    (ANALYST_MODEL, 0.3, {"num_predict": 512}),
        "editor":     (EDITOR_MODEL, 0.3, {"num_predict": 512}),
    }

    @staticmethod
    def get_model(agent_role: str) -> ChatOllama:
        role = agent_role.lower()
        model_name, temp, extra = ModelSelector.MODELS.get(
            role,
            ModelSelector.MODELS["editor"],  # fallback
        )
        return ChatOllama(model=model_name, temperature=temp, **extra)

    @staticmethod
    def get_model_name(agent_role: str) -> str:
        """Return the model name string for display purposes."""
        role = agent_role.lower()
        model_name, _, _ = ModelSelector.MODELS.get(
            role,
            ModelSelector.MODELS["editor"],
        )
        return model_name
