# src/llm_engine.py
from langchain_ollama import ChatOllama


class ModelSelector:
    """
    Selects the best model for each agent role.

    Strategy:
    - supervisor : High reasoning → qwen3-4b-thinking (structured JSON routing)
    - researcher : Fast, lightweight → qwen3:1.7b (mostly bypassed for direct tool calls)
    - analyst   : Code generation  → deepseek-coder:1.3b
    - editor    : Quality reports  → qwen3-4b-thinking (best writing quality)
    """

    # (model_name, temperature, extra_kwargs)
    MODELS = {
        "supervisor": ("pielee/qwen3-4b-thinking-2507_q8:latest", 0.0, {}),
        "researcher": ("qwen3:1.7b", 0.3, {"num_predict": 512}),
        "analyst":    ("deepseek-coder:1.3b", 0.3, {}),
        "editor":     ("pielee/qwen3-4b-thinking-2507_q8:latest", 0.3, {}),
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
