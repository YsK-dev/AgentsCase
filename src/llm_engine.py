from langchain_ollama import Chatollama

class ModelSelector:
    """
    Docstring for ModelSelector
    Selects the best model for the task 
    How selects(Stategy):
    - Supervisor: Needs high reasoning and low hallucination like qwen3:8b
    - Reearcher: Needs speed for reading phi4-mini:latest or other small models
    - Analyst: needs code generation like deepseek coder
    """
    @staticmethod
    def get_model(agent_role: str):
        if agent_role == "Supervisor":
            return Chatollama(model="qwen3:8b")
        elif agent_role == "Researcher":
            return Chatollama(model="phi4-mini:latest")
        elif agent_role == "Analyst":
            return Chatollama(model="deepseek-coder:1.3b")
        else:
            # fallback to qwen since it is the most balanced model
            return Chatollama(model="qwen3:8b")
        
 
        

