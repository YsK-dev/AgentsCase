from langchain_ollama import ChatOllama
class ModelSelector:
    """
    Docstring for ModelSelector
    Selects the best model for the task 
    How selects(Strategy):
    - Supervisor: Needs high reasoning and low hallucination like pielee/qwen3-4b-thinking-2507_q8:latest
    - Researcher: Needs speed for light fast text like qwen3:1.7b
    - Analyst: needs code generation like deepseek-coder:1.3b
    """
    @staticmethod
    def get_model(agent_role: str):
        if agent_role == "Supervisor":
            return ChatOllama(model="pielee/qwen3-4b-thinking-2507_q8:latest",temperature=0)#temp 0 to make it deterministic
        elif agent_role == "Researcher":
            return ChatOllama(model="qwen3:1.7b", temperature=0.4)
        elif agent_role == "Analyst":
            return ChatOllama(model="deepseek-coder:1.3b", temperature=0.4)
        else:
            # fallback to qwen since it is the most balanced model
            return ChatOllama(model="pielee/qwen3-4b-thinking-2507_q8:latest", temperature=0)
        
 
        

