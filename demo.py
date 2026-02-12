import sys
import time
from colorama import Fore, Style, init

# --- LangChain Imports ---
from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Initialize Colors for CLI
init(autoreset=True)

# --- CONFIGURATION: EDIT THIS IF YOU HAVE DIFFERENT MODELS ---
# We use different models to demonstrate "Model Selection Strategy"
MODEL_FAST = "qwen3:1.7b"                          # Small, fast model for research
MODEL_SMART = "pielee/qwen3-4b-thinking-2507_q8:latest"  # Capable model for reasoning
MODEL_CODER = "deepseek-coder:1.3b"                # Code generation model

print(f"{Fore.CYAN}Initializing Agents with models: {MODEL_FAST}, {MODEL_SMART}, {MODEL_CODER}...{Style.RESET_ALL}")

# --- TOOLS ---
def perform_web_search(query):
    """Real web search using DuckDuckGo"""
    try:
        search = DuckDuckGoSearchRun()
        return search.run(query)
    except Exception as e:
        return f"Error connecting to search: {e}"

def execute_python_code(code_snippet):
    """Simulated Python Execution (Safety First for Demo)"""
    # In a real deployed app, use langchain_experimental.utilities.PythonREPL
    # Here we just mock the execution to prevent your computer from running arbitrary LLM code during a test.
    return f"Simulated execution of: {code_snippet[:30]}... [Result: Data Processed Successfully]"

# --- AGENT DEFINITIONS ---

class Agent:
    def __init__(self, name, role, model_name, color):
        self.name = name
        self.role = role
        self.model = ChatOllama(model=model_name, temperature=0)
        self.color = color
        print(f"[{self.color}✓{Style.RESET_ALL}] {self.name} ({self.role}) loaded on {model_name}")

    def think(self, prompt, context=""):
        print(f"\n{self.color}{self.name} is thinking...{Style.RESET_ALL}")
        chain = ChatPromptTemplate.from_template(prompt) | self.model | StrOutputParser()
        try:
            # If context is a dict, use it directly; otherwise wrap in "context" key
            if isinstance(context, dict):
                response = chain.invoke(context)
            else:
                response = chain.invoke({"context": context})
            return response
        except Exception as e:
            return f"Error: {e}. Is Ollama running?"

# --- THE AGENT SQUAD ---

# 1. Researcher: Uses the FAST model.
researcher = Agent(
    name="Researcher",
    role="Information Gathering",
    model_name=MODEL_FAST,
    color=Fore.YELLOW
)

# 2. Analyst: Uses the CODER model.
analyst = Agent(
    name="Analyst",
    role="Data Processing & Code",
    model_name=MODEL_CODER,
    color=Fore.BLUE
)

# 3. Editor: Uses the SMART model.
editor = Agent(
    name="Editor",
    role="Final Reporting",
    model_name=MODEL_SMART,
    color=Fore.GREEN
)

# --- MAIN WORKFLOW SYSTEM ---

def run_workflow(user_query):
    print(f"\n{Fore.MAGENTA}{'='*60}")
    print(f"TASK: {user_query}")
    print(f"{'='*60}{Style.RESET_ALL}")

    # STEP 1: RESEARCHER
    # The researcher decides what to search for based on the user query.
    search_query = researcher.think(
        prompt="You are a research assistant. Based on this user request: '{context}', "
               "generate a single, optimal search query for a search engine. Output ONLY the query, nothing else.",
        context=user_query
    )
    
    print(f"{Fore.YELLOW} > Searching Web for: '{search_query.strip()}'...{Style.RESET_ALL}")
    search_results = perform_web_search(search_query)
    print(f"{Fore.YELLOW} > Found {len(search_results)} characters of data.{Style.RESET_ALL}")

    # STEP 2: ANALYST
    # The analyst takes the search data and extracts structured insights or writes code.
    analyst_insight = analyst.think(
        prompt="You are a Data Analyst. Here is raw text from a web search: \n\n{context}\n\n"
               "Extract the key numbers, dates, and hard facts. "
               "Then, write a short Python snippet that COULD visualize this data (do not execute it).",
        context=search_results
    )
    print(f"{Fore.BLUE} > Analyst has processed the data and generated code.{Style.RESET_ALL}")

    # STEP 3: EDITOR
    # The editor combines everything into the final answer.
    final_report = editor.think(
        prompt="You are a Senior Editor. The user asked: '{original_query}'.\n"
               "Here is the research data: {research}\n"
               "Here is the analyst's technical breakdown: {analyst}\n\n"
               "Write a clear, concise, and professional answer to the user.",
        context={"original_query": user_query, "research": search_results[:500], "analyst": analyst_insight}
    )

    # --- FINAL OUTPUT ---
    print(f"\n{Fore.MAGENTA}{'='*60}")
    print(f"FINAL OUTPUT")
    print(f"{'='*60}{Style.RESET_ALL}")
    print(final_report)

# --- CLI ENTRY POINT ---
if __name__ == "__main__":
    while True:
        try:
            user_input = input(f"\n{Fore.WHITE}Enter your request (or 'q' to quit): {Style.RESET_ALL}")
            if user_input.lower() in ['q', 'quit', 'exit']:
                print("Goodbye!")
                break
            
            run_workflow(user_input)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break