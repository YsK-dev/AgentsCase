from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import tool 


class ToolSelector:
    @staticmethod
    def get_tools(agent_role: str):
        if agent_role == "Supervisor":
            return [search, python_repl]
        elif agent_role == "Researcher":
            return [search]
        elif agent_role == "Analyst":
            return [python_repl]
        else:
            return [search, python_repl]

# web search tool
@tool
def search(query: str):
    """Search the web for information."""
    search = DuckDuckGoSearchRun()
    s = search.run(query)
    return s

# python repl tool
@tool
def python_repl(code: str):
    """
    Run Python code and return the output.
    Usefull for executing python code to calculate things or generate data. input should be a string of python code.
    """
    repl = PythonREPL()
    try:
        result = repl.run(code)
    except Exception as e:
        result = f"Error: {e}"      
    return result 

