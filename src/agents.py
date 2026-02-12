from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from src.llm_engine import ModelSelector
from src.tools import search, python_repl  # Import the correct function names

# Prompts for each agent
RESEARCHER_PROMPT = """You are a research agent. Your task is to search for detailed information about a given topic.
Use the search tool to find relevant information. Be thorough and comprehensive in your research.
Always cite your sources and provide factual, up-to-date information."""

ANALYST_PROMPT = """You are an analyst agent. Your task is to analyze data and provide insights.
Use the Python REPL tool to perform calculations, data analysis, and generate visualizations when needed.
Provide clear, data-driven analysis with supporting evidence."""

EDITOR_PROMPT = """You are an editor agent. Your task is to synthesize research and analysis into a coherent final report.
Combine the research findings and analysis into a well-structured, comprehensive report.
Ensure the report is clear, professional, and actionable."""

def research_agent(state):
    """
    Uses Phi-4 mini to search the web.
    Implements a loop to handle tool calls and tool outputs.
    """
    print("="*25, " RESEARCHER WORKING ", "="*25)
    
    topic = state.get("topic")
    # Bind the specific tool this agent needs
    llm = ModelSelector.get_model("researcher").bind_tools([search])
    
    # Initial conversation state
    messages = [
        SystemMessage(content=RESEARCHER_PROMPT),
        HumanMessage(content=f"Find detailed information about: {topic}")
    ]
    
    # 1. First invocation: See if the LLM wants to search
    ai_msg = llm.invoke(messages)
    messages.append(ai_msg) # Add AI's intent to history

    # 2. Check for tool calls
    if ai_msg.tool_calls:
        print(f"--- Tool Call Detected: {len(ai_msg.tool_calls)} call(s) ---")
        
        # Loop through all requested tool calls
        for tool_call in ai_msg.tool_calls:
            selected_tool = {"search": search}[tool_call["name"].lower()]
            tool_output = selected_tool.invoke(tool_call["args"])
            
            print(f"--- Executing {tool_call['name']} ---")
            
            # Append the result as a ToolMessage
            messages.append(ToolMessage(
                content=str(tool_output),
                tool_call_id=tool_call["id"]
            ))
            
        # 3. Second invocation: Get the final summary based on tool outputs
        final_response = llm.invoke(messages)
        return {"research_output": final_response.content}
    
    # If no tool was called, return the initial response
    return {"research_output": ai_msg.content}

def analyst_agent(state):
    """
    Uses DeepSeek-Coder to analyze data.
    """
    print("="*25, " ANALYST WORKING ", "="*25)
    
    # Get previous research to analyze
    research_data = state.get("research_output", "No data provided.")
    topic = state.get("topic")
    
    llm = ModelSelector.get_model("analyst").bind_tools([python_repl])
    
    messages = [
        SystemMessage(content=ANALYST_PROMPT),
        HumanMessage(content=f"Topic: {topic}\n\nData found by researcher:\n{research_data}")
    ]
    
    ai_msg = llm.invoke(messages)
    messages.append(ai_msg)

    # Tool execution loop
    if ai_msg.tool_calls:
        print(f"--- Analyst generating code ---")
        for tool_call in ai_msg.tool_calls:
            # We assume python_repl is the only tool here
            tool_output = python_repl.invoke(tool_call["args"])
            
            messages.append(ToolMessage(
                content=str(tool_output),
                tool_call_id=tool_call["id"]
            ))
            
        final_response = llm.invoke(messages)
        return {"analysis_output": final_response.content}

    return {"analysis_output": ai_msg.content}

def editor_agent(state):
    """
    Uses a capable model to synthesize the final report.
    """
    print("="*25, " EDITOR WORKING ", "="*25)
    
    research_data = state.get("research_output", "No research data.")
    analysis_data = state.get("analysis_output", "No analysis data.")
    topic = state.get("topic")
    
    llm = ModelSelector.get_model("editor")
    
    messages = [
        SystemMessage(content=EDITOR_PROMPT),
        HumanMessage(content=f"Topic: {topic}\n\nResearch Data:\n{research_data}\n\nAnalysis Data:\n{analysis_data}")
    ]
    
    final_report = llm.invoke(messages)
    return {"final_report": final_report.content}