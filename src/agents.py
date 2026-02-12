# src/agents.py
import json
from colorama import Fore, Style
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from src.llm_engine import ModelSelector
from src.tools import web_search, news_search, python_repl

# ─── Prompts ────────────────────────────────────────────────────────────────

SUPERVISOR_PROMPT = """You are a task router. Given a user query, decide which agents are needed.

Reply with ONLY a JSON object (no markdown, no explanation) in this exact format:
{"agents": ["researcher", "editor"], "query_type": "news", "reason": "short reason"}

Available agents:
- "researcher" — searches the web for current information, news, facts
- "analyst"    — runs Python code for calculations, data analysis, comparisons
- "editor"     — synthesizes information into a polished final answer

query_type must be one of: "news", "factual", "data", "general"

Rules:
- For news/current events queries: ["researcher", "editor"], query_type: "news"
- For factual/how-to queries: ["researcher", "editor"], query_type: "factual"
- For data/comparison/calculation queries: ["researcher", "analyst", "editor"], query_type: "data"
- For opinion/creative/general knowledge: ["editor"], query_type: "general"
- Always include "editor" as the last agent
- Return ONLY valid JSON, nothing else"""

RESEARCHER_PROMPT = """You are a research agent. You have two search tools:
- news_search: Use this for news, current events, recent happenings, headlines
- web_search: Use this for general factual information, how-to, definitions, etc.

Pick the RIGHT tool for the query. For news queries, ALWAYS use news_search.
Be thorough. Return factual findings with sources. Do NOT make up information."""

ANALYST_PROMPT = """You are a data analyst. Analyze the provided data using Python code when calculations are needed.
Focus on extracting numbers, trends, and comparisons. Be precise and data-driven."""

EDITOR_PROMPT = """You are a senior editor. Write a clear, concise, professional answer to the user's question.
Use the research data and analysis provided. Structure your response well.
Do NOT mention internal agent processes or tools. Just deliver the answer naturally."""


# ─── Supervisor ─────────────────────────────────────────────────────────────

def supervisor_agent(state: dict) -> dict:
    """Routes the query to the right agents based on query classification."""
    topic = state["topic"]
    model_name = ModelSelector.get_model_name("supervisor")

    print(f"\n{Fore.MAGENTA}┌─ SUPERVISOR ({model_name}){Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}│  Classifying query...{Style.RESET_ALL}")

    llm = ModelSelector.get_model("supervisor")
    messages = [
        SystemMessage(content=SUPERVISOR_PROMPT),
        HumanMessage(content=f"User query: {topic}"),
    ]

    try:
        response = llm.invoke(messages)
        content = response.content.strip()

        # Try to extract JSON from the response (handle thinking tags, markdown, etc.)
        json_str = content
        # Remove <think>...</think> blocks if present
        if "<think>" in json_str:
            import re
            json_str = re.sub(r"<think>.*?</think>", "", json_str, flags=re.DOTALL).strip()
        # Try to find JSON in the remaining text
        if "{" in json_str:
            start = json_str.index("{")
            end = json_str.rindex("}") + 1
            json_str = json_str[start:end]

        decision = json.loads(json_str)
        agents = decision.get("agents", ["researcher", "editor"])
        query_type = decision.get("query_type", "factual")
        reason = decision.get("reason", "")

        print(f"{Fore.MAGENTA}│  Route: {' → '.join(agents)}{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}│  Query type: {query_type}{Style.RESET_ALL}")
        if reason:
            print(f"{Fore.MAGENTA}│  Reason: {reason}{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}└──────────────────────────────{Style.RESET_ALL}")

    except Exception as e:
        print(f"{Fore.RED}│  Routing failed ({e}), using default pipeline{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}└──────────────────────────────{Style.RESET_ALL}")
        agents = ["researcher", "editor"]
        query_type = "factual"

    return {"route": agents, "query_type": query_type}


# ─── Researcher ─────────────────────────────────────────────────────────────

def research_agent(state: dict) -> dict:
    """Searches the web for information about the topic."""
    topic = state["topic"]
    query_type = state.get("query_type", "factual")
    model_name = ModelSelector.get_model_name("researcher")

    print(f"\n{Fore.YELLOW}┌─ RESEARCHER ({model_name}){Style.RESET_ALL}")

    llm = ModelSelector.get_model("researcher").bind_tools([web_search, news_search])

    # Tailor the instruction based on query type from supervisor
    if query_type == "news":
        instruction = f"Use the news_search tool to find the latest news about: {topic}"
    else:
        instruction = f"Use the web_search tool to find information about: {topic}"

    messages = [
        SystemMessage(content=RESEARCHER_PROMPT),
        HumanMessage(content=instruction),
    ]

    ai_msg = llm.invoke(messages)
    messages.append(ai_msg)

    # Tool dispatch map for researcher
    _research_tools = {"web_search": web_search, "news_search": news_search}

    if ai_msg.tool_calls:
        for tc in ai_msg.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            print(f"{Fore.YELLOW}│  🔧 Tool: {tool_name}({json.dumps(tool_args)}){Style.RESET_ALL}")

            selected_tool = _research_tools.get(tool_name, web_search)
            result = selected_tool.invoke(tool_args)
            print(f"{Fore.YELLOW}│  📄 Got {len(str(result))} chars of results{Style.RESET_ALL}")

            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

        final = llm.invoke(messages)
        output = final.content
    else:
        # Fallback: if LLM didn't call a tool, do it ourselves based on query_type
        print(f"{Fore.YELLOW}│  (LLM skipped tool call — executing {query_type} search directly){Style.RESET_ALL}")
        if query_type == "news":
            result = news_search.invoke({"query": topic})
        else:
            result = web_search.invoke({"query": topic})
        print(f"{Fore.YELLOW}│  📄 Got {len(str(result))} chars of results{Style.RESET_ALL}")
        output = str(result)

    print(f"{Fore.YELLOW}└──────────────────────────────{Style.RESET_ALL}")
    return {"research_output": output}


# ─── Analyst ────────────────────────────────────────────────────────────────

def analyst_agent(state: dict) -> dict:
    """Analyzes data with Python code execution."""
    topic = state["topic"]
    research_data = state.get("research_output", "No data provided.")
    model_name = ModelSelector.get_model_name("analyst")

    print(f"\n{Fore.BLUE}┌─ ANALYST ({model_name}){Style.RESET_ALL}")

    llm = ModelSelector.get_model("analyst").bind_tools([python_repl])
    messages = [
        SystemMessage(content=ANALYST_PROMPT),
        HumanMessage(
            content=f"Topic: {topic}\n\nResearch data:\n{research_data}"
        ),
    ]

    ai_msg = llm.invoke(messages)
    messages.append(ai_msg)

    if ai_msg.tool_calls:
        for tc in ai_msg.tool_calls:
            tool_name = tc["name"]
            print(f"{Fore.BLUE}│  🔧 Tool: {tool_name}(){Style.RESET_ALL}")

            result = python_repl.invoke(tc["args"])
            print(f"{Fore.BLUE}│  📊 Code output: {str(result)[:120]}{Style.RESET_ALL}")

            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

        final = llm.invoke(messages)
        output = final.content
    else:
        print(f"{Fore.BLUE}│  (No code execution needed){Style.RESET_ALL}")
        output = ai_msg.content

    print(f"{Fore.BLUE}└──────────────────────────────{Style.RESET_ALL}")
    return {"analysis_output": output}


# ─── Editor ─────────────────────────────────────────────────────────────────

def editor_agent(state: dict) -> dict:
    """Synthesizes all gathered information into a final answer."""
    topic = state["topic"]
    research = state.get("research_output", "")
    analysis = state.get("analysis_output", "")
    model_name = ModelSelector.get_model_name("editor")

    print(f"\n{Fore.GREEN}┌─ EDITOR ({model_name}){Style.RESET_ALL}")
    print(f"{Fore.GREEN}│  Writing final report...{Style.RESET_ALL}")

    llm = ModelSelector.get_model("editor")

    context_parts = [f"User question: {topic}"]
    if research:
        context_parts.append(f"Research findings:\n{research}")
    if analysis:
        context_parts.append(f"Data analysis:\n{analysis}")

    messages = [
        SystemMessage(content=EDITOR_PROMPT),
        HumanMessage(content="\n\n".join(context_parts)),
    ]

    response = llm.invoke(messages)
    print(f"{Fore.GREEN}└──────────────────────────────{Style.RESET_ALL}")
    return {"final_report": response.content}
