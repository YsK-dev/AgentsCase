# src/agents.py
import json
import os
import re
from datetime import datetime
from typing import Dict, List
from colorama import Fore, Style
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from src.llm_engine import ModelSelector
from src.tools import web_search, news_search, python_repl, searxng_search, duckduckgo_search
from src.logger import get_logger, log_event

_log = get_logger("agents")

# ─── Prompts ────────────────────────────────────────────────────────────────

SUPERVISOR_PROMPT = """/no_think
You are a task router. Given a user query, decide which agents are needed.

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

RESEARCHER_PROMPT = """/no_think
You are a research agent with search tools. Pick the RIGHT tool:
- news_search: for news, current events, headlines
- searxng_search: for private/general web search via SearXNG
- duckduckgo_search: for DuckDuckGo web search
- web_search: default search (SearXNG first, DuckDuckGo fallback)

Return findings with sources. Do NOT make up information. Be brief and factual."""

ANALYST_PROMPT = """/no_think
You are a data analyst. Analyze data using Python code when calculations are needed.
Focus on numbers, trends, and comparisons. Be precise."""

# ─── Query-type specific editor prompts ─────────────────────────────────────

EDITOR_PROMPT_NEWS = """/no_think
You are a professional news editor. Write a comprehensive news report based on the research data provided.

FORMAT YOUR REPORT EXACTLY LIKE THIS:

# [Topic] — News Report
**Date:** [today's date]

## Key Headlines

1. **[Headline]** — [1-2 sentence summary with key details]
   - Source: [source name] | Date: [date]

2. **[Headline]** — [1-2 sentence summary with key details]
   - Source: [source name] | Date: [date]

[Continue for all major stories]

## Analysis & Context
[2-3 sentences providing broader context or connecting themes across the stories]

## Sources
[List all source URLs]

---

RULES:
- Use ONLY information from the research data — do NOT invent facts
- Include ALL relevant stories from the data
- Each headline must have a date and source
- Write professionally and objectively
- If data mentions specific numbers, quotes, or names, include them"""

EDITOR_PROMPT_FINANCIAL = """/no_think
You are a financial analyst editor. Write a professional financial report based on the research data.

FORMAT YOUR REPORT EXACTLY LIKE THIS:

# Financial Report: [Topic]
**Date:** [today's date]

## Market Overview
[Summary of key financial data points]

## Key Developments
1. **[Development]** — [Details with numbers/percentages]
   - Source: [source] | Date: [date]

## Data Points
| Metric | Value | Change |
|--------|-------|--------|
[Fill from research data]

## Outlook & Analysis
[2-3 sentences on implications]

## Sources
[List all URLs]

---

RULES:
- Focus on numbers, percentages, and financial metrics
- Use ONLY data from the research — do NOT invent figures
- Be precise with financial terminology"""

EDITOR_PROMPT_FACTUAL = """/no_think
You are a senior editor. Write a clear, well-structured answer based on the research data.

FORMAT:
- Start with a direct answer to the question
- Use headers (##) to organize sections if the topic is complex
- Include relevant details, examples, and explanations
- Cite sources with URLs where available
- End with a brief summary or key takeaway

RULES:
- Use ONLY information from the research data
- Do NOT mention internal processes or tools
- Be thorough but concise (300-600 words)
- Write professionally"""

EDITOR_PROMPT_GENERAL = """/no_think
You are a senior editor. Write a thoughtful, well-organized response.

- Structure your response with clear sections
- Be informative and engaging
- If research data is provided, incorporate it naturally
- Keep response focused and professional (200-400 words)"""

EDITOR_PROMPTS = {
    "news": EDITOR_PROMPT_NEWS,
    "data": EDITOR_PROMPT_FINANCIAL,
    "factual": EDITOR_PROMPT_FACTUAL,
    "general": EDITOR_PROMPT_GENERAL,
}


def _classify_query_heuristic(topic: str) -> tuple[List[str], str, str]:
    q = topic.lower()
    # Use word-boundary matching to avoid false positives from substrings
    words = set(re.findall(r'\b[a-z]+\b', q))

    news_keywords = {
        "latest", "today", "headline", "news", "breaking", "update", "recent", "yesterday",
    }
    data_keywords = {
        "compare", "comparison", "vs", "versus", "calculate", "analysis", "trend",
        "statistics", "data", "metric", "percentage", "ratio", "forecast",
    }
    general_keywords = {
        "poem", "story", "joke", "creative", "opinion", "brainstorm",
    }

    if words & data_keywords:
        return ["researcher", "analyst", "editor"], "data", "heuristic:data keyword"
    if words & news_keywords:
        return ["researcher", "editor"], "news", "heuristic:news keyword"
    if words & general_keywords:
        return ["editor"], "general", "heuristic:general keyword"
    return ["researcher", "editor"], "factual", "heuristic:default factual"


def _parse_research_items(research: str, max_items: int = 8) -> List[Dict[str, str]]:
    blocks = re.findall(r"(?ms)^\[(\d+)\]\s+(.*?)(?=^\[\d+\]\s+|\Z)", research)
    items: List[Dict[str, str]] = []

    for _, block in blocks:
        lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
        if not lines:
            continue

        title = lines[0]
        url = ""
        date = ""
        source = ""
        snippet_parts = []

        for ln in lines[1:]:
            if ln.startswith("URL:"):
                url = ln.replace("URL:", "", 1).strip()
            elif ln.startswith("Date:"):
                meta = ln.replace("Date:", "", 1).strip()
                if "| Source:" in meta:
                    date, source = [x.strip() for x in meta.split("| Source:", 1)]
                else:
                    date = meta
            else:
                snippet_parts.append(ln)

        items.append(
            {
                "title": title,
                "url": url,
                "date": date,
                "source": source,
                "snippet": " ".join(snippet_parts).strip(),
            }
        )
        if len(items) >= max_items:
            break
    return items


def _build_fallback_report(topic: str, query_type: str, research: str, analysis: str) -> str:
    items = _parse_research_items(research, max_items=8)
    lines = [f"Based on current search results for: {topic}"]

    if query_type == "news":
        lines.append("\n## Key Headlines")
    else:
        lines.append("\n## Findings")

    if items:
        for i, item in enumerate(items, 1):
            lines.append(f"\n{i}. {item['title']}")
            if item["date"] or item["source"]:
                lines.append(f"   - Date: {item['date'] or 'Unknown'} | Source: {item['source'] or 'Unknown'}")
            if item["snippet"]:
                lines.append(f"   - {item['snippet']}")
            if item["url"]:
                lines.append(f"   - URL: {item['url']}")
    else:
        lines.append("\nNo structured results were available.")
        if research:
            lines.append("\nRaw research data:")
            lines.append(research[:2500])

    if analysis:
        lines.append("\n## Analysis")
        lines.append(analysis[:1500])

    return "\n".join(lines).strip()


# ─── Supervisor ─────────────────────────────────────────────────────────────

def supervisor_agent(state: dict) -> dict:
    """Routes the query to the right agents based on query classification."""
    topic = state["topic"]
    model_name = ModelSelector.get_model_name(
        "supervisor",
        task_text=topic,
        query_type="routing",
    )

    print(f"\n{Fore.MAGENTA}┌─ SUPERVISOR ({model_name}){Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}│  Classifying query...{Style.RESET_ALL}")

    # Fast and robust default: deterministic keyword router.
    if os.getenv("SUPERVISOR_MODE", "heuristic").lower() == "heuristic":
        agents, query_type, reason = _classify_query_heuristic(topic)
        print(f"{Fore.MAGENTA}│  Route: {' → '.join(agents)}{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}│  Query type: {query_type}{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}│  Reason: {reason}{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}└──────────────────────────────{Style.RESET_ALL}")
        log_event(_log, "supervisor_routed", topic=topic, route=agents,
                  query_type=query_type, reason=reason, model=model_name)
        return {"route": agents, "query_type": query_type}

    llm = ModelSelector.get_model(
        "supervisor",
        task_text=topic,
        query_type="routing",
    )
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
        log_event(_log, "supervisor_routed", topic=topic, route=agents,
                  query_type=query_type, reason=reason, model=model_name, mode="llm")

    except Exception as e:
        print(f"{Fore.RED}│  Routing failed ({e}), using default pipeline{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}└──────────────────────────────{Style.RESET_ALL}")
        _log.error("Supervisor LLM routing failed: %s", e)
        agents = ["researcher", "editor"]
        query_type = "factual"

    return {"route": agents, "query_type": query_type}


# ─── Researcher 
# ──────────────────────────────────────────────────────────
# ----------------------------------------------------

def research_agent(state: dict) -> dict:
    """Searches the web for information about the topic."""
    topic = state["topic"]
    query_type = state.get("query_type", "factual")
    model_name = ModelSelector.get_model_name(
        "researcher",
        task_text=topic,
        query_type=query_type,
    )

    print(f"\n{Fore.YELLOW}┌─ RESEARCHER ({model_name}){Style.RESET_ALL}")

    # For news queries, call news_search directly — don't rely on small LLMs
    if query_type == "news":
        print(f"{Fore.YELLOW}│  🔧 Tool: news_search (direct, query_type=news){Style.RESET_ALL}")
        result = news_search.invoke({"query": topic})
        log_event(_log, "researcher_tool_call", tool="news_search",
                  args={"query": topic}, result_len=len(str(result)), mode="direct")
        print(f"{Fore.YELLOW}│  📄 Got {len(str(result))} chars of news results{Style.RESET_ALL}")

        # If news_search returned nothing useful, fallback to web_search
        if "No news results found" in str(result):
            print(f"{Fore.YELLOW}│  ⚠️  No news found, falling back to web_search{Style.RESET_ALL}")
            result = web_search.invoke({"query": f"{topic} latest news"})
            log_event(_log, "researcher_tool_call", tool="web_search",
                      args={"query": f"{topic} latest news"}, result_len=len(str(result)),
                      mode="news_fallback")
            print(f"{Fore.YELLOW}│  📄 Got {len(str(result))} chars of web results{Style.RESET_ALL}")

        print(f"{Fore.YELLOW}└──────────────────────────────{Style.RESET_ALL}")
        return {"research_output": str(result)}

    # For factual queries, call web_search directly for speed
    if query_type == "factual":
        print(f"{Fore.YELLOW}│  🔧 Tool: web_search (direct, query_type=factual){Style.RESET_ALL}")
        result = web_search.invoke({"query": topic})
        log_event(_log, "researcher_tool_call", tool="web_search",
                  args={"query": topic}, result_len=len(str(result)), mode="direct")
        print(f"{Fore.YELLOW}│  📄 Got {len(str(result))} chars of results{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}└──────────────────────────────{Style.RESET_ALL}")
        return {"research_output": str(result)}

    # For data/other queries, let the LLM pick the tool
    llm = ModelSelector.get_model(
        "researcher",
        task_text=topic,
        query_type=query_type,
    ).bind_tools(
        [web_search, searxng_search, duckduckgo_search, news_search]
    )
    instruction = f"Search for information about: {topic}"

    messages = [
        SystemMessage(content=RESEARCHER_PROMPT),
        HumanMessage(content=instruction),
    ]

    ai_msg = llm.invoke(messages)
    messages.append(ai_msg)

    _research_tools = {
        "web_search": web_search,
        "searxng_search": searxng_search,
        "duckduckgo_search": duckduckgo_search,
        "news_search": news_search,
    }

    if ai_msg.tool_calls:
        for tc in ai_msg.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            print(f"{Fore.YELLOW}│  🔧 Tool: {tool_name}({json.dumps(tool_args)}){Style.RESET_ALL}")

            selected_tool = _research_tools.get(tool_name, web_search)
            result = selected_tool.invoke(tool_args)
            log_event(_log, "researcher_tool_call", tool=tool_name,
                      args=tool_args, result_len=len(str(result)))
            print(f"{Fore.YELLOW}│  📄 Got {len(str(result))} chars of results{Style.RESET_ALL}")

            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

        final = llm.invoke(messages)
        output = final.content
    else:
        print(f"{Fore.YELLOW}│  (LLM skipped tool call — executing web_search directly){Style.RESET_ALL}")
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
    model_name = ModelSelector.get_model_name(
        "analyst",
        task_text=f"{topic}\n{research_data[:1500]}",
        query_type="data",
    )

    print(f"\n{Fore.BLUE}┌─ ANALYST ({model_name}){Style.RESET_ALL}")

    llm = ModelSelector.get_model(
        "analyst",
        task_text=f"{topic}\n{research_data[:1500]}",
        query_type="data",
    ).bind_tools([python_repl])
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
            log_event(_log, "analyst_tool_call", tool=tool_name,
                      result_preview=str(result)[:200])
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
    """Synthesizes all gathered information into a professional report."""
    topic = state["topic"]
    query_type = state.get("query_type", "general")
    research = state.get("research_output", "")
    analysis = state.get("analysis_output", "")
    model_name = ModelSelector.get_model_name(
        "editor",
        task_text=f"{topic}\n{research[:1500]}\n{analysis[:1500]}",
        query_type=query_type,
    )

    print(f"\n{Fore.GREEN}┌─ EDITOR ({model_name}){Style.RESET_ALL}")
    print(f"{Fore.GREEN}│  Report type: {query_type}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}│  Writing final report...{Style.RESET_ALL}")

    llm = ModelSelector.get_model(
        "editor",
        task_text=f"{topic}\n{research[:1500]}\n{analysis[:1500]}",
        query_type=query_type,
    )

    # Select the right prompt template based on query type
    editor_prompt = EDITOR_PROMPTS.get(query_type, EDITOR_PROMPT_GENERAL)

    # Build context with metadata
    today = datetime.now().strftime("%B %d, %Y")
    context_parts = [
        f"Today's date: {today}",
        f"User question: {topic}",
        f"Report type: {query_type}",
    ]
    if research:
        context_parts.append(f"\n--- RESEARCH DATA ---\n{research}\n--- END RESEARCH DATA ---")
    if analysis:
        context_parts.append(f"\n--- ANALYSIS DATA ---\n{analysis}\n--- END ANALYSIS DATA ---")

    editor_mode = os.getenv("EDITOR_MODE", "llm").lower()
    if editor_mode == "deterministic" and query_type in {"news", "factual", "data"}:
        print(f"{Fore.GREEN}│  Mode: deterministic report synthesis{Style.RESET_ALL}")
        content = _build_fallback_report(topic, query_type, research, analysis)
        log_event(_log, "editor_output", mode="deterministic", query_type=query_type,
                  report_len=len(content), model=model_name)
        print(f"{Fore.GREEN}└──────────────────────────────{Style.RESET_ALL}")
        return {"final_report": content}

    messages = [
        SystemMessage(content=editor_prompt),
        HumanMessage(content="\n\n".join(context_parts)),
    ]

    response = llm.invoke(messages)

    def _extract_text(resp) -> str:
        if hasattr(resp, "content"):
            text = resp.content
        elif hasattr(resp, "text"):
            text = resp.text
        else:
            text = str(resp)

        if "<think>" in text:
            import re
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return text.strip()

    content = _extract_text(response)
    lower_content = content.lower()
    bad_patterns = [
        "i don't have real-time access",
        "i do not have real-time access",
        "i cannot browse",
        "i can't browse",
        "as an ai",
    ]

    # Use deterministic fallback when model returns empty or generic refusal text.
    if not content or any(p in lower_content for p in bad_patterns):
        print(
            f"{Fore.GREEN}│  ⚠️  Model output unusable, generating structured fallback report...{Style.RESET_ALL}"
        )
        _log.warning("Editor LLM output unusable, falling back to deterministic report")
        content = _build_fallback_report(topic, query_type, research, analysis)

    log_event(_log, "editor_output", mode=editor_mode, query_type=query_type,
              report_len=len(content), model=model_name)
    print(f"{Fore.GREEN}└──────────────────────────────{Style.RESET_ALL}")
    return {"final_report": content}
