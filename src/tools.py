# src/tools.py
from ddgs import DDGS
from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import tool


class ToolSelector:
    """Returns the appropriate tools for each agent role."""

    @staticmethod
    def get_tools(agent_role: str):
        if agent_role == "supervisor":
            return []
        elif agent_role == "researcher":
            return [web_search, news_search]
        elif agent_role == "analyst":
            return [python_repl]
        else:
            return [web_search, news_search, python_repl]


@tool
def web_search(query: str) -> str:
    """Search the web for general information. Returns top results with titles, URLs, and snippets."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=10, timelimit="m"))

        # Fallback without time limit if no recent results
        if not results:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=10))

        if not results:
            return "No results found."

        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(
                f"[{i}] {r.get('title', 'No title')}\n"
                f"    URL: {r.get('href', 'N/A')}\n"
                f"    {r.get('body', 'No snippet')}"
            )
        return "\n\n".join(formatted)
    except Exception as e:
        return f"Search error: {e}"


@tool
def news_search(query: str) -> str:
    """Search for recent news articles. Returns headlines with dates, sources, and summaries."""
    try:
        # Try multiple query variations for better coverage
        all_results = []
        queries = [query]

        # Add region-specific variation if not already present
        query_lower = query.lower()
        if "turkey" not in query_lower and "türkiye" not in query_lower:
            # Check if it's a Turkish city/topic
            turkish_cities = ["istanbul", "ankara", "izmir", "antalya", "bursa"]
            if any(city in query_lower for city in turkish_cities):
                queries.append(f"{query} Turkey")

        for q in queries:
            # Try past day first, then week, then month
            for period in ["d", "w", "m"]:
                if len(all_results) >= 10:
                    break
                try:
                    with DDGS() as ddgs:
                        results = list(ddgs.news(q, max_results=10, timelimit=period))
                    if results:
                        # Deduplicate by title
                        existing_titles = {r.get("title", "") for r in all_results}
                        for r in results:
                            if r.get("title", "") not in existing_titles:
                                all_results.append(r)
                                existing_titles.add(r.get("title", ""))
                        if len(all_results) >= 5:
                            break  # Got enough results
                except Exception:
                    continue

        if not all_results:
            return "No news results found."

        # Sort by date (most recent first) and limit to top 10
        all_results = all_results[:10]

        formatted = []
        for i, r in enumerate(all_results, 1):
            date = r.get("date", "Unknown date")
            source = r.get("source", "Unknown source")
            formatted.append(
                f"[{i}] {r.get('title', 'No title')}\n"
                f"    Date: {date} | Source: {source}\n"
                f"    URL: {r.get('url', 'N/A')}\n"
                f"    {r.get('body', 'No snippet')}"
            )
        return "\n\n".join(formatted)
    except Exception as e:
        return f"News search error: {e}"


@tool
def python_repl(code: str) -> str:
    """Run Python code and return the output. Useful for calculations, data processing, or analysis."""
    repl = PythonREPL()
    try:
        result = repl.run(code)
    except Exception as e:
        result = f"Error: {e}"
    return result
