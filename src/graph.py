# src/graph.py
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from src.agents import supervisor_agent, research_agent, analyst_agent, editor_agent


class AgentState(TypedDict):
    topic: str
    route: List[str]          # set by supervisor: e.g. ["researcher", "editor"]
    query_type: str            # set by supervisor: "news", "factual", "data", "general"
    research_output: str
    analysis_output: str
    final_report: str


def _route_after_supervisor(state: AgentState) -> str:
    """Decide the first agent to run based on supervisor's routing decision."""
    route = state.get("route", ["researcher", "editor"])
    if "researcher" in route:
        return "researcher"
    if "analyst" in route:
        return "analyst"
    return "editor"


def _route_after_researcher(state: AgentState) -> str:
    """After researcher, go to analyst if needed, otherwise editor."""
    route = state.get("route", [])
    if "analyst" in route:
        return "analyst"
    return "editor"


def build_graph():
    workflow = StateGraph(AgentState)

    # Add all nodes
    workflow.add_node("supervisor", supervisor_agent)
    workflow.add_node("researcher", research_agent)
    workflow.add_node("analyst", analyst_agent)
    workflow.add_node("editor", editor_agent)

    # Entry: always start with supervisor
    workflow.set_entry_point("supervisor")

    # Conditional edges from supervisor
    workflow.add_conditional_edges(
        "supervisor",
        _route_after_supervisor,
        {"researcher": "researcher", "analyst": "analyst", "editor": "editor"},
    )

    # Conditional edges from researcher
    workflow.add_conditional_edges(
        "researcher",
        _route_after_researcher,
        {"analyst": "analyst", "editor": "editor"},
    )

    # Analyst always goes to editor
    workflow.add_edge("analyst", "editor")

    # Editor always ends
    workflow.add_edge("editor", END)

    return workflow.compile()
