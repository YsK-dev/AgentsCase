# src/graph.py
import time
from typing import Dict, List, NotRequired, TypedDict
from langgraph.graph import StateGraph, END
from src.agents import supervisor_agent, research_agent, analyst_agent, editor_agent


class AgentState(TypedDict):
    topic: str
    route: List[str]          # set by supervisor: e.g. ["researcher", "editor"]
    query_type: str            # set by supervisor: "news", "factual", "data", "general"
    model_mode: NotRequired[str]  # optional: "auto" | "fast" | "accurate"
    research_output: str
    analysis_output: str
    final_report: str
    timings: NotRequired[Dict[str, float]]


def _timed_node(node_name: str, node_fn):
    """Wrap a node and persist elapsed milliseconds in state['timings']."""

    def wrapped(state: AgentState):
        start = time.perf_counter()
        output = node_fn(state)
        elapsed_ms = (time.perf_counter() - start) * 1000

        timings = dict(state.get("timings", {}))
        returned_timings = output.get("timings")
        if isinstance(returned_timings, dict):
            timings.update(returned_timings)
        timings[node_name] = elapsed_ms

        output["timings"] = timings
        return output

    return wrapped


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
    workflow.add_node("supervisor", _timed_node("supervisor", supervisor_agent))
    workflow.add_node("researcher", _timed_node("researcher", research_agent))
    workflow.add_node("analyst", _timed_node("analyst", analyst_agent))
    workflow.add_node("editor", _timed_node("editor", editor_agent))

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
