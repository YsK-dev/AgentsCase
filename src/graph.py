# src/graph.py
from typing import TypedDict
from langgraph.graph import StateGraph, END
from src.agents import research_agent, analyst_agent, editor_agent

class AgentState(TypedDict):
    topic: str
    research_output: str
    analysis_output: str
    final_report: str

def build_graph():
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("researcher", research_agent)
    workflow.add_node("analyst", analyst_agent)
    workflow.add_node("editor", editor_agent)

    # Logic: Research -> Analyst -> Editor -> End
    workflow.set_entry_point("researcher")
    workflow.add_edge("researcher", "analyst")
    workflow.add_edge("analyst", "editor")
    workflow.add_edge("editor", END)

    return workflow.compile()