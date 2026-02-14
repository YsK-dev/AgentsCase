import src.graph as graph_module


def _base_state(topic: str = "test query", model_mode: str = "auto") -> dict:
    return {
        "topic": topic,
        "route": [],
        "query_type": "",
        "model_mode": model_mode,
        "research_output": "",
        "analysis_output": "",
        "final_report": "",
        "timings": {},
    }


def test_graph_route_research_then_editor(monkeypatch):
    def fake_supervisor(state):
        return {"route": ["researcher", "editor"], "query_type": "factual"}

    def fake_researcher(state):
        return {"research_output": "researched facts"}

    def fake_analyst(state):
        raise AssertionError("Analyst should not run for this route")

    def fake_editor(state):
        assert state["research_output"] == "researched facts"
        return {"final_report": "final answer"}

    monkeypatch.setattr(graph_module, "supervisor_agent", fake_supervisor)
    monkeypatch.setattr(graph_module, "research_agent", fake_researcher)
    monkeypatch.setattr(graph_module, "analyst_agent", fake_analyst)
    monkeypatch.setattr(graph_module, "editor_agent", fake_editor)

    app = graph_module.build_graph()
    result = app.invoke(_base_state())

    assert result["final_report"] == "final answer"
    assert "supervisor" in result["timings"]
    assert "researcher" in result["timings"]
    assert "editor" in result["timings"]
    assert "analyst" not in result["timings"]


def test_graph_route_analyst_then_editor(monkeypatch):
    def fake_supervisor(state):
        return {"route": ["analyst", "editor"], "query_type": "data"}

    def fake_researcher(state):
        raise AssertionError("Researcher should not run for this route")

    def fake_analyst(state):
        return {"analysis_output": "analysis complete"}

    def fake_editor(state):
        assert state["analysis_output"] == "analysis complete"
        return {"final_report": "data report"}

    monkeypatch.setattr(graph_module, "supervisor_agent", fake_supervisor)
    monkeypatch.setattr(graph_module, "research_agent", fake_researcher)
    monkeypatch.setattr(graph_module, "analyst_agent", fake_analyst)
    monkeypatch.setattr(graph_module, "editor_agent", fake_editor)

    app = graph_module.build_graph()
    result = app.invoke(_base_state(topic="compare X vs Y"))

    assert result["final_report"] == "data report"
    assert "supervisor" in result["timings"]
    assert "analyst" in result["timings"]
    assert "editor" in result["timings"]
    assert "researcher" not in result["timings"]
