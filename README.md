# Multi-Agent Local LLM Assistant

A local multi-agent assistant built with LangGraph and Ollama.
It routes user queries across specialized agents, uses web search for fresh context, and runs safe Python code for verification.

## Features

- 4-agent workflow: `supervisor`, `researcher`, `analyst`, `editor`
- Multi-model routing: `fast`, `accurate`, `auto`
- Web search tools: SearXNG (primary) + DuckDuckGo (fallback)
- Safe Python execution tool for calculations
- Structured JSON logging
- Unit tests for routing, tools, and error scenarios

## Architecture

```mermaid
flowchart LR
    U[User] --> S[Supervisor]
    S --> R[Researcher]
    S --> A[Analyst]
    S --> E[Editor]
    R --> A
    R --> E
    A --> E
    E --> O[Final Answer]
```

## Tech Stack

- Python
- LangGraph / LangChain
- Ollama (local models)
- SearXNG

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Pull local models (example):

```bash
ollama pull deepseek-coder:1.3b
ollama pull qwen3:1.7b
```

## Configuration

Important `.env` values:

- `MODEL_FAST`
- `MODEL_ACCURATE`
- `MODEL_MODE` (`auto`, `fast`, `accurate`)
- `SEARXNG_URL` (AWS-hosted endpoint supported, e.g. `http://<host>/search`)

## Run

Single query:

```bash
python main.py --mode auto "latest AI chip market news"
```

Interactive:

```bash
python demo.py --mode auto
```

## Tests

```bash
pytest tests/ -v
```

## Notes

- SearXNG is used as the primary engine because it benchmarked better in this project environment.
- DuckDuckGo remains enabled for fallback resilience.
