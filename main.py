# main.py
from colorama import init
from src.graph import build_graph

init(autoreset=True)


def main():
    print("Initializing Multi-Agent System with Smart Routing...")
    app = build_graph()

    user_query = "Istanbul news latest"

    initial_state = {
        "topic": user_query,
        "route": [],
        "query_type": "",
        "research_output": "",
        "analysis_output": "",
        "final_report": "",
    }

    print(f"\nUser Query: {user_query}\n")

    result = app.invoke(initial_state)

    print("\n" + "=" * 50)
    print("FINAL REPORT")
    print("=" * 50)
    print(result["final_report"])


if __name__ == "__main__":
    main()
