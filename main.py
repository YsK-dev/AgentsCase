# main.py
from colorama import init
from src.graph import build_graph
import time

init(autoreset=True)


def main():
    print("Initializing Multi-Agent System with Smart Routing...")
    app = build_graph()

    # Get user query
    import sys
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
    else:
        user_query = input("Enter your query: ").strip()
        if not user_query:
            user_query = "Istanbul news latest"

    initial_state = {
        "topic": user_query,
        "route": [],
        "query_type": "",
        "research_output": "",
        "analysis_output": "",
        "final_report": "",
        "timings": {},
    }

    print(f"\nUser Query: {user_query}\n")

    start = time.perf_counter()
    result = app.invoke(initial_state)
    total_ms = (time.perf_counter() - start) * 1000

    print("\n" + "=" * 50)
    print("FINAL REPORT")
    print("=" * 50)
    print(result["final_report"])
    timings = result.get("timings", {})
    if timings:
        print("\nTIMING SUMMARY")
        for stage in ["supervisor", "researcher", "analyst", "editor"]:
            if stage in timings:
                print(f"- {stage:10s}: {timings[stage]:8.1f} ms")
    print(f"- {'total':10s}: {total_ms:8.1f} ms")


if __name__ == "__main__":
    main()
