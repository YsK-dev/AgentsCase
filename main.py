# main.py
from colorama import init
from src.graph import build_graph
import time

init(autoreset=True)


def main():
    print("Initializing Multi-Agent System with Smart Routing...")
    try:
        app = build_graph()
    except Exception as e:
        print(f"\n❌ Failed to build agent graph: {e}")
        print("Make sure Ollama is running (`ollama serve`) and required models are pulled.")
        return

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

    try:
        start = time.perf_counter()
        result = app.invoke(initial_state)
        total_ms = (time.perf_counter() - start) * 1000
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        return
    except Exception as e:
        print(f"\n❌ Pipeline execution failed: {e}")
        print("Check logs/agent_system.log for details.")
        return

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
