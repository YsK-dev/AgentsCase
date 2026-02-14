# main.py
import argparse
import time
from colorama import init
from src.graph import build_graph

init(autoreset=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the local multi-agent assistant.",
    )
    parser.add_argument(
        "query",
        nargs="*",
        help="Optional query text. If omitted, interactive prompt is used.",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "fast", "accurate"],
        default="auto",
        help="Model selection mode for this request.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("Initializing Multi-Agent System with Smart Routing...")
    try:
        app = build_graph()
    except Exception as e:
        print(f"\n❌ Failed to build agent graph: {e}")
        print("Make sure Ollama is running (`ollama serve`) and required models are pulled.")
        return

    # Get user query
    if args.query:
        user_query = " ".join(args.query)
    else:
        user_query = input("Enter your query: ").strip()
        if not user_query:
            user_query = "Istanbul news latest"

    initial_state = {
        "topic": user_query,
        "route": [],
        "query_type": "",
        "model_mode": args.mode,
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
