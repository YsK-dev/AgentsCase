# demo.py — Interactive Multi-Agent CLI with Smart Routing
import sys
import time
from colorama import Fore, Style, init
from src.graph import build_graph

init(autoreset=True)


def run_query(app, user_query: str):
    """Run a single query through the multi-agent graph."""
    print(f"\n{Fore.MAGENTA}{'=' * 60}")
    print(f"TASK: {user_query}")
    print(f"{'=' * 60}{Style.RESET_ALL}")

    initial_state = {
        "topic": user_query,
        "route": [],
        "query_type": "",
        "research_output": "",
        "analysis_output": "",
        "final_report": "",
        "timings": {},
    }

    start = time.perf_counter()
    result = app.invoke(initial_state)
    total_ms = (time.perf_counter() - start) * 1000

    # ── Final Output ────────────────────────────────────────────
    print(f"\n{Fore.MAGENTA}{'=' * 60}")
    print("FINAL OUTPUT")
    print(f"{'=' * 60}{Style.RESET_ALL}")
    print(result.get("final_report", "No output generated."))
    timings = result.get("timings", {})
    if timings:
        print(f"\n{Fore.CYAN}Timing Summary{Style.RESET_ALL}")
        for stage in ["supervisor", "researcher", "analyst", "editor"]:
            if stage in timings:
                print(f"  - {stage:10s}: {timings[stage]:8.1f} ms")
    print(f"{Fore.CYAN}  - {'total':10s}: {total_ms:8.1f} ms{Style.RESET_ALL}")
    print()


def main():
    print(f"{Fore.CYAN}Initializing Multi-Agent System with Smart Routing...{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Building agent graph (supervisor → researcher/analyst → editor)...{Style.RESET_ALL}")

    try:
        app = build_graph()
    except Exception as e:
        print(f"{Fore.RED}❌ Failed to build agent graph: {e}{Style.RESET_ALL}")
        print(f"{Fore.RED}Make sure Ollama is running (`ollama serve`) and required models are pulled.{Style.RESET_ALL}")
        return

    print(f"{Fore.GREEN}✓ System ready.{Style.RESET_ALL}\n")

    # If a query was passed as CLI argument, run it and exit
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        try:
            run_query(app, query)
        except Exception as e:
            print(f"{Fore.RED}❌ Pipeline error: {e}{Style.RESET_ALL}")
            print(f"{Fore.RED}Check logs/agent_system.log for details.{Style.RESET_ALL}")
        return

    # Interactive loop
    while True:
        try:
            user_input = input(
                f"{Fore.WHITE}Enter your request (or 'q' to quit): {Style.RESET_ALL}"
            )
            if user_input.strip().lower() in ("q", "quit", "exit"):
                print("Goodbye!")
                break
            if not user_input.strip():
                continue

            try:
                run_query(app, user_input.strip())
            except Exception as e:
                print(f"{Fore.RED}❌ Pipeline error: {e}{Style.RESET_ALL}")
                print(f"{Fore.RED}Check logs/agent_system.log for details.{Style.RESET_ALL}")

        except KeyboardInterrupt:
            print("\nExiting...")
            break


if __name__ == "__main__":
    main()
