# demo.py — Interactive Multi-Agent CLI with Smart Routing
import sys
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
    }

    result = app.invoke(initial_state)

    # ── Final Output ────────────────────────────────────────────
    print(f"\n{Fore.MAGENTA}{'=' * 60}")
    print("FINAL OUTPUT")
    print(f"{'=' * 60}{Style.RESET_ALL}")
    print(result.get("final_report", "No output generated."))
    print()


def main():
    print(f"{Fore.CYAN}Initializing Multi-Agent System with Smart Routing...{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Building agent graph (supervisor → researcher/analyst → editor)...{Style.RESET_ALL}")

    app = build_graph()

    print(f"{Fore.GREEN}✓ System ready.{Style.RESET_ALL}\n")

    # If a query was passed as CLI argument, run it and exit
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        run_query(app, query)
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

            run_query(app, user_input.strip())

        except KeyboardInterrupt:
            print("\nExiting...")
            break


if __name__ == "__main__":
    main()
