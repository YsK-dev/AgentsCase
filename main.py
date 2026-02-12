# main.py
from src.graph import build_graph

def main():
    print("Initializing Multi-Agent System...")
    app = build_graph()
    
    user_query = "Compare Apple (AAPL) and Microsoft (MSFT) stock performance."
    
    initial_state = {
        "topic": user_query,
        "research_output": "",
        "analysis_output": "",
        "final_report": ""
    }
    
    print(f"\nUser Query: {user_query}\n")
    
    # Run the graph
    result = app.invoke(initial_state)
    
    print("\n" + "="*50)
    print("FINAL REPORT")
    print("="*50)
    print(result["final_report"])

if __name__ == "__main__":
    main()