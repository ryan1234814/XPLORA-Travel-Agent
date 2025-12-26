import os
import sys
from dotenv import load_dotenv

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

# Force utf-8 encoding for stdout/stderr
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from agents.tools.travel import ALL_TOOLS

def test_all_tools():
    print("\n" + "="*50)
    print("TESTING ALL TRAVEL TOOLS")
    print("="*50 + "\n")
    
    test_destination = "Tokyo"
    
    for tool in ALL_TOOLS:
        print(f"Testing Tool: {tool.name}")
        print(f"Description: {tool.description}")
        
        try:
            # Prepare arguments based on tool name
            args = {"destination": test_destination}
            if tool.name == "search_destination_info":
                # This tool takes 'query' instead of 'destination'
                args = {"query": test_destination}
            elif tool.name == "search_budget_info":
                args = {"destination": test_destination, "duration": "5 days"}
            
            print(f"Invoking with args: {args}")
            result = tool.invoke(input=args)
            
            # Print a snippet of the result
            snippet = str(result)[:300] + "..." if len(str(result)) > 300 else str(result)
            print(f"Result Snippet:\n{snippet}")
            print(f"SUCCESS: {tool.name} worked properly.\n")
            
        except Exception as e:
            import traceback
            print(f"FAILURE: {tool.name} failed with error: {str(e)}")
            traceback.print_exc()
            print()
        
        print("-" * 30 + "\n")

if __name__ == "__main__":
    test_all_tools()
