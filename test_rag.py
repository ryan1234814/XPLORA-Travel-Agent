import os
import sys
import asyncio

# Add the project root to the python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.tools.travel import search_travel_blogs

def test_rag_tool():
    print("Testing the search_travel_blogs tool directly...")
    
    test_queries = [
        "travel blog Tokyo tips",
        "guide Rome cheap",
        "Paris hidden gems blog"
    ]
    
    for query in test_queries:
        print(f"\n--- Testing Tool with Query: '{query}' ---")
        try:
            result = search_travel_blogs.invoke({"query": query})
            print("Tool output:")
            print(result)
        except Exception as e:
            print(f"Error testing tool: {e}")

if __name__ == "__main__":
    # First make sure data is ingested (will just add to mock if no Pinecone)
    import ingest_blogs
    ingest_blogs.main()
    
    print("\n" + "="*50 + "\n")
    
    # Test the tool
    test_rag_tool()
