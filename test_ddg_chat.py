from duckduckgo_search import DDGS

try:
    with DDGS() as ddgs:
        for r in ddgs.chat("Hello, can you help me plan a trip to Paris?", model='gpt-4o-mini'):
            print(r, end="", flush=True)
except Exception as e:
    print(f"\nError: {e}")
