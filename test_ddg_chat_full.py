from duckduckgo_search import DDGS
import time

def ddg_chat_wrapper(prompt, model='gpt-4o-mini'):
    try:
        with DDGS() as ddgs:
            results = ddgs.chat(prompt, model=model)
            return results
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    print("Testing DDG AI Chat...")
    response = ddg_chat_wrapper("Write a 1-day itinerary for Paris in JSON format.")
    print("Response:")
    print(response)
