from ddgs import ddgs

def test_chat():
    results = ddgs.chat("Hello, can you help me plan a trip to Paris?", model='gpt-4o-mini')
    print(results)

if __name__ == "__main__":
    test_chat()
