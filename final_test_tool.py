from agents.tools.travel import search_destination_info
import os
from dotenv import load_dotenv

load_dotenv()

def test():
    print("Testing search_destination_info for 'Tokyo'...")
    # LangChain tools use invoke()
    result = search_destination_info.invoke("Tokyo")
    print(result)

if __name__ == "__main__":
    test()
