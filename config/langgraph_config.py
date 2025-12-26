import os
from dotenv import load_dotenv, find_dotenv
from typing import Dict,Any

load_dotenv(find_dotenv())
class LangGraphConfig:
        GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")
        GEMINI_MODEL="gemini-2.0-flash"
        DUCKDUCKGO_MAX_RESULTS = 10
        DUCKDUCKGO_REGION = "us-en"
        DUCKDUCKGO_SAFESEARCH = "moderate"
        MAX_ITERATIONS = 50
        RECURSION_LIMIT = 100
        WEATHER_SEARCH_ENABLED = True
        ATTRACTION_SEARCH_ENABLED = True
        HOTEL_SEARCH_ENABLED = True
        RESTAURANT_SEARCH_ENABLED = True
        TEMPERATURE=0.7
        MAX_TOKENS=4096
        TOP_P=0.8
        @classmethod
        def get_gemini_config(cls) -> Dict[str, Any]:
            
            return {
                "model": cls.GEMINI_MODEL,
                "temperature": cls.TEMPERATURE,
                "max_output_tokens": cls.MAX_TOKENS,
                "top_p": cls.TOP_P,
            }
        
        @classmethod
        def get_search_config(cls) -> Dict[str, Any]:
            
            return {
                "max_results": cls.DUCKDUCKGO_MAX_RESULTS,
                "region": cls.DUCKDUCKGO_REGION,
                "safesearch": cls.DUCKDUCKGO_SAFESEARCH,
            }
        @classmethod
        def validate_config(cls) -> bool:
            """Validate that all required configurations are present"""
            cls.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
            if not cls.GEMINI_API_KEY:
                print("[WARNING] GEMINI_API_KEY not found in environment variables")
                print("Please set GEMINI_API_KEY in your .env file")
                return False
            return True

# Initialize configuration
langgraph_config = LangGraphConfig()

# Validate configuration on import
if not langgraph_config.validate_config():
    print("[ERROR] Configuration validation failed")
    print("Please check your .env file and ensure GEMINI_API_KEY is set")
else:
    print("[SUCCESS] LangGraph configuration loaded successfully")
