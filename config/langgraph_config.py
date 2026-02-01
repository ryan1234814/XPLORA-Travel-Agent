import os
import warnings
# Suppress the deprecated langchain.verbose warning from langchain_core
warnings.filterwarnings("ignore", category=UserWarning, message=".*Importing verbose from langchain root module.*")
from dotenv import load_dotenv, find_dotenv
from typing import Dict,Any

load_dotenv(find_dotenv())
class LangGraphConfig:
        OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
        # Free model: nvidia/nemotron-3-nano-30b-a3b:free
        OPENROUTER_MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"
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
        def get_search_config(cls) -> Dict[str, Any]:
            
            return {
                "max_results": cls.DUCKDUCKGO_MAX_RESULTS,
                "region": cls.DUCKDUCKGO_REGION,
                "safesearch": cls.DUCKDUCKGO_SAFESEARCH,
            }
        
        @classmethod
        def validate_config(cls) -> bool:
            """Validate configuration"""
            cls.OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
            if not cls.OPENROUTER_API_KEY:
                print("[WARNING] OPENROUTER_API_KEY not found in environment variables")
                return False
            return True

# Initialize configuration
langgraph_config = LangGraphConfig()

# Validate configuration on import
if not langgraph_config.validate_config():
    print("[ERROR] Configuration validation failed")
else:
    print("[SUCCESS] XPLORA configuration loaded successfully (using DuckDuckGo)")
