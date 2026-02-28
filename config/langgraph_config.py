import os
import warnings
# Suppress the deprecated langchain.verbose warning from langchain_core
warnings.filterwarnings("ignore", category=UserWarning, message=".*Importing verbose from langchain root module.*")
from dotenv import load_dotenv, find_dotenv
from typing import Dict,Any

load_dotenv(find_dotenv())
class LangGraphConfig:
        # LLM Provider: "openrouter", "ollama", or "groq"
        # Switch to "groq" for fast, reliable free LLM (recommended!)
        LLM_PROVIDER = "groq"  # Change to "openrouter" or "ollama" if needed
        
        # Groq Configuration (if LLM_PROVIDER = "groq") - RECOMMENDED
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        GROQ_MODEL = "llama-3.3-70b-versatile"  # Fast and powerful
        
        # OpenRouter Configuration (if LLM_PROVIDER = "openrouter")
        OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
        # Free model options (switch if you hit rate limits):
        # - meta-llama/llama-3.2-3b-instruct:free (recommended - better rate limits)
        # - nvidia/nemotron-3-nano-30b-a3b:free (alternative)
        # - google/gemma-2-9b-it:free (alternative)
        OPENROUTER_MODEL = "meta-llama/llama-3.2-3b-instruct:free"
        
        # Ollama Configuration (if LLM_PROVIDER = "ollama")
        OLLAMA_BASE_URL = "http://localhost:11434"
        OLLAMA_MODEL = "llama3.2:3b"  # Options: llama3.2:3b, mistral:7b, phi3:mini
        
        # Search Configuration
        DUCKDUCKGO_MAX_RESULTS = 10
        DUCKDUCKGO_REGION = "us-en"
        DUCKDUCKGO_SAFESEARCH = "moderate"
        
        # Agent Configuration
        MAX_ITERATIONS = 50
        RECURSION_LIMIT = 100
        WEATHER_SEARCH_ENABLED = True
        ATTRACTION_SEARCH_ENABLED = True
        HOTEL_SEARCH_ENABLED = True
        RESTAURANT_SEARCH_ENABLED = True
        
        # LLM Parameters
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
