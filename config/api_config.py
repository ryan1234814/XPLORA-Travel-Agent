import os
from typing import Optional
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
TOMORROW_IO_API_KEY: Optional[str] = os.getenv('TOMORROW_IO_API_KEY')
WEATHER_BASE_URL: str = "https://api.tomorrow.io/v4/weather"

# Google Places API Configuration  
GOOGLE_PLACES_API_KEY: Optional[str] = os.getenv('GOOGLE_PLACES_API_KEY')
PLACES_BASE_URL: str = "https://maps.googleapis.com/maps/api/place"

# Currency Exchange API Configuration
EXCHANGERATE_API_KEY: Optional[str] = os.getenv('EXCHANGERATE_API_KEY')
EXCHANGE_RATE_URL: str = "https://api.exchangerate-api.com/v4/latest"

# Backup free APIs (no key required)
FREE_WEATHER_URL: str = "https://api.open-meteo.com/v1/forecast"
FREE_EXCHANGE_URL: str = "https://api.exchangerate-api.com/v4/latest"

# Pinecone Vector DB Configuration
PINECONE_API_KEY: Optional[str] = os.getenv('PINECONE_API_KEY')
PINECONE_INDEX_NAME: str = os.getenv('PINECONE_INDEX_NAME', 'travel-guides')

# ScrapeGraphAI API Key Configuration
SCRAPEGRAPH_API_KEY: Optional[str] = os.getenv('SCRAPEGRAPH_API_KEY') or os.getenv('SGAI_API_KEY')

def get_api_status() -> dict:
    """Check which APIs have valid keys"""
    return {
        'weather': bool(TOMORROW_IO_API_KEY),
        'places': bool(GOOGLE_PLACES_API_KEY), 
        'exchange': bool(EXCHANGERATE_API_KEY),
        'pinecone': bool(PINECONE_API_KEY),
        'scrapegraph': bool(SCRAPEGRAPH_API_KEY)
    }

# Create API config object for imports
class APIConfig:
    TOMORROW_IO_API_KEY = TOMORROW_IO_API_KEY
    WEATHER_BASE_URL = WEATHER_BASE_URL
    GOOGLE_PLACES_API_KEY = GOOGLE_PLACES_API_KEY
    PLACES_BASE_URL = PLACES_BASE_URL
    EXCHANGERATE_API_KEY = EXCHANGERATE_API_KEY
    EXCHANGE_RATE_URL = EXCHANGE_RATE_URL
    PINECONE_API_KEY = PINECONE_API_KEY
    PINECONE_INDEX_NAME = PINECONE_INDEX_NAME
    SCRAPEGRAPH_API_KEY = SCRAPEGRAPH_API_KEY

# Global instance for importing
api_config = APIConfig()