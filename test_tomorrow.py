import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.api_config import api_config
from agents.tools.travel import search_weather_info

print(f"Key: {api_config.TOMORROW_IO_API_KEY}")
result = search_weather_info.invoke({"destination": "Dubai"})
print(f"Result: {result}")
