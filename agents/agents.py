from typing import List,Dict,Any,Optional,Annotated,TypedDict
import warnings
# Suppress the deprecated langchain.verbose warning from langchain_core
warnings.filterwarnings("ignore", category=UserWarning, message=".*Importing verbose from langchain root module.*")
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage, BaseMessage
from langgraph.graph import StateGraph,END
import json
import re
import time
from datetime import datetime
from langchain_openai import ChatOpenAI
from ddgs import DDGS
from config.langgraph_config import LangGraphConfig as config
from config.api_config import api_config
import requests
import threading

# Generic retry logic for capacity / rate-limit errors (429, 503, etc.)
def is_rate_limit_error(exception):
    """Check if the error is a rate limit or capacity error that should trigger a retry or fallback."""
    error_str = str(exception).lower()
    return any(phrase in error_str for phrase in [
        "ratelimit", "rate limit", "slow down", "429", "503",
        "capacity", "overloaded", "over load", "high load",
        "service unavailable", "try again", "too many requests",
        "temporarily unavailable", "server busy"
    ])

def _safe_message_content(message: Any) -> str:
    """Convert a LangChain message (or any object) into a displayable string."""
    if message is None:
        return ""
    
    # If it has a content attribute (Common for LangChain messages)
    content = getattr(message, "content", None)
    
    if content is not None:
        # If content is a list (e.g., Gemini multi-modal/structured content)
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, str):
                    text_parts.append(part)
                elif isinstance(part, dict) and "text" in part:
                    text_parts.append(part["text"])
            return "".join(text_parts)
        return str(content)
        
    # Fallback to string representation if no content attribute found
    return str(message)

def _try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort JSON parser for model outputs, handling markdown and extra text."""
    if not text or not isinstance(text, str):
        return None
    
    # Strip whitespace
    text = text.strip()
    
    # Try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try cleaning markdown code block markers
    try:
        cleaned = text
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        return json.loads(cleaned)
    except Exception:
        pass

    # Try extraction with regex - greedy match handles nested structures correctly
    try:
        json_match = re.search(r'(\{.*\})', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
    except Exception:
        pass
        
    return None


def _normalize_weather_data(parsed: Any, destination: str, travel_dates: str) -> Dict[str, Any]:
    # Ensure it's a dict
    if not isinstance(parsed, dict):
        parsed = {}
    
    # Standardize values / handle camelCase
    destination_val = parsed.get("destination") or destination
    travel_dates_val = parsed.get("travel_dates") or travel_dates
    
    # extract temperature_c
    temp_c = parsed.get("temperature_c") or parsed.get("temperature")
    if not isinstance(temp_c, dict):
        temp_c = {}
        
    expected_low = temp_c.get("expected_low")
    if expected_low is None:
        expected_low = temp_c.get("expectedLow") or temp_c.get("low") or temp_c.get("min")
        
    expected_high = temp_c.get("expected_high")
    if expected_high is None:
        expected_high = temp_c.get("expectedHigh") or temp_c.get("high") or temp_c.get("max")
        
    typical_range = temp_c.get("typical_range")
    if typical_range is None:
        typical_range = temp_c.get("typicalRange") or temp_c.get("range")
        
    notes = temp_c.get("notes")
    if notes is None:
        notes = temp_c.get("note") or temp_c.get("description")
        
    conditions_summary = parsed.get("conditions_summary")
    if conditions_summary is None:
        conditions_summary = parsed.get("conditionsSummary") or parsed.get("conditions") or parsed.get("summary")
        
    best_times = parsed.get("best_times")
    if best_times is None:
        best_times = parsed.get("bestTimes") or parsed.get("best_time") or []
        
    activity_suggestions = parsed.get("activity_suggestions")
    if activity_suggestions is None:
        activity_suggestions = parsed.get("activitySuggestions") or parsed.get("activities") or []
        
    packing = parsed.get("packing") or parsed.get("packing_list") or []

    # If expected_high/low are still missing or invalid, generate reasonable defaults
    try:
        if expected_low is not None:
            expected_low = float(expected_low)
    except (ValueError, TypeError):
        expected_low = None
        
    try:
        if expected_high is not None:
            expected_high = float(expected_high)
    except (ValueError, TypeError):
        expected_high = None
        
    if expected_low is None or expected_high is None:
        # Default fallback
        expected_low = 12.0
        expected_high = 22.0
        dest_lower = destination.lower()
        if "kyoto" in dest_lower or "tokyo" in dest_lower or "japan" in dest_lower:
            if "spring" in (travel_dates_val or "").lower():
                expected_low = 10.0
                expected_high = 20.0
            elif "summer" in (travel_dates_val or "").lower():
                expected_low = 22.0
                expected_high = 31.0
            elif "autumn" in (travel_dates_val or "").lower() or "fall" in (travel_dates_val or "").lower():
                expected_low = 12.0
                expected_high = 21.0
            elif "winter" in (travel_dates_val or "").lower():
                expected_low = 2.0
                expected_high = 10.0
                
    if not typical_range:
        typical_range = f"{int(expected_low)}°C - {int(expected_high)}°C"
        
    if not notes:
        notes = f"Ideal weather conditions expected during {travel_dates_val}."
        
    if not conditions_summary:
        conditions_summary = "Pleasant weather with clear to partly cloudy skies."
        
    if not isinstance(best_times, list):
        best_times = [str(best_times)] if best_times else ["Morning", "Afternoon"]
        
    if not isinstance(activity_suggestions, list):
        activity_suggestions = [str(activity_suggestions)] if activity_suggestions else ["Sightseeing", "Walking tours"]
        
    if not isinstance(packing, list):
        packing = [str(packing)] if packing else ["Layered clothing", "Comfortable walking shoes"]
        
    return {
        "destination": destination_val,
        "travel_dates": travel_dates_val,
        "temperature_c": {
            "expected_low": expected_low,
            "expected_high": expected_high,
            "typical_range": typical_range,
            "notes": notes
        },
        "conditions_summary": conditions_summary,
        "best_times": best_times,
        "activity_suggestions": activity_suggestions,
        "packing": packing
    }


def _normalize_local_expert_data(parsed: Any, destination: str) -> Dict[str, Any]:
    if not isinstance(parsed, dict):
        parsed = {}
        
    summary = parsed.get("summary")
    if not summary:
        summary = f"Gathering contemporary cultural nuances, unwritten customs, and heritage secrets for {destination} to enhance your perspective."
        
    def get_section(key: str, default_title: str) -> Dict[str, Any]:
        sec = parsed.get(key)
        if not isinstance(sec, dict):
            sec = {}
        title = sec.get("title") or default_title
        insights = sec.get("insights") or sec.get("details") or sec.get("points")
        if not isinstance(insights, list):
            insights = [str(insights)] if insights else []
        return {"title": title, "insights": insights}
        
    sensory = parsed.get("sensory_profile") or parsed.get("sensory")
    if not isinstance(sensory, dict):
        sensory = {}
    sensory_title = sensory.get("title") or "Sensory Signature"
    scents = sensory.get("scents") or sensory.get("scent") or []
    if not isinstance(scents, list):
        scents = [str(scents)] if scents else []
    sounds = sensory.get("sounds") or sensory.get("sound") or []
    if not isinstance(sounds, list):
        sounds = [str(sounds)] if sounds else []
    colors = sensory.get("colors") or sensory.get("color") or []
    if not isinstance(colors, list):
        colors = [str(colors)] if colors else []
        
    # Standard fallbacks for Kyoto if parsing or LLM fails
    if not scents and "kyoto" in destination.lower():
        scents = ["Incense wood in ancient temples", "Damp moss after mountain rain", "Roasted green tea (Hojicha) from traditional shops"]
    if not sounds and "kyoto" in destination.lower():
        sounds = ["Clack of wooden Geta shoes on stone pathways", "Distant resonance of temple bells", "Gentle murmur of the Kamogawa river"]
    if not colors and "kyoto" in destination.lower():
        colors = ["Moss Green (#3D5230)", "Vermilion (#E60012)", "Ink Black (#1C1C1C)"]
        
    return {
        "summary": summary,
        "contemporary_behaviors": get_section("contemporary_behaviors", "Living Rhythms & Emerging Trends"),
        "unwritten_customs": get_section("unwritten_customs", "Unwritten Social Codes & Customs"),
        "folklore_heritage": get_section("folklore_heritage", "Folklore, Beliefs & Hidden Heritage"),
        "sensory_profile": {
            "title": sensory_title,
            "scents": scents or ["Local herbal scents", "Damp morning air"],
            "sounds": sounds or ["Distant street chatter", "Traditional music notes"],
            "colors": colors or ["Earthy brown (#8B5A2B)", "Warm sand (#D2B48C)"]
        },
        "guidebook_vs_reality": get_section("guidebook_vs_reality", "Guidebook Expectations vs. Modern Reality"),
        "authenticity_signals": get_section("authenticity_signals", "Living Authenticity Signals")
    }


# Currency mapping: destination keyword -> (symbol, code, name)
_CURRENCY_MAP = {
    # Europe
    "paris": ("€", "EUR", "Euro"),
    "france": ("€", "EUR", "Euro"),
    "london": ("£", "GBP", "British Pound"),
    "uk": ("£", "GBP", "British Pound"),
    "england": ("£", "GBP", "British Pound"),
    "scotland": ("£", "GBP", "British Pound"),
    "rome": ("€", "EUR", "Euro"),
    "italy": ("€", "EUR", "Euro"),
    "florence": ("€", "EUR", "Euro"),
    "venice": ("€", "EUR", "Euro"),
    "amsterdam": ("€", "EUR", "Euro"),
    "netherlands": ("€", "EUR", "Euro"),
    "berlin": ("€", "EUR", "Euro"),
    "germany": ("€", "EUR", "Euro"),
    "munich": ("€", "EUR", "Euro"),
    "barcelona": ("€", "EUR", "Euro"),
    "spain": ("€", "EUR", "Euro"),
    "madrid": ("€", "EUR", "Euro"),
    "lisbon": ("€", "EUR", "Euro"),
    "portugal": ("€", "EUR", "Euro"),
    "athens": ("€", "EUR", "Euro"),
    "greece": ("€", "EUR", "Euro"),
    "santorini": ("€", "EUR", "Euro"),
    "zurich": ("CHF", "CHF", "Swiss Franc"),
    "switzerland": ("CHF", "CHF", "Swiss Franc"),
    "vienna": ("€", "EUR", "Euro"),
    "austria": ("€", "EUR", "Euro"),
    "prague": ("CZK", "CZK", "Czech Koruna"),
    "czech": ("CZK", "CZK", "Czech Koruna"),
    "budapest": ("HUF", "HUF", "Hungarian Forint"),
    "hungary": ("HUF", "HUF", "Hungarian Forint"),
    "copenhagen": ("kr", "DKK", "Danish Krone"),
    "denmark": ("kr", "DKK", "Danish Krone"),
    "stockholm": ("kr", "SEK", "Swedish Krona"),
    "sweden": ("kr", "SEK", "Swedish Krona"),
    "oslo": ("kr", "NOK", "Norwegian Krone"),
    "norway": ("kr", "NOK", "Norwegian Krone"),
    "dublin": ("€", "EUR", "Euro"),
    "ireland": ("€", "EUR", "Euro"),
    "moscow": ("₽", "RUB", "Russian Ruble"),
    "russia": ("₽", "RUB", "Russian Ruble"),
    "istanbul": ("₺", "TRY", "Turkish Lira"),
    "turkey": ("₺", "TRY", "Turkish Lira"),
    "croatia": ("€", "EUR", "Euro"),
    "split": ("€", "EUR", "Euro"),
    # Asia
    "tokyo": ("¥", "JPY", "Japanese Yen"),
    "japan": ("¥", "JPY", "Japanese Yen"),
    "kyoto": ("¥", "JPY", "Japanese Yen"),
    "osaka": ("¥", "JPY", "Japanese Yen"),
    "hokkaido": ("¥", "JPY", "Japanese Yen"),
    "beijing": ("¥", "CNY", "Chinese Yuan"),
    "china": ("¥", "CNY", "Chinese Yuan"),
    "shanghai": ("¥", "CNY", "Chinese Yuan"),
    "chengdu": ("¥", "CNY", "Chinese Yuan"),
    "hong kong": ("HK$", "HKD", "Hong Kong Dollar"),
    "taiwan": ("NT$", "TWD", "Taiwan Dollar"),
    "taipei": ("NT$", "TWD", "Taiwan Dollar"),
    "bangkok": ("฿", "THB", "Thai Baht"),
    "thailand": ("฿", "THB", "Thai Baht"),
    "phuket": ("฿", "THB", "Thai Baht"),
    "chiang mai": ("฿", "THB", "Thai Baht"),
    "bali": ("Rp", "IDR", "Indonesian Rupiah"),
    "indonesia": ("Rp", "IDR", "Indonesian Rupiah"),
    "singapore": ("S$", "SGD", "Singapore Dollar"),
    "kuala lumpur": ("RM", "MYR", "Malaysian Ringgit"),
    "malaysia": ("RM", "MYR", "Malaysian Ringgit"),
    "manila": ("₱", "PHP", "Philippine Peso"),
    "philippines": ("₱", "PHP", "Philippine Peso"),
    "seoul": ("₩", "KRW", "South Korean Won"),
    "korea": ("₩", "KRW", "South Korean Won"),
    "india": ("₹", "INR", "Indian Rupee"),
    "delhi": ("₹", "INR", "Indian Rupee"),
    "mumbai": ("₹", "INR", "Indian Rupee"),
    "jaipur": ("₹", "INR", "Indian Rupee"),
    "goa": ("₹", "INR", "Indian Rupee"),
    "munnar": ("₹", "INR", "Indian Rupee"),
    "kerala": ("₹", "INR", "Indian Rupee"),
    "kochi": ("₹", "INR", "Indian Rupee"),
    "cochin": ("₹", "INR", "Indian Rupee"),
    "varanasi": ("₹", "INR", "Indian Rupee"),
    "agra": ("₹", "INR", "Indian Rupee"),
    "udaipur": ("₹", "INR", "Indian Rupee"),
    "rishikesh": ("₹", "INR", "Indian Rupee"),
    "ladakh": ("₹", "INR", "Indian Rupee"),
    "kashmir": ("₹", "INR", "Indian Rupee"),
    "himachal": ("₹", "INR", "Indian Rupee"),
    "manali": ("₹", "INR", "Indian Rupee"),
    "shimla": ("₹", "INR", "Indian Rupee"),
    "darjeeling": ("₹", "INR", "Indian Rupee"),
    "ooty": ("₹", "INR", "Indian Rupee"),
    "coorg": ("₹", "INR", "Indian Rupee"),
    "pondicherry": ("₹", "INR", "Indian Rupee"),
    "hampi": ("₹", "INR", "Indian Rupee"),
    "andaman": ("₹", "INR", "Indian Rupee"),
    "bangalore": ("₹", "INR", "Indian Rupee"),
    "bengaluru": ("₹", "INR", "Indian Rupee"),
    "chennai": ("₹", "INR", "Indian Rupee"),
    "hyderabad": ("₹", "INR", "Indian Rupee"),
    "kolkata": ("₹", "INR", "Indian Rupee"),
    "hanoi": ("₫", "VND", "Vietnamese Dong"),
    "vietnam": ("₫", "VND", "Vietnamese Dong"),
    "ho chi minh": ("₫", "VND", "Vietnamese Dong"),
    "cambodia": ("៛", "KHR", "Cambodian Riel"),
    "phnom penh": ("៛", "KHR", "Cambodian Riel"),
    "nepal": ("NPR", "NPR", "Nepalese Rupee"),
    "sri lanka": ("Rs", "LKR", "Sri Lankan Rupee"),
    # Middle East
    "dubai": ("د.إ", "AED", "UAE Dirham"),
    "uae": ("د.إ", "AED", "UAE Dirham"),
    "abu dhabi": ("د.إ", "AED", "UAE Dirham"),
    "qatar": ("QR", "QAR", "Qatari Riyal"),
    "doha": ("QR", "QAR", "Qatari Riyal"),
    "saudi arabia": ("﷼", "SAR", "Saudi Riyal"),
    "riyadh": ("﷼", "SAR", "Saudi Riyal"),
    "oman": ("ر.ع", "OMR", "Omani Rial"),
    "bahrain": ("BD", "BHD", "Bahraini Dinar"),
    "israel": ("₪", "ILS", "Israeli Shekel"),
    "tel aviv": ("₪", "ILS", "Israeli Shekel"),
    "jordan": ("JD", "JOD", "Jordanian Dinar"),
    # Oceania
    "sydney": ("A$", "AUD", "Australian Dollar"),
    "australia": ("A$", "AUD", "Australian Dollar"),
    "melbourne": ("A$", "AUD", "Australian Dollar"),
    "gold coast": ("A$", "AUD", "Australian Dollar"),
    "new zealand": ("NZ$", "NZD", "New Zealand Dollar"),
    "auckland": ("NZ$", "NZD", "New Zealand Dollar"),
    "queenstown": ("NZ$", "NZD", "New Zealand Dollar"),
    # Americas
    "new york": ("$", "USD", "US Dollar"),
    "los angeles": ("$", "USD", "US Dollar"),
    "san francisco": ("$", "USD", "US Dollar"),
    "miami": ("$", "USD", "US Dollar"),
    "las vegas": ("$", "USD", "US Dollar"),
    "chicago": ("$", "USD", "US Dollar"),
    "canada": ("C$", "CAD", "Canadian Dollar"),
    "toronto": ("C$", "CAD", "Canadian Dollar"),
    "vancouver": ("C$", "CAD", "Canadian Dollar"),
    "mexico": ("MX$", "MXN", "Mexican Peso"),
    "cancun": ("MX$", "MXN", "Mexican Peso"),
    "playa del carmen": ("MX$", "MXN", "Mexican Peso"),
    "brazil": ("R$", "BRL", "Brazilian Real"),
    "rio de janeiro": ("R$", "BRL", "Brazilian Real"),
    "sao paulo": ("R$", "BRL", "Brazilian Real"),
    "argentina": ("$", "ARS", "Argentine Peso"),
    "buenos aires": ("$", "ARS", "Argentine Peso"),
    "peru": ("S/", "PEN", "Peruvian Sol"),
    "lima": ("S/", "PEN", "Peruvian Sol"),
    "cuzco": ("S/", "PEN", "Peruvian Sol"),
    "colombia": ("COL$", "COP", "Colombian Peso"),
    "bogota": ("COL$", "COP", "Colombian Peso"),
    "cartagena": ("COL$", "COP", "Colombian Peso"),
    "chile": ("CL$", "CLP", "Chilean Peso"),
    "santiago": ("CL$", "CLP", "Chilean Peso"),
    "costa rica": ("₡", "CRC", "Costa Rican Colón"),
    "caribbean": ("$", "USD", "US Dollar"),
    "cuba": ("CUC", "CUC", "Cuban Peso"),
    # Africa
    "cape town": ("R", "ZAR", "South African Rand"),
    "south africa": ("R", "ZAR", "South African Rand"),
    "johannesburg": ("R", "ZAR", "South African Rand"),
    "cairo": ("E£", "EGP", "Egyptian Pound"),
    "egypt": ("E£", "EGP", "Egyptian Pound"),
    "marrakech": ("د.م.", "MAD", "Moroccan Dirham"),
    "morocco": ("د.م.", "MAD", "Moroccan Dirham"),
    "kenya": ("KSh", "KES", "Kenyan Shilling"),
    "nairobi": ("KSh", "KES", "Kenyan Shilling"),
    "tanzania": ("TSh", "TZS", "Tanzanian Shilling"),
    "zanzibar": ("TSh", "TZS", "Tanzanian Shilling"),
    "ethiopia": ("Br", "ETB", "Ethiopian Birr"),
    "accra": ("GH₵", "GHS", "Ghanaian Cedi"),
    "ghana": ("GH₵", "GHS", "Ghanaian Cedi"),
    # Default
    "usa": ("$", "USD", "US Dollar"),
    "united states": ("$", "USD", "US Dollar"),
}

# Budget tier -> approximate price ranges per day per person in USD (used as fallback reference)
_BUDGET_TIERS = {
    "Essential": (50, 100),
    "Premier": (150, 300),
    "Elite": (300, 600),
    "Legendary": (600, 1500),
}

# Approximate USD to local currency multipliers (rough, for fallback pricing)
_USD_TO_LOCAL_APPROX = {
    "EUR": 0.92, "GBP": 0.79, "CHF": 0.88, "JPY": 150.0, "CNY": 7.25,
    "HKD": 7.82, "TWD": 32.0, "THB": 35.0, "IDR": 15800.0, "SGD": 1.35,
    "MYR": 4.70, "PHP": 56.0, "KRW": 1350.0, "INR": 83.5, "VND": 24500.0,
    "KHR": 4100.0, "LKR": 310.0, "NPR": 133.5,
    "AED": 3.67, "QAR": 3.64, "SAR": 3.75, "OMR": 0.385, "BHD": 0.376,
    "ILS": 3.65, "JOD": 0.709,
    "AUD": 1.55, "NZD": 1.68,
    "CAD": 1.36, "MXN": 17.0, "BRL": 5.0, "ARS": 350.0,
    "PEN": 3.70, "COP": 3950.0, "CLP": 920.0, "CRC": 520.0, "CUC": 1.0,
    "ZAR": 18.5, "EGP": 48.0, "MAD": 10.0, "KES": 153.0, "TZS": 2500.0,
    "ETB": 56.0, "GHS": 15.5, "DKK": 6.85, "SEK": 10.5, "NOK": 10.8,
    "CZK": 23.0, "HUF": 360.0, "TRY": 32.5, "RUB": 92.0, "RSD": 108.0,
}


def _get_currency_for_destination(destination: str) -> tuple:
    """Return (symbol, code, name) for the destination's local currency.
    Falls back to USD if no match is found."""
    dest_lower = destination.lower()
    # Try longest match first to handle multi-word destinations
    for key in sorted(_CURRENCY_MAP.keys(), key=len, reverse=True):
        if key in dest_lower:
            return _CURRENCY_MAP[key]
    return ("$", "USD", "US Dollar")


def _format_price_range(destination: str, budget_tier: str, duration: int) -> str:
    """Generate a price_range string in the destination's local currency."""
    symbol, code, name = _get_currency_for_destination(destination)
    tier_bounds = _BUDGET_TIERS.get(budget_tier, (150, 300))
    multiplier = _USD_TO_LOCAL_APPROX.get(code, 1.0)

    low_usd = tier_bounds[0] * duration
    high_usd = tier_bounds[1] * duration

    low_local = round(low_usd * multiplier)
    high_local = round(high_usd * multiplier)

    # Format with thousand separators for readability
    low_str = f"{low_local:,}"
    high_str = f"{high_local:,}"

    return f"{symbol}{low_str} - {symbol}{high_str} {name}"


def _get_budget_cost_guide(destination: str, budget_tier: str) -> str:
    """Generate budget-specific cost guidelines for activity costs, transport, and meals.
    Returns a string with per-person cost ranges in local currency for the LLM to follow."""
    symbol, code, name = _get_currency_for_destination(destination)
    multiplier = _USD_TO_LOCAL_APPROX.get(code, 1.0)

    # Per-person daily cost ranges in USD for each budget tier
    tiers = {
        'Essential': {
            'meal_breakfast': (3, 8), 'meal_lunch': (5, 12), 'meal_dinner': (8, 18),
            'attraction_entry': (3, 15), 'local_transport': (1, 5),
            'taxi_ride': (3, 10), 'coffee_snack': (1, 3),
        },
        'Premier': {
            'meal_breakfast': (8, 20), 'meal_lunch': (15, 35), 'meal_dinner': (25, 60),
            'attraction_entry': (10, 40), 'local_transport': (3, 10),
            'taxi_ride': (8, 25), 'coffee_snack': (3, 8),
        },
        'Elite': {
            'meal_breakfast': (20, 50), 'meal_lunch': (35, 80), 'meal_dinner': (60, 150),
            'attraction_entry': (25, 80), 'local_transport': (8, 20),
            'taxi_ride': (15, 50), 'coffee_snack': (5, 15),
        },
        'Legendary': {
            'meal_breakfast': (50, 120), 'meal_lunch': (80, 200), 'meal_dinner': (150, 400),
            'attraction_entry': (50, 200), 'local_transport': (15, 40),
            'taxi_ride': (30, 100), 'coffee_snack': (10, 30),
        },
    }

    tier = tiers.get(budget_tier, tiers['Premier'])

    def fmt(usd_range):
        low = round(usd_range[0] * multiplier)
        high = round(usd_range[1] * multiplier)
        return f"{symbol}{low:,}-{symbol}{high:,} {name}"

    guide = (
        f"BUDGET TIER: {budget_tier} — ALL costs below are PER PERSON and MUST stay within these ranges:\n"
        f"- Breakfast: {fmt(tier['meal_breakfast'])}\n"
        f"- Lunch: {fmt(tier['meal_lunch'])}\n"
        f"- Dinner: {fmt(tier['meal_dinner'])}\n"
        f"- Attraction entry: {fmt(tier['attraction_entry'])}\n"
        f"- Local transport (metro/bus): {fmt(tier['local_transport'])}\n"
        f"- Taxi/rideshare ride: {fmt(tier['taxi_ride'])}\n"
        f"- Coffee/snack: {fmt(tier['coffee_snack'])}\n"
        f"CRITICAL: The total price_range ({_format_price_range(destination, budget_tier, 3)}) must match the sum of daily costs. "
        f"Do NOT invent costs outside these ranges. Walking is always Free."
    )
    return guide


def add_message(left: list, right: list) -> list:
    """Helper function to add messages"""
    return left + right

from agents.tools.travel import (
    search_destination_info, 
    search_weather_info, 
    search_hotels, 
    search_restaurants, 
    search_attractions, 
    search_local_tips, 
    search_budget_info,
    search_local_transport_options,
    search_car_rentals,
    search_real_time_transit_info,
    search_travel_blogs,
    geocode_place,
    search_place_comprehensive,
    extract_sources_from_text,
)


def _search_fallback(query: str, max_retries: int = 2) -> str:
    """Search DuckDuckGo and return formatted results as a fallback when LLM is unavailable."""
    import time as _time
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(
                    query,
                    max_results=5,
                    region=config.DUCKDUCKGO_REGION,
                    safesearch=config.DUCKDUCKGO_SAFESEARCH
                ))
                if not results:
                    if attempt < max_retries:
                        _time.sleep(1)
                        continue
                    return f"No search results available for: {query}"
                formatted = []
                for i, r in enumerate(results[:5], 1):
                    formatted.append(f"{i}. {r.get('title', 'N/A')}\n   {r.get('body', 'No details')}\n   Source: {r.get('href', '')}")
                return "\n".join(formatted)
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                _time.sleep(1)
                continue
            return f"Search unavailable: {str(e)}"
    return f"Search unavailable after {max_retries + 1} attempts: {last_error}"


def _build_fallback_itinerary(destination: str, duration: int, interests: List[str], budget_range: str, pace: str, group_type: str, group_size: int, dietary: List[str], accessibility: List[str], accommodation: str, occasion: str) -> Dict[str, Any]:
    """Build a fallback itinerary using DuckDuckGo search when LLM is unavailable."""
    # Search for top attractions
    attraction_query = f"{destination} top attractions must visit places things to do"
    attraction_results = _search_fallback(attraction_query)

    # Search for restaurants
    dietary_str = ', '.join(dietary) if dietary else ''
    restaurant_query = f"{destination} best restaurants local food {dietary_str}"
    restaurant_results = _search_fallback(restaurant_query)

    # Search for local tips
    tips_query = f"{destination} travel tips local guide cultural etiquette"
    tips_results = _search_fallback(tips_query)

    # Build activities from search results
    activities_per_day = {'Relaxed': 2, 'Moderate': 3, 'Active': 4, 'Intense': 5}.get(pace, 3)

    days = []
    for day_num in range(1, duration + 1):
        day_activities = []
        # Determine transport mode and cost based on budget tier
        transport_options = {
            'Essential': {'mode': 'Public Bus', 'cost_base': 2},
            'Premier': {'mode': 'Metro/Train', 'cost_base': 5},
            'Elite': {'mode': 'Private Taxi', 'cost_base': 15},
            'Legendary': {'mode': 'Private Chauffeur', 'cost_base': 40},
        }
        transport = transport_options.get(budget_range, transport_options['Premier'])
        symbol, code, _ = _get_currency_for_destination(destination)
        multiplier = _USD_TO_LOCAL_APPROX.get(code, 1.0)
        transport_cost_local = round(transport['cost_base'] * multiplier)
        transport_cost_str = f"Free" if transport['mode'] == 'Walking' else f"{symbol}{transport_cost_local:,} {name}"

        # Morning activity from attractions
        day_activities.append({
            'time': '09:00 AM',
            'title': f'Day {day_num} Morning - Explore {destination}',
            'description': f'Discover the highlights of {destination}. Search results suggest visiting the top-rated attractions and cultural sites.',
            'location': destination,
            'tag': 'Culture',
            'map_query': destination,
            'transport_to_next': {
                'mode': transport['mode'],
                'duration': '20 min',
                'cost': transport_cost_str,
                'instructions': f'Take {transport["mode"]} to your next destination in {destination}'
            }
        })
        if activities_per_day >= 2:
            day_activities.append({
                'time': '12:30 PM',
                'title': f'Day {day_num} Lunch - Local Dining',
                'description': f'Enjoy local cuisine at recommended restaurants in {destination}. {"Dietary-friendly options: " + ", ".join(dietary) if dietary else "Explore authentic local dishes."}',
                'location': f'{destination} dining area',
                'tag': 'Gastronomy',
                'map_query': f'restaurants in {destination}',
                'transport_to_next': {
                    'mode': 'Walking',
                    'duration': '15 min',
                    'cost': 'Free',
                    'instructions': f'Walk to your afternoon destination in {destination}'
                }
            })
        if activities_per_day >= 3:
            day_activities.append({
                'time': '03:00 PM',
                'title': f'Day {day_num} Afternoon - Discovery',
                'description': f'Continue exploring {destination} with afternoon activities tailored to your interests: {", ".join(interests) if interests else "general sightseeing"}.',
                'location': f'{destination} attractions',
                'tag': interests[0] if interests else 'Adventure',
                'map_query': f'things to do in {destination}'
            })

        day_themes = ['Arrival & Discovery', 'Cultural Immersion', 'Local Exploration', 'Hidden Gems',
                       'Signature Experiences', 'Adventure Day', 'Relaxation & Reflection',
                       'Farewell & Memories']
        theme_idx = (day_num - 1) % len(day_themes)

        days.append({
            'day_number': day_num,
            'day_name': f'Day {day_num}',
            'theme': day_themes[theme_idx],
            'activities': day_activities
        })

    occasion_note = f' Special occasion: {occasion}. We have included thoughtful touches for your celebration.' if occasion else ''
    dietary_note = f' Dietary accommodations ({", ".join(dietary)}) have been considered for all dining recommendations.' if dietary else ''
    acc_note = f' Preferred accommodation type: {accommodation}.' if accommodation and accommodation != 'No preference' else ''

    local_price_range = _format_price_range(destination, budget_range, duration)

    return {
        'trip_title': f'Discover {destination} - Your {duration}-Day Journey',
        'overview': f'Experience the best of {destination} over {duration} days. This itinerary is tailored for {group_size} {group_type.lower()} traveler(s) with interests in {", ".join(interests) if interests else "sightseeing"}.{occasion_note}{dietary_note}{acc_note}',
        'sustainability_score': 75,
        'price_range': local_price_range,
        'concierge_note': f'Welcome to {destination}! We have prepared this itinerary based on the best available local information. For a fully AI-curated experience, please try again when our travel architects are available.',
        'days': days
    }


def _build_fallback_weather(destination: str, travel_dates: str) -> Dict[str, Any]:
    """Build fallback weather data using DuckDuckGo search."""
    query = f"{destination} weather forecast {travel_dates} travel climate"
    results = _search_fallback(query)
    return {
        'destination': destination,
        'travel_dates': travel_dates,
        'temperature_c': {
            'expected_low': 15.0,
            'expected_high': 25.0,
            'typical_range': '15-25°C',
            'notes': 'Based on general climate data. Search results: ' + results[:200]
        },
        'conditions_summary': 'Check local weather services for current conditions',
        'best_times': ['Morning', 'Afternoon'],
        'activity_suggestions': ['Sightseeing', 'Walking tours'],
        'packing': ['Layered clothing', 'Comfortable walking shoes', 'Sun protection']
    }


def _build_fallback_transport(destination: str, origin: str, duration: int, budget_range: str) -> Dict[str, Any]:
    """Build fallback transport data using DuckDuckGo search."""
    query = f"{destination} public transport how to get around getting there from {origin or 'major cities'}"
    results = _search_fallback(query)
    return {
        'flights': {'notes': f'Search for flights from {origin or "your origin"} to {destination}. ' + results[:200]},
        'regional_trains_buses': {'notes': 'Check local rail and bus schedules.', 'cost_vs_time_analysis': 'Varies by route'},
        'car_rentals': {'options': [], 'notes': 'Car rental available at destination.'},
        'airport_transfers': {'options': [{'mode': 'Taxi', 'why': 'Most convenient', 'cost_estimate': 'Varies', 'typical_time_min': None}], 'notes': 'Standard airport transfer options available.'},
        'local_transport': {'how_to_get_around': ['Public transport', 'Walking', 'Taxi/Rideshare'], 'apps': [], 'passes': [], 'notes': results[:200]},
        'route_optimization': {'strategy': 'Visit nearby attractions together to minimize transit time.', 'suggested_area_groupings': [], 'sample_day_route_stops': [], 'google_maps_directions_url': ''}
    }


def _build_fallback_local_expert(destination: str, interests: List[str]) -> Dict[str, Any]:
    """Build fallback local expert data using DuckDuckGo search."""
    query = f"{destination} local culture customs traditions etiquette tips"
    results = _search_fallback(query)
    return {
        'summary': f'Discover the living culture of {destination}. Local insights gathered from travel sources: {results[:300]}',
        'contemporary_behaviors': {'title': 'Living Rhythms & Emerging Trends', 'insights': [results[:150] if results else f'{destination} is a vibrant destination with evolving cultural trends.']},
        'unwritten_customs': {'title': 'Unwritten Social Codes & Customs', 'insights': ['Respect local customs and traditions when visiting.', 'Ask before photographing people or sacred sites.']},
        'folklore_heritage': {'title': 'Folklore, Beliefs & Hidden Heritage', 'insights': [f'{destination} has a rich cultural heritage worth exploring.']},
        'sensory_profile': {'title': 'Sensory Signature', 'scents': ['Local herbal scents', 'Street food aromas'], 'sounds': ['City ambiance', 'Local music'], 'colors': ['Earthy tones', 'Warm colors']},
        'guidebook_vs_reality': {'title': 'Guidebook Expectations vs. Modern Reality', 'insights': ['The real experience often goes beyond guidebook descriptions.']},
        'authenticity_signals': {'title': 'Living Authenticity Signals', 'insights': ['Look for locally-owned establishments.', 'Engage with residents for authentic experiences.']}
    }


class TravelPlanState(TypedDict):
    messages:Annotated[List[HumanMessage|AIMessage|SystemMessage],add_message]
    origin:str
    destination:str
    duration:int
    budget_range:str
    interests:List[str] 
    group_size:int
    travel_dates:str
    group_type:str
    dietary_requirements:List[str]
    accessibility:List[str]
    pace:str
    accommodation_preference:str
    occasion:str
    language_preference:str
    risk_tolerance:str
    current_agent:str
    agent_outputs:Dict[str,Any]
    final_plan:  Dict[str,Any]
    iteration_count:int
    
class LangTravelAgents:
    def __init__(self):
        # Initialize LLM based on configured provider
        if config.LLM_PROVIDER == "groq":
            try:
                from langchain_groq import ChatGroq
                self.llm_type = "groq"
                self.llm = ChatGroq(
                    model=config.GROQ_MODEL,
                    groq_api_key=config.GROQ_API_KEY,
                    temperature=config.TEMPERATURE,
                    max_tokens=config.MAX_TOKENS,
                )
                print(f"[SUCCESS] Using Groq with model: {config.GROQ_MODEL}")
                print(f"[INFO] Fast and reliable - great free tier!")
                
                # Initialize OpenRouter as fallback
                self.fallback_llm = ChatOpenAI(
                    model=config.OPENROUTER_MODEL,
                    openai_api_key=config.OPENROUTER_API_KEY,
                    base_url="https://openrouter.ai/api/v1",
                    temperature=config.TEMPERATURE,
                    max_tokens=config.MAX_TOKENS,
                    timeout=60,
                    max_retries=2,
                    model_kwargs={
                        "extra_headers": {
                            "HTTP-Referer": "https://github.com/ryan1234814/XPLORA-Travel-Agent",
                            "X-Title": "XPLORA Travel Agent",
                        }
                    }
                )
                print("[INFO] Initialized OpenRouter as fallback LLM.")
            except ImportError:
                print("[ERROR] langchain-groq not installed. Run: pip install langchain-groq")
                print("[INFO] Falling back to OpenRouter...")
                config.LLM_PROVIDER = "openrouter"
        
        elif config.LLM_PROVIDER == "ollama":
            try:
                import ollama
                # Use Ollama client directly instead of LangChain wrapper
                self.ollama_client = ollama.Client(host=config.OLLAMA_BASE_URL)
                self.llm_type = "ollama"
                self.llm = None  # We'll handle invocation manually
                print(f"[SUCCESS] Using Ollama (local) with model: {config.OLLAMA_MODEL}")
                print(f"[INFO] No API keys needed - unlimited free usage!")
            except ImportError:
                print("[ERROR] ollama package not installed. Run: pip install ollama")
                print("[INFO] Falling back to OpenRouter...")
                config.LLM_PROVIDER = "openrouter"
        
        if config.LLM_PROVIDER == "openrouter":
            self.llm_type = "openrouter"
            self.llm = ChatOpenAI(
                model=config.OPENROUTER_MODEL,
                openai_api_key=config.OPENROUTER_API_KEY,
                base_url="https://openrouter.ai/api/v1",
                temperature=config.TEMPERATURE,
                max_tokens=config.MAX_TOKENS,
                timeout=60,  # Add 60 second timeout to prevent hanging
                max_retries=2,  # Retry failed requests up to 2 times
                model_kwargs={
                    "extra_headers": {
                        "HTTP-Referer": "https://github.com/ryan1234814/XPLORA-Travel-Agent",
                        "X-Title": "XPLORA Travel Agent",
                    }
                }
            )
            self.fallback_llm = None
            print(f"[SUCCESS] Using OpenRouter with model: {config.OPENROUTER_MODEL}")
        
        # Fallback for unknown providers - LLM will be unavailable, search-only mode
        if not hasattr(self, 'llm') or self.llm is None:
            self.llm_type = "unavailable"
            self.llm = None
            self.fallback_llm = None
            print(f"[WARNING] LLM provider '{config.LLM_PROVIDER}' not recognized. Running in search-only fallback mode.")
        
        self._lock = threading.Lock()
        self._last_request_time = 0
        self.request_interval = 0.1  # Reduced to 0.1 for maximum speed
        self.graph=self.create_agent_graph()

    def _invoke_with_retry(self, llm_instance, messages: List[BaseMessage], provider_name: str, max_attempts: int = 3):
        """Invoke an LLM with retry logic for rate-limit / capacity errors.
        Uses exponential backoff between attempts.
        """
        last_error = None
        for attempt in range(max_attempts):
            try:
                # Proactive Rate Limiting
                with self._lock:
                    elapsed = time.time() - self._last_request_time
                    if elapsed < self.request_interval:
                        time.sleep(self.request_interval - elapsed)
                    self._last_request_time = time.time()

                result = llm_instance.invoke(messages)
                return result
            except Exception as e:
                last_error = e
                
                if is_rate_limit_error(e) and attempt < max_attempts - 1:
                    wait_time = (2 ** attempt) * 2  # 2, 4, 8 seconds
                    print(f"[WARNING] {provider_name} capacity/rate-limit error (attempt {attempt + 1}/{max_attempts}): {str(e)[:100]}")
                    print(f"[INFO] Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                elif is_rate_limit_error(e):
                    print(f"[ERROR] {provider_name} still unavailable after {max_attempts} attempts: {str(e)[:100]}")
                else:
                    # Non-capacity error, don't retry
                    print(f"[ERROR] {provider_name} invocation failed: {type(e).__name__}: {str(e)[:200]}")
                    break
        
        # All retries exhausted or non-retryable error
        raise last_error

    def _invoke_llm(self, messages: List[BaseMessage]):
        """Robust wrapper for LLM invocation with automatic Groq -> OpenRouter fallback.
        
        Flow:
        1. If llm_type is "groq": try Groq once. On ANY failure, immediately fall back to OpenRouter (with retries).
        2. If llm_type is "openrouter": try OpenRouter directly (with retries).
        3. If llm_type is "ollama": use the Ollama client directly.
        """
        # Clean messages: Ensure no empty content
        cleaned_messages = []
        for msg in messages:
            content = _safe_message_content(msg).strip()
            if content:
                if isinstance(msg, SystemMessage):
                    cleaned_messages.append(SystemMessage(content=content))
                elif isinstance(msg, AIMessage):
                    cleaned_messages.append(AIMessage(content=content))
                else:
                    cleaned_messages.append(HumanMessage(content=content))
        
        if not cleaned_messages:
            cleaned_messages = [HumanMessage(content="Please provide travel recommendations.")]
        
        try:
            # --- UNAVAILABLE PATH (search-only fallback mode) ---
            if self.llm_type == "unavailable":
                print(f"[FALLBACK] LLM unavailable (provider: {config.LLM_PROVIDER}). Returning LLM_UNAVAILABLE for search fallback.")
                return AIMessage(content="LLM_UNAVAILABLE")

            # --- OLLAMA PATH ---
            if self.llm_type == "ollama":
                return self._invoke_ollama(cleaned_messages)
            
            # --- GROQ PATH (with automatic OpenRouter fallback) ---
            if self.llm_type == "groq":
                try:
                    # Try Groq once — fail fast on any error
                    with self._lock:
                        elapsed = time.time() - self._last_request_time
                        if elapsed < self.request_interval:
                            time.sleep(self.request_interval - elapsed)
                        self._last_request_time = time.time()
                    
                    result = self.llm.invoke(cleaned_messages)
                    print(f"[DEBUG] Groq invocation successful. Response length: {len(str(result.content)) if hasattr(result, 'content') else 'N/A'}")
                    return result
                except Exception as groq_err:
                    groq_err_str = str(groq_err)[:200]
                    print(f"[WARNING] Groq invocation failed: {type(groq_err).__name__}: {groq_err_str}")
                    
                    if hasattr(self, "fallback_llm") and self.fallback_llm:
                        print(f"[INFO] Automatically falling back to OpenRouter...")
                        # Retry OpenRouter up to 3 times with backoff
                        try:
                            fallback_result = self._invoke_with_retry(
                                self.fallback_llm, cleaned_messages, 
                                provider_name="OpenRouter (Groq fallback)", max_attempts=3
                            )
                            print(f"[SUCCESS] OpenRouter fallback generated response.")
                            return fallback_result
                        except Exception as fb_err:
                            print(f"[ERROR] OpenRouter fallback also failed: {type(fb_err).__name__}: {str(fb_err)[:200]}")
                            return AIMessage(content="LLM_UNAVAILABLE")
                    else:
                        return AIMessage(content="LLM_UNAVAILABLE")
            
            # --- OPENROUTER PATH (primary, with retries) ---
            if self.llm_type == "openrouter":
                try:
                    result = self._invoke_with_retry(
                        self.llm, cleaned_messages,
                        provider_name="OpenRouter", max_attempts=3
                    )
                    print(f"[DEBUG] OpenRouter invocation successful.")
                    return result
                except Exception as or_err:
                    print(f"[ERROR] OpenRouter invocation failed after retries: {type(or_err).__name__}: {str(or_err)[:200]}")
                    return AIMessage(content="LLM_UNAVAILABLE")
                    
        except Exception as e:
            print(f"[ERROR] All LLM invocation paths failed: {type(e).__name__}: {str(e)[:200]}")
            print(f"[ERROR] Full error details: {repr(e)}")
            return AIMessage(content="LLM_UNAVAILABLE")

    def _invoke_ollama(self, cleaned_messages: List[BaseMessage]):
        """Invoke Ollama directly via its Python client."""
        ollama_messages = []
        for msg in cleaned_messages:
            if isinstance(msg, SystemMessage):
                ollama_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, AIMessage):
                ollama_messages.append({"role": "assistant", "content": msg.content})
            else:
                ollama_messages.append({"role": "user", "content": msg.content})
        
        print(f"[DEBUG] Calling Ollama with {len(ollama_messages)} messages")
        if ollama_messages:
            print(f"[DEBUG] First message preview: {ollama_messages[0]['content'][:100]}...")
        
        response = self.ollama_client.chat(
            model=config.OLLAMA_MODEL,
            messages=ollama_messages,
            options={
                "temperature": config.TEMPERATURE,
                "num_predict": config.MAX_TOKENS,
                "num_ctx": 8192,
            }
        )
        
        try:
            content = response.message.content
            print(f"[DEBUG] Ollama API response length: {len(content)}")
            return AIMessage(content=content)
        except AttributeError as e:
            print(f"[ERROR] Failed to extract Ollama response content: {e}")
            print(f"[ERROR] Response object: {response}")
            return AIMessage(content="")

    def create_agent_graph(self)->StateGraph:
        workflow=StateGraph(TravelPlanState)
        workflow.add_node("travel_advisor",self._travel_advisor_agent)
        workflow.add_node("weather_analyst",self._weather_analyst_agent)
        workflow.add_node("budget_optimizer",self._budget_optimizer_agent)
        workflow.add_node("local_expert",self._local_expert_agent)
        workflow.add_node("transport_mobility",self._transport_mobility_agent)
        workflow.add_node("itinerary_planner",self._itinerary_planner_agent)
        workflow.add_node("coordinator",self._coordinator_agent)
        workflow.add_node("tool_executor",self._tool_executor_agent)
        workflow.set_entry_point("coordinator")
        workflow.add_conditional_edges(
            "coordinator",
            self._coordinator_router,
            {
                "travel_advisor": "travel_advisor",
                "weather_analyst": "weather_analyst", 
                "budget_optimizer": "budget_optimizer",
                "local_expert": "local_expert",
                "transport_mobility": "transport_mobility",
                "itinerary_planner": "itinerary_planner",
                "tools": "tool_executor",
                "end": END
            }
        )
        for agent in  ["travel_advisor", "weather_analyst", "budget_optimizer", "local_expert", "transport_mobility", "itinerary_planner"]:
            workflow.add_conditional_edges(
                agent,
                self._agent_router,
                {
                    "tools": "tool_executor",
                    "coordinator": "coordinator",
                    "end": END
                }
            )
        workflow.add_edge("tool_executor","coordinator")
        return workflow.compile()
        
    def _coordinator_agent(self,state:TravelPlanState)->TravelPlanState:
         """Coordinator that ensures all agents run to provide complete data."""
         agent_outputs = state.get('agent_outputs', {})
         iteration = state.get('iteration_count', 0)
         
         weather_status = agent_outputs.get('weather_analyst', {}).get('status')
         transport_status = agent_outputs.get('transport_mobility', {}).get('status')
         local_expert_status = agent_outputs.get('local_expert', {}).get('status')
         
         # Sequential execution checking explicit completion status: weather -> transport -> local_expert -> itinerary
         if not weather_status or weather_status == 'searching':
             response = AIMessage(content="weather_analyst")
         elif not transport_status or transport_status == 'searching':
             response = AIMessage(content="transport_mobility")
         elif not local_expert_status or local_expert_status == 'searching':
             response = AIMessage(content="local_expert")
         elif 'itinerary_planner' not in agent_outputs:
             response = AIMessage(content="itinerary_planner")
         else:
             response = AIMessage(content="FINAL_PLAN")
         
         new_state=state.copy()
         new_state["messages"]=state.get("messages",[])+[response] 
         new_state["current_agent"] = "coordinator"
         new_state["iteration_count"] = iteration + 1
         return new_state
         
    def _coordinator_router(self, state: TravelPlanState) -> str:
        """Router to determine next step from coordinator"""
        messages = state.get("messages", [])
        if not messages:
            return "travel_advisor"
        last = messages[-1]
        content = getattr(last, "content", "") or ""
        content_lower = content.lower()
        if "travel_advisor" in content_lower:
            return "travel_advisor"
        if "weather_analyst" in content_lower:
            return "weather_analyst"
        if "budget_optimizer" in content_lower:
            return "budget_optimizer"
        if "local_expert" in content_lower:
            return "local_expert"
        if "transport_mobility" in content_lower or "transport" in content_lower or "mobility" in content_lower:
            return "transport_mobility"
        if "itinerary_planner" in content_lower:
            return "itinerary_planner"
        if "search" in content_lower:
            return "tools"
        if "final_plan" in content_lower:
            # If they say final plan but haven't run the itinerary planner, force it
            if "itinerary_planner" not in state.get("agent_outputs", {}):
                return "itinerary_planner"
            return "end"
        return "travel_advisor"

    def _travel_advisor_agent(self,state:TravelPlanState)->TravelPlanState:
        group_type = state.get('group_type', 'Couple')
        occasion = state.get('occasion', '')
        pace = state.get('pace', 'Moderate')
        risk_tolerance = state.get('risk_tolerance', 'Balanced')
        dietary = ', '.join(state.get('dietary_requirements', [])) or 'None'
        accommodation = state.get('accommodation_preference', 'No preference')
        accessibility = ', '.join(state.get('accessibility', [])) or 'None'
        language_pref = state.get('language_preference', 'English only')

        system_prompt = f"""You are the Travel Advisor Agent, specialized in destination expertise and recommendations.

Your expertise includes:
- Destination knowledge and highlights
- Attraction recommendations
- Cultural insights and tips
- Best practices for travelers

Current planning request:
- Destination: {state.get('destination')}
- Duration: {state.get('duration')} days
- Interests: {', '.join(state.get('interests', []))}
- Group size: {state.get('group_size')}
- Group type: {group_type}
- Pace: {pace}
- Occasion: {occasion or 'None'}
- Risk tolerance: {risk_tolerance}
- Dietary requirements: {dietary}
- Accessibility needs: {accessibility}
- Accommodation preference: {accommodation}
- Language preference: {language_pref}

Your task: Provide comprehensive destination advice including:
1. ALWAYS use 'NEED_SEARCH: [query]' first to get current, real-world data about the destination, its top attractions, and local gems.
2. Based on the search results, suggest must-see places and cultural insights.
3. Activity recommendations based on interests, group type, pace, and accessibility needs.
4. If a special occasion is mentioned, suggest relevant venues/experiences.
5. Consider dietary requirements when recommending food experiences.

If you have already received search results in the conversation history, proceed with your expert recommendations. Otherwise, start with a search.
"""
        messages=[SystemMessage(content=system_prompt)]
        if state.get("messages"):
            messages.extend(state["messages"][-2:])
        
        response = self._invoke_llm(messages)
        agent_outputs=state.get("agent_outputs",{})
        response_text = _safe_message_content(response)

        # LLM fallback: use DuckDuckGo search if LLM is unavailable
        if 'LLM_UNAVAILABLE' in response_text:
            print(f"[FALLBACK] LLM unavailable for travel_advisor, using DuckDuckGo search...")
            dest = state.get('destination', '')
            interests_str = ', '.join(state.get('interests', []))
            search_q = f"{dest} travel guide attractions {interests_str}"
            search_results = _search_fallback(search_q)
            response_text = f"Destination Guide for {dest}:\n\n{search_results}\n\nInterests: {interests_str}\nGroup: {state.get('group_size', 2)} {state.get('group_type', 'Couple')} travelers\n\nNote: AI-powered personalized recommendations are temporarily unavailable. The above are general destination highlights sourced from web search."
            response = AIMessage(content=response_text)

        agent_outputs["travel_advisor"]={
            "response": response_text,
            "output": response_text,
            "timestamp":datetime.now().isoformat(),
            "status":"completed"
        }
        new_state=state.copy()
        new_state['messages']=state.get('messages',[])+[response]
        new_state['agent_outputs']=agent_outputs
        new_state['current_agent']='travel_advisor'
        
        return new_state

    def _weather_analyst_agent(self, state: TravelPlanState) -> TravelPlanState:
        # 1. Try real-time weather from Tomorrow.io when available.
        try:
            if api_config.TOMORROW_IO_API_KEY and state.get('destination'):
                params = {
                    "location": state.get('destination'),
                    "apikey": api_config.TOMORROW_IO_API_KEY
                }
                resp = requests.get(f"{api_config.WEATHER_BASE_URL}/realtime", params=params, timeout=15)
                if resp.status_code == 200:
                    data = resp.json() or {}
                    values = data.get("data", {}).get("values", {})
                    
                    parsed = {
                        "destination": state.get('destination'),
                        "travel_dates": state.get('travel_dates'),
                        "temperature_c": {
                            "expected_low": values.get("temperature"),
                            "expected_high": values.get("temperature"),
                            "typical_range": f"{values.get('temperature', 'N/A')}°C",
                            "notes": f"Current: {values.get('temperature')}°C (Apparent: {values.get('temperatureApparent')}°C)"
                        },
                        "conditions_summary": f"Cloud cover: {values.get('cloudCover')}%, Wind: {values.get('windSpeed')} m/s",
                        "best_times": ["Morning exploration", "Evening walk"],
                        "activity_suggestions": ["Standard activities based on current conditions"],
                        "packing": ["Standard travel wear based on temperature"],
                        "source": "tomorrow_io"
                    }

                    normalized = _normalize_weather_data(parsed, state.get('destination', ''), state.get('travel_dates', ''))
                    agent_outputs = state.get("agent_outputs", {})
                    agent_outputs["weather_analyst"] = {
                        "response": json.dumps(normalized),
                        "output": normalized,
                        "timestamp": datetime.now().isoformat(),
                        "status": "completed"
                    }
                    new_state = state.copy()
                    new_state["messages"] = state.get("messages", []) + [AIMessage(content=json.dumps(normalized))]
                    new_state["current_agent"] = "weather_analyst"
                    new_state["agent_outputs"] = agent_outputs
                    return new_state
        except Exception:
            pass

        
        agent_outputs = state.get("agent_outputs", {})
        search_results = agent_outputs.get("weather_analyst", {}).get("search_results")
        
        # 3. If no search results and no OpenWeather, request a search
        if not search_results and state.get('destination'):
            response_content = f"NEED_SEARCH: {state.get('destination')} current weather and climate forecast for {state.get('travel_dates') or 'upcoming days'}"
            new_state = state.copy()
            new_state["messages"] = state.get("messages", []) + [AIMessage(content=response_content)]
            new_state["current_agent"] = "weather_analyst"
            
            # Initialize the output entry so tool executor knows where to put results
            if "weather_analyst" not in agent_outputs:
                agent_outputs["weather_analyst"] = {}
            agent_outputs["weather_analyst"]["status"] = "searching"
            new_state["agent_outputs"] = agent_outputs
            return new_state

        # 4. Final Fallback/Synthesis with search results or general knowledge
        system_prompt = f"""You are the Weather Analyst Agent. 
Provide weather-intelligent recommendations for {state.get('destination')} during {state.get('travel_dates')}.

Search Results provided to you: {search_results or "No specific search results found, use general climate knowledge."}

Your task: Return a STRICT JSON object. 
DO NOT include any markdown code blocks, backticks, or conversational text. 
ONLY return the JSON object itself.

Schema:
{{
  "destination": string,
  "travel_dates": string,
  "temperature_c": {{
    "expected_low": number | null,
    "expected_high": number | null,
    "typical_range": string,
    "notes": string
  }},
  "conditions_summary": string,
  "best_times": [string],
  "activity_suggestions": [string],
  "packing": [string]
}}
"""
        messages = [SystemMessage(content=system_prompt)]
        if state.get("messages"):
            messages.extend(state["messages"][-2:])
        
        response = self._invoke_llm(messages)
        response_text = _safe_message_content(response)

        # LLM fallback: use DuckDuckGo search if LLM is unavailable
        if 'LLM_UNAVAILABLE' in response_text:
            print(f"[FALLBACK] LLM unavailable for weather_analyst, using search data...")
            # Use existing search results directly to avoid redundant DuckDuckGo calls
            if search_results and search_results != 'Search fallback used':
                parsed_normalized = _normalize_weather_data({}, state.get('destination', ''), state.get('travel_dates', ''))
                parsed_normalized['temperature_c']['notes'] = f"Based on search results: {str(search_results)[:300]}"
                parsed_normalized['conditions_summary'] = f"Weather info gathered from local sources for {state.get('destination', '')}."
            else:
                parsed_normalized = _build_fallback_weather(state.get('destination', ''), state.get('travel_dates', ''))
            agent_outputs["weather_analyst"] = {
                "response": json.dumps(parsed_normalized),
                "output": parsed_normalized,
                "timestamp": datetime.now().isoformat(),
                "status": "completed",
                "search_results": search_results or 'Search fallback used'
            }
            new_state = state.copy()
            new_state["messages"] = state.get("messages", []) + [AIMessage(content=json.dumps(parsed_normalized))]
            new_state["current_agent"] = "weather_analyst"
            new_state["agent_outputs"] = agent_outputs
            return new_state

        parsed = _try_parse_json(response_text)
        
        parsed_normalized = _normalize_weather_data(parsed, state.get('destination', ''), state.get('travel_dates', ''))
        agent_outputs["weather_analyst"] = {
            "response": response_text,
            "output": parsed_normalized,
            "timestamp": datetime.now().isoformat(),
            "status": "completed",
            "search_results": search_results
        }
        new_state = state.copy()
        new_state["messages"] = state.get("messages", []) + [response]
        new_state["current_agent"] = "weather_analyst"
        new_state["agent_outputs"] = agent_outputs
        return new_state
    
    def _budget_optimizer_agent(self, state: TravelPlanState) -> TravelPlanState:
        """Budget optimizer agent - stub implementation"""
        group_type = state.get('group_type', 'Couple')
        dietary = ', '.join(state.get('dietary_requirements', [])) or 'None'
        pace = state.get('pace', 'Moderate')
        accommodation = state.get('accommodation_preference', 'No preference')
        occasion = state.get('occasion', '')

        system_prompt = f"""You are the Budget Optimizer Agent, specialized in cost analysis and money-saving strategies.

Your expertise includes:
- Travel cost analysis and budgeting
- Money-saving tips and strategies
- Budget allocation recommendations
- Cost-effective alternatives

Current planning request:
- Destination: {state.get('destination')}
- Duration: {state.get('duration')} days
- Budget range: {state.get('budget_range')}
- Group size: {state.get('group_size')}
- Group type: {group_type}
- Pace: {pace}
- Accommodation preference: {accommodation}
- Dietary requirements: {dietary}
- Occasion: {occasion or 'None'}

Your task: Provide budget optimization recommendations including:
1. Estimated daily and total costs
2. Budget breakdown by category (accommodation, food, activities, transport)
3. Money-saving tips and strategies
4. Cost-effective alternatives for expensive activities
5. Consider group size and type for shared costs and discounts
6. Factor in accommodation preference for accurate lodging estimates
7. If a special occasion is noted, budget for relevant extras

If you need current pricing information, respond with 'NEED_SEARCH: [budget search query]'
Otherwise, provide your budget analysis and recommendations.
"""
        
        messages = [SystemMessage(content=system_prompt)]
        if state.get("messages"):
            messages.extend(state["messages"][-2:])
        
        response = self._invoke_llm(messages)
        response_text = _safe_message_content(response)

        # LLM fallback: use DuckDuckGo search if LLM is unavailable
        if 'LLM_UNAVAILABLE' in response_text:
            print(f"[FALLBACK] LLM unavailable for budget_optimizer, using DuckDuckGo search...")
            dest = state.get('destination', '')
            dur = state.get('duration', 3)
            budget_q = f"{dest} travel budget {dur} days costs expenses {state.get('budget_range', '')}"
            search_results = _search_fallback(budget_q)
            response_text = f"Budget Analysis for {dest} ({state.get('budget_range', 'Premier')} tier, {dur} days, {state.get('group_size', 2)} travelers):\n\n{search_results}\n\nNote: AI-powered budget analysis is temporarily unavailable. The above are general cost estimates sourced from web search."
            response = AIMessage(content=response_text)

        agent_outputs = state.get("agent_outputs", {})
        agent_outputs["budget_optimizer"] = {
            "response": response_text,
            "output": response_text,
            "timestamp": datetime.now().isoformat(),
            "status": "completed"
        }

        new_state = state.copy()
        new_state["messages"] = state.get("messages", []) + [response]
        new_state["current_agent"] = "budget_optimizer"
        new_state["agent_outputs"] = agent_outputs
        return new_state


    def _transport_mobility_agent(self, state: TravelPlanState) -> TravelPlanState:
        """Transport & Mobility agent - produces structured JSON for end-to-end movement planning."""
        destination = state.get('destination', '')
        origin = state.get('origin', '')

        # Use DuckDuckGo search to gather real transport data for the destination
        print(f"[INFO] Using DuckDuckGo search for {destination} mobility planning...")
        search_results_combined = []
        try:
            # Search for local transport options
            transport_result = search_local_transport_options.invoke({"destination": f"{destination} public transport metro bus taxi how to get around"})
            if transport_result:
                search_results_combined.append("LOCAL TRANSPORT:\n" + str(transport_result))
            # Search for car rental options
            car_result = search_car_rentals.invoke({"destination": f"{destination} car rental options prices"})
            if car_result:
                search_results_combined.append("CAR RENTALS:\n" + str(car_result))
            # Search for flights and train options
            transit_result = search_real_time_transit_info.invoke({"destination": f"{destination} flights trains bus routes from {origin or 'major cities'}"})
            if transit_result:
                search_results_combined.append("FLIGHTS & TRAINS:\n" + str(transit_result))
        except Exception as search_err:
            print(f"[WARNING] DuckDuckGo transport search encountered an issue: {search_err}")

        # Fallback implementation using the default LLM with search context
        print(f"[INFO] Synthesizing transport data via LLM for {destination}...")
        search_context = chr(10).join(search_results_combined) if search_results_combined else "No specific search results found, use general transport knowledge."

        group_type = state.get('group_type', 'Couple')
        accessibility = ', '.join(state.get('accessibility', [])) or 'None'
        accommodation = state.get('accommodation_preference', 'No preference')
        pace = state.get('pace', 'Moderate')

        system_prompt = f"""You are the Transport & Mobility Agent.

Purpose: End-to-end movement planning for a trip.

Trip context:
- Origin (if provided): {origin}
- Destination: {destination}
- Duration: {state.get('duration')} days
- Group size: {state.get('group_size')}
- Group type: {group_type}
- Budget tier: {state.get('budget_range')}
- Interests: {', '.join(state.get('interests', []))}
- Accessibility needs: {accessibility}
- Pace: {pace}

Search Results from DuckDuckGo:
{search_context}

Your tasks:
1) Flight search & comparison guidance (best sites/strategies, what to compare).
2) Train/bus options (region-specific tips, when rail is better than flying).
3) Airport transfer suggestions (best options, rough timing, pitfalls).
4) Local transport guidance (cards/passes, apps, etiquette, safety).
5) Route optimization between attractions: propose an efficient order and grouping strategy.
6) Consider accessibility needs when recommending transport modes.
7) For larger groups, suggest cost-effective shared transport options.

Use the search results above to populate real, specific transport details. If search results are sparse, supplement with your knowledge.

IMPORTANT: Return STRICT JSON with this schema (no markdown):
{{
  "flights": {{
    "recommended_search_queries": [string],
    "comparison_tips": [string],
    "notes": string
  }},
  "regional_trains_buses": {{
    "recommended_search_queries": [string],
    "provider_hints": [string],
    "cost_vs_time_analysis": string,
    "notes": string
  }},
  "car_rentals": {{
    "recommended_search_queries": [string],
    "options": [{{ "company": string, "estimated_daily_rate": string, "pros_cons": string }}],
    "notes": string
  }},
  "airport_transfers": {{
    "recommended_search_queries": [string],
    "options": [{{ "mode": string, "why": string, "cost_estimate": string, "typical_time_min": number|null }}],
    "notes": string
  }},
  "local_transport": {{
    "recommended_search_queries": [string],
    "how_to_get_around": [string],
    "apps": [string],
    "passes": [string],
    "real_time_info_links": [string],
    "cost_vs_time_comparison": string,
    "notes": string
  }},
  "route_optimization": {{
    "strategy": string,
    "suggested_area_groupings": [string],
    "sample_day_route_stops": [string],
    "google_maps_directions_url": string
  }}
}}"""

        messages = [SystemMessage(content=system_prompt)]
        if state.get("messages"):
            messages.extend(state["messages"][-2:])
            
        response = self._invoke_llm(messages)
        response_text = _safe_message_content(response)

        # LLM fallback: use DuckDuckGo search data if LLM is unavailable
        if 'LLM_UNAVAILABLE' in response_text:
            print(f"[FALLBACK] LLM unavailable for transport_mobility, using search data...")
            fallback_data = _build_fallback_transport(destination, origin, state.get('duration', 3), state.get('budget_range', 'Premier'))
            agent_outputs = state.get("agent_outputs", {})
            agent_outputs["transport_mobility"] = {
                "response": json.dumps(fallback_data),
                "output": fallback_data,
                "timestamp": datetime.now().isoformat(),
                "status": "completed",
                "source": "duckduckgo_fallback"
            }
            new_state = state.copy()
            new_state["messages"] = state.get("messages", []) + [AIMessage(content=json.dumps(fallback_data))]
            new_state["current_agent"] = "transport_mobility"
            new_state["agent_outputs"] = agent_outputs
            return new_state

        parsed = _try_parse_json(response_text)

        agent_outputs = state.get("agent_outputs", {})
        agent_outputs["transport_mobility"] = {
            "response": response_text,
            "output": parsed if isinstance(parsed, dict) else response_text,
            "timestamp": datetime.now().isoformat(),
            "status": "completed",
            "source": "duckduckgo_search"
        }

        new_state = state.copy()
        new_state["messages"] = state.get("messages", []) + [response]
        new_state["current_agent"] = "transport_mobility"
        new_state["agent_outputs"] = agent_outputs
        return new_state
    
    def _local_expert_agent(self, state: TravelPlanState) -> TravelPlanState:
        agent_outputs = state.get("agent_outputs", {})
        search_results = agent_outputs.get("local_expert", {}).get("search_results")
        
        # 1. Request a search if search results are not present
        if not search_results and state.get('destination'):
            response_content = f"NEED_SEARCH: {state.get('destination')} contemporary local culture unwritten customs folklore sensory characteristics"
            new_state = state.copy()
            new_state["messages"] = state.get("messages", []) + [AIMessage(content=response_content)]
            new_state["current_agent"] = "local_expert"
            
            if "local_expert" not in agent_outputs:
                agent_outputs["local_expert"] = {}
            agent_outputs["local_expert"]["status"] = "searching"
            new_state["agent_outputs"] = agent_outputs
            return new_state

        # 2. Synthesize with search results
        language_pref = state.get('language_preference', 'English only')
        dietary = ', '.join(state.get('dietary_requirements', [])) or 'None'
        occasion = state.get('occasion', '')
        cultural_mode = state.get('risk_tolerance', 'Balanced')

        system_prompt = f"""You are the Local Expert Agent, specialized in deep cultural intelligence, local behaviors, and heritage.

Your task: Propose a deep cultural-intelligence brief for a trip to {state.get('destination')} with interests: {', '.join(state.get('interests', []))}.

User preferences:
- Language: {language_pref}
- Dietary requirements: {dietary}
- Occasion: {occasion or 'None'}
- Exploration style: {cultural_mode}

Search Results provided to you: {search_results or "No specific search results found, use general cultural knowledge."}

You must return a STRICT JSON object. DO NOT include any markdown code blocks, backticks, or conversational text. ONLY return the JSON object itself.

Schema:
{{
  "summary": "A concise, evocative narrative summary (3-4 sentences) that helps the user understand how the destination feels, its living identity, and what locals experience today.",
  "contemporary_behaviors": {{
    "title": "Living Rhythms & Emerging Trends",
    "insights": [string]
  }},
  "unwritten_customs": {{
    "title": "Unwritten Social Codes & Customs",
    "insights": [string]
  }},
  "folklore_heritage": {{
    "title": "Folklore, Beliefs & Hidden Heritage",
    "insights": [string]
  }},
  "sensory_profile": {{
    "title": "Sensory Signature (Scent, Sound, Color)",
    "scents": [string],
    "sounds": [string],
    "colors": [string]
  }},
  "guidebook_vs_reality": {{
    "title": "Guidebook Expectations vs. Modern Reality",
    "insights": [string]
  }},
  "authenticity_signals": {{
    "title": "Living Authenticity Signals",
    "insights": [string]
  }}
}}

Provide realistic, contemporary local insights. In 'colors', include evocative names followed by their hex codes in parentheses, e.g. "Moss Green (#3D5230)", "Vermilion (#E60012)".
"""
        messages = [SystemMessage(content=system_prompt)]
        if state.get("messages"):
            messages.extend(state["messages"][-2:])
            
        response = self._invoke_llm(messages)
        response_text = _safe_message_content(response)

        # LLM fallback: use DuckDuckGo search data if LLM is unavailable
        if 'LLM_UNAVAILABLE' in response_text:
            print(f"[FALLBACK] LLM unavailable for local_expert, using search data...")
            # Use existing search results directly to avoid redundant DuckDuckGo calls
            if search_results and search_results != 'Search fallback used':
                parsed_normalized = _normalize_local_expert_data({}, state.get('destination', ''))
                parsed_normalized['summary'] = f"Local cultural insights for {state.get('destination', '')} based on travel sources: {str(search_results)[:300]}"
            else:
                parsed_normalized = _build_fallback_local_expert(state.get('destination', ''), state.get('interests', []))
            agent_outputs["local_expert"] = {
                "response": json.dumps(parsed_normalized),
                "output": parsed_normalized,
                "timestamp": datetime.now().isoformat(),
                "status": "completed",
                "search_results": search_results or 'Search fallback used'
            }
            new_state = state.copy()
            new_state["messages"] = state.get("messages", []) + [AIMessage(content=json.dumps(parsed_normalized))]
            new_state["current_agent"] = "local_expert"
            new_state["agent_outputs"] = agent_outputs
            return new_state

        parsed = _try_parse_json(response_text)
        
        parsed_normalized = _normalize_local_expert_data(parsed, state.get('destination', ''))
        
        agent_outputs["local_expert"] = {
            "response": response_text,
            "output": parsed_normalized,
            "timestamp": datetime.now().isoformat(),
            "status": "completed",
            "search_results": search_results
        }
        
        new_state = state.copy()
        new_state["messages"] = state.get("messages", []) + [response]
        new_state["current_agent"] = "local_expert"
        new_state["agent_outputs"] = agent_outputs
        return new_state
    
    def _itinerary_planner_agent(self, state: TravelPlanState) -> TravelPlanState:
        """Itinerary planner agent - produces structured JSON for the UI"""
        
        # Gather context from other agents
        agent_outputs = state.get("agent_outputs", {})
        # Make the planner self-sufficient - it can work without other agents for speed
        destination = state.get('destination', 'Dubai')
        duration = state.get('duration', 3)
        budget_tier = state.get('budget_range', 'Premier')
        interests = ', '.join(state.get('interests', []))
        group_type = state.get('group_type', 'Couple')
        group_size = state.get('group_size', 2)
        dietary = ', '.join(state.get('dietary_requirements', [])) or 'None'
        accessibility = ', '.join(state.get('accessibility', [])) or 'None'
        pace = state.get('pace', 'Moderate')
        accommodation = state.get('accommodation_preference', 'No preference')
        occasion = state.get('occasion', '')
        language_pref = state.get('language_preference', 'English only')
        risk_tolerance = state.get('risk_tolerance', 'Balanced')
        
        pace_guidance = {
            'Relaxed': '1-2 activities per day with long breaks and downtime',
            'Moderate': '2-3 activities per day with balanced schedule',
            'Active': '3-4 activities per day with packed itinerary',
            'Intense': '4-5 activities per day, early start to late finish'
        }.get(pace, '2-3 activities per day with balanced schedule')
        
        # Determine local currency for the destination
        local_currency_symbol, local_currency_code, local_currency_name = _get_currency_for_destination(destination)
        local_price_range = _format_price_range(destination, budget_tier, duration)

        budget_cost_guide = _get_budget_cost_guide(destination, budget_tier)

        system_prompt = f"""Create a {duration}-day travel itinerary for {destination} in JSON format.

REQUIREMENTS:
- Return ONLY valid JSON (no markdown, no explanations)
- Each day must have DIFFERENT real locations in {destination}
- Include {duration} days with activities per day
- Budget: {budget_tier} ({local_currency_name}), Interests: {interests}
- Group: {group_size} {group_type} travelers
- Pace: {pace} ({pace_guidance})
- Dietary: {dietary}
- Accessibility: {accessibility}
- Accommodation: {accommodation}
- Occasion: {occasion or 'None'}
- Language: {language_pref}
- Exploration style: {risk_tolerance}

CURRENCY: The destination is {destination}. Use {local_currency_name} ({local_currency_symbol}) for ALL prices, costs, and price_range in the JSON. Do NOT use USD ($).

BUDGET COST RULES (MUST FOLLOW — per person costs):
{budget_cost_guide}

IMPORTANT GUIDELINES:
- For dietary requirements, recommend restaurants and food experiences that accommodate them.
- For accessibility needs, ensure activities and venues are accessible.
- For special occasions, include relevant special touches or venues.
- Match the pace: Relaxed = fewer activities, Active = more activities per day.
- For risk_tolerance 'Conservative', stay in established tourist areas; 'Adventurous' can include hidden gems and local neighborhoods.
- Transport costs in transport_to_next MUST match the budget tier above (e.g., Essential = cheap local buses, Legendary = private taxis/charters).

JSON FORMAT:
{{
  "trip_title": "Title for {destination}",
  "overview": "Brief trip description",
  "sustainability_score": 75,
  "price_range": "{local_price_range}",
  "concierge_note": "Welcome message",
  "days": [
    {{
      "day_number": 1,
      "day_name": "Day 1",
      "theme": "Theme for day",
      "activities": [
        {{
          "time": "09:00 AM",
          "title": "Real attraction name",
          "description": "What to do here",
          "location": "Specific location",
          "tag": "Culture",
          "map_query": "Location for maps",
          "transport_to_next": {{
            "mode": "Walking",
            "duration": "15 min",
            "cost": "Free",
            "instructions": "How to get there"
          }}
        }}
      ]
    }}
  ]
}}

Generate the JSON for {destination} now:"""
        
        messages = [SystemMessage(content=system_prompt)]
        # Don't include message history for faster, more focused responses

            
        response = self._invoke_llm(messages)
        response_text = _safe_message_content(response)

        # LLM fallback: use DuckDuckGo search-based itinerary if LLM is unavailable
        if 'LLM_UNAVAILABLE' in response_text:
            print(f"[FALLBACK] LLM unavailable for itinerary_planner, building search-based itinerary...")
            parsed = _build_fallback_itinerary(
                destination, duration, state.get('interests', []), budget_tier,
                pace, group_type, group_size,
                state.get('dietary_requirements', []), state.get('accessibility', []),
                accommodation, occasion
            )
            agent_outputs = state.get("agent_outputs", {})
            agent_outputs["itinerary_planner"] = {
                "response": json.dumps(parsed),
                "output": parsed,
                "timestamp": datetime.now().isoformat(),
                "status": "completed"
            }
            new_state = state.copy()
            new_state["messages"] = state.get("messages", []) + [AIMessage(content=json.dumps(parsed))]
            new_state["current_agent"] = "itinerary_planner"
            new_state["agent_outputs"] = agent_outputs
            return new_state

        # Debug: Print first 500 chars of response to see what Groq is generating
        print(f"[DEBUG] LLM Response (first 500 chars): {response_text[:500]}")
        
        # AGGRESSIVE HTML REMOVAL - Strip ALL HTML tags using regex
        import re
        # Remove all HTML tags
        response_text = re.sub(r'<[^>]+>', '', response_text)
        # Remove common HTML entities
        response_text = response_text.replace('&nbsp;', ' ').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
        
        print(f"[DEBUG] After HTML removal (first 500 chars): {response_text[:500]}")
        
        # Force JSON parsing using the improved helper
        parsed = _try_parse_json(response_text)
        
        if not parsed:
            print(f"[WARNING] JSON parsing failed. Response was: {response_text[:200]}")
        else:
            print(f"[SUCCESS] JSON parsed successfully with {len(parsed.get('days', []))} days")

        # Validate the parsed structure
        if parsed and isinstance(parsed, dict):
            # Ensure required fields exist
            if "days" not in parsed or not isinstance(parsed.get("days"), list):
                parsed = None
            elif len(parsed.get("days", [])) == 0:
                parsed = None
            else:
                # Validate each day has activities
                valid_days = []
                for day in parsed.get("days", []):
                    if isinstance(day, dict) and "activities" in day and isinstance(day["activities"], list):
                        if len(day["activities"]) > 0:
                            valid_days.append(day)
                
                if len(valid_days) == 0:
                    parsed = None
                else:
                    parsed["days"] = valid_days

        # If parsing fails or output is invalid, provide a structured fallback
        if not parsed or not isinstance(parsed, dict):
            destination = state.get('destination', 'your destination')
            duration = state.get('duration', 3)
            
            # Create a basic but valid structure with sample activities
            parsed = {
                "trip_title": f"Discover {destination}",
                "overview": f"Experience the best of {destination} with this carefully curated {duration}-day itinerary featuring local culture, cuisine, and iconic landmarks.",
                "sustainability_score": 80,
                "price_range": state.get("budget_range", "Premier"),
                "concierge_note": f"Welcome to {destination}! We've prepared a wonderful journey showcasing the destination's highlights and hidden gems.",
                "days": []
            }
            
            # Add sample days with activities
            for day_num in range(1, min(duration + 1, 4)):  # Cap at 3 days for fallback
                sample_day = {
                    "day_number": day_num,
                    "day_name": f"Day {day_num}",
                    "theme": "Exploration & Discovery",
                    "activities": [
                        {
                            "time": "09:00 AM",
                            "title": f"Morning Exploration in {destination}",
                            "description": f"Start your day discovering the highlights and local culture of {destination}.",
                            "location": f"Central {destination}",
                            "tag": "Culture",
                            "map_query": destination,
                            "transport_to_next": {
                                "mode": "Walking",
                                "duration": "15 min",
                                "cost": "Free",
                                "instructions": "Stroll through the local area"
                            }
                        },
                        {
                            "time": "12:30 PM",
                            "title": "Local Cuisine Experience",
                            "description": f"Enjoy authentic local flavors at a recommended restaurant in {destination}.",
                            "location": f"{destination} dining district",
                            "tag": "Gastronomy",
                            "map_query": f"restaurants in {destination}",
                            "transport_to_next": {
                                "mode": "Local Transport",
                                "duration": "20 min",
                                "cost": "$3.00",
                                "instructions": "Take local transit to afternoon destination"
                            }
                        },
                        {
                            "time": "03:00 PM",
                            "title": "Afternoon Activities",
                            "description": f"Explore popular attractions and landmarks in {destination}.",
                            "location": f"{destination} attractions",
                            "tag": "Adventure",
                            "map_query": f"attractions in {destination}"
                        }
                    ]
                }
                parsed["days"].append(sample_day)

        agent_outputs = state.get("agent_outputs", {})
        agent_outputs["itinerary_planner"] = {
            "response": response_text,
            "output": parsed,
            "timestamp": datetime.now().isoformat(),
            "status": "completed"
        }
        
        new_state = state.copy()
        new_state["messages"] = state.get("messages", []) + [response]
        new_state["current_agent"] = "itinerary_planner"
        new_state["agent_outputs"] = agent_outputs
        return new_state

    
    def _tool_executor_agent(self, state: TravelPlanState) -> TravelPlanState:
        last_message = state['messages'][-1] if state.get("messages") else None
        if not last_message:
            return state
            
        content = _safe_message_content(last_message)
        if "NEED_SEARCH" in content:
            try:
                # Extract search query
                search_query = content.split("NEED_SEARCH:")[1].strip()
                current_agent = state.get("current_agent", "")
                
                print(f"[TOOL] Executing search for {current_agent}: {search_query}")
                
                # Determine which tool to use
                search_query_lower = search_query.lower()
                if "weather" in search_query_lower or current_agent == "weather_analyst":
                    tool_result = search_weather_info.invoke({"destination": search_query})
                elif "hotel" in search_query_lower or "stay" in search_query_lower:
                    tool_result = search_hotels.invoke({"destination": search_query, "budget": state.get('budget_range', 'Premier'), "accommodation_type": state.get('accommodation_preference', '')})
                elif "restaurant" in search_query_lower or "food" in search_query_lower:
                    tool_result = search_restaurants.invoke({"destination": search_query, "dietary": ', '.join(state.get('dietary_requirements', []))})
                elif "attraction" in search_query_lower or "activity" in search_query_lower:
                    tool_result = search_attractions.invoke({"destination": search_query})
                elif "budget" in search_query_lower or "cost" in search_query_lower or current_agent == "budget_optimizer":
                    tool_result = search_budget_info.invoke({"destination": search_query})
                elif "tip" in search_query_lower or "culture" in search_query_lower or current_agent == "local_expert":
                    tool_result = search_local_tips.invoke({"destination": search_query})
                elif "car rental" in search_query_lower:
                    tool_result = search_car_rentals.invoke({"destination": search_query})
                elif "transit" in search_query_lower and "real-time" in search_query_lower:
                    tool_result = search_real_time_transit_info.invoke({"destination": search_query})
                elif "transport" in search_query_lower or "metro" in search_query_lower or "taxi" in search_query_lower:
                    tool_result = search_local_transport_options.invoke({"destination": search_query})
                elif "blog" in search_query_lower or "guide" in search_query_lower or "rag" in search_query_lower:
                    tool_result = search_travel_blogs.invoke({"query": search_query})
                else:
                    tool_result = search_destination_info.invoke(search_query)
                
                # Create result message
                result_message = AIMessage(content=f"Search Results:\n\n{tool_result}")
                
                new_state = state.copy()
                new_state["messages"] = state.get("messages", []) + [result_message]
                
                # Update agent outputs with search status
                agent_outputs = state.get("agent_outputs", {})
                if current_agent:
                    agent_outputs[current_agent]["search_results"] = tool_result
                new_state["agent_outputs"] = agent_outputs
                
                return new_state
                
            except Exception as e:
                error_msg = f"Error executing tool: {str(e)}"
                error_message = AIMessage(content=error_msg)
                new_state = state.copy()
                new_state["messages"] = state.get("messages", []) + [error_message]
                return new_state
        
        return state
    def _agent_router(self, state: TravelPlanState) -> str:
        """Router to determine next step from specialized agents"""
        # Route to tool_executor if a search is requested
        messages = state.get("messages", [])
        if messages:
            last = messages[-1]
            content = getattr(last, "content", "") or ""
            if "NEED_SEARCH" in content:
                return "tools"
        return "coordinator"

    def answer_place_question(self, place: str, question: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> dict:
        """Answer a question about a specific place using web research + LLM.
        This is an isolated method — NOT added to the StateGraph.
        Returns dict with answer_markdown, location, facts, sources, followup_suggestions.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Step 1 & 2: Geocode AND search IN PARALLEL (biggest speed win)
        location: dict = {}
        web_context: str = ""
        with ThreadPoolExecutor(max_workers=2) as executor:
            geo_future = executor.submit(geocode_place, place)
            search_future = executor.submit(search_place_comprehensive, place, question)
            for future in as_completed([geo_future, search_future], timeout=45):
                try:
                    if future is geo_future:
                        location = future.result()
                    else:
                        web_context = future.result()
                except Exception as e:
                    print(f"[WARNING] Parallel task failed: {e}")

        # Step 3: Build conversation history string (truncated)
        history_str = ""
        if conversation_history:
            history_parts = []
            for msg in conversation_history[-4:]:
                role = msg.get("role", "user")
                content = msg.get("content", "")[:200]
                history_parts.append(f"{role.capitalize()}: {content}")
            history_str = "\n".join(history_parts)

        # Step 4: Build messages for LLM (truncated web_context to save tokens)
        location_str = (
            f"{location.get('display_name', 'Unknown')} "
            f"({location.get('lat', 'N/A')}, {location.get('lng', 'N/A')})"
        ) if location else "Unknown location"

        # Truncate web context to fit within LLM context window efficiently
        truncated_context = web_context[:3000] if len(web_context) > 3000 else web_context

        system_prompt = (
            "You are XPLORA, an expert travel assistant. Answer from the web context below. "
            "Cite sources as [1][2]. If info missing, say 'Not found'. "
            "Format: ## Direct Answer, ## Key Details (bullets), ## Practical Info."
        )

        human_prompt = (
            f"Place: {place} ({location_str})\n"
            f"Question: {question}\n"
            f"Web Context:\n{truncated_context}\n"
        )
        if history_str:
            human_prompt += f"\nConversation:\n{history_str}\n"

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]

        # Step 5: Invoke LLM
        llm_response = self._invoke_llm(messages)
        answer_text = _safe_message_content(llm_response)

        # Step 6: Handle LLM_UNAVAILABLE — synthesize answer from web context
        if "LLM_UNAVAILABLE" in answer_text:
            print("[FALLBACK] LLM unavailable for place Q&A, synthesizing from web context")
            answer_text = (
                f"## Answer about {place}\n\n"
                f"*Note: AI assistant is temporarily unavailable. Below are web search results.*\n\n"
                f"### Raw Search Results\n\n{web_context}\n\n"
                f"---\n"
                f"For a more detailed AI-curated answer, please try again in a few moments."
            )

        # Step 7: Extract sources (dedup already done in search_place_comprehensive)
        sources = extract_sources_from_text(web_context)[:10]

        # Step 8: Generate facts from answer
        facts: list = []
        answer_lower = answer_text.lower()
        fact_patterns = [
            ("entry fee", "Entry Fee"), ("admission", "Admission"),
            ("opening hours", "Opening Hours"), ("best time", "Best Time"),
            ("wheelchair", "Accessibility"), ("accessible", "Accessibility"),
            ("timings", "Timings"), ("free entry", "Entry Fee"),
            ("ticket", "Ticket Info"),
        ]
        for pattern, label in fact_patterns:
            if pattern in answer_lower:
                for line in answer_text.split("\n"):
                    if pattern in line.lower() and len(line.strip()) < 300:
                        clean = line.strip().lstrip("#*-• ").strip()
                        if clean and not any(f["label"] == label for f in facts):
                            facts.append({"label": label, "value": clean[:150]})
                            break
        if location and not facts:
            facts = [
                {"label": "Location", "value": location.get("display_name", place)},
                {"label": "Type", "value": location.get("type", "place")},
            ]

        # Step 9: Build follow-up suggestions
        followup_suggestions: list = []
        if place:
            followup_suggestions = [
                f"What are the opening hours for {place}?",
                f"Is {place} crowded on weekends?",
                f"Nearby food options to {place}?",
            ]

        # Step 10: Build Google Maps URLs
        lat = location.get("lat")
        lng = location.get("lng")
        google_maps_url = ""
        directions_url = ""
        if lat and lng:
            google_maps_url = f"https://www.google.com/maps/search/?api=1&query={lat},{lng}"
            directions_url = f"https://www.google.com/maps/dir/?api=1&destination={lat},{lng}"

        location_data = {
            "display_name": location.get("display_name", place) if location else place,
            "lat": lat if lat else 0,
            "lng": lng if lng else 0,
            "address": location.get("address", "") if location else "",
            "type": location.get("type", "place") if location else "place",
            "google_maps_url": google_maps_url,
            "directions_url": directions_url,
        }

        return {
            "place": place,
            "question": question,
            "answer_markdown": answer_text,
            "location": location_data,
            "facts": facts,
            "sources": sources,
            "followup_suggestions": followup_suggestions,
        }
