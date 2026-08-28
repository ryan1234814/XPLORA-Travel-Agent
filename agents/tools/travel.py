import asyncio
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
from ddgs import DDGS
import json
import re
import requests
from datetime import datetime
from config.langgraph_config import langgraph_config as config
from config.api_config import api_config

@tool
def search_destination_info(query: str):
    """Search for general information about a travel destination including attractions and guides."""
    try:
        with DDGS() as ddgs:
            search_query = query
            if "travel" not in query.lower() and "attraction" not in query.lower():
                search_query += " travel destination guide attractions"
                
            results = list(ddgs.text(
                search_query,
                max_results=config.DUCKDUCKGO_MAX_RESULTS,
                region=config.DUCKDUCKGO_REGION,
                safesearch=config.DUCKDUCKGO_SAFESEARCH
            ))
            
            if not results:
                return f"No search results found for the destination: {query}"
            
            formatted_results = []
            for i, result in enumerate(results[:5], 1):
                formatted_results.append(
                    f"{i}. {result.get('title', 'No title')}\n"
                    f"   {result.get('body', 'No description')}\n"
                    f"   Source: {result.get('href', 'No URL')}\n"
                )
        
            return "\n".join(formatted_results)
    except Exception as e:
        return f"Error searching for destination info: {str(e)}"

@tool
def search_weather_info(destination: str, dates: str = "") -> str:
    """Search for current weather information and forecasts for a destination."""
    try:
        # Try Tomorrow.io API first if key exists and no specific dates are requested (current weather)
        if api_config.TOMORROW_IO_API_KEY and not dates:
            try:
                params = {
                    "location": destination,
                    "apikey": api_config.TOMORROW_IO_API_KEY
                }
                response = requests.get(f"{api_config.WEATHER_BASE_URL}/realtime", params=params)
                if response.status_code == 200:
                    data = response.json()
                    values = data.get("data", {}).get("values", {})
                    return (f"Current Weather in {destination}:\n"
                            f"Temperature: {values.get('temperature')}°C (Apparent: {values.get('temperatureApparent')}°C)\n"
                            f"Humidity: {values.get('humidity')}%\n"
                            f"Wind Speed: {values.get('windSpeed')} m/s\n"
                            f"Cloud Cover: {values.get('cloudCover')}%")
            except Exception as api_err:
                print(f"Tomorrow.io API error: {api_err}")

        # Fallback to DuckDuckGo search
        weather_query = f"{destination} weather forecast {dates} travel climate"
        with DDGS() as ddgs:
            results = list(ddgs.text(
                weather_query,
                max_results=config.DUCKDUCKGO_MAX_RESULTS,
                region=config.DUCKDUCKGO_REGION,
                safesearch=config.DUCKDUCKGO_SAFESEARCH
            ))
            
            if not results:
                return f"No weather results found for: {destination}"
            
            formatted_results = [f"Weather information for {destination}:"]
            for i, result in enumerate(results[:3], 1):
                formatted_results.append(
                    f"{i}. {result.get('title', 'No title')}\n"
                    f"   {result.get('body', 'No description')}\n"
                )
        
            return "\n".join(formatted_results)
    except Exception as e:
        return f"Error searching for weather info: {str(e)}"

@tool
def search_hotels(destination: str, budget: str = "mid-range", accommodation_type: str = "") -> str:
    """Search for hotel information and pricing in a specific destination."""
    try:
        acc_type = f" {accommodation_type}" if accommodation_type and accommodation_type != "No preference" else ""
        hotel_query = f"{destination} {budget} hotels{acc_type} best places to stay accommodation"
        with DDGS() as ddgs:
            results = list(ddgs.text(
                hotel_query,
                max_results=6,
                region=config.DUCKDUCKGO_REGION,
                safesearch=config.DUCKDUCKGO_SAFESEARCH
            ))
            
            if not results:
                return f"No hotel information found for {destination}"
            
            hotels = [f"Hotel options in {destination} ({budget} budget{acc_type}):"]
            for i, result in enumerate(results[:4], 1):
                hotels.append(
                    f"{i}. {result.get('title', 'Hotel')}\n"
                    f"   {result.get('body', 'No details')[:180]}...\n"
                )
            
            return "\n".join(hotels)
    except Exception as e:
        return f"Error searching hotels: {str(e)}"

@tool
def search_restaurants(destination: str, cuisine: str = "", dietary: str = "") -> str:
    """Search for restaurants and dining options in a specific destination."""
    try:
        dietary_filter = f" {dietary}" if dietary and dietary != "No restrictions" else ""
        restaurant_query = f"{destination} best restaurants {cuisine}{dietary_filter} local food dining where to eat"
        with DDGS() as ddgs:
            results = list(ddgs.text(
                restaurant_query,
                max_results=6,
                region=config.DUCKDUCKGO_REGION,
                safesearch=config.DUCKDUCKGO_SAFESEARCH
            ))
            
            if not results:
                return f"No restaurant information found for {destination}"
            
            restaurants = [f"Restaurant recommendations in {destination}{dietary_filter}:"]
            for i, result in enumerate(results[:4], 1):
                restaurants.append(
                    f"{i}. {result.get('title', 'Restaurant')}\n"
                    f"   {result.get('body', 'No details')[:180]}...\n"
                )
            
            return "\n".join(restaurants)
    except Exception as e:
        return f"Error searching restaurants: {str(e)}"

@tool
def search_attractions(destination: str, accessibility: str = "") -> str:
    """Search for top attractions and things to do in a specific destination."""
    try:
        acc_filter = f" accessible {accessibility}" if accessibility and accessibility != "None" else ""
        attraction_query = f"{destination} top attractions must see places things to do{acc_filter}"
        with DDGS() as ddgs:
            results = list(ddgs.text(
                attraction_query,
                max_results=6,
                region=config.DUCKDUCKGO_REGION,
                safesearch=config.DUCKDUCKGO_SAFESEARCH
            ))
            
            if not results:
                return f"No attraction information found for {destination}"
            
            attractions = [f"Top attractions in {destination}{acc_filter}:"]
            for i, result in enumerate(results[:5], 1):
                attractions.append(
                    f"{i}. {result.get('title', 'Attraction')}\n"
                    f"   {result.get('body', 'No details')[:200]}...\n"
                )
            
            return "\n".join(attractions)
    except Exception as e:
        return f"Error searching attractions: {str(e)}"

@tool
def search_local_tips(destination: str) -> str:
    """Search for local tips, culture, and insider information about a destination."""
    try:
        tips_query = f"{destination} local tips insider guide cultural etiquette what to know"
        with DDGS() as ddgs:
            results = list(ddgs.text(
                tips_query,
                max_results=5,
                region=config.DUCKDUCKGO_REGION,
                safesearch=config.DUCKDUCKGO_SAFESEARCH
            ))
            
            if not results:
                return f"No local tips found for {destination}"
            
            tips = [f"Local tips for {destination}:"]
            for result in results[:3]:
                tips.append(
                    f"• {result.get('title', 'Local Tip')}\n"
                    f"  {result.get('body', 'No details')[:200]}...\n"
                )
            
            return "\n".join(tips)
    except Exception as e:
        return f"Error searching local tips: {str(e)}"

@tool
def search_budget_info(destination: str, duration: str = "7 days") -> str:
    """Search for travel budget information and estimated expenses for a destination."""
    try:
        budget_query = f"{destination} travel budget for {duration} estimated expenses"
        with DDGS() as ddgs:
            results = list(ddgs.text(
                budget_query,
                max_results=5,
                region=config.DUCKDUCKGO_REGION,
                safesearch=config.DUCKDUCKGO_SAFESEARCH
            ))
            
            if not results:
                return f"No budget info found for {destination}"
            
            budget_info = [f"Budget information for {destination}:"]
            for result in results[:3]:
                budget_info.append(
                    f"• {result.get('title', 'Budget Info')}\n"
                    f"  {result.get('body', 'No details available')}\n"
                )
            
            return "\n".join(budget_info)
    except Exception as e:
        return f"Error searching budget info: {str(e)}"

@tool
def search_flights(origin: str, destination: str, travel_dates: str = "") -> str:
    """Search for flight options and comparison pages between an origin and destination."""
    try:
        if not origin or not destination:
            return "Missing origin or destination for flight search."

        query = f"flights {origin} to {destination} {travel_dates} price compare"
        with DDGS() as ddgs:
            results = list(ddgs.text(
                query,
                max_results=8,
                region=config.DUCKDUCKGO_REGION,
                safesearch=config.DUCKDUCKGO_SAFESEARCH
            ))

            if not results:
                return f"No flight search results found for {origin} → {destination}."

            formatted = [f"Flight search results for {origin} → {destination} ({travel_dates or 'dates flexible'}):"]
            for i, r in enumerate(results[:5], 1):
                formatted.append(
                    f"{i}. {r.get('title', 'No title')}\n"
                    f"   {r.get('body', 'No description')[:220]}...\n"
                    f"   Source: {r.get('href', 'No URL')}\n"
                )
            return "\n".join(formatted)
    except Exception as e:
        return f"Error searching flights: {str(e)}"

@tool
def search_train_bus_options(origin: str, destination: str, region_hint: str = "") -> str:
    """Search for train/bus options between an origin and destination (region-specific where possible)."""
    try:
        if not origin or not destination:
            return "Missing origin or destination for train/bus search."

        query = f"train bus {origin} to {destination} {region_hint} tickets schedule"
        with DDGS() as ddgs:
            results = list(ddgs.text(
                query,
                max_results=8,
                region=config.DUCKDUCKGO_REGION,
                safesearch=config.DUCKDUCKGO_SAFESEARCH
            ))

            if not results:
                return f"No train/bus results found for {origin} → {destination}."

            formatted = [f"Train/Bus results for {origin} → {destination}:"]
            for i, r in enumerate(results[:5], 1):
                formatted.append(
                    f"{i}. {r.get('title', 'No title')}\n"
                    f"   {r.get('body', 'No description')[:220]}...\n"
                    f"   Source: {r.get('href', 'No URL')}\n"
                )
            return "\n".join(formatted)
    except Exception as e:
        return f"Error searching train/bus options: {str(e)}"

@tool
def suggest_airport_transfers(destination: str, airport_code_or_name: str = "") -> str:
    """Search for airport transfer options (train, taxi, rideshare, shuttle) for a destination."""
    try:
        if not destination:
            return "Missing destination for airport transfer suggestions."

        airport_part = f" {airport_code_or_name}" if airport_code_or_name else ""
        query = f"{destination}{airport_part} airport transfer options train bus taxi shuttle rideshare"

        with DDGS() as ddgs:
            results = list(ddgs.text(
                query,
                max_results=8,
                region=config.DUCKDUCKGO_REGION,
                safesearch=config.DUCKDUCKGO_SAFESEARCH
            ))

            if not results:
                return f"No airport transfer results found for {destination}."

            formatted = [f"Airport transfer options for {destination}:"]
            for i, r in enumerate(results[:5], 1):
                formatted.append(
                    f"{i}. {r.get('title', 'No title')}\n"
                    f"   {r.get('body', 'No description')[:220]}...\n"
                    f"   Source: {r.get('href', 'No URL')}\n"
                )
            return "\n".join(formatted)
    except Exception as e:
        return f"Error searching airport transfers: {str(e)}"

@tool
def search_local_transport_guidance(destination: str) -> str:
    """Search for local transport guidance (metro cards, passes, apps, safety) for a destination."""
    try:
        if not destination:
            return "Missing destination for local transport guidance."

        query = f"{destination} public transport guide metro pass IC card apps how to use"
        with DDGS() as ddgs:
            results = list(ddgs.text(
                query,
                max_results=8,
                region=config.DUCKDUCKGO_REGION,
                safesearch=config.DUCKDUCKGO_SAFESEARCH
            ))

            if not results:
                return f"No local transport guidance found for {destination}."

            formatted = [f"Local transport guidance for {destination}:"]
            for i, r in enumerate(results[:5], 1):
                formatted.append(
                    f"{i}. {r.get('title', 'No title')}\n"
                    f"   {r.get('body', 'No description')[:220]}...\n"
                    f"   Source: {r.get('href', 'No URL')}\n"
                )
            return "\n".join(formatted)
    except Exception as e:
        return f"Error searching local transport guidance: {str(e)}"

@tool
def search_local_transport_options(destination: str, origin_point: str = "", destination_point: str = "") -> str:
    """Search for specific local transport options (taxi, metro, bus) with cost and time estimates."""
    try:
        if not destination:
            return "Missing destination for local transport options."
        
        query = f"{destination} {origin_point} to {destination_point} transport options cost price time duration"
        if not origin_point:
            query = f"{destination} public transport vs taxi vs uber cost comparison and travel times"

        with DDGS() as ddgs:
            results = list(ddgs.text(
                query,
                max_results=8,
                region=config.DUCKDUCKGO_REGION,
                safesearch=config.DUCKDUCKGO_SAFESEARCH
            ))

            if not results:
                return f"No specific transport options found for {destination}."

            formatted = [f"Local transport options for {destination}:"]
            for i, r in enumerate(results[:5], 1):
                formatted.append(
                    f"{i}. {r.get('title', 'No title')}\n"
                    f"   {r.get('body', 'No description')[:250]}...\n"
                    f"   Source: {r.get('href', 'No URL')}\n"
                )
            return "\n".join(formatted)
    except Exception as e:
        return f"Error searching transport options: {str(e)}"

@tool
def search_car_rentals(destination: str, car_type: str = "standard") -> str:
    """Search for car rental options, companies, and estimated daily prices in a destination."""
    try:
        if not destination:
            return "Missing destination for car rental search."

        query = f"{destination} car rental price per day {car_type} companies best deals"
        with DDGS() as ddgs:
            results = list(ddgs.text(
                query,
                max_results=6,
                region=config.DUCKDUCKGO_REGION,
                safesearch=config.DUCKDUCKGO_SAFESEARCH
            ))

            if not results:
                return f"No car rental information found for {destination}."

            rentals = [f"Car rental options in {destination}:"]
            for i, r in enumerate(results[:4], 1):
                rentals.append(
                    f"{i}. {r.get('title', 'Rental Info')}\n"
                    f"   {r.get('body', 'No details')[:220]}...\n"
                )
            return "\n".join(rentals)
    except Exception as e:
        return f"Error searching car rentals: {str(e)}"

@tool
def search_real_time_transit_info(destination: str) -> str:
    """Search for real-time transit information, service alerts, and live maps for a destination's transport network."""
    try:
        query = f"{destination} real-time transit info live bus train metro status service alerts"
        with DDGS() as ddgs:
            results = list(ddgs.text(
                query,
                max_results=5,
                region=config.DUCKDUCKGO_REGION,
                safesearch=config.DUCKDUCKGO_SAFESEARCH
            ))
            
            if not results:
                return f"No real-time transit info found for {destination}."
                
            info = [f"Real-time transit information for {destination}:"]
            for r in results[:3]:
                info.append(
                    f"• {r.get('title', 'Transit Update')}\n"
                    f"  {r.get('body', 'No details')[:250]}...\n"
                    f"  Link: {r.get('href', 'No URL')}\n"
                )
            return "\n".join(info)
    except Exception as e:
        return f"Error searching real-time transit: {str(e)}"

@tool
def build_google_maps_directions_link(stops: List[str]) -> str:
    """Build a Google Maps Directions URL for up to 10 stops (origin + waypoints + destination) using text queries."""
    try:
        cleaned = [s.strip() for s in (stops or []) if isinstance(s, str) and s.strip()]
        if len(cleaned) < 2:
            return ""
        origin = cleaned[0].replace(" ", "+")
        destination = cleaned[-1].replace(" ", "+")
        waypoints = [s.replace(" ", "+") for s in cleaned[1:-1]]
        url = f"https://www.google.com/maps/dir/?api=1&origin={origin}&destination={destination}"
        if waypoints:
            url += f"&waypoints={'%7C'.join(waypoints)}"
        return url
    except Exception:
        return ""

@tool
def search_travel_blogs(query: str) -> str:
    """Search the internal Vector Database (Pinecone) for scraped travel blogs, guides, and richer recommendations."""
    try:
        from db.rag import rag_db
        return rag_db.query(query, k=3)
    except Exception as e:
        return f"Error searching travel knowledge base: {str(e)}"

def geocode_place(place: str) -> dict:
    """Geocode a place name using Nominatim (OpenStreetMap) free API.
    Returns {display_name, lat, lng, address, type} or {} on failure.
    """
    import time as _time
    try:
        params = {
            "q": place,
            "format": "json",
            "limit": 1,
        }
        headers = {"User-Agent": "XPLORA/1.0"}
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params=params,
            headers=headers,
            timeout=8,
        )
        if resp.status_code == 429:
            # Rate limited — wait and retry once
            _time.sleep(2)
            resp = requests.get(
                "https://nominatim.openstreetmap.org/search",
                params=params,
                headers=headers,
                timeout=8,
            )
        if resp.status_code != 200:
            print(f"[WARNING] Nominatim returned status {resp.status_code}")
            return {}
        results = resp.json()
        if not results:
            return {}
        r = results[0]
        return {
            "display_name": r.get("display_name", place),
            "lat": float(r.get("lat", 0)),
            "lng": float(r.get("lon", 0)),
            "address": r.get("display_name", ""),
            "type": r.get("type", "place"),
        }
    except Exception as e:
        print(f"[WARNING] Geocoding failed for '{place}': {e}")
        return {}


def _ddgs_search(query: str, max_results: int = 5) -> list:
    """Run a single DDGS search. Returns list of result dicts. Never throws."""
    try:
        with DDGS() as ddgs:
            return list(ddgs.text(
                query,
                max_results=max_results,
                region=config.DUCKDUCKGO_REGION,
                safesearch=config.DUCKDUCKGO_SAFESEARCH,
            ))
    except Exception as e:
        print(f"[WARNING] DDGS search failed for '{query[:50]}': {e}")
        return []


def search_place_comprehensive(place: str, question: str) -> str:
    """Perform comprehensive search for a place + question.
    Runs all 3 searches IN PARALLEL for maximum speed.
    Returns formatted string. Never throws.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    all_sources: list = []

    # Define 3 parallel search tasks
    def search_place_question() -> list:
        return _ddgs_search(f"{place} {question}", max_results=5)

    def search_travel_guide() -> list:
        return _ddgs_search(f"{place} travel guide history entry fee timings accessibility", max_results=5)

    def search_rag() -> str:
        try:
            result = search_travel_blogs.invoke({"query": f"{place} {question}"})
            return str(result) if result and "Error" not in str(result) else ""
        except Exception as e:
            print(f"[WARNING] RAG search failed: {e}")
            return ""

    # Execute all 3 in parallel with a generous timeout
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(search_place_question): "place_question",
            executor.submit(search_travel_guide): "travel_guide",
            executor.submit(search_rag): "rag",
        }
        for future in as_completed(futures, timeout=15):
            try:
                result = future.result()
                label = futures[future]
                if label == "rag":
                    if result:
                        all_sources.append({
                            "title": "Travel Knowledge Base",
                            "body": result[:800],
                            "url": "",
                        })
                else:
                    for r in result:
                        all_sources.append({
                            "title": r.get("title", "N/A"),
                            "body": r.get("body", "No details"),
                            "url": r.get("href", ""),
                        })
            except Exception as e:
                print(f"[WARNING] Parallel search task failed: {e}")

    # Deduplicate by URL
    seen_urls: set = set()
    unique_sources: list = []
    for src in all_sources:
        url = src.get("url", "")
        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)
        unique_sources.append(src)

    # Format numbered output
    formatted_parts: list = []
    for i, src in enumerate(unique_sources, 1):
        formatted_parts.append(
            f"{i}. {src['title']}\n"
            f"   {src['body']}\n"
            f"   Source: {src['url']}"
        )

    return "\n".join(formatted_parts) if formatted_parts else f"No search results found for {place}"


def extract_sources_from_text(text: str) -> list:
    """Extract source URLs from numbered search results text.
    Returns list of {title, url, snippet}.
    """
    import re as _re
    sources = []
    lines = text.split("\n")
    current_title = ""
    current_snippet = ""
    current_url = ""
    for line in lines:
        stripped = line.strip()
        # Match numbered title lines like "1. Title" or "1.  Title"
        title_match = _re.match(r'^\d+\.\s+(.+)$', stripped)
        if title_match:
            # Save previous source if exists
            if current_title and current_url:
                sources.append({
                    "title": current_title,
                    "url": current_url,
                    "snippet": current_snippet[:200].strip(),
                })
            elif current_title and not current_url:
                sources.append({
                    "title": current_title,
                    "url": "",
                    "snippet": current_snippet[:200].strip(),
                })
            current_title = title_match.group(1).strip()
            current_snippet = ""
            current_url = ""
            continue
        if stripped.startswith("Source:"):
            url = stripped.replace("Source:", "").strip()
            current_url = url
            continue
        if stripped and not stripped.startswith("Source:"):
            current_snippet += " " + stripped
    # Save last source
    if current_title:
        sources.append({
            "title": current_title,
            "url": current_url,
            "snippet": current_snippet[:200].strip(),
        })
    return sources


# Export all tools in a single list
ALL_TOOLS = [
    search_destination_info,
    search_weather_info,
    search_hotels,
    search_restaurants,
    search_attractions,
    search_local_tips,
    search_budget_info,
    search_flights,
    search_train_bus_options,
    suggest_airport_transfers,
    search_local_transport_guidance,
    build_google_maps_directions_link,
    search_local_transport_options,
    search_car_rentals,
    search_real_time_transit_info,
    search_travel_blogs
]