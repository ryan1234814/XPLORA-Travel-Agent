import os
import requests
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY", "")
PLACES_BASE_URL = os.getenv("PLACES_BASE_URL", "https://maps.googleapis.com/maps/api/place")

# Fallback unsplash images by category
FALLBACK_IMAGES = {
    "default": [
        "https://images.unsplash.com/photo-1488085061387-422e29b40080?w=800&q=80",
        "https://images.unsplash.com/photo-1476514525535-07fb6b4ae8f1?w=800&q=80",
    ],
    "temple": ["https://images.unsplash.com/photo-1492571350019-22de08371fd3?w=800&q=80"],
    "restaurant": ["https://images.unsplash.com/photo-1414235077428-338989a2e8c0?w=800&q=80"],
    "nature": ["https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800&q=80"],
}

def _google_text_search(query: str) -> Optional[Dict[str, Any]]:
    if not GOOGLE_PLACES_API_KEY:
        return None
    try:
        resp = requests.get(f"{PLACES_BASE_URL}/textsearch/json", params={"query": query, "key": GOOGLE_PLACES_API_KEY}, timeout=8)
        if resp.status_code != 200:
            print(f"[Places] textsearch {resp.status_code}: {resp.text[:200]}")
            return None
        data = resp.json()
        results = data.get("results", [])
        if not results:
            return None
        return results[0]
    except Exception as e:
        print(f"[Places] textsearch error: {e}")
        return None

def _google_details(place_id: str) -> Optional[Dict[str, Any]]:
    if not GOOGLE_PLACES_API_KEY or not place_id:
        return None
    try:
        fields = "name,rating,user_ratings_total,photos,reviews,formatted_address,url,geometry,types,price_level"
        resp = requests.get(f"{PLACES_BASE_URL}/details/json", params={"place_id": place_id, "fields": fields, "key": GOOGLE_PLACES_API_KEY}, timeout=8)
        if resp.status_code != 200:
            return None
        data = resp.json()
        return data.get("result")
    except Exception as e:
        print(f"[Places] details error: {e}")
        return None

def _build_photo_url(photo_reference: str, maxwidth: int = 800) -> str:
    if not GOOGLE_PLACES_API_KEY:
        return ""
    return f"{PLACES_BASE_URL}/photo?maxwidth={maxwidth}&photo_reference={photo_reference}&key={GOOGLE_PLACES_API_KEY}"

def enrich_place(query: str) -> Dict[str, Any]:
    """Enrich a place/activity query with photos, rating, reviews via Google Places.
    Falls back to mock/social proof if API unavailable.
    Cached via DB layer caller.
    """
    query = (query or "").strip()
    if not query:
        return {"query": query, "rating": None, "user_ratings_total": 0, "photos": [], "reviews": [], "source": "none"}

    # Try Google Places
    hit = _google_text_search(query)
    if hit:
        place_id = hit.get("place_id", "")
        details = _google_details(place_id) if place_id else None
        # Prefer details; fallback to textsearch result
        src = details or hit
        photos = []
        for p in (src.get("photos") or hit.get("photos") or [])[:4]:
            ref = p.get("photo_reference")
            if ref:
                photos.append(_build_photo_url(ref))
        rating = src.get("rating") or hit.get("rating")
        total = src.get("user_ratings_total") or hit.get("user_ratings_total") or 0
        reviews_raw = src.get("reviews") or []
        reviews = []
        for r in reviews_raw[:3]:
            reviews.append({
                "author_name": r.get("author_name", "Google User"),
                "rating": r.get("rating"),
                "text": r.get("text", "")[:280],
                "time": r.get("relative_time_description", ""),
                "profile_photo": r.get("profile_photo_url", ""),
            })
        name = src.get("name") or hit.get("name") or query
        maps_url = src.get("url") or f"https://www.google.com/maps/search/?api=1&query={query.replace(' ','+')}"
        return {
            "query": query,
            "place_id": place_id,
            "place_name": name,
            "rating": rating,
            "user_ratings_total": total,
            "photos": photos,
            "reviews": reviews,
            "google_maps_url": maps_url,
            "source": "google_places",
            "raw": {"formatted_address": src.get("formatted_address", "")}
        }

    # Fallback: generate deterministic mock social proof (so UI always works without quota)
    import hashlib
    h = int(hashlib.md5(query.encode()).hexdigest()[:8], 16)
    mock_rating = round(3.8 + (h % 12) / 10, 1)  # 3.8 - 4.9
    mock_total = 200 + (h % 5000)
    # Use Unsplash source as fallback images (no key needed)
    fallback_photos = [
        f"https://picsum.photos/seed/{abs(hash(query)) % 10000}/800/600",
        f"https://picsum.photos/seed/{abs(hash(query+'2')) % 10000}/800/600",
    ]
    mock_reviews = [
        {"author_name": "Traveler", "rating": 5, "text": f"Amazing experience at {query}! Highly recommended for first-time visitors.", "time": "a month ago"},
        {"author_name": "Explorer", "rating": 4, "text": f"Great spot in {query.split(',')[-1].strip() if ',' in query else query}. Go early to avoid crowds.", "time": "2 weeks ago"},
    ]
    return {
        "query": query,
        "place_id": None,
        "place_name": query,
        "rating": mock_rating,
        "user_ratings_total": mock_total,
        "photos": fallback_photos,
        "reviews": mock_reviews,
        "google_maps_url": f"https://www.google.com/maps/search/?api=1&query={query.replace(' ','+')}",
        "source": "fallback_mock",
        "raw": {}
    }
