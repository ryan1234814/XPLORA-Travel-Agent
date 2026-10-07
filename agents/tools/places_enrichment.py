import os
import re
import requests
from typing import Dict, Any, Optional, List
from urllib.parse import quote_plus
from dotenv import load_dotenv

load_dotenv()

GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY", "")
PLACES_BASE_URL = os.getenv("PLACES_BASE_URL", "https://maps.googleapis.com/maps/api/place")

# Wikimedia blocks generic/empty user agents, so send a descriptive one.
WIKI_HEADERS = {
    "User-Agent": "XploraTravelBot/1.0 (travel itinerary place lookup; https://xplora.travel)"
}

# Queries that name a *category* rather than a real venue. Generic wording must
# never be mistaken for the name of the place being visited.
_GENERIC_QUERY_RE = re.compile(
    r"^(restaurants?|caf[eé]s?|hotels?|attractions?|things to do|places?|spots?|bars?|markets?)\s+(in|near|around)\s+",
    re.IGNORECASE,
)
_STOPWORDS = {
    "the", "and", "for", "with", "near", "best", "top", "visit", "things", "place", "places",
    "restaurant", "restaurants", "hotel", "hotels", "attraction", "attractions", "travel",
    "tour", "tours", "cafe", "cafes", "where", "what", "guide",
}


def _is_generic_query(query: str) -> bool:
    return bool(_GENERIC_QUERY_RE.match(query.strip()))


def _search_term(query: str) -> str:
    """Strip category wording so 'restaurants in Dehradun' searches as 'Dehradun'."""
    term = _GENERIC_QUERY_RE.sub("", query.strip())
    return term or query.strip()


def _is_relevant_match(query: str, title: str) -> bool:
    """Require genuine overlap so a city article can't stand in for a venue.

    Only the part before the last comma is the venue: 'Louvre, Paris' must match on
    'Louvre' alone, while 'Clock Tower, Dehradun' must share two words so that the
    'History of Dehradun' city article cannot impersonate the monument.
    """
    def tokens(text: str) -> set:
        parts = re.split(r"[^a-z0-9']+", (text or "").lower())
        return {p for p in parts if len(p) > 2 and p not in _STOPWORDS}
    primary = query.rsplit(",", 1)[0] if "," in query else query
    a, b = tokens(primary), tokens(title)
    if not a or not b:
        return False
    required = 2 if len(a) >= 2 else 1
    return len(a & b) >= required

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


def _wikipedia_lookup(query: str) -> Optional[Dict[str, Any]]:
    """Resolve the canonical place name and its lead image via Wikipedia (no key needed)."""
    term = _search_term(query)
    try:
        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "list": "search", "srsearch": term,
                    "srlimit": 1, "format": "json", "redirects": 1},
            headers=WIKI_HEADERS, timeout=8,
        )
        if r.status_code != 200:
            return None
        hits = ((r.json().get("query") or {}).get("search") or [])
        if not hits:
            return None
        title = hits[0].get("title") or ""
        if not _is_relevant_match(term, title):
            return None
        s = requests.get(
            "https://en.wikipedia.org/api/rest_v1/page/summary/" + quote_plus(title.replace(" ", "_")),
            headers=WIKI_HEADERS, timeout=8,
        )
        if s.status_code != 200:
            return {"title": _clean_place_title(title), "image": None, "page_url": None, "description": None}
        j = s.json()
        image = ((j.get("thumbnail") or {}).get("source")) or ((j.get("originalimage") or {}).get("source"))
        page_url = ((j.get("content_urls") or {}).get("desktop") or {}).get("page")
        return {
            "title": _clean_place_title(j.get("title") or title),
            "image": image,
            "page_url": page_url,
            "description": j.get("description"),
        }
    except Exception as e:
        print(f"[Places] wikipedia error: {e}")
        return None


def _commons_images(query: str, limit: int = 8) -> List[str]:
    """Photographs of this exact place from Wikimedia Commons, pre-sized for the web."""
    term = _search_term(query)
    try:
        r = requests.get(
            "https://commons.wikimedia.org/w/api.php",
            params={"action": "query", "generator": "search",
                    "gsrsearch": f"filetype:bitmap {term}", "gsrnamespace": 6,
                    "gsrlimit": limit, "prop": "imageinfo",
                    "iiprop": "url|mime|size", "iiurlwidth": 900, "format": "json"},
            headers=WIKI_HEADERS, timeout=10,
        )
        if r.status_code != 200:
            return []
        pages = ((r.json().get("query") or {}).get("pages") or {})
        urls = []
        for page in sorted(pages.values(), key=lambda p: p.get("index", 0)):
            info = (page.get("imageinfo") or [{}])[0]
            url = info.get("thumburl")
            mime = str(info.get("mime") or "")
            w, h = info.get("width") or 0, info.get("height") or 0
            if not url or not mime.startswith("image/") or mime in ("image/svg+xml", "image/gif"):
                continue
            if w and h and (h / w > 2.2 or w / h > 2.6):
                continue  # banners/logos/crops that read badly inside a card
            urls.append(url)
        return urls
    except Exception as e:
        print(f"[Places] commons error: {e}")
        return []


# Titles that are catalogues/meta pages rather than a visitable venue.
_NOT_A_PLACE_RE = re.compile(
    r"^(list of|category:|template:|portal:|file:|tourism in|index of|timeline of|outline of|history of)|"
    r"\bdisambiguation\b|^\d{4} in ",
    re.IGNORECASE,
)

# A venue article defines itself with a copula plus a place-type noun in its opening
# sentence ("X is a temple in ..."); people, events and organisations are not venues.
# The noun is matched with optional plurals: "is one of the oldest coffeehouses".
_PLACE_TYPE_RE = re.compile(
    r"\b(temple|mandir|monastery|convent|abbey|shrine|mosque|church|cathedral|gurdwara|synagogue|"
    r"cave|waterfall|lake|reservoir|beach|cliff|gorge|valley|peak|hill|doon|"
    r"hill station|national park|wildlife sanctuary|sanctuary|nature reserve|botanical garden|garden|"
    r"museum|gallery|library|theatre|theater|amphitheatre|observatory|planetarium|zoo|aquarium|"
    r"monument|memorial|fort|fortress|palace|mansion|haveli|castle|tower|bridge|arch|"
    r"market|bazaar|plaza|square|street|road|mall|bakery|confectionery|restaurant|caf[eé]|diner|pub|brewery|"
    r"park|golf course|resort|spa|retreat|stadium|arena|pier|harbour|lighthouse|coffee house|coffeehouse|teahouse|"
    r"hotel|inn|guest house|guesthouse|homestay|lodge|hostel|winery|vineyard|farmhouse|"
    r"viewpoint|dam|barrage|ashram|matha|tomb|mausoleum|ruins|complex|campus|institute|sanatorium)"
    r"(?:s|es)?\b",
    re.IGNORECASE,
)
# "X is a type of cafe" / "X was the description of" describe a concept, not a venue.
_ABSTRACT_RE = re.compile(
    r"\b(type|kind|form|sort|style|term|phrase|description|category|concept|tradition|"
    r"culture|scene|industry|movement|phenomenon|example|list|name|word|group|set)"
    r"(?:s|es)?\b",
    re.IGNORECASE,
)
# Creative works and businesses that borrow venue wording ("The Big Restaurant is a
# 1966 film", "Café de Paris Sauce is a butter-based sauce") — never a place to visit.
_WORK_ORG_RE = re.compile(
    r"\b(film|movie|novel|book|poem|song|album|opera|painting|photograph|character|franchise|"
    r"series|season|director|actor|sauce|dish|recipe|butter|condiment|spread|cuisine|menu|"
    r"company|corporation|conglomerate|brand|retailer|manufacturer|magazine|newspaper|journal|"
    r"band|orchestra|award|prize|festival|event|election|team|club|association|organisation|"
    r"organization|institution|university|college|school|hospital|clinic|denomination)"
    r"(?:s|es)?\b",
    re.IGNORECASE,
)
_PERSON_RE = re.compile(
    r"\b(politician|actor|actress|singer|writer|poet|author|novelist|playwright|cricketer|footballer|"
    r"tennis player|player|scientist|researcher|academic|professor|teacher|educator|lawyer|advocate|judge|"
    r"journalist|businessman|industrialist|composer|musician|dancer|painter|sculptor|artist|architect|"
    r"engineer|director|producer|photographer|critic|philosopher|mathematician|physicist|biologist|"
    r"general|colonel|captain|soldier|martyr|freedom fighter|activist|philanthropist|rebel|pirate|"
    r"chef|cook|sommelier|waiter|host|proprietor|owner|founder|resident|inhabitant|citizen|"
    r"priest|pandit|pundit|padri|imam|maulvi|mufti|rabbi|pastor|reverend|bishop|deacon|vicar|abbot|abbess|"
    r"president|secretary|treasurer|officer|inspector|commissioner|mayor|councillor|member of parliament|"
    r"explorer|settler|colonist|merchant|trader|banker|clerk|farmer|landowner|zamindar|archaeologist|"
    r"saint|guru|baba|monk|scholar|civil servant|bureaucrat|minister|governor|chief minister|"
    r"monarch|king|queen|prince|princess|emperor|nawab|raja|rani|dynasty|clan|tribe)"
    r"(?:s|es)?\b",
    re.IGNORECASE,
)


def _clean_place_title(title: str) -> str:
    """Drop Wikipedia disambiguators: 'Robber's Cave, India' -> "Robber's Cave".

    The itinerary appends the destination to every map query, so the qualifier is
    redundant in the search and noisy in the UI.
    """
    clean = re.sub(r"\s*\([^()]*\)\s*$", "", (title or "").strip()).strip()
    if "," in clean:
        head, tail = clean.rsplit(",", 1)
        if len(head.strip()) >= 3 and len(tail.split()) <= 3:
            clean = head.strip()
    return clean


def _reads_as_venue(title: str, extract: str, anchor: Optional[str] = None) -> bool:
    """Accept only articles that actually describe a place someone can visit.

    A venue is defined by its opening sentences ("X is a temple in ..."), so the
    place-type noun must follow the copula directly and must not be introduced as a
    mere *type* of something — that phrasing marks a concept article instead.
    """
    if not title or _NOT_A_PLACE_RE.search(title):
        return False
    text = (extract or "").strip()
    if not text:
        return False
    # A venue in Agra must mention Agra; without this, a city article thousands of
    # kilometres away (or a train named after one) leaks into the itinerary.
    if anchor and anchor.lower() not in (title + " " + text).lower():
        return False
    # Only the first sentence defines the subject; later sentences can mention any
    # number of venues while describing a film, a dish or a business. Only its first
    # copula counts, otherwise "X was a fighter who is honoured at a temple" reads as
    # the temple.
    first_sentence = re.split(r"(?<=[.])\s+", text, maxsplit=1)[0]
    copula = re.search(r"\b(?:is|are|was|were)\b", first_sentence, re.IGNORECASE)
    if not copula:
        return False
    window = first_sentence[copula.end():copula.end() + 120]
    venue = _PLACE_TYPE_RE.search(window)
    if not venue:
        return False
    if _ABSTRACT_RE.search(window) or _WORK_ORG_RE.search(window):
        return False
    # Wording that defines the *subject* ahead of the venue noun demotes it:
    # "is an architect who designed a temple" is a person, whereas "is one of the
    # oldest coffeehouses, patronised by writers" is a place.
    person = _PERSON_RE.search(window)
    if person and person.start() < venue.start():
        return False
    return True


def wikipedia_place_names(query: str, limit: int = 8, anchor: Optional[str] = None) -> List[str]:
    """Real, named venues for a search term, taken from Wikipedia article titles.

    Used to give itineraries actual place names when the LLM path is degraded.
    Titles are precision-filtered: a wrong name shown to a traveller is worse than
    a generic label, so anything that does not read as a venue is discarded.
    """
    try:
        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "generator": "search", "gsrsearch": query,
                    "gsrlimit": limit, "gsrnamespace": 0, "prop": "extracts",
                    "exintro": 1, "explaintext": 1, "exsentences": 3, "exlimit": "max",
                    "redirects": 1, "format": "json"},
            headers=WIKI_HEADERS, timeout=10,
        )
        if r.status_code != 200:
            return []
        pages = ((r.json().get("query") or {}).get("pages") or {})
        names: List[str] = []
        for page in sorted(pages.values(), key=lambda p: p.get("index", 0)):
            raw = (page.get("title") or "").strip()
            extract = page.get("extract") or ""
            # Filter on the original title (its disambiguator often carries the
            # place-type wording), then present the cleaned name to the traveller.
            if not _reads_as_venue(raw, extract, anchor):
                continue
            clean = _clean_place_title(raw)
            if not clean or clean.lower() in {n.lower() for n in names}:
                continue
            names.append(clean)
        return names
    except Exception as e:
        print(f"[Places] wikipedia titles error: {e}")
        return []


def _rotate(items: List[str], offset: int) -> List[str]:
    """Shift the gallery start so repeated places surface different imagery."""
    if not items or not offset:
        return items
    k = offset % len(items)
    return items[k:] + items[:k]

def enrich_place(query: str, offset: int = 0) -> Dict[str, Any]:
    """Enrich a place query with the real place name and place-specific photos.

    Sources in order: Google Places (when a key is configured), then Wikipedia
    (canonical name + lead image) and Wikimedia Commons (gallery). Nothing is
    invented — when a source has no rating or reviews those come back empty.
    Cached via DB layer caller.
    """
    query = (query or "").strip()
    if not query:
        return {"query": query, "place_name": "", "rating": None, "user_ratings_total": 0,
                "photos": [], "reviews": [], "google_maps_url": "", "source": "none", "raw": {}}

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
            "raw": {"source": "google_places", "formatted_address": src.get("formatted_address", "")}
        }

    # No Google key (or no Google match): resolve the real place and its photography
    # from Wikipedia / Wikimedia Commons.
    wiki = _wikipedia_lookup(query)
    gallery: List[str] = []
    if wiki and wiki.get("image"):
        gallery.append(wiki["image"])
    for url in _commons_images(query):
        if url not in gallery:
            gallery.append(url)

    gallery = _rotate(gallery, offset)

    # Only trust the Wikipedia title as the place name when the query actually named a
    # venue — "restaurants in Dehradun" must not silently become "Dehradun".
    resolved_name = (wiki or {}).get("title") if (wiki and not _is_generic_query(query)) else None
    place_name = resolved_name or query
    source = "wikimedia" if gallery else "none"

    return {
        "query": query,
        "place_id": None,
        "place_name": place_name,
        "rating": None,
        "user_ratings_total": 0,
        "photos": gallery[:4],
        "reviews": [],
        "google_maps_url": f"https://www.google.com/maps/search/?api=1&query={quote_plus(place_name)}",
        "source": source,
        "raw": {
            "source": source,
            "wikipedia_url": (wiki or {}).get("page_url"),
            "description": (wiki or {}).get("description"),
            "formatted_address": "",
        },
    }
