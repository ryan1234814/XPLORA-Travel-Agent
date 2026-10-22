import sys
import os
import time
from collections import defaultdict
from typing import Annotated
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, EmailStr, Field, field_validator
from typing import List, Optional, Dict, Any, ClassVar
import json
from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.agents import LangTravelAgents, TravelPlanState
from db.database import (
    bootstrap as db_bootstrap,
    create_user, delete_itinerary, get_itinerary, get_place_cache, get_rating_stats,
    get_trip_ratings, get_user_by_id, list_user_itineraries, save_itinerary,
    save_trip_rating, set_place_cache, verify_credentials,
)
from backend import auth
from config.api_config import get_api_status

app = FastAPI()

# Singleton agent system — initialized once, reused across requests
_agent_system: Optional[LangTravelAgents] = None

def get_agent_system() -> LangTravelAgents:
    global _agent_system
    if _agent_system is None:
        _agent_system = LangTravelAgents()
    return _agent_system


@app.on_event("startup")
async def _connect_neon():
    """Create the Neon schema up front. Never fatal: the health check must still
    answer even with a bad DATABASE_URL so the deploy is not marked unhealthy."""
    db_bootstrap()


# Enable CORS for the React frontend. allow_credentials is required for the
# session cookie to survive the cross-origin call from the Vercel frontend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        o.strip().rstrip("/")
        for o in (os.getenv("CORS_ORIGINS") or "http://localhost:3000").split(",")
        if o.strip()
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def health_check():
    """Lightweight health endpoint — Render's deploy health check hits `/`.
    Must stay free of DB/agent calls so cold boots respond quickly."""
    return {
        "status": "ok",
        "service": "xplora-backend",
        "api_status": get_api_status(),
    }

# ════════════════════════ accounts & sessions ════════════════════════

_dummy_hash = None


def _dummy_hash():
    """Hashed once so a sign-in attempt against an unknown email still costs the
    same as a real password check (no user-enumeration timing oracle)."""
    global _dummy_hash
    if _dummy_hash is None:
        _dummy_hash = auth.hash_password("xplora-timing-equaliser")
    return _dummy_hash


# Cookie is the primary transport; Bearer is accepted for API clients and tests.
_bearer = HTTPBearer(auto_error=False)
BearerCreds = Annotated[Optional[HTTPAuthorizationCredentials], Depends(_bearer)]


def _token_from_request(request: Request, bearer_creds: Optional[HTTPAuthorizationCredentials]) -> Optional[str]:
    if bearer_creds is not None:
        return bearer_creds.credentials
    return request.cookies.get(auth.COOKIE_NAME)


async def _resolve_user(request: Request, bearer_creds: BearerCreds) -> Optional[dict]:
    user_id = auth.decode_token(_token_from_request(request, bearer_creds))
    if user_id is None:
        return None
    try:
        return get_user_by_id(user_id)
    except Exception as e:
        print(f"session lookup failed: {e}")
        return None


OptionalUser = Annotated[Optional[dict], Depends(_resolve_user)]


async def require_user(user: OptionalUser) -> dict:
    if user is None:
        raise HTTPException(status_code=401, detail="Sign in to continue")
    return user


CurrentUser = Annotated[dict, Depends(require_user)]


def _issue_session(response: Response, user: dict) -> None:
    response.set_cookie(
        key=auth.COOKIE_NAME,
        value=auth.create_token(user),
        max_age=auth.cookie_max_age(),
        path="/",
        httponly=True,
        secure=auth.cookie_secure(),
        samesite="lax",
    )


class SignUpRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=200)
    confirm_password: Optional[str] = None
    display_name: Optional[str] = Field(default=None, max_length=80)


class SignInRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=200)


# Very small in-process guard against credential stuffing. Multi-worker
# deployments should put this behind a real rate limiter.
_LOGIN_WINDOW_SECONDS = 300
_LOGIN_MAX_FAILURES = 8
_login_failures: Dict[str, List[float]] = defaultdict(list)


def _too_many_attempts(key: str) -> bool:
    now = time.time()
    hits = [t for t in _login_failures[key] if now - t < _LOGIN_WINDOW_SECONDS]
    _login_failures[key] = hits
    return len(hits) >= _LOGIN_MAX_FAILURES


def _record_attempt(key: str) -> None:
    _login_failures[key].append(time.time())


auth_router = APIRouter(prefix="/api/auth", tags=["auth"])


@auth_router.post("/signup", status_code=201)
async def signup(req: SignUpRequest, response: Response):
    problem = auth.password_problem(req.password, req.confirm_password)
    if problem:
        raise HTTPException(status_code=422, detail=problem)
    user = create_user(req.email, auth.hash_password(req.password), req.display_name)
    if user is None:
        raise HTTPException(status_code=409, detail="That email is already registered. Sign in instead.")
    _issue_session(response, user)
    return {"user": user}


@auth_router.post("/login")
async def login(req: SignInRequest, response: Response):
    key = req.email.lower().strip()
    if _too_many_attempts(key):
        raise HTTPException(status_code=429, detail="Too many sign-in attempts. Try again in a few minutes.")
    user, reason = verify_credentials(req.email, req.password)
    if reason == "db_error":
        raise HTTPException(status_code=503, detail="Database unavailable. Please retry.")
    if user is None:
        _record_attempt(key)
        if reason == "unknown_email":
            _dummy_hash()  # equalise response time for unknown emails
        # Deliberately vague: never reveal whether the email exists.
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    _login_failures.pop(key, None)
    _issue_session(response, user)
    return {"user": user}


@auth_router.post("/logout")
async def logout(response: Response):
    response.delete_cookie(auth.COOKIE_NAME, path="/")
    return {"ok": True}


@auth_router.get("/me")
async def me(user: CurrentUser):
    return {"user": user}


@auth_router.get("/itineraries")
async def my_itineraries(user: CurrentUser, limit: int = 50):
    return {"itineraries": list_user_itineraries(user["id"], limit=limit)}


@auth_router.get("/itineraries/{itinerary_id}")
async def my_itinerary(itinerary_id: int, user: CurrentUser):
    stored = get_itinerary(itinerary_id, user_id=user["id"])
    if not stored:
        raise HTTPException(status_code=404, detail="Itinerary not found")
    stored["itinerary_data"] = stored.get("itinerary_data") or {}
    return stored


@auth_router.delete("/itineraries/{itinerary_id}")
async def delete_my_itinerary(itinerary_id: int, user: CurrentUser):
    if not delete_itinerary(itinerary_id, user["id"]):
        raise HTTPException(status_code=404, detail="Itinerary not found")
    return {"ok": True}


app.include_router(auth_router)


class PlanRequest(BaseModel):
    origin: Optional[str] = ""
    destination: str
    duration: int = 3
    budget: str = "Premier"
    interests: List[str] = ["Wellness", "Gastronomy"]
    travel_dates: Optional[str] = ""
    group_size: Optional[int] = 2
    group_type: Optional[str] = "Couple"
    dietary_requirements: Optional[List[str]] = []
    accessibility: Optional[List[str]] = []
    pace: Optional[str] = "Moderate"
    accommodation_preference: Optional[str] = "No preference"
    occasion: Optional[str] = ""
    language_preference: Optional[str] = "English only"
    risk_tolerance: Optional[str] = "Balanced"
    title: Optional[str] = ""

    VALID_BUDGETS: ClassVar[List[str]] = ["Essential", "Premier", "Elite", "Legendary"]
    VALID_PACES: ClassVar[List[str]] = ["Relaxed", "Moderate", "Active", "Intense"]
    VALID_GROUP_TYPES: ClassVar[List[str]] = ["Solo", "Couple", "Family", "Friends", "Business"]
    VALID_RISK_TOLERANCES: ClassVar[List[str]] = ["Conservative", "Balanced", "Adventurous"]
    VALID_OCCASIONS: ClassVar[List[str]] = ["", "Honeymoon", "Birthday", "Anniversary", "Graduation", "Proposal", "Retirement", "Festival/Celebration"]
    VALID_ACCOMMODATION: ClassVar[List[str]] = ["No preference", "Hotel", "Hostel", "Airbnb/Vacation Rental", "Boutique/Heritage Stay", "Camping/Glamping", "Luxury Resort"]
    VALID_LANGUAGE: ClassVar[List[str]] = ["English only", "Basic local phrases", "Conversational local", "Fluent local"]
    VALID_INTERESTS: ClassVar[List[str]] = [
        "Wellness", "Gastronomy", "Photography", "History", "Adventure", "Art",
        "Nature & Outdoors", "Nightlife", "Shopping", "Sports", "Architecture",
        "Music & Festivals", "Wildlife", "Spirituality"
    ]
    VALID_DIETARY: ClassVar[List[str]] = [
        "Vegetarian", "Vegan", "Halal", "Kosher", "Gluten-free",
        "Nut allergy", "Lactose intolerant", "No restrictions"
    ]
    VALID_ACCESSIBILITY: ClassVar[List[str]] = [
        "Wheelchair", "Limited mobility", "Stroller-friendly",
        "Visual support", "Hearing support", "None"
    ]

    @field_validator('destination')
    @classmethod
    def destination_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError('Destination is required.')
        return v.strip()

    @field_validator('duration')
    @classmethod
    def duration_must_be_valid(cls, v: int) -> int:
        if v < 1 or v > 14:
            raise ValueError('Duration must be between 1 and 14 days.')
        return v

    @field_validator('group_size')
    @classmethod
    def group_size_must_be_valid(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and (v < 1 or v > 12):
            raise ValueError('Group size must be between 1 and 12.')
        return v

    @field_validator('budget')
    @classmethod
    def budget_must_be_valid(cls, v: str) -> str:
        if v not in cls.VALID_BUDGETS:
            raise ValueError(f'Budget must be one of: {", ".join(cls.VALID_BUDGETS)}')
        return v

    @field_validator('pace')
    @classmethod
    def pace_must_be_valid(cls, v: Optional[str]) -> Optional[str]:
        if v and v not in cls.VALID_PACES:
            raise ValueError(f'Pace must be one of: {", ".join(cls.VALID_PACES)}')
        return v

    @field_validator('group_type')
    @classmethod
    def group_type_must_be_valid(cls, v: Optional[str]) -> Optional[str]:
        if v and v not in cls.VALID_GROUP_TYPES:
            raise ValueError(f'Group type must be one of: {", ".join(cls.VALID_GROUP_TYPES)}')
        return v

    @field_validator('risk_tolerance')
    @classmethod
    def risk_tolerance_must_be_valid(cls, v: Optional[str]) -> Optional[str]:
        if v and v not in cls.VALID_RISK_TOLERANCES:
            raise ValueError(f'Risk tolerance must be one of: {", ".join(cls.VALID_RISK_TOLERANCES)}')
        return v

    @field_validator('occasion')
    @classmethod
    def occasion_must_be_valid(cls, v: Optional[str]) -> Optional[str]:
        if v and v not in cls.VALID_OCCASIONS:
            raise ValueError(f'Occasion must be one of: {", ".join(cls.VALID_OCCASIONS)}')
        return v

    @field_validator('accommodation_preference')
    @classmethod
    def accommodation_must_be_valid(cls, v: Optional[str]) -> Optional[str]:
        if v and v not in cls.VALID_ACCOMMODATION:
            raise ValueError(f'Accommodation must be one of: {", ".join(cls.VALID_ACCOMMODATION)}')
        return v

    @field_validator('language_preference')
    @classmethod
    def language_must_be_valid(cls, v: Optional[str]) -> Optional[str]:
        if v and v not in cls.VALID_LANGUAGE:
            raise ValueError(f'Language preference must be one of: {", ".join(cls.VALID_LANGUAGE)}')
        return v

    @field_validator('interests')
    @classmethod
    def interests_must_be_valid(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError('At least one interest is required.')
        invalid = [i for i in v if i not in cls.VALID_INTERESTS]
        if invalid:
            raise ValueError(f'Invalid interests: {", ".join(invalid)}. Must be from: {", ".join(cls.VALID_INTERESTS)}')
        return v

    @field_validator('dietary_requirements')
    @classmethod
    def dietary_must_be_valid(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v:
            invalid = [i for i in v if i not in cls.VALID_DIETARY]
            if invalid:
                raise ValueError(f'Invalid dietary requirements: {", ".join(invalid)}')
        return v or []

    @field_validator('accessibility')
    @classmethod
    def accessibility_must_be_valid(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v:
            invalid = [i for i in v if i not in cls.VALID_ACCESSIBILITY]
            if invalid:
                raise ValueError(f'Invalid accessibility options: {", ".join(invalid)}')
        return v or []

class ItineraryDescriptionRequest(BaseModel):
    destination: str
    origin: Optional[str] = ""
    travel_dates: Optional[str] = ""
    interests: Optional[List[str]] = None
    budget: Optional[str] = "Premier"
    duration: Optional[int] = 3

    @field_validator('destination')
    @classmethod
    def validate_dest(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError('Destination is required.')
        return v.strip()

    @field_validator('duration')
    @classmethod
    def validate_dur(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and (v < 1 or v > 14):
            raise ValueError('Duration must be 1-14')
        return v


class AskPlaceRequest(BaseModel):
    place: str
    question: str
    conversation_id: Optional[str] = None

    @field_validator('place')
    @classmethod
    def validate_place(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError('Place is required.')
        v = v.strip()
        if len(v) < 1 or len(v) > 120:
            raise ValueError('Place must be between 1 and 120 characters.')
        return v

    @field_validator('question')
    @classmethod
    def validate_question(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError('Question is required.')
        v = v.strip()
        if len(v) < 3:
            raise ValueError('Question must be at least 3 characters.')
        if len(v) > 500:
            raise ValueError('Question must be at most 500 characters.')
        return v


@app.post("/api/ask-place")
async def ask_place(req: AskPlaceRequest):
    """Answer a question about a specific place using web research + LLM."""
    try:
        agent_system = get_agent_system()
        result = agent_system.answer_place_question(
            place=req.place,
            question=req.question,
            conversation_history=[],
        )

        conversation_id = req.conversation_id or str(uuid.uuid4())

        # Safely parse lat/lng as floats
        location = result.get("location", {})
        try:
            lat = float(location.get("lat", 0))
            lng = float(location.get("lng", 0))
        except (ValueError, TypeError):
            lat = 0.0
            lng = 0.0

        return {
            "place": result.get("place", req.place),
            "question": result.get("question", req.question),
            "conversation_id": conversation_id,
            "answer_markdown": result.get("answer_markdown", ""),
            "location": {
                "display_name": location.get("display_name", req.place),
                "lat": lat,
                "lng": lng,
                "address": location.get("address", ""),
                "type": location.get("type", "place"),
                "google_maps_url": location.get("google_maps_url", ""),
                "directions_url": location.get("directions_url", ""),
            },
            "facts": result.get("facts", []),
            "sources": result.get("sources", []),
            "followup_suggestions": result.get("followup_suggestions", []),
        }

    except Exception as e:
        print(f"Error in ask-place: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Place research temporarily unavailable. Try again."
        )


@app.post("/api/itinerary-description")
async def itinerary_description(req: ItineraryDescriptionRequest):
    """FAST itinerary description with travel details, flight booking details and links via Scrapling."""
    try:
        agent_system = get_agent_system()
        result = agent_system.generate_itinerary_description(
            destination=req.destination,
            origin=req.origin or "",
            travel_dates=req.travel_dates or "",
            interests=req.interests or [],
            budget=req.budget or "Premier",
            duration=req.duration or 3,
        )
        return result
    except Exception as e:
        print(f"Error in itinerary-description: {str(e)}")
        raise HTTPException(status_code=500, detail="Itinerary description temporarily unavailable. Try again.")


@app.get("/api/scrapling-search")
async def scrapling_search_endpoint(q: str = "", origin: str = "", destination: str = "", travel_dates: str = ""):
    """Direct FAST Scrapling search — returns travel details + flight links for any destination."""
    if not q and not destination:
        raise HTTPException(status_code=400, detail="Query or destination required")
    dest = destination or q
    try:
        from agents.tools.scrapling_search import scrapling_travel_search
        agent_system = get_agent_system()  # ensure init
        data = scrapling_travel_search(dest, origin=origin, travel_dates=travel_dates)
        return data
    except Exception as e:
        print(f"Error in scrapling-search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-itinerary")
async def generate_itinerary(req: PlanRequest, user: OptionalUser):
    try:
        agent_system = get_agent_system()
        
        state = TravelPlanState(
            messages=[],
            origin=req.origin,
            destination=req.destination,
            duration=req.duration,
            budget_range=req.budget,
            interests=req.interests,
            group_size=req.group_size,
            travel_dates=req.travel_dates or "",
            group_type=req.group_type,
            dietary_requirements=req.dietary_requirements,
            accessibility=req.accessibility,
            pace=req.pace,
            accommodation_preference=req.accommodation_preference,
            occasion=req.occasion,
            language_preference=req.language_preference,
            risk_tolerance=req.risk_tolerance,
            current_agent="",
            agent_outputs={},
            final_plan={},
            iteration_count=0
        )

        # Execution logic
        # We'll use graph.invoke(state) to get the final state once
        # If we wanted to stream progress, we could use a WebSocket/EventSource
        final_state = agent_system.graph.invoke(state, config={"recursion_limit": 50})
        
        # Extract the results from agent_outputs
        itinerary_data = final_state.get("agent_outputs", {})

        # Ensure Scrapling FAST travel/flight details are present (fallback enrichment)
        try:
            has_flight = False
            # Check itinerary planner output
            itin_out = itinerary_data.get("itinerary_planner", {}).get("output", {})
            if isinstance(itin_out, dict) and itin_out.get("flight_booking_details"):
                has_flight = True
            if not has_flight:
                from agents.tools.scrapling_search import scrapling_travel_search as _sts
                sc_data = _sts(req.destination, origin=req.origin or "", travel_dates=req.travel_dates or "", interests=req.interests, budget=req.budget)
                # Attach at top-level and into itinerary_planner for frontend
                itinerary_data["scrapling_travel_search"] = sc_data
                if isinstance(itin_out, dict):
                    itin_out["flight_booking_details"] = itin_out.get("flight_booking_details") or sc_data.get("flight_booking_details", {})
                    itin_out["hotel_booking_details"] = itin_out.get("hotel_booking_details") or sc_data.get("hotel_booking_details", {})
                    itin_out["useful_links"] = itin_out.get("useful_links") or sc_data.get("useful_links", [])
                    itin_out["sources"] = itin_out.get("sources") or sc_data.get("sources", [])[:10]
                    itin_out["travel_details"] = itin_out.get("travel_details") or sc_data.get("travel_details", "")
        except Exception as enrich_e:
            print(f"[WARNING] Post-enrichment failed: {enrich_e}")
        
        # Store in Neon Postgres, owned by the signed-in user when there is one
        save_itinerary(
            req.origin, 
            req.destination, 
            req.duration, 
            req.budget, 
            req.interests, 
            itinerary_data,
            travel_dates=req.travel_dates,
            group_size=req.group_size,
            group_type=req.group_type,
            dietary_requirements=req.dietary_requirements,
            accessibility=req.accessibility,
            pace=req.pace,
            accommodation_preference=req.accommodation_preference,
            occasion=req.occasion,
            title=req.title,
            user_id=(user or {}).get("id")
        )
        
        # Clean the output (some components might return strings instead of dicts)
        # The invoke function handles most of the robust parsing
        
        return itinerary_data

    except Exception as e:
        print(f"Error in generation: {str(e)}")
        # Instead of crashing, try to return whatever partial data we have
        try:
            # If the agent system was created but graph failed, return partial results
            if 'agent_system' in dir() and agent_system:
                itinerary_data = state.get("agent_outputs", {})
                if itinerary_data:
                    return itinerary_data
        except Exception:
            pass
        # Return a graceful error that the frontend can handle
        raise HTTPException(status_code=500, detail="Travel planning encountered an issue. Please try again.")

class TripRatingRequest(BaseModel):
    destination: str
    rating: int
    feedback: Optional[str] = ""
    origin: Optional[str] = ""
    trip_title: Optional[str] = ""

    @field_validator('rating')
    @classmethod
    def validate_rating(cls, v: int):
        if v < 1 or v > 5:
            raise ValueError('Rating must be 1-5')
        return v


@app.post("/api/trip-rating")
async def post_trip_rating(req: TripRatingRequest, user: OptionalUser):
    ok = save_trip_rating(
        req.destination, req.rating, req.feedback or "", req.origin or "",
        req.trip_title or "", user_id=(user or {}).get("id"),
    )
    if not ok:
        raise HTTPException(status_code=500, detail="Could not save rating")
    stats = get_rating_stats(req.destination)
    return {"ok": True, "stats": stats}


@app.get("/api/trip-rating")
async def get_trip_rating(destination: str = "", limit: int = 20):
    ratings = get_trip_ratings(destination or None, limit=min(limit, 50))
    stats = get_rating_stats(destination or None)
    return {"ratings": ratings, "stats": stats}


@app.get("/api/place-enrichment")
async def place_enrichment(q: str, offset: int = 0, use_cache: bool = True):
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="q required")
    q = q.strip()[:300]
    offset = max(0, min(int(offset or 0), 50))
    # Check cache first. Rows are only trusted when they record which source
    # produced them, which retires the older invented ratings/photos.
    if use_cache:
        cached = get_place_cache(q)
        cached_source = (cached.get("raw") or {}).get("source") if cached else None
        if cached and cached_source and (cached.get("rating") is not None or cached.get("photos")):
            from agents.tools.places_enrichment import _rotate
            return {
                **cached,
                "query": q,
                "source": cached_source,
                "photos": _rotate(cached.get("photos") or [], offset),
                "cached": True,
            }
    try:
        from agents.tools.places_enrichment import enrich_place
        data = enrich_place(q, offset=offset)
        # Save to cache (best-effort)
        try:
            set_place_cache(q, data)
        except Exception:
            pass
        return {**data, "cached": False}
    except Exception as e:
        print(f"place-enrichment error: {e}")
        raise HTTPException(status_code=500, detail="Enrichment failed")


@app.post("/api/places-batch")
async def places_batch(body: Dict[str, Any]):
    queries = body.get("queries", [])
    if not isinstance(queries, list) or not queries:
        raise HTTPException(status_code=400, detail="queries list required")
    queries = [str(x).strip() for x in queries if str(x).strip()][:10]
    try:
        from agents.tools.places_enrichment import enrich_place
        from concurrent.futures import ThreadPoolExecutor
        results = {}
        # Try cache first
        uncached = []
        for qq in queries:
            c = get_place_cache(qq)
            if c and c.get("rating"):
                results[qq] = {**c, "query": qq, "cached": True}
            else:
                uncached.append(qq)
        if uncached:
            with ThreadPoolExecutor(max_workers=4) as ex:
                futs = {ex.submit(enrich_place, qq): qq for qq in uncached}
                for fut, qq in futs.items():
                    try:
                        data = fut.result(timeout=10)
                        results[qq] = {**data, "cached": False}
                        try:
                            set_place_cache(qq, data)
                        except Exception:
                            pass
                    except Exception as e:
                        print(f"batch enrich {qq} failed: {e}")
                        results[qq] = {"query": qq, "rating": None, "photos": [], "reviews": [], "error": str(e)}
        return {"results": results}
    except Exception as e:
        print(f"places-batch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
