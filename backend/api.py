import sys
import os
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from typing import List, Optional, Dict, Any, ClassVar
import json
import mysql.connector
from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.agents import LangTravelAgents, TravelPlanState
from db.database import save_itinerary

app = FastAPI()

# Enable CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, this should be more restricted
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        agent_system = LangTravelAgents()
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


@app.post("/api/generate-itinerary")
async def generate_itinerary(req: PlanRequest):
    try:
        agent_system = LangTravelAgents()
        
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
        
        # Store in MySQL database
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
            occasion=req.occasion
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
