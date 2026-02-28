import sys
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json

# Add parent directory to path to import agents and config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.agents import LangTravelAgents, TravelPlanState

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

@app.post("/generate-itinerary")
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
            group_size=2, # Default
            travel_dates="Season: Spring 2024", # Example
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
        
        # Clean the output (some components might return strings instead of dicts)
        # We can use the logic from the streamlit app for robust parsing but the invoke already handled most of it
        
        return itinerary_data

    except Exception as e:
        print(f"Error in generation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
