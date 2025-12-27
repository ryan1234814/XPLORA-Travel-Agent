from typing import List,Dict,Any,Optional,Annotated,TypedDict
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from langgraph.graph import StateGraph,END
import json
import re
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from config.langgraph_config import LangGraphConfig as config
from config.api_config import api_config
import requests

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
    
    # Try direct parse
    try:
        return json.loads(text.strip())
    except Exception:
        pass

    # Try extraction with regex
    try:
        # Look for the first { and the last }
        json_match = re.search(r'(\{.*\})', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
    except Exception:
        pass
        
    return None

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
    search_budget_info
)

class TravelPlanState(TypedDict):
    messages:Annotated[List[HumanMessage|AIMessage|SystemMessage],add_message]
    origin:str
    destination:str
    duration:int
    budget_range:str
    interests:List[str] 
    group_size:int
    travel_dates:str
    current_agent:str
    agent_outputs:Dict[str,Any]
    final_plan:  Dict[str,Any]
    iteration_count:int
    
class LangTravelAgents:
    def __init__(self):
        self.llm=ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            google_api_key=config.GEMINI_API_KEY,
            temperature=config.TEMPERATURE,
            max_output_tokens=config.MAX_TOKENS,
            top_p=config.TOP_P,
        )
        self.graph=self.create_agent_graph()
        
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
         system_prompt = f"""You are the Coordinator Agent for a multi-agent travel planning system.

Your role is to:
1. Analyze the travel planning request
2. Determine which specialized agents need to contribute
3. Coordinate the workflow between agents
4. Synthesize final recommendations

Current request:
- Destination: {state.get('destination', 'Not specified')}
- Duration: {state.get('duration', 'Not specified')} days
- Budget: {state.get('budget_range', 'Not specified')}
- Interests: {', '.join(state.get('interests', []))}
- Group size: {state.get('group_size', 1)}
- Travel dates: {state.get('travel_dates', 'Not specified')}

Available agents:
- travel_advisor: Destination expertise and attraction recommendations
- weather_analyst: Weather forecasting and activity planning
- budget_optimizer: Cost analysis and money-saving strategies
- local_expert: Local insights and cultural tips
- transport_mobility: End-to-end movement planning (flights, rail/bus, transfers, local transport, route optimization)
- itinerary_planner: Schedule optimization and logistics

Agent outputs so far: {json.dumps(state.get('agent_outputs', {}), indent=2)}

Based on the current state, decide what to do next:
1. If you need more information or specific analysis, specify which agent should work next.
2. IMPORTANT: You MUST call the 'itinerary_planner' to generate the final structured JSON itinerary before ending the process.
3. If you have enough information from all other agents, your next step should be 'itinerary_planner'.
4. ONLY respond with 'FINAL_PLAN' if the 'itinerary_planner' has already completed its work and you are ready to conclude.

Your response should be either:
- Agent name to call next (travel_advisor, weather_analyst, budget_optimizer, local_expert, transport_mobility, itinerary_planner)
- 'FINAL_PLAN' if the itinerary is already created and you are ready to conclude
- 'SEARCH' if you need to search for information first
"""
         messages=[SystemMessage(content=system_prompt)]
         if state.get("messages"):
             messages.extend(state["messages"][-3:])
         else:
             # Add a human message to start the conversation
             messages.append(HumanMessage(content="Please analyze the travel request and determine which agents should contribute."))
         response=self.llm.invoke(messages)  
         new_state=state.copy()
         new_state["messages"]=state.get("messages",[])+[response] 
         new_state["current_agent"] = "coordinator"
         new_state["iteration_count"] = state.get("iteration_count", 0) + 1
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

Your task: Provide comprehensive destination advice including:
1. Top attractions and must-see places
2. Cultural insights and etiquette tips
3. Best areas to stay and explore
4. Activity recommendations based on interests

If you need to search for current information about the destination, respond with 'NEED_SEARCH: [search query]'
Otherwise, provide your expert recommendations based on your knowledge.
"""
        messages=[SystemMessage(content=system_prompt)]
        if state.get("messages"):
            messages.extend(state["messages"][-2:])
        response=self.llm.invoke(messages)
        agent_outputs=state.get("agent_outputs",{})
        response_text = _safe_message_content(response)
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
    def _weather_analyst_agent(self,state:TravelPlanState)->TravelPlanState:
        # Prefer real-time weather from OpenWeather when available.
        try:
            if api_config.OPENWEATHER_API_KEY and state.get('destination'):
                params = {
                    "q": state.get('destination'),
                    "appid": api_config.OPENWEATHER_API_KEY,
                    "units": "metric"
                }
                resp = requests.get(f"{api_config.WEATHER_BASE_URL}/weather", params=params, timeout=15)
                if resp.status_code == 200:
                    data = resp.json() or {}
                    main = data.get("main", {}) or {}
                    wind = data.get("wind", {}) or {}
                    weather = (data.get("weather") or [{}])[0] or {}
                    sys_data = data.get("sys", {}) or {}

                    parsed = {
                        "destination": data.get("name") or state.get('destination'),
                        "travel_dates": state.get('travel_dates'),
                        "temperature_c": {
                            "expected_low": main.get("temp_min"),
                            "expected_high": main.get("temp_max"),
                            "typical_range": f"{main.get('temp_min', 'N/A')}–{main.get('temp_max', 'N/A')}°C",
                            "notes": f"Current: {main.get('temp')}°C (feels like {main.get('feels_like')}°C)"
                        },
                        "conditions_summary": weather.get("description") or "",
                        "best_times": [],
                        "activity_suggestions": [],
                        "packing": [],
                        "source": {
                            "provider": "openweather",
                            "country": sys_data.get("country"),
                            "humidity_pct": main.get("humidity"),
                            "wind_speed_mps": wind.get("speed")
                        }
                    }

                    agent_outputs = state.get("agent_outputs", {})
                    agent_outputs["weather_analyst"] = {
                        "response": json.dumps(parsed),
                        "output": parsed,
                        "timestamp": datetime.now().isoformat(),
                        "status": "completed"
                    }
                    new_state = state.copy()
                    new_state["messages"] = state.get("messages", []) + [AIMessage(content=json.dumps(parsed))]
                    new_state["current_agent"] = "weather_analyst"
                    new_state["agent_outputs"] = agent_outputs
                    return new_state
        except Exception:
            pass

        system_prompt = f"""You are the Weather Analyst Agent, specialized in weather intelligence and climate-aware planning.

Your expertise includes:
- Weather pattern analysis
- Seasonal travel recommendations
- Activity planning based on weather conditions
- Climate considerations for destinations

Current planning request:
- Destination: {state.get('destination')}
- Travel dates: {state.get('travel_dates')}
- Duration: {state.get('duration')} days
- Planned activities: {', '.join(state.get('interests', []))}

Your task: Provide weather-intelligent recommendations including:
1. Expected weather conditions during travel dates
2. Temperature details (in Celsius) for the destination during the travel dates
3. Best times of day for outdoor activities
4. Weather-appropriate activity suggestions
5. Packing recommendations based on climate

Return your result as STRICT JSON with this schema:
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

If you cannot provide exact temperatures, use typical seasonal ranges and set expected_low/high to null.

If you need current weather data, respond with 'NEED_SEARCH: [weather search query]'
Otherwise, provide your analysis based on climate knowledge.
"""
        messages=[SystemMessage(content=system_prompt)]
        if state.get("messages"):
            messages.extend(state["messages"][-2:])
        response=self.llm.invoke(messages)
        response_text = _safe_message_content(response)
        parsed = _try_parse_json(response_text)
        agent_outputs=state.get("agent_outputs",{})
        agent_outputs["weather_analyst"] = {
            "response": response_text,
            "output": parsed if isinstance(parsed, dict) else response_text,
            "timestamp": datetime.now().isoformat(),
            "status": "completed"
        }
        new_state=state.copy()
        new_state["messages"]=state.get("messages",[])+[response]
        new_state["current_agent"]="weather_analyst"
        new_state["agent_outputs"]=agent_outputs
        return new_state
    
    def _budget_optimizer_agent(self, state: TravelPlanState) -> TravelPlanState:
        """Budget optimizer agent - stub implementation"""
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

Your task: Provide budget optimization recommendations including:
1. Estimated daily and total costs
2. Budget breakdown by category (accommodation, food, activities, transport)
3. Money-saving tips and strategies
4. Cost-effective alternatives for expensive activities

If you need current pricing information, respond with 'NEED_SEARCH: [budget search query]'
Otherwise, provide your budget analysis and recommendations.
"""
        
        messages = [SystemMessage(content=system_prompt)]
        if state.get("messages"):
            messages.extend(state["messages"][-2:])
        response = self.llm.invoke(messages)
        response_text = _safe_message_content(response)
        
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
        system_prompt = f"""You are the Transport & Mobility Agent.

Purpose: End-to-end movement planning for a trip.

Trip context:
- Origin (if provided): {state.get('origin', '')}
- Destination: {state.get('destination')}
- Duration: {state.get('duration')} days
- Group size: {state.get('group_size')}
- Budget tier: {state.get('budget_range')}
- Interests: {', '.join(state.get('interests', []))}

Your tasks:
1) Flight search & comparison guidance (best sites/strategies, what to compare).
2) Train/bus options (region-specific tips, when rail is better than flying).
3) Airport transfer suggestions (best options, rough timing, pitfalls).
4) Local transport guidance (cards/passes, apps, etiquette, safety).
5) Route optimization between attractions: propose an efficient order and grouping strategy.

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
    "notes": string
  }},
  "airport_transfers": {{
    "recommended_search_queries": [string],
    "options": [{{"mode": string, "why": string, "typical_time_min": number|null}}],
    "notes": string
  }},
  "local_transport": {{
    "recommended_search_queries": [string],
    "how_to_get_around": [string],
    "apps": [string],
    "passes": [string],
    "notes": string
  }},
  "route_optimization": {{
    "strategy": string,
    "suggested_area_groupings": [string],
    "sample_day_route_stops": [string],
    "google_maps_directions_url": string
  }}
}}

If you need live data, respond with 'NEED_SEARCH: [query]'.
"""

        messages = [SystemMessage(content=system_prompt)]
        if state.get("messages"):
            messages.extend(state["messages"][-2:])
        response = self.llm.invoke(messages)
        response_text = _safe_message_content(response)
        parsed = _try_parse_json(response_text)

        agent_outputs = state.get("agent_outputs", {})
        agent_outputs["transport_mobility"] = {
            "response": response_text,
            "output": parsed if isinstance(parsed, dict) else response_text,
            "timestamp": datetime.now().isoformat(),
            "status": "completed"
        }

        new_state = state.copy()
        new_state["messages"] = state.get("messages", []) + [response]
        new_state["current_agent"] = "transport_mobility"
        new_state["agent_outputs"] = agent_outputs
        return new_state
    
    def _local_expert_agent(self, state: TravelPlanState) -> TravelPlanState:
        """Local expert agent - stub implementation"""
        system_prompt = f"""You are the Local Expert Agent, specialized in insider knowledge and local insights.

Your expertise includes:
- Local customs and cultural nuances
- Hidden gems and off-the-beaten-path recommendations
- Local dining and entertainment scene
- Practical local tips and advice

Current planning request:
- Destination: {state.get('destination')}
- Interests: {', '.join(state.get('interests', []))}
- Duration: {state.get('duration')} days

Your task: Provide local expert insights including:
1. Hidden gems and local favorites
2. Cultural etiquette and customs
3. Local dining recommendations
4. Insider tips for getting around and saving money

If you need current local information, respond with 'NEED_SEARCH: [local tips search query]'
Otherwise, provide your local expertise and insights.
"""
        
        messages = [SystemMessage(content=system_prompt)]
        if state.get("messages"):
            messages.extend(state["messages"][-2:])
        response = self.llm.invoke(messages)
        response_text = _safe_message_content(response)
        
        agent_outputs = state.get("agent_outputs", {})
        agent_outputs["local_expert"] = {
            "response": response_text,
            "output": response_text,
            "timestamp": datetime.now().isoformat(),
            "status": "completed"
        }
        
        new_state = state.copy()
        new_state["messages"] = state.get("messages", []) + [response]
        new_state["current_agent"] = "local_expert"
        new_state["agent_outputs"] = agent_outputs
        return new_state
    
    def _itinerary_planner_agent(self, state: TravelPlanState) -> TravelPlanState:
        """Itinerary planner agent - produces structured JSON for the UI"""
        system_prompt = f"""You are the Itinerary Planner Agent, a world-class luxury travel architect.
        
Your task: Create a definitive, structured itinerary for {state.get('destination')}.
Duration: {state.get('duration')} days.

Your task is to provide a complete, high-end travel narrative.

IMPORTANT: Your response must be a single, valid JSON object containing the itinerary. 
This is critical for the premium user interface to display your curated work.

Schema:
{{
  "trip_title": "Elegant naming",
  "overview": "2-3 sentence teaser",
  "sustainability_score": 70-98,
  "price_range": "e.g., $2,500 - $4,000",
  "concierge_note": "A personalized greeting addressing the traveler's interests.",
  "days": [
    {{
      "day_number": 1,
      "day_name": "e.g., Friday",
      "theme": "Daily focus",
      "activities": [
        {{
          "time": "09:00 AM",
          "title": "Name",
          "description": "Engaging text",
          "location": "Venue Name",
          "tag": "Category",
          "map_query": "Search query for Google Maps"
        }}
      ]
    }}
  ]
}}
"""
        
        messages = [SystemMessage(content=system_prompt)]
        if state.get("messages"):
            # Only keep recent history to stay focused
            messages.extend(state["messages"][-5:])
            
        response = self.llm.invoke(messages)
        response_text = _safe_message_content(response)
        
        # Force JSON parsing using the improved helper
        parsed = _try_parse_json(response_text)

        # If parsing fails or output is empty, provide a basic fallback structure
        if not parsed or not response_text.strip():
            if not response_text.strip():
                response_text = "The AI agent encountered a brief interruption. Please regenerate."
            
            # Basic structure if we really can't get JSON
            parsed = {
                "trip_title": f"Journey to {state.get('destination')}",
                "overview": response_text[:500] if len(response_text) > 0 else "Curating your bespoke travel experience.",
                "sustainability_score": 85,
                "price_range": state.get("budget_range", "Luxury"),
                "concierge_note": "A bespoke plan is being finalized.",
                "days": []
            }

        agent_outputs = state.get("agent_outputs", {})
        agent_outputs["itinerary_planner"] = {
            "response": response_text,
            "output": parsed if parsed else response_text,
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
                    tool_result = search_hotels.invoke({"destination": search_query})
                elif "restaurant" in search_query_lower or "food" in search_query_lower:
                    tool_result = search_restaurants.invoke({"destination": search_query})
                elif "attraction" in search_query_lower or "activity" in search_query_lower:
                    tool_result = search_attractions.invoke({"destination": search_query})
                elif "budget" in search_query_lower or "cost" in search_query_lower or current_agent == "budget_optimizer":
                    tool_result = search_budget_info.invoke({"destination": search_query})
                elif "tip" in search_query_lower or "culture" in search_query_lower or current_agent == "local_expert":
                    tool_result = search_local_tips.invoke({"destination": search_query})
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
        # For now, always return to coordinator
        return "coordinator"
