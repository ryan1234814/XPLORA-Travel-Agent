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
from config.langgraph_config import LangGraphConfig as config
from config.api_config import api_config
import requests
import threading
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

# Generic retry logic for Gemini Rate Limits (429)
def is_rate_limit_error(exception):
    """Check if the error is a rate limit error."""
    error_str = str(exception).lower()
    return "ratelimit" in error_str or "slow down" in error_str or "429" in error_str

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

    # Remove markdown code blocks if present
    try:
        # Pattern 1: ```json ... ```
        if '```json' in text:
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
        
        # Pattern 2: ``` ... ```
        if '```' in text:
            json_match = re.search(r'```\s*(\{.*?\})\s*```', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
    except Exception:
        pass

    # Try extraction with regex - look for the first { and the last }
    try:
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
    search_budget_info,
    search_local_transport_options,
    search_car_rentals,
    search_real_time_transit_info
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
            print(f"[SUCCESS] Using OpenRouter with model: {config.OPENROUTER_MODEL}")
        
        self._lock = threading.Lock()
        self._last_request_time = 0
        self.request_interval = 0.1  # Reduced to 0.1 for maximum speed
        self.graph=self.create_agent_graph()

    @retry(
        retry=retry_if_exception(is_rate_limit_error),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        stop=stop_after_attempt(5),
        reraise=True
    )
    def _invoke_llm(self, messages: List[BaseMessage]):
        """Robust wrapper for LLM invocation using OpenRouter or Ollama."""
        # Proactive Rate Limiting
        with self._lock:
            elapsed = time.time() - self._last_request_time
            if elapsed < self.request_interval:
                time.sleep(self.request_interval - elapsed)
            self._last_request_time = time.time()

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
            # Handle Ollama direct API
            if self.llm_type == "ollama":
                # Convert LangChain messages to Ollama format
                ollama_messages = []
                for msg in cleaned_messages:
                    if isinstance(msg, SystemMessage):
                        ollama_messages.append({"role": "system", "content": msg.content})
                    elif isinstance(msg, AIMessage):
                        ollama_messages.append({"role": "assistant", "content": msg.content})
                    else:
                        ollama_messages.append({"role": "user", "content": msg.content})
                
                print(f"[DEBUG] Calling Ollama with {len(ollama_messages)} messages")
                print(f"[DEBUG] First message preview: {ollama_messages[0]['content'][:100]}...")
                
                # Call Ollama API directly
                response = self.ollama_client.chat(
                    model=config.OLLAMA_MODEL,
                    messages=ollama_messages,
                    options={
                        "temperature": config.TEMPERATURE,
                        "num_predict": config.MAX_TOKENS,
                        "num_ctx": 8192,
                    }
                )
                
                # Debug: Print full response structure
                print(f"[DEBUG] Ollama response type: {type(response)}")
                print(f"[DEBUG] Ollama response attributes: {dir(response)}")
                
                # Extract content from response - it's an object, not a dict!
                try:
                    content = response.message.content
                    print(f"[DEBUG] Ollama API response length: {len(content)}")
                    print(f"[DEBUG] Response preview: {content[:200]}")
                except AttributeError as e:
                    print(f"[ERROR] Failed to extract content: {e}")
                    print(f"[ERROR] Response object: {response}")
                    content = ""
                
                # Return as AIMessage for compatibility
                return AIMessage(content=content)
            
            # Handle OpenRouter or Groq via LangChain
            else:
                result = self.llm.invoke(cleaned_messages)
                print(f"[DEBUG] LLM invocation successful. Response type: {type(result)}")
                print(f"[DEBUG] Response content length: {len(str(result.content)) if hasattr(result, 'content') else 'N/A'}")
                return result
                
        except Exception as e:
            print(f"[ERROR] LLM invocation failed: {type(e).__name__}: {str(e)}")
            print(f"[ERROR] Full error details: {repr(e)}")
            # Return a fallback message instead of crashing
            return AIMessage(content="Unable to generate response due to connection error. Please try again.")

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
         
         # Sequential execution: weather -> transport -> itinerary
         if iteration == 0 or len(agent_outputs) == 0:
             # First, get weather/climate data
             response = AIMessage(content="weather_analyst")
         elif 'weather_analyst' in agent_outputs and 'transport_mobility' not in agent_outputs:
             # Then get transport data
             response = AIMessage(content="transport_mobility")
         elif 'weather_analyst' in agent_outputs and 'transport_mobility' in agent_outputs and 'itinerary_planner' not in agent_outputs:
             # Finally, create itinerary with all data available
             response = AIMessage(content="itinerary_planner")
         elif 'itinerary_planner' in agent_outputs:
             # All done
             response = AIMessage(content="FINAL_PLAN")
         else:
             # Fallback: go to itinerary planner
             response = AIMessage(content="itinerary_planner")
         
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
1. ALWAYS use 'NEED_SEARCH: [query]' first to get current, real-world data about the destination, its top attractions, and local gems.
2. Based on the search results, suggest must-see places and cultural insights.
3. Activity recommendations based on interests.

If you have already received search results in the conversation history, proceed with your expert recommendations. Otherwise, start with a search.
"""
        messages=[SystemMessage(content=system_prompt)]
        if state.get("messages"):
            messages.extend(state["messages"][-2:])
        
        response = self._invoke_llm(messages)
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

    def _weather_analyst_agent(self, state: TravelPlanState) -> TravelPlanState:
        # 1. Try real-time weather from OpenWeather when available.
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
                    weather = (data.get("weather") or [{}])[0] or {}
                    
                    parsed = {
                        "destination": data.get("name") or state.get('destination'),
                        "travel_dates": state.get('travel_dates'),
                        "temperature_c": {
                            "expected_low": main.get("temp_min"),
                            "expected_high": main.get("temp_max"),
                            "typical_range": f"{main.get('temp_min', 'N/A')}–{main.get('temp_max', 'N/A')}°C",
                            "notes": f"Current: {main.get('temp')}°C ({weather.get('description')})"
                        },
                        "conditions_summary": weather.get("description") or "",
                        "best_times": ["Morning exploration", "Evening walk"],
                        "activity_suggestions": [f"Outdoor activities are ideal given {weather.get('description')}"],
                        "packing": ["Standard travel wear"],
                        "source": "openweather"
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

        # 2. Check if we have search results already
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
        parsed = _try_parse_json(response_text)
        
        agent_outputs["weather_analyst"] = {
            "response": response_text,
            "output": parsed if isinstance(parsed, dict) else response_text,
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
        
        response = self._invoke_llm(messages)
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
}}

If you need live data, respond with 'NEED_SEARCH: [query]'. Specifically use 'NEED_SEARCH: [destination] transport cost comparison' for comparative analysis.
"""

        messages = [SystemMessage(content=system_prompt)]
        if state.get("messages"):
            messages.extend(state["messages"][-2:])
            
        response = self._invoke_llm(messages)
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
            
        response = self._invoke_llm(messages)
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
        
        # Gather context from other agents
        agent_outputs = state.get("agent_outputs", {})
        # Make the planner self-sufficient - it can work without other agents for speed
        destination = state.get('destination', 'Dubai')
        duration = state.get('duration', 3)
        budget_tier = state.get('budget_range', 'Premier')
        interests = ', '.join(state.get('interests', []))
        
        system_prompt = f"""Create a {duration}-day travel itinerary for {destination} in JSON format.

REQUIREMENTS:
- Return ONLY valid JSON (no markdown, no explanations)
- Each day must have DIFFERENT real locations in {destination}
- Include {duration} days with 2-3 activities per day
- Budget: {budget_tier}, Interests: {interests}

JSON FORMAT:
{{
  "trip_title": "Title for {destination}",
  "overview": "Brief trip description",
  "sustainability_score": 75,
  "price_range": "$2000-$4000",
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
                    tool_result = search_hotels.invoke({"destination": search_query})
                elif "restaurant" in search_query_lower or "food" in search_query_lower:
                    tool_result = search_restaurants.invoke({"destination": search_query})
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
