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
    search_travel_blogs
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
                            raise  # Let outer handler catch this
                    else:
                        raise groq_err
            
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
                    raise
                    
        except Exception as e:
            print(f"[ERROR] All LLM invocation paths failed: {type(e).__name__}: {str(e)[:200]}")
            print(f"[ERROR] Full error details: {repr(e)}")
            # Return a fallback message instead of crashing
            return AIMessage(content="Unable to generate response due to connection error. Please try again.")

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
        destination = state.get('destination', '')
        origin = state.get('origin', '')

        # Define JSON schema for ScrapeGraphAI extraction
        schema = {
            "type": "object",
            "properties": {
                "flights": {
                    "type": "object",
                    "properties": {
                        "recommended_search_queries": {"type": "array", "items": {"type": "string"}},
                        "comparison_tips": {"type": "array", "items": {"type": "string"}},
                        "notes": {"type": "string"}
                    },
                    "required": ["recommended_search_queries", "comparison_tips", "notes"]
                },
                "regional_trains_buses": {
                    "type": "object",
                    "properties": {
                        "recommended_search_queries": {"type": "array", "items": {"type": "string"}},
                        "provider_hints": {"type": "array", "items": {"type": "string"}},
                        "cost_vs_time_analysis": {"type": "string"},
                        "notes": {"type": "string"}
                    },
                    "required": ["recommended_search_queries", "provider_hints", "cost_vs_time_analysis", "notes"]
                },
                "car_rentals": {
                    "type": "object",
                    "properties": {
                        "recommended_search_queries": {"type": "array", "items": {"type": "string"}},
                        "options": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "company": {"type": "string"},
                                    "estimated_daily_rate": {"type": "string"},
                                    "pros_cons": {"type": "string"}
                                },
                                "required": ["company", "estimated_daily_rate", "pros_cons"]
                            }
                        },
                        "notes": {"type": "string"}
                    },
                    "required": ["recommended_search_queries", "options", "notes"]
                },
                "airport_transfers": {
                    "type": "object",
                    "properties": {
                        "recommended_search_queries": {"type": "array", "items": {"type": "string"}},
                        "options": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "mode": {"type": "string"},
                                    "why": {"type": "string"},
                                    "cost_estimate": {"type": "string"},
                                    "typical_time_min": {"type": ["integer", "null"]}
                                },
                                "required": ["mode", "why", "cost_estimate", "typical_time_min"]
                            }
                        },
                        "notes": {"type": "string"}
                    },
                    "required": ["recommended_search_queries", "options", "notes"]
                },
                "local_transport": {
                    "type": "object",
                    "properties": {
                        "recommended_search_queries": {"type": "array", "items": {"type": "string"}},
                        "how_to_get_around": {"type": "array", "items": {"type": "string"}},
                        "apps": {"type": "array", "items": {"type": "string"}},
                        "passes": {"type": "array", "items": {"type": "string"}},
                        "real_time_info_links": {"type": "array", "items": {"type": "string"}},
                        "cost_vs_time_comparison": {"type": "string"},
                        "notes": {"type": "string"}
                    },
                    "required": ["recommended_search_queries", "how_to_get_around", "apps", "passes", "real_time_info_links", "cost_vs_time_comparison", "notes"]
                },
                "route_optimization": {
                    "type": "object",
                    "properties": {
                        "strategy": {"type": "string"},
                        "suggested_area_groupings": {"type": "array", "items": {"type": "string"}},
                        "sample_day_route_stops": {"type": "array", "items": {"type": "string"}},
                        "google_maps_directions_url": {"type": "string"}
                    },
                    "required": ["strategy", "suggested_area_groupings", "sample_day_route_stops", "google_maps_directions_url"]
                }
            },
            "required": ["flights", "regional_trains_buses", "car_rentals", "airport_transfers", "local_transport", "route_optimization"]
        }

        # Attempt ScrapeGraphAI structured search first if API key is present
        scrapegraph_key = getattr(api_config, "SCRAPEGRAPH_API_KEY", None)
        if scrapegraph_key:
            try:
                print(f"[INFO] Accessing ScrapeGraphAI for {destination} web research...")
                from scrapegraph_py import ScrapeGraphAI
                sg = ScrapeGraphAI(api_key=scrapegraph_key)
                
                query = f"{destination} public transit options flights trains car rentals airport transfer guide"
                prompt = (
                    f"Perform web research to extract comprehensive and real details for public transit, flights, trains, "
                    f"car rental, and airport transfer options for a trip to {destination} originating from {origin or 'any origin'}. "
                    f"Populate the fields in the schema with accurate local transport and route optimization details."
                )
                
                res = sg.search(
                    query=query,
                    prompt=prompt,
                    schema=schema,
                    num_results=2
                )
                if res.status == "success" and res.data and getattr(res.data, "json_data", None):
                    parsed = res.data.json_data
                    if isinstance(parsed, dict) and all(key in parsed for key in schema["required"]):
                        print(f"[SUCCESS] ScrapeGraphAI successfully researched mobility details for {destination}")
                        
                        agent_outputs = state.get("agent_outputs", {})
                        agent_outputs["transport_mobility"] = {
                            "response": json.dumps(parsed),
                            "output": parsed,
                            "timestamp": datetime.now().isoformat(),
                            "status": "completed",
                            "source": "scrapegraph_ai"
                        }

                        new_state = state.copy()
                        new_state["messages"] = state.get("messages", []) + [AIMessage(content=json.dumps(parsed))]
                        new_state["current_agent"] = "transport_mobility"
                        new_state["agent_outputs"] = agent_outputs
                        return new_state
                print(f"[WARNING] ScrapeGraphAI returned status {res.status} or invalid data. Falling back to default LLM.")
            except Exception as sg_err:
                print(f"[WARNING] ScrapeGraphAI request failed: {sg_err}. Falling back to default LLM.")

        # Fallback implementation using the default LLM
        print(f"[INFO] Using fallback LLM for {destination} mobility planning...")
        system_prompt = f"""You are the Transport & Mobility Agent.

Purpose: End-to-end movement planning for a trip.

Trip context:
- Origin (if provided): {origin}
- Destination: {destination}
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
            "status": "completed",
            "source": "fallback_llm"
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
        system_prompt = f"""You are the Local Expert Agent, specialized in deep cultural intelligence, local behaviors, and heritage.

Your task: Propose a deep cultural-intelligence brief for a trip to {state.get('destination')} with interests: {', '.join(state.get('interests', []))}.

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
