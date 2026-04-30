import streamlit as st
import warnings
# Suppress the deprecated langchain.verbose warning from langchain_core
warnings.filterwarnings("ignore", category=UserWarning, message=".*Importing verbose from langchain root module.*")
import sys
import os
import json
import re
from datetime import datetime
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.agents import LangTravelAgents, TravelPlanState
from db.database import save_itinerary
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Page Configuration
st.set_page_config(
    page_title="XPLORA - Premium Travel Concierge",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Robust message content extractor
def get_content(msg):
    """Extract and sanitize content from messages, removing HTML/JSON code blocks."""
    if msg is None:
        return ""
    
    content = ""
    if hasattr(msg, 'content'):
        content = msg.content
    elif isinstance(msg, dict) and 'output' in msg:
        content = msg['output']
    else:
        content = str(msg)
    
    # If content is a dict or list, don't display it as raw JSON
    if isinstance(content, (dict, list)):
        return "[Structured data - please regenerate itinerary]"
    
    # Convert to string and sanitize
    content = str(content)
    
    # Remove markdown code blocks (both ```json and ```)
    import re
    content = re.sub(r'```json\s*.*?\s*```', '', content, flags=re.DOTALL)
    content = re.sub(r'```\s*.*?\s*```', '', content, flags=re.DOTALL)
    
    # Remove ALL HTML tags aggressively - match any <...> pattern
    content = re.sub(r'<[^>]+>', '', content, flags=re.DOTALL)
    # Also remove any remaining < or > characters that might be part of broken HTML
    content = re.sub(r'</?div[^>]*>', '', content, flags=re.DOTALL)
    content = re.sub(r'</?span[^>]*>', '', content, flags=re.DOTALL)
    
    # Remove JSON-like structures if they appear
    if content.strip().startswith('{') and content.strip().endswith('}'):
        return "[JSON data - please regenerate itinerary]"
    
    # If content looks like HTML (contains multiple HTML-like patterns), reject it
    if content.count('<') > 3 or content.count('class=') > 2:
        return "[HTML content detected - please regenerate itinerary]"
    
    return content.strip()

# Custom CSS for the Premium "Velura" Look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    :root {
        --primary: #8e7dbe;
        --bg-dark: #07090d;
        --card-bg: rgba(22, 26, 33, 0.7);
        --accent: #c695fa;
    }

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
        background-color: var(--bg-dark);
        color: #e2e8f0;
    }

    .main {
        background: radial-gradient(circle at top right, #1a152e 0%, #07090d 100%);
    }

    /* Premium Card Effect */
    .premium-card {
        background: var(--card-bg);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }

    /* Trip Header */
    .trip-header {
        padding: 40px 0;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        margin-bottom: 30px;
    }
    .trip-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #fff 0%, #a48cf4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
    }
    .trip-overview {
        color: #94a3b8;
        font-size: 1.25rem;
        margin-top: 15px;
        max-width: 900px;
        line-height: 1.6;
    }

    /* Badges */
    .badge-container {
        display: flex;
        gap: 12px;
        margin-top: 20px;
    }
    .badge {
        padding: 8px 16px;
        border-radius: 12px;
        font-size: 0.9rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 8px;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
    }

    /* Activity Design */
    .activity-row {
        display: flex;
        gap: 24px;
        margin-bottom: 30px;
        padding-left: 20px;
        border-left: 2px solid rgba(164, 140, 244, 0.2);
        position: relative;
    }
    .activity-row::before {
        content: '';
        position: absolute;
        left: -6px;
        top: 0;
        width: 10px;
        height: 10px;
        background: #a48cf4;
        border-radius: 50%;
        box-shadow: 0 0 15px #a48cf4;
    }
    .activity-time {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        color: #a48cf4;
        min-width: 80px;
        font-size: 1rem;
    }
    .activity-content {
        flex: 1;
    }
    .activity-name {
        font-size: 1.5rem;
        font-weight: 600;
        color: white;
        margin-bottom: 8px;
    }
    .activity-description {
        color: #94a3b8;
        line-height: 1.6;
        margin-bottom: 12px;
    }
    .activity-tags {
        display: flex;
        gap: 8px;
        align-items: center;
    }
    .tag {
        font-size: 0.75rem;
        background: rgba(164, 140, 244, 0.1);
        color: #a48cf4;
        padding: 4px 10px;
        border-radius: 6px;
        text-transform: uppercase;
        font-weight: 700;
    }
    .maps-link {
        color: #60a5fa;
        text-decoration: none;
        font-size: 0.85rem;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 4px;
        margin-top: 10px;
        transition: opacity 0.2s;
    }
    .maps-link:hover {
        opacity: 0.8;
        text-decoration: underline;
    }

    /* Transport Info Styling */
    .transport-info {
        background: rgba(164, 140, 244, 0.05);
        border: 1px dashed rgba(164, 140, 244, 0.3);
        border-radius: 12px;
        padding: 12px;
        margin-top: 15px;
        font-size: 0.85rem;
        display: flex;
        flex-direction: column;
        gap: 4px;
    }
    .transport-header {
        color: #a48cf4;
        font-weight: 700;
        text-transform: uppercase;
        font-size: 0.7rem;
        letter-spacing: 1px;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    /* Sidebar Overrides */
    [data-testid="stSidebar"] {
        background-color: #0c0e12;
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 12px !important;
        background: linear-gradient(135deg, #a48cf4 0%, #6e56cf 100%) !important;
        border: none !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 12px 24px !important;
        width: 100%;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        box-shadow: 0 0 20px rgba(164, 140, 244, 0.4);
        transform: translateY(-2px);
    }

    /* Custom Tab */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        margin-bottom: 30px;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 10px 24px;
        color: #94a3b8;
        border: 1px solid rgba(255,255,255,0.05);
    }
    .stTabs [aria-selected="true"] {
        background: #a48cf4 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper function for Maps
def get_map_html(location, height=300):
    if not location:
        return ""
    encoded_location = location.replace(" ", "+")
    url = f"https://www.google.com/maps?q={encoded_location}&output=embed"
    return f'<iframe width="100%" height="{height}" frameborder="0" style="border:0; border-radius: 12px;" src="{url}" allowfullscreen></iframe>'

# Sidebar Inputs (Styled)
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/diamond.png", width=60)
    st.title("VELURA")
    st.markdown("*Your Personal Travel Architect*")
    st.markdown("---")
    
    origin = st.text_input("Origin (Optional)", placeholder="e.g. New Delhi (DEL)")
    destination = st.text_input("Destination", placeholder="e.g. Kyoto, Japan")
    duration = st.slider("Duration (Days)", 1, 14, 3)
    budget = st.selectbox("Tier", ["Essential", "Premier", "Elite", "Legendary"])
    interests = st.multiselect(
        "Focus",
        ["Wellness", "Gastronomy", "Photography", "History", "Adventure", "Art"],
        default=["Wellness", "Gastronomy"]
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    generate_btn = st.button("DESIGN ITINERARY", type="primary")
    
    if st.button("RESET CONCIERGE", type="secondary"):
        st.session_state.agent_system = None
        st.session_state.itinerary_data = None
        st.rerun()

# Initial State
if "agent_system" not in st.session_state or st.session_state.agent_system is None:
    st.session_state.agent_system = LangTravelAgents()
    st.session_state.itinerary_data = None

if generate_btn:
    if not destination:
        st.error("Please define a destination.")
    else:
        with st.spinner(f"Curating your elite {destination} experience..."):
            state = TravelPlanState(
                messages=[],
                origin=origin,
                destination=destination,
                duration=duration,
                budget_range=budget,
                interests=interests,
                group_size=2, # Default
                travel_dates="Season: Spring 2024", # Example
                current_agent="",
                agent_outputs={},
                final_plan={},
                iteration_count=0
            )

            # Execution logic
            progress_container = st.container()
            status_area = st.empty()
            
            try:
                events = st.session_state.agent_system.graph.stream(state, config={"recursion_limit": 50})
                
                for event in events:
                    for node_name, node_state in event.items():
                        status_area.markdown(f"**Fine-tuning:** `{node_name.replace('_', ' ').title()}`")
                    final_state = list(event.values())[0]

                st.session_state.itinerary_data = final_state.get("agent_outputs", {})

                # Store in MySQL database
                save_itinerary(
                    origin, 
                    destination, 
                    duration, 
                    budget, 
                    interests, 
                    st.session_state.itinerary_data
                )

                st.rerun()
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "ratelimit" in error_msg.lower():
                    st.error("✦ **Concierge is busy.** Our elite planning servers are currently at capacity. Please wait a moment and try again.")
                else:
                    st.error(f"✦ **Concierge Interruption:** {error_msg}")
                st.session_state.agent_system = None # Reset to allow fresh retry

# RENDER UI
if st.session_state.itinerary_data:
    itinerary = st.session_state.itinerary_data.get("itinerary_planner", {}).get("output")
    mobility = st.session_state.itinerary_data.get("transport_mobility", {}).get("output")
    
    # Re-attempt parsing if it's a string that looks like JSON or if it's messy
    if isinstance(itinerary, str):
        try:
            # 1. Try direct parse
            itinerary = json.loads(itinerary.strip())
        except:
            try:
                # 2. Try regex extraction (look for first { and last })
                json_match = re.search(r'(\{.*\})', itinerary, re.DOTALL)
                if json_match:
                    itinerary = json.loads(json_match.group(1))
            except:
                pass

    if isinstance(mobility, str):
        try:
            mobility = json.loads(mobility.strip())
        except:
            try:
                json_match = re.search(r'(\{.*\})', mobility, re.DOTALL)
                if json_match:
                    mobility = json.loads(json_match.group(1))
            except:
                pass
            
    if isinstance(itinerary, dict):
        col_main, col_side = st.columns([2.5, 1])
        
        with col_main:
            # Trip Header
            st.markdown(f"""
<div class="trip-header">
    <div class="trip-title">{itinerary.get('trip_title', 'The Ultimate Journey')}</div>
    <div class="trip-overview">{itinerary.get('overview', '')}</div>
    <div class="badge-container">
        <div class="badge"> Sustainable Choice: {itinerary.get('sustainability_score', 85)}%</div>
        <div class="badge"> Range: {itinerary.get('price_range', 'Luxury')}</div>
    </div>
</div>

<div class="premium-card">
    <div style="color: #a48cf4; font-weight: 700; text-transform: uppercase; font-size: 0.8rem; letter-spacing: 2px; margin-bottom: 12px;">✦ Concierge Perspective</div>
    <div style="font-style: italic; color: #cbd5e1; font-size: 1.1rem; border-left: 2px solid #a48cf4; padding-left: 20px;">
        "{itinerary.get('concierge_note', 'Welcome to your bespoke adventure.')}"
    </div>
</div>
""", unsafe_allow_html=True)
            
            # Day Selection
            day_names = [f"Day {d.get('day_number')}" for d in itinerary.get('days', [])]
            if day_names:
                day_tabs = st.tabs(day_names)
                
                for i, day_tab in enumerate(day_tabs):
                    day_data = itinerary.get('days', [])[i]
                    with day_tab:
                        st.markdown(f"### {day_data.get('theme', 'Daily Explorations')}")
                        st.markdown(f"*{day_data.get('day_name', 'Plan')}*")
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                        for act in day_data.get('activities', []):
                            map_url = f"https://www.google.com/maps/search/?api=1&query={act.get('map_query', act.get('location')).replace(' ', '+')}"
                            # activity card
                            st.markdown(f"""
<div class="activity-row">
    <div class="activity-time">{act.get('time')}</div>
    <div class="activity-content">
        <div class="activity-name">{act.get('title')}</div>
        <div class="activity-description">{act.get('description')}</div>
        <div class="activity-tags">
            <div class="tag">{act.get('tag', 'Included')}</div>
            <span style="color: #475569;">•</span>
            <div style="color: #94a3b8; font-size:0.85rem;">📍 {act.get('location')}</div>
        </div>
        {f'''
        <div class="transport-info">
            <div class="transport-header">
                <span>🚀 Next Movement</span>
                <span style="opacity: 0.5;">|</span>
                <span>{act.get('transport_to_next', {}).get('mode', 'Standard Transit')}</span>
            </div>
            <div style="color: #cbd5e1;">{act.get('transport_to_next', {}).get('instructions', 'Proceed to your next destination.')}</div>
            <div style="display: flex; gap: 15px; margin-top: 5px; color: #94a3b8; font-family: 'Inter', sans-serif; font-weight: 600; font-size: 0.75rem;">
                <span>⏱ {act.get('transport_to_next', {}).get('duration', 'N/A')}</span>
                <span>💰 {act.get('transport_to_next', {}).get('cost', 'N/A')}</span>
            </div>
        </div>''' if act.get('transport_to_next') else ''}
    </div>
</div>
""", unsafe_allow_html=True)
                            
                            # Integrated Map for each activity (optional expander)
                            with st.expander(f"Explore {act.get('title')} 📍"):
                                st.components.v1.html(get_map_html(act.get('map_query', act.get('location'))), height=400)

        with col_side:
            # RIGHT PANEL
            st.markdown('<div class="side-panel">', unsafe_allow_html=True)
            st.markdown('<div class="side-panel-title">📊 Trip Insights</div>', unsafe_allow_html=True)
            
            # Mock chart like in image
            chart_data = pd.DataFrame({
                'Category': ['Culture', 'Rest', 'Activity'],
                'Balance': [70, 45, 85]
            })
            st.bar_chart(chart_data, x='Category', y='Balance', color="#5856d6")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Destination Insight
            st.markdown('<div class="side-panel">', unsafe_allow_html=True)
            st.markdown('<div class="side-panel-title">🕯️ Local Soul</div>', unsafe_allow_html=True)
            local_info = st.session_state.itinerary_data.get("local_expert", {}).get("output", "Discover the deep history and the hidden gems of your destination.")
            st.markdown(f'<div class="insight-text">{get_content(local_info)[:300]}...</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Weather Widget
            st.markdown('<div class="side-panel">', unsafe_allow_html=True)
            st.markdown('<div class="side-panel-title">🌦️ Climate Outlook</div>', unsafe_allow_html=True)
            weather_info = st.session_state.itinerary_data.get("weather_analyst", {}).get("output")
            if isinstance(weather_info, dict):
                temp_high = weather_info.get('temperature_c', {}).get('expected_high')
                temp_display = f"{temp_high}°C" if temp_high is not None else "N/A"
                st.write(f"High: {temp_display}")
                st.write(f"Conditions: {weather_info.get('conditions_summary', 'Clear skies')}")
            else:
                st.write(get_content(weather_info)[:150] + "...")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="side-panel">', unsafe_allow_html=True)
            st.markdown('<div class="side-panel-title">🚆 Transport & Mobility</div>', unsafe_allow_html=True)

            if isinstance(mobility, dict):
                flights = mobility.get("flights", {})
                trains = mobility.get("regional_trains_buses", {})
                transfers = mobility.get("airport_transfers", {})
                local = mobility.get("local_transport", {})
                route = mobility.get("route_optimization", {})

                with st.expander("Flight search & comparison"):
                    tips = flights.get("comparison_tips", [])
                    if tips:
                        st.markdown("\n".join([f"- {t}" for t in tips]))
                    queries = flights.get("recommended_search_queries", [])
                    if queries:
                        st.markdown("\n".join([f"- `{q}`" for q in queries]))
                    if flights.get("notes"):
                        st.write(flights.get("notes"))

                with st.expander("Train / bus options"):
                    hints = trains.get("provider_hints", [])
                    if hints:
                        st.markdown("\n".join([f"- {h}" for h in hints]))
                    queries = trains.get("recommended_search_queries", [])
                    if queries:
                        st.markdown("\n".join([f"- `{q}`" for q in queries]))
                    if trains.get("cost_vs_time_analysis"):
                        st.info(f"**Analysis:** {trains.get('cost_vs_time_analysis')}")
                    if trains.get("notes"):
                        st.write(trains.get("notes"))

                with st.expander("Car rental options"):
                    rentals = mobility.get("car_rentals", {})
                    options = rentals.get("options", [])
                    if options:
                        for opt in options:
                            st.markdown(f"**{opt.get('company', 'Rental')}**")
                            st.markdown(f"- Rate: {opt.get('estimated_daily_rate', 'N/A')}")
                            st.markdown(f"- Details: {opt.get('pros_cons', 'N/A')}")
                    queries = rentals.get("recommended_search_queries", [])
                    if queries:
                        st.markdown("\n".join([f"- `{q}`" for q in queries]))
                    if rentals.get("notes"):
                        st.write(rentals.get("notes"))

                with st.expander("Airport transfers"):
                    options = transfers.get("options", [])
                    if options:
                        for opt in options[:6]:
                            mode = opt.get("mode", "")
                            why = opt.get("why", "")
                            tmin = opt.get("typical_time_min", None)
                            cost = opt.get("cost_estimate", "")
                            time_txt = f" (~{int(tmin)} min)" if isinstance(tmin, (int, float)) else ""
                            cost_txt = f" [{cost}]" if cost else ""
                            st.markdown(f"- **{mode}{time_txt}{cost_txt}**: {why}")
                    queries = transfers.get("recommended_search_queries", [])
                    if queries:
                        st.markdown("\n".join([f"- `{q}`" for q in queries]))
                    if transfers.get("notes"):
                        st.write(transfers.get("notes"))

                with st.expander("Local transport guidance"):
                    passes = local.get("passes", [])
                    apps = local.get("apps", [])
                    how = local.get("how_to_get_around", [])
                    if passes:
                        st.markdown("\n".join([f"- {p}" for p in passes]))
                    if apps:
                        st.markdown("\n".join([f"- {a}" for a in apps]))
                    if how:
                        st.markdown("\n".join([f"- {h}" for h in how]))
                    if local.get("cost_vs_time_comparison"):
                        st.info(f"**Comparison:** {local.get('cost_vs_time_comparison')}")
                    queries = local.get("recommended_search_queries", [])
                    if queries:
                        st.markdown("\n".join([f"- `{q}`" for q in queries]))
                    if local.get("notes"):
                        st.write(local.get("notes"))

                with st.expander("Route optimization"):
                    if route.get("strategy"):
                        st.write(route.get("strategy"))
                    groupings = route.get("suggested_area_groupings", [])
                    if groupings:
                        st.markdown("\n".join([f"- {g}" for g in groupings]))
                    stops = route.get("sample_day_route_stops", [])
                    if stops:
                        st.markdown("\n".join([f"- {s}" for s in stops]))
                    url = route.get("google_maps_directions_url", "")
                    if url:
                        st.markdown(f"[Open in Google Maps →]({url})")
            else:
                if mobility:
                    st.write(get_content(mobility)[:300] + "...")
                else:
                    st.write("Movement planning will appear here after itinerary generation.")

            st.markdown('</div>', unsafe_allow_html=True)

    elif itinerary:
        # If itinerary exists but is not a dict, show a clean error message
        st.markdown(f"""
            <div class="premium-card">
                <div class="trip-title" style="font-size: 2rem;">Itinerary Generation Issue</div>
                <div class="activity-description" style="margin-top:20px;">
                    We encountered an issue generating your structured itinerary. 
                    Please click the <strong>RESET CONCIERGE</strong> button in the sidebar and try again.
                    <br><br>
                    If the issue persists, try adjusting your destination or preferences.
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.info("✦ Our concierge is meticulously assembling your bespoke experience. One moment...")

else:
    # Landing Page
    st.markdown("<br><br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c2:
        st.markdown("""
            <div style="text-align: center;">
                <img src="https://img.icons8.com/fluency/96/diamond.png" width="80">
                <h1 style="color: white;">Velura</h1>
                <p style="color: #94a3b8;">Enter your desired destination on the left to begin your bespoke travel narrative.</p>
            </div>
        """, unsafe_allow_html=True)
