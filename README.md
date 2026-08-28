# XPLORA — Intelligent Travel Architect

XPLORA is an AI-powered travel planning application that uses a team of specialized AI agents to research, design, and craft personalized travel itineraries. It also features **Ask XPLORA**, a place-specific Q&A tool that answers travel questions with real-time web research.

## What It Does

- **AI Itinerary Planning** — Tell XPLORA where you want to go, for how long, and what you're into. Six AI agents collaborate to build a day-by-day itinerary with activities, restaurants, transport, weather, budget breakdowns, and cultural insights.

- **Ask XPLORA** — Ask any question about any place in the world. XPLORA searches the web, geocodes the location, and returns a cited answer with location details, facts, and Google Maps links.

- **Follow-up Conversations** — Ask follow-up questions in the same session to dive deeper into any topic.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 19, TypeScript, Vite, Tailwind CSS 3, Framer Motion |
| Backend | Python, FastAPI, Pydantic |
| AI Orchestration | LangGraph (StateGraph), LangChain |
| LLM Providers | Groq (default), OpenRouter (fallback), Ollama (local) |
| Web Search | DuckDuckGo (free, no API key needed) |
| Geocoding | Nominatim / OpenStreetMap (free) |
| RAG Knowledge Base | Pinecone + Sentence Transformers (optional) |
| Database | MySQL (for storing generated itineraries) |

## Project Structure

```
XPLORA-Travel-Agent/
├── backend/
│   └── api.py                  # FastAPI routes (itinerary + ask-place)
├── agents/
│   ├── agents.py               # LangGraph multi-agent system
│   └── tools/
│       └── travel.py           # Search tools, geocoding, web research
├── config/
│   ├── api_config.py           # API key configuration
│   └── langgraph_config.py     # LLM provider settings
├── db/
│   └── database.py             # MySQL helpers
├── src/
│   ├── main.tsx                # Entry point with React Router
│   ├── App.tsx                 # Main itinerary planner UI
│   ├── App.css                 # Component styles
│   ├── index.css               # Global styles & CSS variables
│   └── pages/
│       ├── Landing.tsx         # Landing page (/)
│       └── AskPlace.tsx        # Ask XPLORA feature (/app → Ask tab)
├── .env                        # Environment variables (you create this)
├── requirements.txt            # Python dependencies
├── package.json                # Node.js dependencies
├── start.sh                    # Start both servers
└── tailwind.config.js          # Tailwind configuration
```

## Getting Started

### Prerequisites

- **Python 3.9+**
- **Node.js 20+** and npm
- **MySQL Server** (running on default port 3306)
- **A Groq API Key** — [Get one free](https://console.groq.com/keys)

### 1. Clone & Install

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Node.js dependencies
npm install
```

### 2. Set Up Environment Variables

Create a `.env` file in the project root:

```env
# Required
GROQ_API_KEY="your_groq_api_key_here"

# Database
MYSQL_HOST="localhost"
MYSQL_PORT=3306
MYSQL_USER="root"
MYSQL_PASSWORD="your_password"
MYSQL_DATABASE="travel_agent"

# Optional — fallback LLM when Groq is at capacity
OPENROUTER_API_KEY="your_openrouter_api_key_here"

# Optional — real-time weather
TOMORROW_IO_API_KEY="your_tomorrow_io_key_here"

# Optional — RAG knowledge base
PINECONE_API_KEY="your_pinecone_api_key_here"
PINECONE_INDEX_NAME="travel-guides"
```

### 3. Set Up the Database

```sql
CREATE DATABASE IF NOT EXISTS travel_agent;
```

The app creates required tables automatically on first run.

### 4. Start the App

```bash
bash start.sh
```

This launches both servers and opens the app in your browser:

- **Frontend:** http://localhost:3000
- **Backend API:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs

### Alternative: Start Manually

```bash
# Terminal 1 — Backend
python3 -m uvicorn backend.api:app --host 0.0.0.0 --port 8000

# Terminal 2 — Frontend
npm run dev
```

## How It Works

### Itinerary Planning

When you click **Design Itinerary**, the backend orchestrates six specialized AI agents:

1. **Travel Advisor** — Researches destination highlights and attractions
2. **Weather Analyst** — Fetches real-time weather and climate data
3. **Budget Optimizer** — Calculates cost breakdowns in local currency
4. **Local Expert** — Uncovers cultural customs, sensory profiles, and hidden heritage
5. **Transport Planner** — Maps routes, transit options, and airport transfers
6. **Itinerary Architect** — Assembles everything into a day-by-day plan

If the LLM is unavailable, the system falls back to DuckDuckGo web search to still provide useful results.

### Ask XPLORA

1. Enter a **place** and a **question**
2. The system geocodes the location via Nominatim
3. It searches the web (DuckDuckGo) + internal travel knowledge base
4. An LLM synthesizes the research into a cited answer
5. You get: markdown answer, map links, key facts, source citations, and follow-up suggestions

## LLM Providers

The project supports three providers. Edit `config/langgraph_config.py` to switch:

```python
LLM_PROVIDER = "groq"       # Default — fast, free tier
# LLM_PROVIDER = "openrouter"  # Fallback — free models available
# LLM_PROVIDER = "ollama"      # Local — unlimited, no API key needed
```

## Deployment

- **Render.com** — `render.yaml` included for one-click deployment
- **Vercel** — `vercel.json` included for frontend-only deployment

## License

This project is for educational and personal use.
