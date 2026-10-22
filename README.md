# XPLORA — Intelligent Travel Architect

XPLORA is an AI-powered travel planning application that uses a team of specialized AI agents to research, design, and craft personalized travel itineraries. It also features **Ask XPLORA**, a place-specific Q&A tool that answers travel questions with real-time web research.

## What It Does

- **AI Itinerary Planning** — Tell XPLORA where you want to go, for how long, and what you're into. Six AI agents collaborate to build a day-by-day itinerary with activities, restaurants, transport, weather, budget breakdowns, and cultural insights.

- **Ask XPLORA** — Ask any question about any place in the world. XPLORA searches the web, geocodes the location, and returns a cited answer with location details, facts, and Google Maps links.

- **Follow-up Conversations** — Ask follow-up questions in the same session to dive deeper into any topic.

- **Accounts & Dashboard** — Sign up with email and password, and every itinerary you generate is archived to your own dashboard, where you can reopen or delete any past trip.

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
| Database | Neon (serverless Postgres) — itineraries, accounts, ratings, place cache |

## Project Structure

```
XPLORA-Travel-Agent/
├── backend/
│   ├── api.py                  # FastAPI routes (itinerary + ask-place + auth)
│   └── auth.py                 # bcrypt hashing, JWT session cookie
├── agents/
│   ├── agents.py               # LangGraph multi-agent system
│   └── tools/
│       └── travel.py           # Search tools, geocoding, web research
├── config/
│   ├── api_config.py           # API key configuration
│   └── langgraph_config.py     # LLM provider settings
├── db/
│   ├── database.py             # Neon Postgres helpers (psycopg pool)
│   └── rag.py                  # Pinecone vector store
├── src/
│   ├── main.tsx                # Entry point with React Router
│   ├── App.tsx                 # Main itinerary planner UI
│   ├── App.css                 # Component styles
│   ├── index.css               # Global styles & CSS variables
│   ├── utils/
│   │   └── auth.ts             # Session store + /api/auth client
│   └── pages/
│       ├── Landing.tsx         # Landing page (/)
│       ├── AskPlace.tsx        # Ask XPLORA feature (/app → Ask tab)
│       ├── Login.tsx           # Sign in (/login)
│       ├── Signup.tsx          # Create account (/signup)
│       └── Dashboard.tsx       # Saved itineraries (/dashboard)
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
- **A Neon account** (free tier is enough) — [console.neon.tech](https://console.neon.tech)
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

# Database — Neon (serverless Postgres). Use the POOLED endpoint.
DATABASE_URL="postgresql://user:pass@ep-xxx-pooler.ap-southeast-1.aws.neon.tech/travel_agent?sslmode=require"

# Accounts / sessions
JWT_SECRET="long-random-string"          # openssl rand -hex 32
CORS_ORIGINS="http://localhost:3000"     # comma-separated frontend origins
AUTH_COOKIE_SECURE="false"               # true once served over HTTPS

# Optional — fallback LLM when Groq is at capacity
OPENROUTER_API_KEY="your_openrouter_api_key_here"

# Optional — real-time weather
TOMORROW_IO_API_KEY="your_tomorrow_io_key_here"

# Optional — RAG knowledge base
PINECONE_API_KEY="your_pinecone_api_key_here"
PINECONE_INDEX_NAME="travel-guides"
```

### 3. Set Up the Database

Create a free project at [console.neon.tech](https://console.neon.tech), copy its pooled
connection string into `DATABASE_URL`, and the app creates the `users`, `itineraries`,
`trip_ratings` and `place_reviews_cache` tables automatically on first start.

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

### Accounts & Saved Trips

- `POST /api/auth/signup` and `POST /api/auth/login` hash the password with bcrypt and return
  an HTTP-only `xplora_session` JWT cookie — the token is never exposed to JavaScript.
- Sign-in failures use one generic message, run a dummy hash for unknown emails (no timing
  oracle) and are rate-limited per email.
- `/api/generate-itinerary` stores the plan with the caller's `user_id` when a session cookie
  is present, so `/api/auth/itineraries` returns only your own trips (the heavy itinerary JSON is
  left out of the list and fetched per trip when you reopen one).
- `/dashboard` is the protected route; `/app` still works signed out, it just doesn't archive.

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
