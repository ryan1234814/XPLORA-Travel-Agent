# Setup & Run Guide 🚀

This guide explains how to set up and run the **XPLORA Travel Agent** locally. The project consists of a **FastAPI backend** (Python), a **React frontend** (Node/Vite), and a **Neon (serverless Postgres)** database for accounts, saved itineraries, ratings and the place-review cache.

> **Web search is powered by DuckDuckGo — no API keys required for search functionality.**

---

## 📁 Project Structure

```
XPLORA-Travel-Agent/
├── backend/
│   ├── api.py              # FastAPI application & routes
│   └── auth.py             # Password hashing + JWT session cookie
├── agents/
│   ├── agents.py            # LangGraph multi-agent orchestration
│   └── tools/
│       └── travel.py        # DuckDuckGo search tools
├── config/
│   ├── api_config.py        # API keys & external service config
│   └── langgraph_config.py  # LLM provider & agent settings
├── db/
│   ├── database.py          # Neon Postgres helpers & schema
│   └── rag.py               # Pinecone vector store (optional RAG)
├── src/                     # React frontend source
│   ├── App.tsx              # Main application component
│   ├── App.css              # Component styles
│   ├── index.css            # Global styles & CSS variables
│   ├── utils/auth.ts        # Frontend session store
│   └── pages/               # Landing, Login, Signup, Dashboard, AskPlace
├── data/                    # Travel knowledge base (blog content)
├── .env                     # Environment variables (create this)
├── requirements.txt         # Python dependencies
├── package.json             # Node.js dependencies
├── tailwind.config.js       # Tailwind CSS configuration
├── run_app.py               # Script to start both services
└── render.yaml              # Render.com deployment config
```

---

## 🛠 Prerequisites

- **Python 3.9+** (tested with 3.14)
- **Node.js 20+** and npm
- **A Neon account** — free tier is enough: <https://console.neon.tech>
- **A Groq API Key** — [Get one free here](https://console.groq.com/keys)

### Optional (for enhanced features)

| Service | Purpose | Required? |
|---------|---------|-----------|
| [OpenRouter API Key](https://openrouter.ai/keys) | Fallback LLM when Groq is at capacity | Highly recommended |
| [Tomorrow.io API Key](https://www.tomorrow.io/weather-api/) | Real-time weather data | Optional |
| [Pinecone API Key](https://www.pinecone.io/) | RAG knowledge base for travel blogs | Optional |

---

## ⚙️ Environment Configuration

Create a `.env` file in the project root directory:

```env
# ─── Database Configuration (Required) ───
# Neon -> Project -> Connection Details -> *pooled* endpoint
DATABASE_URL="postgresql://user:pass@ep-xxx-pooler.ap-southeast-1.aws.neon.tech/travel_agent?sslmode=require"

# ─── Accounts / Sessions (Required for sign-up & dashboard) ───
JWT_SECRET="run: openssl rand -hex 32"
CORS_ORIGINS="http://localhost:3000"   # comma-separated frontend origins
AUTH_COOKIE_SECURE="false"             # set true once served over HTTPS

# ─── Main LLM Configuration (Required) ───
# Choose one provider and set its key:
GROQ_API_KEY="your_groq_api_key_here"

# ─── Fallback LLM (Recommended) ───
# Used automatically if Groq hits rate limits
OPENROUTER_API_KEY="your_openrouter_api_key_here"

# ─── Weather Data (Optional) ───
# Provides real-time weather; falls back to free Open-Meteo API without it
TOMORROW_IO_API_KEY="your_tomorrow_io_key_here"

# ─── RAG Knowledge Base (Optional) ───
PINECONE_API_KEY="your_pinecone_api_key_here"
PINECONE_INDEX_NAME="travel-guides"
```

> **Minimum required:** `GROQ_API_KEY`, `DATABASE_URL` and `JWT_SECRET` are essential to run the app with accounts. All other keys unlock enhanced features.

### Creating the Neon database

Either from the console, or with the Neon CLI:

```bash
npm install -g neon
neon login
neon projects create --name xplora-travel --region-id aws-ap-southeast-1 --database travel_agent
neon connection-string --pooled     # paste this into DATABASE_URL
```

No manual SQL is needed: `users`, `itineraries`, `trip_ratings` and `place_reviews_cache` are
created by the backend on startup (`db/database.py`).

---

## 📦 Installation

### 1. Backend Dependencies (Python)

From the project root:

```bash
pip install -r requirements.txt
```

> If you encounter `pydantic_v1` errors, run:
> ```bash
> pip install --upgrade langchain langchain-core langchain-openai langchain-groq langgraph
> ```

### 2. Frontend Dependencies (Node.js)

From the project root:

```bash
npm install
```

### 3. Neon Database

Point `DATABASE_URL` in your `.env` at a Neon (serverless Postgres) project — see
[Creating the Neon database](#creating-the-neon-database) above.

No manual SQL is required: the `users`, `itineraries`, `trip_ratings` and
`place_reviews_cache` tables are created automatically on first backend startup.

---

## 🏃 Running the Application

### Option A: Start Both Services at Once (Recommended)

```bash
python3 run_app.py
```

This launches:
- **Frontend:** `http://localhost:3000`
- **Backend API:** `http://localhost:8000`

Press `Ctrl+C` to shut down both services.

### Option B: Start Services Individually

**Backend (FastAPI):**

```bash
python3 -m uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload
```

**Frontend (Vite dev server):**

```bash
npm run dev
```

### Option C: Production Build

```bash
# Build the frontend
npm run build

# Preview the production build
npm run preview
```

---

## 🤖 LLM Provider Configuration

The project supports three LLM providers. Edit `config/langgraph_config.py` to switch:

| Provider | Speed | Cost | Setup |
|----------|-------|------|-------|
| **Groq** (default) | ⚡ Fastest | Free tier | Set `GROQ_API_KEY` in `.env` |
| **OpenRouter** | Moderate | Free models available | Set `OPENROUTER_API_KEY` in `.env` |
| **Ollama** | Varies | Free (local) | Install [Ollama](https://ollama.com/) and pull a model |

Change the provider in `config/langgraph_config.py`:

```python
LLM_PROVIDER = "groq"      # or "openrouter" or "ollama"
```

---

## 🔍 How Search Works

All web search functionality uses **DuckDuckGo** via the `duckduckgo_search` library. No API keys are required for search.

The following search tools are used by the agents:

| Tool | Purpose |
|------|---------|
| `search_destination_info` | General destination guides & attractions |
| `search_weather_info` | Weather data (DuckDuckGo + Tomorrow.io fallback) |
| `search_hotels` | Hotel recommendations |
| `search_restaurants` | Restaurant suggestions |
| `search_attractions` | Points of interest |
| `search_local_tips` | Cultural tips & customs |
| `search_budget_info` | Cost & budget data |
| `search_local_transport_options` | Public transit info |
| `search_car_rentals` | Car rental options |
| `search_real_time_transit_info` | Flights, trains, routes |
| `search_travel_blogs` | Travel blog content (Pinecone RAG) |

---

## 🌐 Deployment

The canonical setup is **frontend on Vercel + backend on Render** — see `deploy_instructions.md` for the full walkthrough and post-deploy checklist.

### Render (Backend only)

The project includes a `render.yaml` blueprint that provisions **only** the `xplora-backend` Python web service (Python 3.12 via `PYTHON_VERSION`, health check at `/`). Set the required environment variables in the Render dashboard when the blueprint prompts for them.

### Vercel (Frontend)

The `vercel.json` configuration builds and hosts the React frontend. You **must** set `VITE_API_BASE_URL` in the Vercel project environment variables to the Render backend URL (e.g. `https://xplora-backend.onrender.com`) — it is baked in at build time, so redeploy after changing it. Without it, production API calls fail and the browser console logs an explicit error.

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'agents.tools.travel'` | Run `pip install -r requirements.txt` from the project root |
| `GROQ_API_KEY not found` | Add your Groq API key to the `.env` file |
| Neon/Postgres connection error | Ensure `DATABASE_URL` uses the **pooled** endpoint (`...-pooler...`) and `sslmode=require` |
| `npm run build` fails | Run `npm install` first, then `npm run build` |
| Port 8000 already in use | Kill the existing process: `lsof -ti:8000 | xargs kill -9` |
| Search returns no results | DuckDuckGo may be rate-limiting; wait a moment and retry |
