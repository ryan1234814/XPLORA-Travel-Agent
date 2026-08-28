# Setup & Run Guide 🚀

This guide explains how to set up and run the **XPLORA Travel Agent** locally. The project consists of a **FastAPI backend** (Python), a **React frontend** (Node/Vite), and **MySQL database** integration for storing generated itineraries.

> **Web search is powered by DuckDuckGo — no API keys required for search functionality.**

---

## 📁 Project Structure

```
XPLORA-Travel-Agent/
├── backend/
│   └── api.py              # FastAPI application & routes
├── agents/
│   ├── agents.py            # LangGraph multi-agent orchestration
│   └── tools/
│       └── travel.py        # DuckDuckGo search tools
├── config/
│   ├── api_config.py        # API keys & external service config
│   └── langgraph_config.py  # LLM provider & agent settings
├── db/
│   └── database.py          # MySQL database models & helpers
├── src/                     # React frontend source
│   ├── App.tsx              # Main application component
│   ├── App.css              # Component styles
│   └── index.css            # Global styles & CSS variables
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
- **MySQL Server** (running on default port 3306)
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
MYSQL_HOST="localhost"
MYSQL_PORT=3306
MYSQL_USER="root"
MYSQL_PASSWORD="your_password"
MYSQL_DATABASE="travel_agent"

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

> **Minimum required:** Only `GROQ_API_KEY` and the MySQL credentials are essential to run the app. All other keys unlock enhanced features.

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

### 3. MySQL Database

Create the database (if it doesn't exist):

```sql
CREATE DATABASE IF NOT EXISTS travel_agent;
```

The app will automatically create the required tables on first run.

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

### Render.com

The project includes a `render.yaml` for one-click deployment. Set the required environment variables in the Render dashboard.

### Vercel (Frontend only)

The `vercel.json` configuration is included for deploying the built React frontend separately.

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'agents.tools.travel'` | Run `pip install -r requirements.txt` from the project root |
| `GROQ_API_KEY not found` | Add your Groq API key to the `.env` file |
| MySQL connection error | Ensure MySQL is running and credentials in `.env` are correct |
| `npm run build` fails | Run `npm install` first, then `npm run build` |
| Port 8000 already in use | Kill the existing process: `lsof -ti:8000 | xargs kill -9` |
| Search returns no results | DuckDuckGo may be rate-limiting; wait a moment and retry |
