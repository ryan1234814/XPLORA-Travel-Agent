# Setup & Run Guide 🚀

This guide explains how to start the XPLORA Travel Agent locally. The project consists of a **FastAPI backend** (Python), a **React frontend** (Node/Vite), and **MySQL database** integration for storing generated itineraries.

## 📁 Project Structure

The project has been organized into specific directories for clarity:
- `backend/`: Contains the FastAPI application logic (`api.py`).
- `frontend/`: Contains the React UI (`client/`).
- `db/`: Contains the MySQL database configuration and models (`database.py`).
- `config/`: Configuration files for APIs and LLM behavior.
- `agents/`: LangGraph agents and external tools logic.

## 🛠 Prerequisites

- Python 3.9+
- Node.js 18+ and npm
- MySQL Server (running on default port 3306)
- [Groq API Key](https://console.groq.com/keys)
- (Optional) OpenWeather API Key for real-time weather
- (Optional) OpenRouter API Key as a fallback in case Groq is at capacity

## ⚙️ Environment Configuration

Create a `.env` file in the root directory (`XPLORA-Travel-Agent/`) with your API keys and database credentials:

```env
# Database Configuration (Required)
MYSQL_HOST="localhost"
MYSQL_PORT=3306
MYSQL_USER="root"
MYSQL_PASSWORD="newpassword"
MYSQL_DATABASE="travel_agent"

# Main LLM Configuration (Required)
GROQ_API_KEY="your_groq_api_key_here"

# Fallback LLM via OpenRouter (Highly Recommended)
# Automatically used if Groq hits rate limits or is at capacity
OPENROUTER_API_KEY="your_openrouter_api_key_here"

# Real-time Web Search / Weather (Optional but recommended)
OPENWEATHER_API_KEY="your_openweather_api_key_here"
```

## 📦 Installation

### 1. Backend Dependencies
From the root directory, install Python packages:
```bash
pip install -r requirements.txt
```
*(Note: If you run into `pydantic_v1` ModuleNotFoundErrors, run `pip install --upgrade langchain langchain-community langchain-openai langchain-groq langchain-text-splitters` to align the dependencies.)*

### 2. Frontend Dependencies
Navigate to the client folder and install npm packages:
```bash
cd frontend/client
npm install
```

---

## 🏃 Running the Application

The fastest and easiest way to start both the FastAPI backend and the React frontend simultaneously is using the `run_app.py` script from the root directory.

```bash
# From the root directory (XPLORA-Travel-Agent)
python3 run_app.py
```

This script will:
1. Automatically scaffold and launch the Vite/React frontend.
2. Initialize the FastAPI backend.
3. Keep track of both services in the same terminal window.

*The React UI will typically start on `http://localhost:3000` (or `3001` if busy).*
*The API will start running on `http://localhost:8000`.*

---

## 🤖 Modifying LLM Provider
By default, the project is configured to use **Groq** for extreme speed and accuracy, with **OpenRouter** enabled as an automatic fallback.
If you need to change the primary provider, edit `config/langgraph_config.py` and change the `LLM_PROVIDER` variable (e.g., to `"openrouter"` or `"ollama"`).
