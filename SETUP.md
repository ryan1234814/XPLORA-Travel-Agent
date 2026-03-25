# Setup & Run Guide 🚀

This guide explains how to start the XPLORA Travel Agent locally. The project consists of a **FastAPI backend** (Python) and a **React frontend** (Node/Vite).

## 🛠 Prerequisites

- Python 3.9+
- Node.js 18+ and npm
- [Groq API Key](https://console.groq.com/keys)
- (Optional) OpenWeather API Key for real-time weather

## ⚙️ Environment Configuration

Create a `.env` file in the root directory (`XPLORA-Travel-Agent/`) with your API keys:

```env
# Main LLM Configuration (Required)
GROQ_API_KEY="your_groq_api_key_here"

# Real-time Web Search / Weather (Optional but recommended)
OPENWEATHER_API_KEY="your_openweather_api_key_here"

# Fallback LLM via OpenRouter (Optional)
OPENROUTER_API_KEY="your_openrouter_api_key_here"
```

## 📦 Installation

### 1. Backend Dependencies
From the root directory, install Python packages:
```bash
python3 -m pip install -r requirements.txt
```

### 2. Frontend Dependencies
Navigate to the client folder and install npm packages:
```bash
cd frontend/client
npm install
```

---

## 🏃 Running the Application

For the best development experience, run the backend and frontend in two separate terminals.

### Terminal 1: Start the Backend (FastAPI)
```bash
# From the root directory XPLORA-Travel-Agent
cd frontend
python3 -m uvicorn api:app --reload
```
*The API will start running on `http://localhost:8000`*

### Terminal 2: Start the Frontend (Vite/React)
```bash
# From the root directory XPLORA-Travel-Agent
cd frontend/client
npm run dev
```
*The Velura Concierge UI will start, typically on `http://localhost:3000` or `http://localhost:5173`.*

---

## 🤖 Modifying LLM Provider
By default, the project is configured to use **Groq** for extreme speed and accuracy.
If you need to change this, edit `config/langgraph_config.py` and change the `LLM_PROVIDER` variable (e.g., to `"openrouter"` or `"ollama"`).
