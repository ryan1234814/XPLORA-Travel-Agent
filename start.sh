#!/bin/bash
# XPLORA - Start both backend and frontend servers
cd "$(dirname "$0")"

echo "🚀 Starting XPLORA..."

# Kill any existing processes on these ports
lsof -ti:8000 | xargs kill -9 2>/dev/null
lsof -ti:3000 | xargs kill -9 2>/dev/null
sleep 1

# Start backend
echo "📦 Starting backend on http://localhost:8000 ..."
python3 -m uvicorn backend.api:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

sleep 3

# Start frontend
echo "🎨 Starting frontend on http://localhost:3000 ..."
npx vite --host 0.0.0.0 --port 3000 &
FRONTEND_PID=$!

sleep 3

echo ""
echo "✅ Both servers are running!"
echo "   Frontend: http://localhost:3000"
echo "   Backend:  http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "   Press Ctrl+C to stop both servers."

# Open browser
open http://localhost:3000 2>/dev/null

# Wait for both processes
wait
