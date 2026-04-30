import subprocess
import os
import sys
import time
import signal

def run_services():
    # Base path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    client_dir = os.path.join(base_dir, "frontend", "client")

    print("🚀 Starting XPLORA Premium Concierge...")

    # Use the Python binary with all dependencies
    python_path = sys.executable

    # 1. Start FastAPI Backend
    print("📡 Initializing Backend API (FastAPI)...")
    backend_proc = subprocess.Popen(
        [python_path, "-m", "uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=base_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )

    # 2. Start React Frontend
    print("✨ Scaffolding React Frontend (Vite)...")
    frontend_proc = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=client_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )

    print("\n✅ Services are initializing!")
    print("🔗 Frontend: http://localhost:3000")
    print("🔗 Backend:  http://localhost:8000")
    print("\nPress Ctrl+C to shut down.")

    # Function to print output from a process
    def print_output(proc, prefix):
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            print(f"[{prefix}] {line.strip()}")

    import threading
    threading.Thread(target=print_output, args=(backend_proc, "BACKEND"), daemon=True).start()
    threading.Thread(target=print_output, args=(frontend_proc, "FRONTEND"), daemon=True).start()

    try:
        while True:
            # Check if processes are still running
            if backend_proc.poll() is not None:
                print("\n❌ Backend process crashed.")
                break
            if frontend_proc.poll() is not None:
                print("\n❌ Frontend process crashed.")
                break
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down services...")
        backend_proc.terminate()
        frontend_proc.terminate()
        print("Done.")

if __name__ == "__main__":
    run_services()
