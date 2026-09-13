#!/usr/bin/env python3
"""
Chess Mate Vision - Unified Application Launcher
Starts both the FastAPI backend and React Vite frontend in a single terminal.
Handles graceful shutdown (Ctrl+C), pre-flight checks, and port cleanup.
"""

import os
import signal
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
FRONTEND_DIR = PROJECT_ROOT / "frontend"


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def kill_process_on_port(port: int):
    try:
        cmd = f"fuser -k -9 {port}/tcp"
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def check_preflight() -> Path:
    # 1. Determine Python executable
    python_bin = VENV_PYTHON if VENV_PYTHON.exists() else Path(sys.executable)
    print(f"🐍 Python Engine: {python_bin}")

    # 2. Check frontend dependencies
    node_modules = FRONTEND_DIR / "node_modules"
    if not node_modules.exists():
        print("📦 Installing frontend dependencies (npm install)...")
        subprocess.run("npm install", shell=True, cwd=str(FRONTEND_DIR), check=True)

    # 3. Clean up busy ports if an old session was left hanging
    for port in (8000, 5173):
        if is_port_in_use(port):
            print(f"⚠️  Port {port} is currently busy. Clearing port...")
            kill_process_on_port(port)
            time.sleep(0.5)

    return python_bin


def main():
    print("\n" + "=" * 62)
    print("        ♟️   CHESS MATE VISION — ALL-IN-ONE LAUNCHER   ♟️")
    print("=" * 62)

    python_bin = check_preflight()

    processes = []

    def cleanup(signum=None, frame=None):
        print("\n\n🛑 Shutting down Chess Mate Vision services...")
        for p in processes:
            try:
                if p.poll() is None:
                    p.terminate()
            except Exception:
                pass
        time.sleep(0.5)
        for p in processes:
            try:
                if p.poll() is None:
                    p.kill()
            except Exception:
                pass
        # Final port cleanup
        kill_process_on_port(8000)
        kill_process_on_port(5173)
        print("✅ All services stopped cleanly. Goodbye!\n")
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    # 1. Launch FastAPI Backend
    print("\n🚀 [1/2] Launching FastAPI Backend (Port 8000)...")
    backend_proc = subprocess.Popen(
        [str(python_bin), "backend/app.py"],
        cwd=str(PROJECT_ROOT),
    )
    processes.append(backend_proc)

    # Wait for backend to be listening on 8000
    backend_ready = False
    for _ in range(40):
        if is_port_in_use(8000):
            backend_ready = True
            break
        time.sleep(0.25)

    if backend_ready:
        print("   ✅ Backend is LIVE on http://127.0.0.1:8000")
    else:
        print("   ⚠️  Backend startup pending, proceeding to frontend...")

    # 2. Launch Vite Frontend
    print("🚀 [2/2] Launching React + Vite Visualizer (Port 5173)...")
    frontend_proc = subprocess.Popen(
        ["npm", "run", "dev", "--", "--host"],
        cwd=str(FRONTEND_DIR),
    )
    processes.append(frontend_proc)

    # Wait for frontend to be listening on 5173
    for _ in range(40):
        if is_port_in_use(5173):
            break
        time.sleep(0.25)

    print("\n" + "=" * 62)
    print("  🎉 CHESS MATE VISION IS RUNNING!")
    print("  🌐 App Visualizer : http://localhost:5173")
    print("  ⚡ Backend API    : http://localhost:8000")
    print("  📖 API Docs       : http://localhost:8000/docs")
    print("=" * 62)
    print("  👉 Press [Ctrl+C] at any time in this terminal to quit.\n")

    # Try opening default browser
    try:
        time.sleep(0.5)
        webbrowser.open("http://localhost:5173")
    except Exception:
        pass

    # Monitor subprocesses
    while True:
        for p in processes:
            if p.poll() is not None:
                print(f"\n⚠️ Service exited unexpectedly (return code: {p.returncode})")
                cleanup()
        time.sleep(1)


if __name__ == "__main__":
    main()
