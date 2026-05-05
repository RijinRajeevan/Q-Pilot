"""
Main execution script for Q-Pilot Live.
Orchestrates the FastAPI backend and Vite frontend servers.
"""
import os
import sys
import argparse
import subprocess
import time

def setup_environment():
    """Setup environment and directories"""
    directories = [
        'data',
        'models',
        'training',
        'evaluation',
        'results',
        'api'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Ensured directory exists: {directory}")

def run_backend():
    """Launch FastAPI Backend"""
    print("Starting Q-Pilot Live Backend (FastAPI)...")
    try:
        cmd = [sys.executable, "-m", "uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
        # Use Popen to run in background
        return subprocess.Popen(cmd)
    except Exception as e:
        print(f"Error launching backend: {e}")
        return None

def run_frontend():
    """Launch Vite Frontend"""
    print("Starting Q-Pilot Live Frontend (React/Vite)...")
    frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
    
    if not os.path.exists(frontend_dir):
        print(f"Frontend directory not found at {frontend_dir}. Please run 'npm create vite' setup.")
        return None
        
    try:
        # Use shell=True for npm commands on Windows
        return subprocess.Popen("npm run dev", cwd=frontend_dir, shell=True)
    except Exception as e:
        print(f"Error launching frontend: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Q-Pilot Live: Quantum Vehicle Trajectory Prediction System")
    parser.add_argument('--mode', choices=['backend', 'frontend', 'full'], default='full',
                        help='Execution mode: backend (FastAPI), frontend (Vite), or full. (Default: full)')
    args = parser.parse_args()

    setup_environment()

    processes = []
    
    if args.mode in ['backend', 'full']:
        backend_proc = run_backend()
        if backend_proc:
            processes.append(backend_proc)
            
    if args.mode in ['frontend', 'full']:
        if args.mode == 'full':
            print("Waiting for backend to initialize...")
            time.sleep(3)
        frontend_proc = run_frontend()
        if frontend_proc:
            processes.append(frontend_proc)

    try:
        # Keep main thread alive
        if processes:
            print("\nSystem running! Press Ctrl+C to terminate all services.")
            while True:
                time.sleep(1)
        else:
            print("No processes started.")
    except KeyboardInterrupt:
        print("\nStopping services...")
        for p in processes:
            p.terminate()
        print("Shutdown complete.")

if __name__ == "__main__":
    main()