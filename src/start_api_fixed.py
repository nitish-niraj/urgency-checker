"""
Fixed API Server Startup Script
===============================

This script starts the API server with proper error handling for Windows systems.
"""

import subprocess
import sys
import os
import time
from pathlib import Path

def ensure_logs_directory():
    """Ensure logs directory exists"""
    logs_dir = Path("../logs")
    logs_dir.mkdir(exist_ok=True)
    print(f"✅ Logs directory ready: {logs_dir.absolute()}")

def start_api_server_fixed():
    """Start API server with proper Windows support"""
    
    print("🏛️ Starting Civic Issue Urgency Classifier API")
    print("=" * 50)
    
    # Ensure logs directory exists
    ensure_logs_directory()
    
    # Set environment variable for proper encoding
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    
    try:
        print("🚀 Starting FastAPI server...")
        print("📍 Server will be available at: http://localhost:8000")
        print("📖 API Documentation: http://localhost:8000/docs")
        print("🔍 Health Check: http://localhost:8000/health")
        print("📊 Statistics: http://localhost:8000/stats")
        print()
        print("⏳ Please wait while models are being loaded and trained...")
        print("   (This may take 30-60 seconds on first startup)")
        print()
        print("Press Ctrl+C to stop the server")
        print("-" * 50)
        
        # Start the server with proper encoding
        cmd = [
            sys.executable, "-m", "uvicorn", 
            "api_server:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload",
            "--log-level", "info"
        ]
        
        # Run with UTF-8 encoding
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            encoding='utf-8',
            errors='replace'
        )
        
        # Stream output
        try:
            for line in process.stdout:
                # Filter out problematic characters
                clean_line = line.encode('ascii', 'ignore').decode('ascii')
                print(clean_line, end='')
        except KeyboardInterrupt:
            print("\n👋 Stopping server...")
            process.terminate()
            process.wait()
            print("✅ Server stopped successfully")
            
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return False
    
    return True

if __name__ == "__main__":
    start_api_server_fixed()