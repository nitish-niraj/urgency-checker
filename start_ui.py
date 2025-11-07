"""
═══════════════════════════════════════════════════════════════
  Civic Issue Urgency Classifier - Easy Startup Script
  iOS 26 Liquid Design UI
═══════════════════════════════════════════════════════════════
"""

import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def print_banner():
    """Print startup banner"""
    print("\n" + "="*70)
    print("🏛️  CIVIC ISSUE URGENCY CLASSIFIER")
    print("="*70)
    print("   AI-Powered Government Response Prioritization System")
    print("   iOS 26 Liquid Design Interface")
    print("="*70 + "\n")

def check_dependencies():
    """Check if required packages are installed"""
    print("📦 Checking dependencies...")
    
    required_packages = [
        'fastapi',
        'uvicorn',
        'pydantic',
        'jinja2'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"   ❌ {package} (missing)")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("📥 Installing missing packages...")
        
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', 
                *missing_packages
            ])
            print("✅ All packages installed successfully!\n")
        except subprocess.CalledProcessError:
            print("❌ Failed to install packages. Please install manually:")
            print(f"   pip install {' '.join(missing_packages)}")
            sys.exit(1)
    else:
        print("✅ All dependencies are installed!\n")

def check_structure():
    """Check if required directories and files exist"""
    print("📁 Checking project structure...")
    
    base_dir = Path(__file__).parent
    required_paths = [
        base_dir / "src" / "demo_api_browser.py",
        base_dir / "static" / "css" / "styles.css",
        base_dir / "static" / "js" / "main.js",
        base_dir / "templates" / "index.html"
    ]
    
    all_exist = True
    for path in required_paths:
        if path.exists():
            print(f"   ✅ {path.relative_to(base_dir)}")
        else:
            print(f"   ❌ {path.relative_to(base_dir)} (missing)")
            all_exist = False
    
    if not all_exist:
        print("\n⚠️  Some required files are missing!")
        print("   Please ensure all project files are in place.")
        sys.exit(1)
    
    print("✅ Project structure is complete!\n")

def start_server():
    """Start the FastAPI server"""
    print("🚀 Starting application server...")
    print("="*70)
    print()
    print("   🌐 Web Interface:  http://localhost:8001")
    print("   📖 API Docs:       http://localhost:8001/docs")
    print("   💚 Health Check:   http://localhost:8001/health")
    print("   📊 Statistics:     http://localhost:8001/stats")
    print()
    print("="*70)
    print()
    print("✨ Features:")
    print("   • iOS 26 Liquid Design UI")
    print("   • Real-time AI Classification")
    print("   • Interactive Form with Animations")
    print("   • Glassmorphism Effects")
    print("   • Responsive Mobile Design")
    print()
    print("="*70)
    print()
    print("⏳ Starting server in 3 seconds...")
    print("   (Opening browser automatically...)")
    print()
    print("💡 Press Ctrl+C to stop the server")
    print("="*70 + "\n")
    
    # Wait and open browser
    time.sleep(3)
    
    try:
        webbrowser.open('http://localhost:8001')
    except:
        pass
    
    # Start server
    try:
        subprocess.run([
            sys.executable, 
            '-m', 'uvicorn',
            'src.demo_api_browser:app',
            '--host', '0.0.0.0',
            '--port', '8001',
            '--reload'
        ])
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
        print("👋 Thank you for using Civic Issue Urgency Classifier!")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        print("\n💡 Try running manually:")
        print("   python -m uvicorn src.demo_api_browser:app --host 0.0.0.0 --port 8001")
        sys.exit(1)

def main():
    """Main startup function"""
    print_banner()
    check_dependencies()
    check_structure()
    start_server()

if __name__ == "__main__":
    main()
