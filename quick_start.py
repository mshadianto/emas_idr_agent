# quick_start.py - Quick Setup Script for Premium Gold Analysis
import os
import sys
import subprocess
import platform

def print_header():
    print("🏆" * 50)
    print("    RAG AGENTIC AI - GOLD ANALYSIS PRO")
    print("         with OpenRouter Qwen3")
    print("🏆" * 50)
    print()

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher required")
        print(f"   Current version: {version.major}.{version.minor}")
        print("   Please upgrade Python")
        return False
    
    print(f"✅ Python {version.major}.{version.minor} - Compatible")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\n🔧 Installing dependencies...")
    
    requirements = [
        "streamlit==1.28.1",
        "pandas==2.1.3", 
        "plotly==5.17.0",
        "requests==2.31.0",
        "numpy==1.24.3",
        "scikit-learn==1.3.2",
        "yfinance==0.2.18",
        "python-dateutil==2.8.2",
        "beautifulsoup4==4.12.2",
        "lxml==4.9.3",
        "openai==1.12.0",
        "aiohttp==3.9.1"
    ]
    
    try:
        for package in requirements:
            print(f"   Installing {package.split('==')[0]}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package], 
                                stdout=subprocess.DEVNULL, 
                                stderr=subprocess.DEVNULL)
        
        print("✅ All dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        print("   Try running: pip install -r requirements.txt")
        return False

def setup_api_guide():
    """Display API setup guide"""
    print("\n🔑 API KEYS SETUP GUIDE")
    print("=" * 40)
    print("For premium features, get these FREE API keys:")
    print()
    
    apis = [
        {
            "name": "🥇 Alpha Vantage",
            "url": "https://www.alphavantage.co/support/#api-key",
            "benefit": "Professional gold & currency data (500 calls/day FREE)"
        },
        {
            "name": "🧠 OpenRouter", 
            "url": "https://openrouter.ai/",
            "benefit": "Qwen3 AI analysis (~$0.01 per query)"
        },
        {
            "name": "💰 Fixer.io",
            "url": "https://fixer.io/signup/free", 
            "benefit": "Bank-grade exchange rates (100 calls/month FREE)"
        },
        {
            "name": "📊 Finnhub",
            "url": "https://finnhub.io/register",
            "benefit": "Real-time market data (60 calls/minute FREE)"
        }
    ]
    
    for api in apis:
        print(f"{api['name']}")
        print(f"   URL: {api['url']}")
        print(f"   Benefit: {api['benefit']}")
        print()
    
    print("💡 TIP: You can run the app without API keys!")
    print("   Premium features will be available after adding keys in the UI.")

def create_sample_config():
    """Create sample configuration file"""
    config_content = '''# Sample API Configuration
# Copy your API keys here (optional)

# Alpha Vantage - Professional financial data
ALPHA_VANTAGE_KEY = "your_alpha_vantage_key_here"

# OpenRouter - For Qwen3 AI analysis  
OPENROUTER_API_KEY = "your_openrouter_key_here"

# Fixer.io - Bank-grade exchange rates
FIXER_API_KEY = "your_fixer_key_here"

# Finnhub - Real-time market data
FINNHUB_API_KEY = "your_finnhub_key_here"

# Note: You can also enter these keys directly in the app UI
# The app will work without these keys using free data sources
'''
    
    try:
        with open('config_sample.py', 'w') as f:
            f.write(config_content)
        print("✅ Created config_sample.py for API keys")
    except Exception as e:
        print(f"⚠️ Could not create config file: {e}")

def check_streamlit():
    """Check if Streamlit is working"""
    try:
        result = subprocess.run([sys.executable, "-c", "import streamlit"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Streamlit is ready")
            return True
        else:
            print("❌ Streamlit import failed")
            return False
    except Exception as e:
        print(f"❌ Error checking Streamlit: {e}")
        return False

def launch_app():
    """Launch the Streamlit application"""
    print("\n🚀 LAUNCHING APPLICATION")
    print("=" * 30)
    
    if not os.path.exists('app.py'):
        print("❌ app.py not found in current directory")
        print("   Make sure you're in the correct folder")
        return False
    
    print("✅ Found app.py")
    print("🌐 Starting Streamlit server...")
    print("\n" + "="*50)
    print("🏆 GOLD ANALYSIS PRO STARTING...")
    print("📊 Premium APIs: Alpha Vantage, OpenRouter, Fixer.io, Finnhub")
    print("🧠 AI Engine: Qwen3 via OpenRouter")  
    print("💻 Open your browser and go to: http://localhost:8501")
    print("🔑 Add your API keys in the sidebar for premium features")
    print("="*50)
    
    try:
        # Launch Streamlit
        os.system("streamlit run app.py")
        return True
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
        return True
    except Exception as e:
        print(f"❌ Error launching app: {e}")
        return False

def main():
    """Main setup function"""
    print_header()
    
    # Check Python version
    if not check_python_version():
        return
    
    # Install dependencies
    print("\n📦 Checking dependencies...")
    if not install_dependencies():
        print("\n❌ Dependency installation failed")
        print("   Please run manually: pip install -r requirements.txt")
        return
    
    # Check Streamlit
    if not check_streamlit():
        print("   Please install Streamlit manually: pip install streamlit")
        return
    
    # Setup guide
    setup_api_guide()
    
    # Create sample config
    create_sample_config()
    
    # Ready to launch
    print("\n🎯 SETUP COMPLETE!")
    print("=" * 20)
    print("✅ Python version compatible")
    print("✅ Dependencies installed") 
    print("✅ Streamlit ready")
    print("✅ Configuration template created")
    print()
    
    # Ask user if they want to launch
    try:
        choice = input("🚀 Launch the application now? (y/n): ").lower().strip()
        if choice in ['y', 'yes', '']:
            launch_app()
        else:
            print("\n💡 To launch later, run: streamlit run app.py")
            print("📖 Check the setup guide for API configuration")
    except KeyboardInterrupt:
        print("\n👋 Setup completed. Run 'streamlit run app.py' to start!")

if __name__ == "__main__":
    main()