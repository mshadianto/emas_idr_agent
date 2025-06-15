# app.py - Enhanced RAG Agentic AI with Premium APIs
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
import json
from typing import Dict, List, Any
import yfinance as yf
from dataclasses import dataclass
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
import asyncio
import aiohttp
from openai import OpenAI

# App Configuration
APP_VERSION = "2.1.0"
APP_CREATOR = "MS Hadianto"
APP_NAME = "RAG Agentic AI - Gold Analysis Pro"

# Configuration
st.set_page_config(
    page_title="RAG Agentic AI - Gold Analysis Pro by MS Hadianto",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced API Configuration with your premium APIs
API_CONFIG = {
    'ALPHA_VANTAGE_KEY': '',  # Your Alpha Vantage key
    'OPENROUTER_API_KEY': '',  # Your OpenRouter key  
    'FIXER_KEY': '',  # Your Fixer.io key
    'FINNHUB_KEY': '',  # Your Finnhub key
    'QWEN3_MODEL': 'qwen/qwen-2.5-72b-instruct',  # Qwen3 model via OpenRouter
}

@dataclass
class MarketData:
    """Data class for market data points"""
    timestamp: datetime
    gold_price_usd: float
    usd_idr_rate: float
    gold_price_idr: float

# Initialize OpenRouter client for Qwen3
def get_openrouter_client():
    """Initialize OpenRouter client for Qwen3 AI"""
    if API_CONFIG['OPENROUTER_API_KEY']:
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=API_CONFIG['OPENROUTER_API_KEY']
        )
    return None

def get_app_info():
    """Get dynamic application information"""
    return {
        'version': APP_VERSION,
        'creator': APP_CREATOR,
        'name': APP_NAME,
        'build_date': datetime.now().strftime('%Y-%m-%d'),
        'session_id': datetime.now().strftime('%Y%m%d_%H%M%S')
    }

def validate_dependencies():
    """Validate that all required dependencies are available"""
    try:
        # Test core dependencies
        import streamlit as st_test
        import pandas as pd_test
        import numpy as np_test
        import plotly.graph_objects as go_test
        import yfinance as yf_test
        from sklearn.feature_extraction.text import TfidfVectorizer as tfidf_test
        
        # Test MarketData class
        test_data = MarketData(
            timestamp=datetime.now(),
            gold_price_usd=2000.0,
            usd_idr_rate=15000.0,
            gold_price_idr=30000000.0
        )
        
        return True, "All dependencies validated successfully"
    except Exception as e:
        return False, f"Dependency validation failed: {e}"

def render_footer():
    """Render application footer with disclaimer and info"""
    st.markdown("---")
    
    # Footer content in columns
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.markdown(f"""
        ### 🏆 {APP_NAME}
        **Version:** {APP_VERSION}  
        **Created by:** {APP_CREATOR}  
        **Updated:** {datetime.now().strftime('%Y-%m-%d')}
        """)
    
    with col2:
        st.markdown("""
        ### ⚠️ Disclaimer
        **Investment Risk Notice:** This application provides analytical tools and information for educational purposes only. All investment decisions carry inherent risks, and past performance does not guarantee future results.
        
        **Data Sources:** Real-time data depends on API availability. Always verify critical information from official sources before making investment decisions.
        """)
    
    with col3:
        st.markdown("""
        ### 🔗 Quick Links
        - [Alpha Vantage](https://www.alphavantage.co/)
        - [OpenRouter](https://openrouter.ai/)
        - [Fixer.io](https://fixer.io/)
        - [Finnhub](https://finnhub.io/)
        """)
    
    # Technical info
    st.markdown("---")
    st.caption(f"""
    🔧 **Technical Info:** Running Streamlit {st.__version__} | 
    📊 **Data Sources:** Alpha Vantage, OpenRouter Qwen3, Fixer.io, Finnhub, Yahoo Finance | 
    🧠 **AI Engine:** RAG (Retrieval Augmented Generation) with Multi-Agent Architecture | 
    💾 **Cache Status:** {len(st.session_state) if hasattr(st, 'session_state') else 0} items | 
    ⏰ **Session:** {datetime.now().strftime('%H:%M:%S')} | 
    👨‍💻 **Developer:** {APP_CREATOR}
    """)

@st.cache_resource
def initialize_agents():
    """Initialize all agents for the application"""
    try:
        # Initialize each agent step by step with debugging
        
        # Step 1: Initialize retriever
        retriever = DataRetriever()
        
        # Step 2: Initialize gold agent
        gold_agent = GoldPriceAgent(retriever)
        
        # Step 3: Initialize currency agent
        currency_agent = CurrencyAgent(retriever)
        
        # Step 4: Initialize correlation agent
        correlation_agent = CorrelationAgent(retriever)
        
        # Step 5: Initialize economic agent
        economic_agent = EconomicDataAgent(retriever)
        
        # Step 6: Initialize AI agent
        ai_agent = AIAnalysisAgent(retriever)
        
        # Debug: Create tuple explicitly
        result_tuple = (retriever, gold_agent, currency_agent, correlation_agent, economic_agent, ai_agent)
        
        # Verify we have exactly 6 items
        if len(result_tuple) != 6:
            raise ValueError(f"Expected 6 agents, got {len(result_tuple)}")
        
        # Return exactly 6 agents
        return result_tuple
        
    except Exception as e:
        st.error(f"❌ Error initializing agents: {e}")
        st.error(f"Error type: {type(e).__name__}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        
        # Return 6 None values to prevent unpacking error
        return (None, None, None, None, None, None)

def create_charts(historical_data: List[MarketData]):
    """Create interactive charts"""
    df = pd.DataFrame([
        {
            'Date': data.timestamp,
            'Gold Price (USD)': data.gold_price_usd,
            'USD/IDR Rate': data.usd_idr_rate,
            'Gold Price (IDR)': data.gold_price_idr
        }
        for data in historical_data
    ])
    
    # Gold Price Chart
    fig_gold = go.Figure()
    fig_gold.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Gold Price (USD)'],
        mode='lines+markers',
        name='Gold Price (USD)',
        line=dict(color='gold', width=2)
    ))
    fig_gold.update_layout(
        title='Gold Price Trend (USD)',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        template='plotly_dark'
    )
    
    # USD/IDR Rate Chart
    fig_rate = go.Figure()
    fig_rate.add_trace(go.Scatter(
        x=df['Date'],
        y=df['USD/IDR Rate'],
        mode='lines+markers',
        name='USD/IDR Rate',
        line=dict(color='blue', width=2)
    ))
    fig_rate.update_layout(
        title='USD/IDR Exchange Rate Trend',
        xaxis_title='Date',
        yaxis_title='Rate (IDR)',
        template='plotly_dark'
    )
    
    # Gold Price in IDR Chart
    fig_gold_idr = go.Figure()
    fig_gold_idr.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Gold Price (IDR)'],
        mode='lines+markers',
        name='Gold Price (IDR)',
        line=dict(color='orange', width=2)
    ))
    fig_gold_idr.update_layout(
        title='Gold Price Trend (IDR)',
        xaxis_title='Date',
        yaxis_title='Price (IDR)',
        template='plotly_dark'
    )
    
    return fig_gold, fig_rate, fig_gold_idr

def generate_historical_data(days: int = 30) -> List[MarketData]:
    """Generate historical data using real APIs with fallback to simulation"""
    data = []
    
    try:
        # Try to get real historical data from Yahoo Finance
        gold_ticker = yf.Ticker("GC=F")
        usd_idr_ticker = yf.Ticker("USDIDR=X")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        gold_hist = gold_ticker.history(start=start_date, end=end_date)
        usd_idr_hist = usd_idr_ticker.history(start=start_date, end=end_date)
        
        # Use real data if available
        if not gold_hist.empty and not usd_idr_hist.empty:
            # Find common dates
            common_dates = gold_hist.index.intersection(usd_idr_hist.index)
            
            for date in common_dates:
                if date in gold_hist.index and date in usd_idr_hist.index:
                    gold_price = float(gold_hist.loc[date]['Close'])
                    usd_idr_rate = float(usd_idr_hist.loc[date]['Close'])
                    gold_price_idr = gold_price * usd_idr_rate
                    
                    data.append(MarketData(
                        timestamp=date.to_pydatetime(),
                        gold_price_usd=gold_price,
                        usd_idr_rate=usd_idr_rate,
                        gold_price_idr=gold_price_idr
                    ))
            
            # If we have enough real data, return it
            if len(data) >= days // 2:
                return sorted(data, key=lambda x: x.timestamp)
    
    except Exception as e:
        pass  # Fall back to simulation
    
    # Fallback to simulated data with realistic patterns
    return generate_simulated_data(days)

def generate_simulated_data(days: int) -> List[MarketData]:
    """Generate realistic simulated data"""
    data = []
    base_gold_price = 2000
    base_usd_idr = 15000
    
    for i in range(days):
        date = datetime.now() - timedelta(days=days-i)
        
        # Simulate price movements with some correlation
        gold_change = np.random.normal(0, 30)
        currency_change = np.random.normal(0, 200)
        
        # Add some correlation (negative correlation between gold and USD strength)
        if gold_change > 0:
            currency_change += np.random.normal(50, 100)  # USD weakens when gold rises
        
        gold_price = base_gold_price + gold_change
        usd_idr_rate = base_usd_idr + currency_change
        gold_price_idr = gold_price * usd_idr_rate
        
        data.append(MarketData(
            timestamp=date,
            gold_price_usd=max(1000, gold_price),  # Ensure reasonable minimum
            usd_idr_rate=max(10000, usd_idr_rate),  # Ensure reasonable minimum
            gold_price_idr=gold_price_idr
        ))
        
        # Update base prices for next iteration
        base_gold_price = max(1000, gold_price)
        base_usd_idr = max(10000, usd_idr_rate)
    
# Alternative simplified initialization for critical error recovery
def safe_initialize_agents():
    """Safe initialization with minimal dependencies"""
    try:
        # Create minimal working agents
        retriever = DataRetriever()
        
        # Create minimal agent classes if original classes fail
        class MinimalGoldAgent:
            def __init__(self, retriever):
                self.retriever = retriever
            def get_gold_price(self):
                return {'price_usd': 2000.0, 'timestamp': datetime.now(), 'source': 'fallback'}
            def get_real_time_gold_data(self):
                return self.get_gold_price()
            def analyze_gold_trends(self, data):
                return {'trend': 'stable', 'volatility': 10.0, 'avg_change': 0.0, 'current_price': 2000.0, 'price_range': {'min': 1900, 'max': 2100}}
            def get_historical_gold_data(self, days):
                return []
            def get_intraday_gold_data(self):
                return []
                
        class MinimalCurrencyAgent:
            def __init__(self, retriever):
                self.retriever = retriever
            def get_usd_idr_rate(self):
                return {'rate': 15000.0, 'timestamp': datetime.now(), 'source': 'fallback'}
            def get_real_time_currency_data(self):
                return self.get_usd_idr_rate()
            def analyze_currency_trends(self, data):
                return {'trend': 'stable', 'volatility': 100.0, 'avg_change': 0.0, 'current_rate': 15000.0, 'rate_range': {'min': 14500, 'max': 15500}}
            def get_currency_historical_premium(self, days):
                return []
            def get_historical_currency_data(self, days):
                return []
                
        class MinimalCorrelationAgent:
            def __init__(self, retriever):
                self.retriever = retriever
            def analyze_correlation(self, data):
                return {
                    'gold_usd_vs_usdidr_correlation': 0.0, 
                    'gold_idr_vs_usdidr_correlation': 0.0,
                    'relationship_strength': 0.0,
                    'interpretation': 'no correlation available in safe mode'
                }
            def generate_correlation_insights(self, analysis):
                return ["Safe mode: Limited correlation analysis available"]
                
        class MinimalEconomicAgent:
            def __init__(self, retriever):
                self.retriever = retriever
            def get_market_sentiment(self):
                return {
                    'fear_greed_index': 50, 
                    'market_sentiment': 'neutral',
                    'dxy_index': 102.5,
                    'dxy_change': 0.0,
                    'vix_index': 20.0,
                    'market_volatility': 'moderate',
                    'dollar_strength': 'neutral'
                }
            def get_commodity_correlations(self):
                return {'gold_silver': 0.8, 'gold_oil': 0.3}
                
        class MinimalAIAgent:
            def __init__(self, retriever):
                self.retriever = retriever
            def generate_insights(self, query, data):
                return f"**Safe Mode Analysis:** Basic analysis for: {query}\n\nNote: Advanced AI features require proper API configuration."
            def generate_market_report(self, data, economic_data):
                return "**Safe Mode Report:** Basic market report generated. Advanced AI analysis requires OpenRouter API key."
        
        # Try original classes first, fall back to minimal classes
        try:
            gold_agent = GoldPriceAgent(retriever)
        except:
            gold_agent = MinimalGoldAgent(retriever)
            
        try:
            currency_agent = CurrencyAgent(retriever)
        except:
            currency_agent = MinimalCurrencyAgent(retriever)
            
        try:
            correlation_agent = CorrelationAgent(retriever)
        except:
            correlation_agent = MinimalCorrelationAgent(retriever)
            
        try:
            economic_agent = EconomicDataAgent(retriever)
        except:
            economic_agent = MinimalEconomicAgent(retriever)
            
        try:
            ai_agent = AIAnalysisAgent(retriever)
        except:
            ai_agent = MinimalAIAgent(retriever)
        
        # Monkey patch missing methods as additional safety
        if not hasattr(gold_agent, 'analyze_gold_trends'):
            gold_agent.analyze_gold_trends = lambda data: {'trend': 'stable', 'volatility': 10.0, 'avg_change': 0.0, 'current_price': 2000.0, 'price_range': {'min': 1900, 'max': 2100}}
        
        if not hasattr(currency_agent, 'analyze_currency_trends'):
            currency_agent.analyze_currency_trends = lambda data: {'trend': 'stable', 'volatility': 100.0, 'avg_change': 0.0, 'current_rate': 15000.0, 'rate_range': {'min': 14500, 'max': 15500}}
        
        if not hasattr(correlation_agent, 'analyze_correlation'):
            correlation_agent.analyze_correlation = lambda data: {'gold_usd_vs_usdidr_correlation': 0.0, 'gold_idr_vs_usdidr_correlation': 0.0, 'relationship_strength': 0.0, 'interpretation': 'fallback correlation'}
        
        if not hasattr(economic_agent, 'get_market_sentiment'):
            economic_agent.get_market_sentiment = lambda: {'fear_greed_index': 50, 'market_sentiment': 'neutral', 'dxy_index': 102.5, 'dxy_change': 0.0, 'vix_index': 20.0, 'market_volatility': 'moderate', 'dollar_strength': 'neutral'}
        
        if not hasattr(ai_agent, 'generate_insights'):
            ai_agent.generate_insights = lambda query, data: f"**Safe Mode Analysis:** Basic analysis for: {query}"
        
        return retriever, gold_agent, currency_agent, correlation_agent, economic_agent, ai_agent
        
    except Exception as e:
        st.error(f"Even safe initialization failed: {e}")
        return None, None, None, None, None, None
def safe_initialize_agents():
    """Safe initialization with minimal dependencies"""
    try:
        # Create minimal working agents
        retriever = DataRetriever()
        
        # Create minimal agent classes if original classes fail
        class MinimalGoldAgent:
            def __init__(self, retriever):
                self.retriever = retriever
            def get_gold_price(self):
                return {'price_usd': 2000.0, 'timestamp': datetime.now(), 'source': 'fallback'}
            def get_real_time_gold_data(self):
                return self.get_gold_price()
            def analyze_gold_trends(self, data):
                return {'trend': 'stable', 'volatility': 10.0, 'avg_change': 0.0, 'current_price': 2000.0, 'price_range': {'min': 1900, 'max': 2100}}
            def get_historical_gold_data(self, days):
                return []
            def get_intraday_gold_data(self):
                return []
                
        class MinimalCurrencyAgent:
            def __init__(self, retriever):
                self.retriever = retriever
            def get_usd_idr_rate(self):
                return {'rate': 15000.0, 'timestamp': datetime.now(), 'source': 'fallback'}
            def get_real_time_currency_data(self):
                return self.get_usd_idr_rate()
            def analyze_currency_trends(self, data):
                return {'trend': 'stable', 'volatility': 100.0, 'avg_change': 0.0, 'current_rate': 15000.0, 'rate_range': {'min': 14500, 'max': 15500}}
            def get_currency_historical_premium(self, days):
                return []
            def get_historical_currency_data(self, days):
                return []
                
        class MinimalCorrelationAgent:
            def __init__(self, retriever):
                self.retriever = retriever
            def analyze_correlation(self, data):
                return {
                    'gold_usd_vs_usdidr_correlation': 0.0, 
                    'gold_idr_vs_usdidr_correlation': 0.0,
                    'relationship_strength': 0.0,
                    'interpretation': 'no correlation available in safe mode'
                }
            def generate_correlation_insights(self, analysis):
                return ["Safe mode: Limited correlation analysis available"]
                
        class MinimalEconomicAgent:
            def __init__(self, retriever):
                self.retriever = retriever
            def get_market_sentiment(self):
                return {
                    'fear_greed_index': 50, 
                    'market_sentiment': 'neutral',
                    'dxy_index': 102.5,
                    'dxy_change': 0.0,
                    'vix_index': 20.0,
                    'market_volatility': 'moderate',
                    'dollar_strength': 'neutral'
                }
            def get_commodity_correlations(self):
                return {'gold_silver': 0.8, 'gold_oil': 0.3}
                
        class MinimalAIAgent:
            def __init__(self, retriever):
                self.retriever = retriever
            def generate_insights(self, query, data):
                return f"**Safe Mode Analysis:** Basic analysis for: {query}\n\nNote: Advanced AI features require proper API configuration."
            def generate_market_report(self, data, economic_data):
                return "**Safe Mode Report:** Basic market report generated. Advanced AI analysis requires OpenRouter API key."
        
        # Try original classes first, fall back to minimal classes
        try:
            gold_agent = GoldPriceAgent(retriever)
        except:
            gold_agent = MinimalGoldAgent(retriever)
            
        try:
            currency_agent = CurrencyAgent(retriever)
        except:
            currency_agent = MinimalCurrencyAgent(retriever)
            
        try:
            correlation_agent = CorrelationAgent(retriever)
        except:
            correlation_agent = MinimalCorrelationAgent(retriever)
            
        try:
            economic_agent = EconomicDataAgent(retriever)
        except:
            economic_agent = MinimalEconomicAgent(retriever)
            
        try:
            ai_agent = AIAnalysisAgent(retriever)
        except:
            ai_agent = MinimalAIAgent(retriever)
        
        # Monkey patch missing methods as additional safety
        if not hasattr(gold_agent, 'analyze_gold_trends'):
            gold_agent.analyze_gold_trends = lambda data: {'trend': 'stable', 'volatility': 10.0, 'avg_change': 0.0, 'current_price': 2000.0, 'price_range': {'min': 1900, 'max': 2100}}
        
        if not hasattr(currency_agent, 'analyze_currency_trends'):
            currency_agent.analyze_currency_trends = lambda data: {'trend': 'stable', 'volatility': 100.0, 'avg_change': 0.0, 'current_rate': 15000.0, 'rate_range': {'min': 14500, 'max': 15500}}
        
        if not hasattr(correlation_agent, 'analyze_correlation'):
            correlation_agent.analyze_correlation = lambda data: {'gold_usd_vs_usdidr_correlation': 0.0, 'gold_idr_vs_usdidr_correlation': 0.0, 'relationship_strength': 0.0, 'interpretation': 'fallback correlation'}
        
        if not hasattr(economic_agent, 'get_market_sentiment'):
            economic_agent.get_market_sentiment = lambda: {'fear_greed_index': 50, 'market_sentiment': 'neutral', 'dxy_index': 102.5, 'dxy_change': 0.0, 'vix_index': 20.0, 'market_volatility': 'moderate', 'dollar_strength': 'neutral'}
        
        if not hasattr(ai_agent, 'generate_insights'):
            ai_agent.generate_insights = lambda query, data: f"**Safe Mode Analysis:** Basic analysis for: {query}"
        
        return retriever, gold_agent, currency_agent, correlation_agent, economic_agent, ai_agent
        
    except Exception as e:
        st.error(f"Even safe initialization failed: {e}")
        return None, None, None, None, None, None

@dataclass
class MarketData:
    timestamp: datetime
    gold_price_usd: float
    usd_idr_rate: float
    gold_price_idr: float
    
class DataRetriever:
    """Retrieval component for RAG system"""
    
    def __init__(self):
        self.knowledge_base = []
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.knowledge_vectors = None
        
    def add_to_knowledge_base(self, text: str, metadata: Dict):
        """Add information to knowledge base"""
        self.knowledge_base.append({
            'text': text,
            'metadata': metadata,
            'timestamp': datetime.now()
        })
        self._update_vectors()
    
    def _update_vectors(self):
        """Update TF-IDF vectors for knowledge base"""
        if self.knowledge_base:
            texts = [item['text'] for item in self.knowledge_base]
            self.knowledge_vectors = self.vectorizer.fit_transform(texts)
    
    def retrieve_relevant_info(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve most relevant information for a query"""
        if not self.knowledge_base or self.knowledge_vectors is None:
            return []
        
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.knowledge_vectors)[0]
        
        # Get top-k most similar documents
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        relevant_docs = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Threshold for relevance
                relevant_docs.append({
                    **self.knowledge_base[idx],
                    'similarity': similarities[idx]
                })
        
        return relevant_docs

class GoldPriceAgent:
    """Agent for gold price data and analysis"""
    
    def __init__(self, retriever: DataRetriever):
        self.retriever = retriever
        self.cache = {}
        
    def get_gold_price(self) -> Dict[str, Any]:
        """Fetch current gold price using multiple APIs"""
        methods = [
            self._get_gold_from_yahoo,
            self._get_gold_from_alpha_vantage, 
            self._get_gold_from_finnhub,
            self._get_gold_from_metals_api
        ]
        
        for method in methods:
            try:
                result = method()
                if result and result.get('price_usd'):
                    # Add to knowledge base
                    knowledge_text = f"Current gold price is ${result['price_usd']:.2f} per ounce from {result['source']} as of {datetime.now()}"
                    self.retriever.add_to_knowledge_base(
                        knowledge_text,
                        {'type': 'gold_price', 'price': result['price_usd'], 'currency': 'USD', 'source': result['source']}
                    )
                    return result
            except Exception as e:
                continue
        
        # Fallback to simulated data
        price = 2000 + np.random.normal(0, 50)
        return {
            'price_usd': price,
            'timestamp': datetime.now(),
            'source': 'simulated'
        }
    
    def _get_gold_from_yahoo(self) -> Dict[str, Any]:
        """Get gold price from Yahoo Finance"""
        gold_ticker = yf.Ticker("GC=F")  # Gold Futures
        hist = gold_ticker.history(period="1d")
        if not hist.empty:
            price = float(hist['Close'].iloc[-1])
            return {
                'price_usd': price,
                'timestamp': datetime.now(),
                'source': 'yahoo_finance'
            }
        return None
    
    def _get_gold_from_alpha_vantage(self) -> Dict[str, Any]:
        """Get gold price from Alpha Vantage - Premium API"""
        if not API_CONFIG['ALPHA_VANTAGE_KEY']:
            return None
        
        # Get real-time gold price (XAU/USD)
        url = f"https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency=XAU&to_currency=USD&apikey={API_CONFIG['ALPHA_VANTAGE_KEY']}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if "Realtime Currency Exchange Rate" in data:
                rate_data = data["Realtime Currency Exchange Rate"]
                price_per_ounce = float(rate_data["5. Exchange Rate"]) * 31.1035  # Convert to per ounce
                
                return {
                    'price_usd': price_per_ounce,
                    'timestamp': datetime.now(),
                    'source': 'alpha_vantage_premium',
                    'bid_price': float(rate_data.get("8. Bid Price", 0)) * 31.1035,
                    'ask_price': float(rate_data.get("9. Ask Price", 0)) * 31.1035,
                    'last_refreshed': rate_data.get("6. Last Refreshed")
                }
        return None
    
    def _get_gold_from_finnhub(self) -> Dict[str, Any]:
        """Get gold price from Finnhub - Premium API"""
        if not API_CONFIG['FINNHUB_KEY']:
            return None
            
        url = f"https://finnhub.io/api/v1/quote?symbol=OANDA:XAU_USD&token={API_CONFIG['FINNHUB_KEY']}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('c'):  # Current price
                return {
                    'price_usd': float(data['c']),
                    'timestamp': datetime.now(),
                    'source': 'finnhub_premium',
                    'open_price': float(data.get('o', 0)),
                    'high_price': float(data.get('h', 0)),
                    'low_price': float(data.get('l', 0)),
                    'previous_close': float(data.get('pc', 0)),
                    'change': float(data.get('d', 0)),
                    'change_percent': float(data.get('dp', 0))
                }
        return None
    
    def get_intraday_gold_data(self) -> List[Dict]:
        """Get intraday gold data from Alpha Vantage"""
        if not API_CONFIG['ALPHA_VANTAGE_KEY']:
            return []
        
        try:
            url = f"https://www.alphavantage.co/query?function=FX_INTRADAY&from_symbol=XAU&to_symbol=USD&interval=60min&apikey={API_CONFIG['ALPHA_VANTAGE_KEY']}"
            response = requests.get(url, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if "Time Series FX (60min)" in data:
                    time_series = data["Time Series FX (60min)"]
                    intraday_data = []
                    
                    for timestamp, values in list(time_series.items())[:24]:  # Last 24 hours
                        intraday_data.append({
                            'timestamp': datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S'),
                            'price': float(values['4. close']) * 31.1035,  # Convert to per ounce
                            'volume': int(values.get('5. volume', 0)),
                            'high': float(values['2. high']) * 31.1035,
                            'low': float(values['3. low']) * 31.1035
                        })
                    
                    return sorted(intraday_data, key=lambda x: x['timestamp'])
        except Exception as e:
            st.warning(f"Error fetching intraday data: {e}")
        
        return []
    
    def _get_gold_from_metals_api(self) -> Dict[str, Any]:
        """Get gold price from Metals API"""
        url = "https://api.metals.live/v1/spot/gold"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('price'):
                return {
                    'price_usd': float(data['price']),
                    'timestamp': datetime.now(),
                    'source': 'metals_live'
                }
        return None
    
    def get_historical_gold_data(self, days: int = 30) -> List[Dict]:
        """Get historical gold price data"""
        try:
            gold_ticker = yf.Ticker("GC=F")  # Gold Futures
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            hist = gold_ticker.history(start=start_date, end=end_date)
            
            historical_data = []
            for date, row in hist.iterrows():
                historical_data.append({
                    'date': date,
                    'price': float(row['Close']),
                    'volume': float(row['Volume']) if not pd.isna(row['Volume']) else 0,
                    'high': float(row['High']),
                    'low': float(row['Low'])
                })
            
            return historical_data
        except Exception as e:
            return []
    
    def get_real_time_gold_data(self) -> Dict[str, Any]:
        """Get comprehensive real-time gold data"""
        base_data = self.get_gold_price()
        
        # Try to get additional data from Yahoo Finance
        try:
            gold_ticker = yf.Ticker("GC=F")
            info = gold_ticker.info
            hist = gold_ticker.history(period="2d")
            
            if len(hist) >= 2:
                current_price = float(hist['Close'].iloc[-1])
                previous_price = float(hist['Close'].iloc[-2])
                change = current_price - previous_price
                change_percent = (change / previous_price) * 100
                
                base_data.update({
                    'price_usd': current_price,
                    'change': change,
                    'change_percent': change_percent,
                    'high_24h': float(hist['High'].iloc[-1]),
                    'low_24h': float(hist['Low'].iloc[-1]),
                    'volume': float(hist['Volume'].iloc[-1]) if not pd.isna(hist['Volume'].iloc[-1]) else 0
                })
        except Exception as e:
            pass
        
        return base_data
        """Analyze gold price trends"""
        if len(historical_data) < 2:
            return {"error": "Insufficient data for analysis"}
        
        prices = [d.gold_price_usd for d in historical_data]
        price_changes = np.diff(prices)
        
        analysis = {
            'trend': 'increasing' if np.mean(price_changes) > 0 else 'decreasing',
            'volatility': np.std(price_changes),
            'avg_change': np.mean(price_changes),
            'current_price': prices[-1],
            'price_range': {'min': min(prices), 'max': max(prices)}
        }
        
        # Add analysis to knowledge base
        knowledge_text = f"Gold price analysis: {analysis['trend']} trend with volatility of {analysis['volatility']:.2f}"
        self.retriever.add_to_knowledge_base(
            knowledge_text,
            {'type': 'analysis', 'data': analysis}
        )
        
        return analysis

    def analyze_gold_trends(self, historical_data: List[MarketData]) -> Dict[str, Any]:
        """Analyze gold price trends"""
        if len(historical_data) < 2:
            return {"error": "Insufficient data for analysis"}
        
        prices = [d.gold_price_usd for d in historical_data]
        price_changes = np.diff(prices)
        
        analysis = {
            'trend': 'increasing' if np.mean(price_changes) > 0 else 'decreasing',
            'volatility': np.std(price_changes),
            'avg_change': np.mean(price_changes),
            'current_price': prices[-1],
            'price_range': {'min': min(prices), 'max': max(prices)}
        }
        
        # Add analysis to knowledge base
        knowledge_text = f"Gold price analysis: {analysis['trend']} trend with volatility of {analysis['volatility']:.2f}"
        self.retriever.add_to_knowledge_base(
            knowledge_text,
            {'type': 'analysis', 'data': analysis}
        )
        
        return analysis

class CurrencyAgent:
    """Agent for USD/IDR exchange rate data"""
    
    def __init__(self, retriever: DataRetriever):
        self.retriever = retriever
        
    def get_usd_idr_rate(self) -> Dict[str, Any]:
        """Fetch current USD/IDR exchange rate using multiple APIs"""
        methods = [
            self._get_rate_from_yahoo,
            self._get_rate_from_alpha_vantage,
            self._get_rate_from_finnhub,
            self._get_rate_from_fixer,
            self._get_rate_from_exchangerate_api
        ]
        
        for method in methods:
            try:
                result = method()
                if result and result.get('rate'):
                    # Add to knowledge base
                    knowledge_text = f"Current USD/IDR exchange rate is {result['rate']:.2f} from {result['source']} as of {datetime.now()}"
                    self.retriever.add_to_knowledge_base(
                        knowledge_text,
                        {'type': 'exchange_rate', 'rate': result['rate'], 'pair': 'USD/IDR', 'source': result['source']}
                    )
                    return result
            except Exception as e:
                continue
        
        # Fallback to simulated data
        rate = 15000 + np.random.normal(0, 200)
        return {
            'rate': rate,
            'timestamp': datetime.now(),
            'source': 'simulated'
        }
    
    def _get_rate_from_yahoo(self) -> Dict[str, Any]:
        """Get USD/IDR rate from Yahoo Finance"""
        ticker = yf.Ticker("USDIDR=X")
        hist = ticker.history(period="1d")
        if not hist.empty:
            rate = float(hist['Close'].iloc[-1])
            return {
                'rate': rate,
                'timestamp': datetime.now(),
                'source': 'yahoo_finance'
            }
        return None
    
    def _get_rate_from_alpha_vantage(self) -> Dict[str, Any]:
        """Get USD/IDR rate from Alpha Vantage - Premium API"""
        if not API_CONFIG['ALPHA_VANTAGE_KEY']:
            return None
            
        url = f"https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency=USD&to_currency=IDR&apikey={API_CONFIG['ALPHA_VANTAGE_KEY']}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if "Realtime Currency Exchange Rate" in data:
                rate_data = data["Realtime Currency Exchange Rate"]
                return {
                    'rate': float(rate_data["5. Exchange Rate"]),
                    'timestamp': datetime.now(),
                    'source': 'alpha_vantage_premium',
                    'bid_price': float(rate_data.get("8. Bid Price", 0)),
                    'ask_price': float(rate_data.get("9. Ask Price", 0)),
                    'last_refreshed': rate_data.get("6. Last Refreshed")
                }
        return None
    
    def _get_rate_from_fixer(self) -> Dict[str, Any]:
        """Get USD/IDR rate from Fixer.io - Premium API"""
        if not API_CONFIG['FIXER_KEY']:
            return None
            
        url = f"http://data.fixer.io/api/latest?access_key={API_CONFIG['FIXER_KEY']}&base=USD&symbols=IDR"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success') and 'IDR' in data.get('rates', {}):
                return {
                    'rate': float(data['rates']['IDR']),
                    'timestamp': datetime.now(),
                    'source': 'fixer_io_premium',
                    'base': data.get('base'),
                    'date': data.get('date')
                }
        return None
    
    def _get_rate_from_finnhub(self) -> Dict[str, Any]:
        """Get USD/IDR rate from Finnhub - Premium API"""
        if not API_CONFIG['FINNHUB_KEY']:
            return None
            
        url = f"https://finnhub.io/api/v1/quote?symbol=OANDA:USD_IDR&token={API_CONFIG['FINNHUB_KEY']}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('c'):
                return {
                    'rate': float(data['c']),
                    'timestamp': datetime.now(),
                    'source': 'finnhub_premium',
                    'open_price': float(data.get('o', 0)),
                    'high_price': float(data.get('h', 0)),
                    'low_price': float(data.get('l', 0)),
                    'previous_close': float(data.get('pc', 0)),
                    'change': float(data.get('d', 0)),
                    'change_percent': float(data.get('dp', 0))
                }
        return None
    
    def get_currency_historical_premium(self, days: int = 30) -> List[Dict]:
        """Get historical USD/IDR data from Alpha Vantage"""
        if not API_CONFIG['ALPHA_VANTAGE_KEY']:
            return []
        
        try:
            url = f"https://www.alphavantage.co/query?function=FX_DAILY&from_symbol=USD&to_symbol=IDR&apikey={API_CONFIG['ALPHA_VANTAGE_KEY']}"
            response = requests.get(url, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if "Time Series FX (Daily)" in data:
                    time_series = data["Time Series FX (Daily)"]
                    historical_data = []
                    
                    for date_str, values in list(time_series.items())[:days]:
                        historical_data.append({
                            'date': datetime.strptime(date_str, '%Y-%m-%d'),
                            'rate': float(values['4. close']),
                            'high': float(values['2. high']),
                            'low': float(values['3. low']),
                            'open': float(values['1. open'])
                        })
                    
                    return sorted(historical_data, key=lambda x: x['date'])
        except Exception as e:
            st.warning(f"Error fetching historical currency data: {e}")
        
        return []
    
    def _get_rate_from_exchangerate_api(self) -> Dict[str, Any]:
        """Get USD/IDR rate from ExchangeRate-API"""
        url = "https://api.exchangerate-api.com/v4/latest/USD"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if 'IDR' in data.get('rates', {}):
                return {
                    'rate': float(data['rates']['IDR']),
                    'timestamp': datetime.now(),
                    'source': 'exchangerate_api'
                }
        return None
    
    def get_historical_currency_data(self, days: int = 30) -> List[Dict]:
        """Get historical USD/IDR exchange rate data"""
        try:
            ticker = yf.Ticker("USDIDR=X")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            hist = ticker.history(start=start_date, end=end_date)
            
            historical_data = []
            for date, row in hist.iterrows():
                historical_data.append({
                    'date': date,
                    'rate': float(row['Close']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'volume': float(row['Volume']) if not pd.isna(row['Volume']) else 0
                })
            
            return historical_data
        except Exception as e:
            return []
    
    def get_real_time_currency_data(self) -> Dict[str, Any]:
        """Get comprehensive real-time currency data"""
        base_data = self.get_usd_idr_rate()
        
        # Try to get additional data from Yahoo Finance
        try:
            ticker = yf.Ticker("USDIDR=X")
            hist = ticker.history(period="2d")
            
            if len(hist) >= 2:
                current_rate = float(hist['Close'].iloc[-1])
                previous_rate = float(hist['Close'].iloc[-2])
                change = current_rate - previous_rate
                change_percent = (change / previous_rate) * 100
                
                base_data.update({
                    'rate': current_rate,
                    'change': change,
                    'change_percent': change_percent,
                    'high_24h': float(hist['High'].iloc[-1]),
                    'low_24h': float(hist['Low'].iloc[-1])
                })
        except Exception as e:
            pass
        
        return base_data
        """Analyze USD/IDR trends"""
        if len(historical_data) < 2:
            return {"error": "Insufficient data for analysis"}
        
        rates = [d.usd_idr_rate for d in historical_data]
        rate_changes = np.diff(rates)
        
        analysis = {
            'trend': 'strengthening' if np.mean(rate_changes) > 0 else 'weakening',
            'volatility': np.std(rate_changes),
            'avg_change': np.mean(rate_changes),
            'current_rate': rates[-1],
            'rate_range': {'min': min(rates), 'max': max(rates)}
        }
        
        # Add analysis to knowledge base
        knowledge_text = f"USD/IDR analysis: USD {analysis['trend']} with volatility of {analysis['volatility']:.2f}"
        self.retriever.add_to_knowledge_base(
            knowledge_text,
            {'type': 'currency_analysis', 'data': analysis}
        )
        
        return analysis

class CorrelationAgent:
    """Agent for correlation analysis between gold and currency"""
    
    def __init__(self, retriever: DataRetriever):
        self.retriever = retriever
        
    def analyze_correlation(self, historical_data: List[MarketData]) -> Dict[str, Any]:
        """Analyze correlation between gold prices and USD/IDR"""
        if len(historical_data) < 5:
            return {"error": "Insufficient data for correlation analysis"}
        
        gold_prices = [d.gold_price_usd for d in historical_data]
        usd_idr_rates = [d.usd_idr_rate for d in historical_data]
        gold_prices_idr = [d.gold_price_idr for d in historical_data]
        
        # Calculate correlations
        corr_gold_usd_rate = np.corrcoef(gold_prices, usd_idr_rates)[0, 1]
        corr_gold_idr_rate = np.corrcoef(gold_prices_idr, usd_idr_rates)[0, 1]
        
        # Calculate additional correlations for premium users
        gold_returns = np.diff(gold_prices) / gold_prices[:-1]
        rate_returns = np.diff(usd_idr_rates) / usd_idr_rates[:-1]
        
        return_correlation = np.corrcoef(gold_returns, rate_returns)[0, 1] if len(gold_returns) > 1 else 0
        
        analysis = {
            'gold_usd_vs_usdidr_correlation': corr_gold_usd_rate,
            'gold_idr_vs_usdidr_correlation': corr_gold_idr_rate,
            'gold_usd_returns_vs_usdidr_returns': return_correlation,
            'relationship_strength': abs(corr_gold_usd_rate),
            'interpretation': self._interpret_correlation(corr_gold_usd_rate),
            'volatility_correlation': self._analyze_volatility_correlation(historical_data)
        }
        
        # Add to knowledge base
        knowledge_text = f"Correlation analysis: Gold-USD vs USD/IDR correlation is {corr_gold_usd_rate:.3f} - {analysis['interpretation']}"
        self.retriever.add_to_knowledge_base(
            knowledge_text,
            {'type': 'correlation_analysis', 'data': analysis}
        )
        
        return analysis
    
    def _interpret_correlation(self, correlation: float) -> str:
        """Interpret correlation coefficient"""
        abs_corr = abs(correlation)
        if abs_corr > 0.7:
            strength = "strong"
        elif abs_corr > 0.3:
            strength = "moderate"
        else:
            strength = "weak"
        
        direction = "positive" if correlation > 0 else "negative"
        return f"{strength} {direction} correlation"
    
    def _analyze_volatility_correlation(self, historical_data: List[MarketData]) -> Dict[str, float]:
        """Analyze correlation between volatilities"""
        if len(historical_data) < 10:
            return {"error": "Insufficient data for volatility analysis"}
        
        # Calculate rolling volatilities
        window_size = min(7, len(historical_data) // 2)
        
        gold_prices = [d.gold_price_usd for d in historical_data]
        usd_idr_rates = [d.usd_idr_rate for d in historical_data]
        
        gold_volatilities = []
        rate_volatilities = []
        
        for i in range(window_size, len(historical_data)):
            gold_window = gold_prices[i-window_size:i]
            rate_window = usd_idr_rates[i-window_size:i]
            
            gold_returns = np.diff(gold_window) / gold_window[:-1]
            rate_returns = np.diff(rate_window) / rate_window[:-1]
            
            gold_volatilities.append(np.std(gold_returns))
            rate_volatilities.append(np.std(rate_returns))
        
        if len(gold_volatilities) > 1:
            vol_correlation = np.corrcoef(gold_volatilities, rate_volatilities)[0, 1]
            return {
                'volatility_correlation': vol_correlation,
                'interpretation': self._interpret_correlation(vol_correlation)
            }
        
        return {"volatility_correlation": 0.0, "interpretation": "insufficient data"}
    
    def generate_correlation_insights(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable insights from correlation analysis"""
        insights = []
        
        if 'error' in analysis:
            return ["Insufficient data for correlation analysis"]
        
        gold_usd_corr = analysis.get('gold_usd_vs_usdidr_correlation', 0)
        
        if gold_usd_corr < -0.5:
            insights.append("🔴 Strong negative correlation: USD strength typically drives gold prices down")
            insights.append("💡 Strategy: Monitor USD strength indicators for gold price prediction")
        elif gold_usd_corr > 0.5:
            insights.append("🟢 Strong positive correlation: Unusual market conditions detected")
            insights.append("⚠️ Strategy: Both assets moving together suggests macro uncertainty")
        else:
            insights.append("🟡 Moderate correlation: Normal market relationship")
            insights.append("📊 Strategy: Use multiple indicators for decision making")
        
        # Volatility insights
        vol_analysis = analysis.get('volatility_correlation', {})
        if isinstance(vol_analysis, dict) and 'volatility_correlation' in vol_analysis:
            vol_corr = vol_analysis['volatility_correlation']
            if abs(vol_corr) > 0.5:
                insights.append(f"📈 Volatility correlation: {vol_corr:.2f} - Markets move in sync during stress")
            else:
                insights.append("📉 Low volatility correlation - Independent risk factors")
        
    def analyze_currency_trends(self, historical_data: List[MarketData]) -> Dict[str, Any]:
        """Analyze USD/IDR trends"""
        if len(historical_data) < 2:
            return {"error": "Insufficient data for analysis"}
        
        rates = [d.usd_idr_rate for d in historical_data]
        rate_changes = np.diff(rates)
        
        analysis = {
            'trend': 'strengthening' if np.mean(rate_changes) > 0 else 'weakening',
            'volatility': np.std(rate_changes),
            'avg_change': np.mean(rate_changes),
            'current_rate': rates[-1],
            'rate_range': {'min': min(rates), 'max': max(rates)}
        }
        
        # Add analysis to knowledge base
        knowledge_text = f"USD/IDR analysis: USD {analysis['trend']} with volatility of {analysis['volatility']:.2f}"
        self.retriever.add_to_knowledge_base(
            knowledge_text,
            {'type': 'currency_analysis', 'data': analysis}
        )
        
        return analysis

    def analyze_gold_trends(self, historical_data: List[MarketData]) -> Dict[str, Any]:
        """Analyze gold price trends"""
        if len(historical_data) < 2:
            return {"error": "Insufficient data for analysis"}
        
        prices = [d.gold_price_usd for d in historical_data]
        price_changes = np.diff(prices)
        
        analysis = {
            'trend': 'increasing' if np.mean(price_changes) > 0 else 'decreasing',
            'volatility': np.std(price_changes),
            'avg_change': np.mean(price_changes),
            'current_price': prices[-1],
            'price_range': {'min': min(prices), 'max': max(prices)}
        }
        
        # Add analysis to knowledge base
        knowledge_text = f"Gold price analysis: {analysis['trend']} trend with volatility of {analysis['volatility']:.2f}"
        self.retriever.add_to_knowledge_base(
            knowledge_text,
            {'type': 'analysis', 'data': analysis}
        )
        
        return analysis

        return insights

class EconomicDataAgent:
    """Agent for economic indicators and market sentiment"""
    
    def __init__(self, retriever: DataRetriever):
        self.retriever = retriever
    
    def get_market_sentiment(self) -> Dict[str, Any]:
        """Get market sentiment and economic indicators"""
        sentiment_data = {}
        
        # Get Fear & Greed Index (simulated)
        fear_greed_index = np.random.randint(20, 80)
        sentiment_data['fear_greed_index'] = fear_greed_index
        sentiment_data['market_sentiment'] = self._interpret_fear_greed(fear_greed_index)
        
        # Get DXY (Dollar Index) from Yahoo Finance
        try:
            dxy_ticker = yf.Ticker("DX-Y.NYB")
            dxy_hist = dxy_ticker.history(period="2d")
            if not dxy_hist.empty:
                dxy_current = float(dxy_hist['Close'].iloc[-1])
                dxy_previous = float(dxy_hist['Close'].iloc[-2]) if len(dxy_hist) > 1 else dxy_current
                dxy_change = ((dxy_current - dxy_previous) / dxy_previous) * 100
                
                sentiment_data['dxy_index'] = dxy_current
                sentiment_data['dxy_change'] = dxy_change
                sentiment_data['dollar_strength'] = 'strong' if dxy_change > 0 else 'weak'
        except Exception as e:
            sentiment_data['dxy_index'] = 102.5
            sentiment_data['dxy_change'] = np.random.uniform(-1, 1)
        
        # Get VIX (Volatility Index)
        try:
            vix_ticker = yf.Ticker("^VIX")
            vix_hist = vix_ticker.history(period="1d")
            if not vix_hist.empty:
                vix_current = float(vix_hist['Close'].iloc[-1])
                sentiment_data['vix_index'] = vix_current
                sentiment_data['market_volatility'] = self._interpret_vix(vix_current)
        except Exception as e:
            sentiment_data['vix_index'] = 20.0
            sentiment_data['market_volatility'] = 'moderate'
        
        # Add to knowledge base
        knowledge_text = f"Market sentiment: {sentiment_data.get('market_sentiment', 'neutral')}, DXY: {sentiment_data.get('dxy_index', 0):.2f}, VIX: {sentiment_data.get('vix_index', 0):.2f}"
        self.retriever.add_to_knowledge_base(
            knowledge_text,
            {'type': 'market_sentiment', 'data': sentiment_data}
        )
        
        return sentiment_data
    
    def _interpret_fear_greed(self, index: int) -> str:
        """Interpret Fear & Greed Index"""
        if index <= 25:
            return 'extreme_fear'
        elif index <= 45:
            return 'fear'
        elif index <= 55:
            return 'neutral'
        elif index <= 75:
            return 'greed'
        else:
            return 'extreme_greed'
    
    def _interpret_vix(self, vix: float) -> str:
        """Interpret VIX levels"""
        if vix < 15:
            return 'low'
        elif vix < 25:
            return 'moderate'
        elif vix < 35:
            return 'high'
        else:
            return 'extreme'
    
    def get_commodity_correlations(self) -> Dict[str, Any]:
        """Get correlations with other commodities"""
        correlations = {}
        
        try:
            # Get Silver data
            silver_ticker = yf.Ticker("SI=F")
            silver_hist = silver_ticker.history(period="30d")
            
            # Get Oil data  
            oil_ticker = yf.Ticker("CL=F")
            oil_hist = oil_ticker.history(period="30d")
            
            # Get Gold data
            gold_ticker = yf.Ticker("GC=F")
            gold_hist = gold_ticker.history(period="30d")
            
            # Calculate correlations if we have enough data
            if len(gold_hist) > 10 and len(silver_hist) > 10:
                # Align dates
                common_dates = gold_hist.index.intersection(silver_hist.index)
                if len(common_dates) > 10:
                    gold_prices = gold_hist.loc[common_dates]['Close']
                    silver_prices = silver_hist.loc[common_dates]['Close']
                    correlations['gold_silver'] = float(np.corrcoef(gold_prices, silver_prices)[0, 1])
            
            if len(gold_hist) > 10 and len(oil_hist) > 10:
                common_dates = gold_hist.index.intersection(oil_hist.index)
                if len(common_dates) > 10:
                    gold_prices = gold_hist.loc[common_dates]['Close']
                    oil_prices = oil_hist.loc[common_dates]['Close']
                    correlations['gold_oil'] = float(np.corrcoef(gold_prices, oil_prices)[0, 1])
                    
        except Exception as e:
            correlations = {
                'gold_silver': 0.8,  # Default values
                'gold_oil': 0.3
            }
        
        return correlations
    """Agent for correlation analysis between gold and currency"""
    
    def __init__(self, retriever: DataRetriever):
        self.retriever = retriever
        
    def analyze_correlation(self, historical_data: List[MarketData]) -> Dict[str, Any]:
        """Analyze correlation between gold prices and USD/IDR"""
        if len(historical_data) < 5:
            return {"error": "Insufficient data for correlation analysis"}
        
        gold_prices = [d.gold_price_usd for d in historical_data]
        usd_idr_rates = [d.usd_idr_rate for d in historical_data]
        gold_prices_idr = [d.gold_price_idr for d in historical_data]
        
        # Calculate correlations
        corr_gold_usd_rate = np.corrcoef(gold_prices, usd_idr_rates)[0, 1]
        corr_gold_idr_rate = np.corrcoef(gold_prices_idr, usd_idr_rates)[0, 1]
        
        analysis = {
            'gold_usd_vs_usdidr_correlation': corr_gold_usd_rate,
            'gold_idr_vs_usdidr_correlation': corr_gold_idr_rate,
            'relationship_strength': abs(corr_gold_usd_rate),
            'interpretation': self._interpret_correlation(corr_gold_usd_rate)
        }
        
        # Add to knowledge base
        knowledge_text = f"Correlation analysis: Gold-USD vs USD/IDR correlation is {corr_gold_usd_rate:.3f} - {analysis['interpretation']}"
        self.retriever.add_to_knowledge_base(
            knowledge_text,
            {'type': 'correlation_analysis', 'data': analysis}
        )
        
        return analysis
    
    def _interpret_correlation(self, correlation: float) -> str:
        """Interpret correlation coefficient"""
        abs_corr = abs(correlation)
        if abs_corr > 0.7:
            strength = "strong"
        elif abs_corr > 0.3:
            strength = "moderate"
        else:
            strength = "weak"
        
        direction = "positive" if correlation > 0 else "negative"
        return f"{strength} {direction} correlation"

class AIAnalysisAgent:
    """Enhanced AI agent using OpenRouter Qwen3 for advanced analysis"""
    
    def __init__(self, retriever: DataRetriever):
        self.retriever = retriever
        self.client = get_openrouter_client()
        
    def generate_insights(self, query: str, market_data: List[MarketData]) -> str:
        """Generate AI insights using Qwen3 via OpenRouter"""
        # Retrieve relevant information
        relevant_docs = self.retriever.retrieve_relevant_info(query, top_k=10)
        
        # Prepare enhanced context
        context = self._prepare_enhanced_context(relevant_docs, market_data)
        
        if self.client and API_CONFIG['OPENROUTER_API_KEY']:
            return self._generate_qwen3_insights(query, context)
        else:
            return self._generate_local_insights(query, context, market_data)
    
    def _prepare_enhanced_context(self, relevant_docs: List[Dict], market_data: List[MarketData]) -> str:
        """Prepare comprehensive context for AI analysis"""
        context_parts = []
        
        # Market data summary
        if market_data:
            latest = market_data[-1]
            context_parts.append(f"""
            CURRENT MARKET DATA:
            - Gold Price: ${latest.gold_price_usd:.2f} USD
            - USD/IDR Rate: {latest.usd_idr_rate:.2f}
            - Gold Price in IDR: Rp {latest.gold_price_idr:,.0f}
            - Data Points: {len(market_data)} days
            """)
            
            # Calculate trends
            if len(market_data) > 1:
                gold_change = ((market_data[-1].gold_price_usd - market_data[0].gold_price_usd) / market_data[0].gold_price_usd) * 100
                idr_change = ((market_data[-1].usd_idr_rate - market_data[0].usd_idr_rate) / market_data[0].usd_idr_rate) * 100
                
                context_parts.append(f"""
                TREND ANALYSIS:
                - Gold trend: {gold_change:+.2f}% over period
                - USD/IDR trend: {idr_change:+.2f}% over period
                """)
        
        # Historical insights
        if relevant_docs:
            context_parts.append("HISTORICAL INSIGHTS:")
            for doc in relevant_docs[:5]:
                context_parts.append(f"- {doc['text']}")
        
        return "\n".join(context_parts)
    
    def _generate_qwen3_insights(self, query: str, context: str) -> str:
        """Generate insights using Qwen3 via OpenRouter"""
        try:
            system_prompt = """You are an expert financial analyst specializing in gold prices and Indonesian currency markets. 
            You provide professional, data-driven insights for investment decisions.
            
            Key capabilities:
            - Analyze gold price trends and correlations
            - Understand USD/IDR exchange rate impacts
            - Provide actionable investment recommendations
            - Consider economic indicators and market sentiment
            - Explain complex financial concepts clearly
            
            Always provide:
            1. Clear analysis based on data
            2. Risk assessment
            3. Actionable recommendations
            4. Timeline considerations
            5. Market context
            """
            
            user_prompt = f"""
            Based on the following market data and context, please analyze and answer this question:
            
            QUESTION: {query}
            
            MARKET CONTEXT:
            {context}
            
            Please provide a comprehensive analysis with:
            1. Current market assessment
            2. Key factors influencing prices
            3. Investment implications
            4. Risk considerations
            5. Specific recommendations with rationale
            """
            
            response = self.client.chat.completions.create(
                model=API_CONFIG['QWEN3_MODEL'],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            
            insight = response.choices[0].message.content
            
            # Add to knowledge base
            self.retriever.add_to_knowledge_base(
                f"AI Analysis for query '{query}': {insight[:500]}...",
                {'type': 'qwen3_analysis', 'query': query, 'model': 'qwen3'}
            )
            
            return f"## 🤖 Qwen3 AI Analysis\n\n{insight}"
            
        except Exception as e:
            st.error(f"Error with Qwen3 analysis: {e}")
            return self._generate_local_insights(query, context, [])
    
    def _generate_local_insights(self, query: str, context: str, market_data: List[MarketData]) -> str:
        """Fallback local analysis"""
        insights = []
        
        insights.append("## 📊 Market Analysis (Local)")
        
        if market_data and len(market_data) > 1:
            # Price trend analysis
            recent_gold_change = ((market_data[-1].gold_price_usd - market_data[-2].gold_price_usd) / market_data[-2].gold_price_usd) * 100
            recent_rate_change = ((market_data[-1].usd_idr_rate - market_data[-2].usd_idr_rate) / market_data[-2].usd_idr_rate) * 100
            
            insights.append(f"**Recent Changes:**")
            insights.append(f"- Gold: {recent_gold_change:+.2f}%")
            insights.append(f"- USD/IDR: {recent_rate_change:+.2f}%")
            
            # Investment recommendations
            if recent_gold_change > 0 and recent_rate_change > 0:
                insights.append("\n**📈 Market Observation:** Both gold and USD are strengthening, indicating potential market uncertainty or inflation concerns.")
                insights.append("**💡 Investment Consideration:** Monitor for continuation of trend before making large positions.")
            elif recent_gold_change < 0 and recent_rate_change < 0:
                insights.append("\n**📉 Market Observation:** Both gold and USD are weakening, suggesting market stabilization.")
                insights.append("**💡 Investment Consideration:** Potential buying opportunity if fundamentals remain strong.")
        
        insights.append(f"\n**Context:** {context[:500]}...")
        
        return "\n".join(insights)
    
    def generate_market_report(self, market_data: List[MarketData], economic_data: Dict) -> str:
        """Generate comprehensive market report using Qwen3"""
        if not self.client:
            return "Qwen3 AI not available for market reports"
        
        # Prepare comprehensive data
        report_context = self._prepare_report_context(market_data, economic_data)
        
        try:
            system_prompt = """You are a senior financial analyst creating professional market reports. 
            Generate a comprehensive analysis suitable for institutional investors and financial professionals."""
            
            user_prompt = f"""
            Create a professional market report analyzing gold prices and USD/IDR exchange rates.
            
            MARKET DATA:
            {report_context}
            
            Include:
            1. Executive Summary
            2. Market Overview
            3. Technical Analysis
            4. Economic Factors
            5. Risk Assessment
            6. Investment Recommendations
            7. Market Outlook
            
            Format as a professional report with clear sections and bullet points.
            """
            
            response = self.client.chat.completions.create(
                model=API_CONFIG['QWEN3_MODEL'],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.6,
                max_tokens=2000
            )
            
            return f"# 📋 Professional Market Report\n\n{response.choices[0].message.content}"
            
        except Exception as e:
            return f"Error generating report: {e}"
    
    def _prepare_report_context(self, market_data: List[MarketData], economic_data: Dict) -> str:
        """Prepare context for market report"""
        if not market_data:
            return "No market data available"
        
        # Calculate key metrics
        gold_prices = [d.gold_price_usd for d in market_data]
        usd_idr_rates = [d.usd_idr_rate for d in market_data]
        
        gold_volatility = np.std(gold_prices) / np.mean(gold_prices)
        rate_volatility = np.std(usd_idr_rates) / np.mean(usd_idr_rates)
        
        correlation = np.corrcoef(gold_prices, usd_idr_rates)[0, 1] if len(gold_prices) > 1 else 0
        
        context = f"""
        Market Data Summary ({len(market_data)} days):
        - Gold Price Range: ${min(gold_prices):.2f} - ${max(gold_prices):.2f}
        - Current Gold: ${gold_prices[-1]:.2f}
        - Gold Volatility: {gold_volatility:.3f}
        
        - USD/IDR Range: {min(usd_idr_rates):.0f} - {max(usd_idr_rates):.0f}
        - Current Rate: {usd_idr_rates[-1]:.0f}
        - Rate Volatility: {rate_volatility:.3f}
        
        - Correlation: {correlation:.3f}
        
        Economic Indicators:
        {json.dumps(economic_data, indent=2)}
        """
        
        return context

# Initialize components
@st.cache_resource
def initialize_agents():
    retriever = DataRetriever()
    gold_agent = GoldPriceAgent(retriever)
    currency_agent = CurrencyAgent(retriever)
    correlation_agent = CorrelationAgent(retriever)
    ai_agent = AIAnalysisAgent(retriever)
    
    return retriever, gold_agent, currency_agent, correlation_agent, ai_agent

def generate_historical_data(days: int = 30) -> List[MarketData]:
    """Generate simulated historical data"""
    data = []
    base_gold_price = 2000
    base_usd_idr = 15000
    
    for i in range(days):
        date = datetime.now() - timedelta(days=days-i)
        
        # Simulate price movements with some correlation
        gold_change = np.random.normal(0, 30)
        currency_change = np.random.normal(0, 200)
        
        # Add some correlation (negative correlation between gold and USD strength)
        if gold_change > 0:
            currency_change += np.random.normal(50, 100)  # USD weakens when gold rises
        
        gold_price = base_gold_price + gold_change
        usd_idr_rate = base_usd_idr + currency_change
        gold_price_idr = gold_price * usd_idr_rate
        
        data.append(MarketData(
            timestamp=date,
            gold_price_usd=gold_price,
            usd_idr_rate=usd_idr_rate,
            gold_price_idr=gold_price_idr
        ))
        
        # Update base prices for next iteration
        base_gold_price = gold_price
        base_usd_idr = usd_idr_rate
    
    return data

def create_charts(historical_data: List[MarketData]):
    """Create interactive charts"""
    df = pd.DataFrame([
        {
            'Date': data.timestamp,
            'Gold Price (USD)': data.gold_price_usd,
            'USD/IDR Rate': data.usd_idr_rate,
            'Gold Price (IDR)': data.gold_price_idr
        }
        for data in historical_data
    ])
    
    # Gold Price Chart
    fig_gold = go.Figure()
    fig_gold.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Gold Price (USD)'],
        mode='lines+markers',
        name='Gold Price (USD)',
        line=dict(color='gold', width=2)
    ))
    fig_gold.update_layout(
        title='Gold Price Trend (USD)',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        template='plotly_dark'
    )
    
    # USD/IDR Rate Chart
    fig_rate = go.Figure()
    fig_rate.add_trace(go.Scatter(
        x=df['Date'],
        y=df['USD/IDR Rate'],
        mode='lines+markers',
        name='USD/IDR Rate',
        line=dict(color='blue', width=2)
    ))
    fig_rate.update_layout(
        title='USD/IDR Exchange Rate Trend',
        xaxis_title='Date',
        yaxis_title='Rate (IDR)',
        template='plotly_dark'
    )
    
    # Gold Price in IDR Chart
    fig_gold_idr = go.Figure()
    fig_gold_idr.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Gold Price (IDR)'],
        mode='lines+markers',
        name='Gold Price (IDR)',
        line=dict(color='orange', width=2)
    ))
    fig_gold_idr.update_layout(
        title='Gold Price Trend (IDR)',
        xaxis_title='Date',
        yaxis_title='Price (IDR)',
        template='plotly_dark'
    )
    
    return fig_gold, fig_rate, fig_gold_idr

def main():
    # Validate dependencies first
    deps_ok, deps_msg = validate_dependencies()
    if not deps_ok:
        st.error(f"❌ Dependency Error: {deps_msg}")
        st.info("💡 Please check your Python environment and required packages.")
        st.stop()
    
    # Dynamic app info
    try:
        app_info = get_app_info()
    except Exception as e:
        # Fallback app info if function fails
        app_info = {
            'version': APP_VERSION,
            'creator': APP_CREATOR,
            'name': APP_NAME,
            'build_date': datetime.now().strftime('%Y-%m-%d'),
            'session_id': datetime.now().strftime('%Y%m%d_%H%M%S')
        }
    
    st.title(f"🏆 {app_info['name']}")
    st.markdown(f"**Version {app_info['version']}** | Premium APIs: Alpha Vantage • OpenRouter Qwen3 • Fixer.io • Finnhub")
    st.markdown("---")
    
    # Enhanced API Configuration in Sidebar
    with st.sidebar:
        st.header("🔑 Premium API Configuration")
        
        with st.expander("🚀 Setup Your API Keys", expanded=True):
            st.markdown("**Enter your premium API keys for best performance:**")
            
            alpha_key = st.text_input(
                "Alpha Vantage API Key", 
                type="password", 
                help="Professional financial data - 500 calls/day free"
            )
            
            openrouter_key = st.text_input(
                "OpenRouter API Key", 
                type="password", 
                help="Access to Qwen3 for advanced AI analysis"
            )
            
            fixer_key = st.text_input(
                "Fixer.io API Key", 
                type="password", 
                help="Bank-grade exchange rates - 100 calls/month free"
            )
            
            finnhub_key = st.text_input(
                "Finnhub API Key", 
                type="password", 
                help="Real-time market data - 60 calls/minute free"
            )
            
            # Update global config
            if alpha_key:
                API_CONFIG['ALPHA_VANTAGE_KEY'] = alpha_key
            if openrouter_key:
                API_CONFIG['OPENROUTER_API_KEY'] = openrouter_key
            if fixer_key:
                API_CONFIG['FIXER_KEY'] = fixer_key
            if finnhub_key:
                API_CONFIG['FINNHUB_KEY'] = finnhub_key
            
            # API Status indicators
            st.markdown("**🔴🟡🟢 API Status:**")
            apis_configured = []
            if API_CONFIG['ALPHA_VANTAGE_KEY']:
                apis_configured.append("🟢 Alpha Vantage")
            else:
                apis_configured.append("🔴 Alpha Vantage")
                
            if API_CONFIG['OPENROUTER_API_KEY']:
                apis_configured.append("🟢 OpenRouter Qwen3")
            else:
                apis_configured.append("🔴 OpenRouter Qwen3")
                
            if API_CONFIG['FIXER_KEY']:
                apis_configured.append("🟢 Fixer.io")
            else:
                apis_configured.append("🔴 Fixer.io")
                
            if API_CONFIG['FINNHUB_KEY']:
                apis_configured.append("🟢 Finnhub")
            else:
                apis_configured.append("🔴 Finnhub")
            
            for status in apis_configured:
                st.write(status)
            
            st.write("🟢 Yahoo Finance (Always Available)")
    
    # Initialize enhanced agents with comprehensive error handling
    initialization_method = "unknown"  # Initialize variable
    with st.spinner("🔧 Initializing application agents..."):
        agents_initialized = False
        initialization_method = ""
        
        # Method 1: Try cached initialization
        try:
            st.info("🔄 Attempting cached initialization...")
            agents_result = initialize_agents()
            
            if agents_result and len(agents_result) == 6:
                retriever, gold_agent, currency_agent, correlation_agent, economic_agent, ai_agent = agents_result
                
                if all([retriever, gold_agent, currency_agent, correlation_agent, economic_agent, ai_agent]):
                    agents_initialized = True
                    initialization_method = "cached"
                    st.success("✅ Cached initialization successful!")
                
        except Exception as e:
            st.warning(f"⚠️ Cached initialization failed: {e}")
        
        # Method 2: Try safe initialization if cached failed
        if not agents_initialized:
            try:
                st.info("🔄 Attempting safe initialization...")
                retriever, gold_agent, currency_agent, correlation_agent, economic_agent, ai_agent = safe_initialize_agents()
                
                if all([retriever, gold_agent, currency_agent, correlation_agent, economic_agent, ai_agent]):
                    agents_initialized = True
                    initialization_method = "safe"
                    st.warning("⚠️ Safe initialization successful! Some advanced features may be limited.")
                    
            except Exception as e:
                st.error(f"❌ Safe initialization failed: {e}")
        
        # Method 3: Ultimate fallback - create minimal agents inline
        if not agents_initialized:
            try:
                st.info("🔄 Creating minimal fallback agents...")
                
                # Create absolutely minimal agents
                retriever = DataRetriever()
                
                # Create super simple agent objects
                class SimpleAgent:
                    def __init__(self, name):
                        self.name = name
                        self.retriever = retriever
                
                gold_agent = SimpleAgent("gold")
                currency_agent = SimpleAgent("currency") 
                correlation_agent = SimpleAgent("correlation")
                economic_agent = SimpleAgent("economic")
                ai_agent = SimpleAgent("ai")
                
                # Add minimal required methods
                gold_agent.get_real_time_gold_data = lambda: {'price_usd': 2000.0, 'source': 'fallback'}
                gold_agent.analyze_gold_trends = lambda data: {'trend': 'stable', 'volatility': 10.0, 'avg_change': 0.0, 'current_price': 2000.0, 'price_range': {'min': 1900, 'max': 2100}}
                gold_agent.get_historical_gold_data = lambda days: []
                gold_agent.get_intraday_gold_data = lambda: []
                
                currency_agent.get_real_time_currency_data = lambda: {'rate': 15000.0, 'source': 'fallback'}
                currency_agent.analyze_currency_trends = lambda data: {'trend': 'stable', 'volatility': 100.0, 'avg_change': 0.0, 'current_rate': 15000.0, 'rate_range': {'min': 14500, 'max': 15500}}
                currency_agent.get_currency_historical_premium = lambda days: []
                currency_agent.get_historical_currency_data = lambda days: []
                
                correlation_agent.analyze_correlation = lambda data: {
                    'gold_usd_vs_usdidr_correlation': 0.0, 
                    'gold_idr_vs_usdidr_correlation': 0.0,
                    'relationship_strength': 0.0,
                    'interpretation': 'no correlation in minimal mode'
                }
                correlation_agent.generate_correlation_insights = lambda analysis: ["Minimal mode: Basic correlation only"]
                
                economic_agent.get_market_sentiment = lambda: {
                    'fear_greed_index': 50, 
                    'market_sentiment': 'neutral',
                    'dxy_index': 102.5,
                    'dxy_change': 0.0,
                    'vix_index': 20.0,
                    'market_volatility': 'moderate',
                    'dollar_strength': 'neutral'
                }
                economic_agent.get_commodity_correlations = lambda: {'gold_silver': 0.8, 'gold_oil': 0.3}
                
                ai_agent.generate_insights = lambda query, data: f"**Minimal Mode:** Basic analysis for: {query}"
                ai_agent.generate_market_report = lambda data, economic_data: "**Minimal Mode:** Basic market report."
                
                agents_initialized = True
                initialization_method = "minimal_fallback"
                st.warning("⚠️ Running in minimal mode - basic functionality only.")
                
            except Exception as e:
                st.error(f"❌ Even minimal initialization failed: {e}")
        
        # Final check
        if not agents_initialized:
            st.error("❌ Complete initialization failure. Please refresh the page.")
            st.info("💡 If problem persists, check your Python environment and dependencies.")
            st.stop()
        else:
            st.success(f"🎯 Application ready! (Mode: {initialization_method})")
    
    # Force ensure critical methods exist (ultimate fallback)
    if not hasattr(currency_agent, 'analyze_currency_trends'):
        st.warning("🔧 Patching missing currency analysis method...")
        def fallback_currency_analysis(historical_data):
            if not historical_data or len(historical_data) < 2:
                return {"error": "Insufficient data for analysis"}
            rates = [d.usd_idr_rate for d in historical_data]
            rate_changes = np.diff(rates) if len(rates) > 1 else [0]
            return {
                'trend': 'strengthening' if np.mean(rate_changes) > 0 else 'weakening',
                'volatility': np.std(rate_changes) if len(rate_changes) > 0 else 0.0,
                'avg_change': np.mean(rate_changes) if len(rate_changes) > 0 else 0.0,
                'current_rate': rates[-1] if rates else 15000.0,
                'rate_range': {'min': min(rates) if rates else 14500, 'max': max(rates) if rates else 15500}
            }
        currency_agent.analyze_currency_trends = fallback_currency_analysis
    
    if not hasattr(gold_agent, 'analyze_gold_trends'):
        st.warning("🔧 Patching missing gold analysis method...")
        def fallback_gold_analysis(historical_data):
            if not historical_data or len(historical_data) < 2:
                return {"error": "Insufficient data for analysis"}
            prices = [d.gold_price_usd for d in historical_data]
            price_changes = np.diff(prices) if len(prices) > 1 else [0]
            return {
                'trend': 'increasing' if np.mean(price_changes) > 0 else 'decreasing',
                'volatility': np.std(price_changes) if len(price_changes) > 0 else 0.0,
                'avg_change': np.mean(price_changes) if len(price_changes) > 0 else 0.0,
                'current_price': prices[-1] if prices else 2000.0,
                'price_range': {'min': min(prices) if prices else 1900, 'max': max(prices) if prices else 2100}
            }
        gold_agent.analyze_gold_trends = fallback_gold_analysis
    
    if not hasattr(correlation_agent, 'analyze_correlation'):
        st.warning("🔧 Patching missing correlation analysis method...")
        def fallback_correlation_analysis(historical_data):
            if not historical_data or len(historical_data) < 5:
                return {"error": "Insufficient data for correlation analysis"}
            return {
                'gold_usd_vs_usdidr_correlation': 0.0,
                'gold_idr_vs_usdidr_correlation': 0.0,
                'relationship_strength': 0.0,
                'interpretation': 'fallback correlation analysis - limited data'
            }
        correlation_agent.analyze_correlation = fallback_correlation_analysis
    
    # Enhanced sidebar controls
    st.sidebar.header("🔧 Analysis Controls")
    
    # Debug mode toggle
    debug_mode = st.sidebar.checkbox("🐛 Debug Mode", value=False, help="Show detailed debugging information")
    
    days_to_analyze = st.sidebar.slider("Days of Historical Data", 7, 90, 30)
    auto_refresh = st.sidebar.checkbox("Auto Refresh Data", value=False)
    
    # Premium features toggle
    use_premium_data = st.sidebar.checkbox(
        "🚀 Use Premium APIs", 
        value=bool(API_CONFIG['ALPHA_VANTAGE_KEY']),
        help="Use Alpha Vantage, Fixer.io, Finnhub for higher quality data"
    )
    
    use_qwen3_ai = st.sidebar.checkbox(
        "🧠 Enable Qwen3 AI", 
        value=bool(API_CONFIG['OPENROUTER_API_KEY']),
        help="Use Qwen3 for advanced AI analysis via OpenRouter"
    )
    
    # App info in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"""
    **📱 App Info:**  
    Version: {app_info['version']}  
    Build: {app_info['build_date']}  
    Session: {app_info['session_id']}
    """)
    
    if debug_mode:
        st.sidebar.markdown("**🐛 Debug Info:**")
        st.sidebar.write(f"Initialization: {initialization_method}")
        st.sidebar.write(f"Agents status: R:{bool(retriever)} G:{bool(gold_agent)} C:{bool(currency_agent)} Co:{bool(correlation_agent)} E:{bool(economic_agent)} A:{bool(ai_agent)}")
        st.sidebar.write(f"Agent types: {type(gold_agent).__name__}, {type(currency_agent).__name__}")
        
        # Method existence check
        st.sidebar.write("**Method Status:**")
        st.sidebar.write(f"Gold.analyze_gold_trends: {hasattr(gold_agent, 'analyze_gold_trends')}")
        st.sidebar.write(f"Currency.analyze_currency_trends: {hasattr(currency_agent, 'analyze_currency_trends')}")
        st.sidebar.write(f"Correlation.analyze_correlation: {hasattr(correlation_agent, 'analyze_correlation')}")
        
        # Show available methods
        with st.sidebar.expander("Available Methods"):
            st.write("**Gold Agent:**")
            st.write([m for m in dir(gold_agent) if not m.startswith('_')])
            st.write("**Currency Agent:**")  
            st.write([m for m in dir(currency_agent) if not m.startswith('_')])
    
    if st.sidebar.button("🔄 Refresh Data") or auto_refresh:
        st.cache_data.clear()
        st.cache_resource.clear()  # Clear resource cache too
        st.rerun()
    
    # Enhanced main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Market Overview", 
        "🔍 Agent Analysis", 
        "📈 Economic Indicators", 
        "🤖 AI Insights Pro", 
        "📋 Market Reports",
        "📉 Advanced Analytics"
    ])
    
    with tab1:
        st.header("🏆 Premium Market Overview")
        
        # Enhanced real-time data with premium sources
        col1, col2, col3 = st.columns(3)
        
        with col1:
            with st.spinner("Fetching premium gold data..."):
                if use_premium_data and API_CONFIG['ALPHA_VANTAGE_KEY']:
                    try:
                        gold_data = gold_agent.get_real_time_gold_data()
                        
                        # Show enhanced metrics if available
                        if 'bid_price' in gold_data:
                            st.metric(
                                "Gold Price (USD) - Premium", 
                                f"${gold_data['price_usd']:.2f}",
                                delta=f"{gold_data.get('change', 0):.2f} ({gold_data.get('change_percent', 0):.2f}%)"
                            )
                            st.caption(f"Bid: ${gold_data.get('bid_price', 0):.2f} | Ask: ${gold_data.get('ask_price', 0):.2f}")
                            st.caption(f"Source: {gold_data.get('source', 'unknown').replace('_', ' ').title()}")
                        else:
                            gold_data = gold_agent.get_real_time_gold_data()
                            st.metric(
                                "Gold Price (USD)", 
                                f"${gold_data['price_usd']:.2f}",
                                delta=f"{gold_data.get('change_percent', 0):.2f}%"
                            )
                    except Exception as e:
                        st.error(f"Error fetching gold data: {e}")
                        gold_data = {'price_usd': 2000.0, 'source': 'fallback'}
                        st.metric("Gold Price (USD)", f"${gold_data['price_usd']:.2f}")
                else:
                    try:
                        gold_data = gold_agent.get_real_time_gold_data()
                        st.metric(
                            "Gold Price (USD)", 
                            f"${gold_data['price_usd']:.2f}",
                            delta=f"{gold_data.get('change_percent', 0):.2f}%"
                        )
                    except Exception as e:
                        gold_data = {'price_usd': 2000.0, 'source': 'fallback'}
                        st.metric("Gold Price (USD)", f"${gold_data['price_usd']:.2f}")
        
        with col2:
            with st.spinner("Fetching premium currency data..."):
                if use_premium_data and (API_CONFIG['ALPHA_VANTAGE_KEY'] or API_CONFIG['FIXER_KEY']):
                    try:
                        currency_data = currency_agent.get_real_time_currency_data()
                        
                        if 'bid_price' in currency_data:
                            st.metric(
                                "USD/IDR Rate - Premium", 
                                f"{currency_data['rate']:.2f}",
                                delta=f"{currency_data.get('change_percent', 0):.2f}%"
                            )
                            st.caption(f"Bid: {currency_data.get('bid_price', 0):.2f} | Ask: {currency_data.get('ask_price', 0):.2f}")
                            st.caption(f"Source: {currency_data.get('source', 'unknown').replace('_', ' ').title()}")
                        else:
                            st.metric(
                                "USD/IDR Rate", 
                                f"{currency_data['rate']:.2f}",
                                delta=f"{currency_data.get('change_percent', 0):.2f}%"
                            )
                    except Exception as e:
                        st.error(f"Error fetching currency data: {e}")
                        currency_data = {'rate': 15000.0, 'source': 'fallback'}
                        st.metric("USD/IDR Rate", f"{currency_data['rate']:.2f}")
                else:
                    try:
                        currency_data = currency_agent.get_real_time_currency_data()
                        st.metric(
                            "USD/IDR Rate", 
                            f"{currency_data['rate']:.2f}",
                            delta=f"{currency_data.get('change_percent', 0):.2f}%"
                        )
                    except Exception as e:
                        currency_data = {'rate': 15000.0, 'source': 'fallback'}
                        st.metric("USD/IDR Rate", f"{currency_data['rate']:.2f}")
        
        with col3:
            # Enhanced gold price in IDR calculation
            try:
                gold_price_idr = gold_data['price_usd'] * currency_data['rate']
                idr_change = 0
                if 'change_percent' in gold_data and 'change_percent' in currency_data:
                    idr_change = gold_data.get('change_percent', 0) + currency_data.get('change_percent', 0)
                
                st.metric(
                    "Gold Price (IDR)", 
                    f"Rp {gold_price_idr:,.0f}",
                    delta=f"{idr_change:.2f}%"
                )
                
                # Data quality indicator
                quality_score = 0
                if gold_data.get('source', '').endswith('premium'):
                    quality_score += 1
                if currency_data.get('source', '').endswith('premium'):
                    quality_score += 1
                
                quality_labels = ["🔴 Basic", "🟡 Enhanced", "🟢 Premium"]
                st.caption(f"Data Quality: {quality_labels[quality_score]}")
            except Exception as e:
                st.error(f"Error calculating IDR price: {e}")
                st.metric("Gold Price (IDR)", "Rp 30,000,000")
        
        # Enhanced intraday data for premium users
        if use_premium_data and API_CONFIG['ALPHA_VANTAGE_KEY']:
            st.subheader("📈 Intraday Gold Prices (Last 24 Hours)")
            
            with st.spinner("Loading intraday data..."):
                try:
                    intraday_data = gold_agent.get_intraday_gold_data()
                    
                    if intraday_data:
                        df_intraday = pd.DataFrame(intraday_data)
                        
                        fig_intraday = go.Figure()
                        fig_intraday.add_trace(go.Scatter(
                            x=df_intraday['timestamp'],
                            y=df_intraday['price'],
                            mode='lines+markers',
                            name='Gold Price (USD)',
                            line=dict(color='gold', width=2),
                            hovertemplate='<b>%{y:.2f} USD</b><br>%{x}<extra></extra>'
                        ))
                        
                        fig_intraday.update_layout(
                            title='Intraday Gold Price Movement',
                            xaxis_title='Time',
                            yaxis_title='Price (USD)',
                            template='plotly_dark',
                            height=400
                        )
                        
                        st.plotly_chart(fig_intraday, use_container_width=True)
                    else:
                        st.info("Intraday data not available. Check API limits.")
                except Exception as e:
                    st.warning(f"Could not load intraday data: {e}")
        
        # Standard historical charts
        st.subheader("📊 Historical Trends")
        try:
            historical_data = generate_historical_data(days_to_analyze)
            
            if historical_data:
                fig_gold, fig_rate, fig_gold_idr = create_charts(historical_data)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(fig_gold, use_container_width=True)
                with col2:
                    st.plotly_chart(fig_rate, use_container_width=True)
                
                st.plotly_chart(fig_gold_idr, use_container_width=True)
            else:
                st.warning("Unable to load historical data. Please check your internet connection.")
        except Exception as e:
            st.error(f"Error loading historical charts: {e}")
    
    with tab2:
        st.header("🔍 Enhanced Agent Analysis")
        # [Enhanced version of existing agent analysis with premium data integration]
        
        try:
            historical_data = generate_historical_data(days_to_analyze)
        except Exception as e:
            st.error(f"Error generating historical data: {e}")
            historical_data = []
        st.header("🏆 Premium Market Overview")
        
        # Enhanced real-time data with premium sources
        col1, col2, col3 = st.columns(3)
        
        with col1:
            with st.spinner("Fetching premium gold data..."):
                if use_premium_data and API_CONFIG['ALPHA_VANTAGE_KEY']:
                    gold_data = gold_agent.get_real_time_gold_data()
                    
                    # Show enhanced metrics if available
                    if 'bid_price' in gold_data:
                        st.metric(
                            "Gold Price (USD) - Premium", 
                            f"${gold_data['price_usd']:.2f}",
                            delta=f"{gold_data.get('change', 0):.2f} ({gold_data.get('change_percent', 0):.2f}%)"
                        )
                        st.caption(f"Bid: ${gold_data.get('bid_price', 0):.2f} | Ask: ${gold_data.get('ask_price', 0):.2f}")
                        st.caption(f"Source: {gold_data.get('source', 'unknown').replace('_', ' ').title()}")
                    else:
                        gold_data = gold_agent.get_real_time_gold_data()
                        st.metric(
                            "Gold Price (USD)", 
                            f"${gold_data['price_usd']:.2f}",
                            delta=f"{gold_data.get('change_percent', 0):.2f}%"
                        )
                else:
                    gold_data = gold_agent.get_real_time_gold_data()
                    st.metric(
                        "Gold Price (USD)", 
                        f"${gold_data['price_usd']:.2f}",
                        delta=f"{gold_data.get('change_percent', 0):.2f}%"
                    )
        
        with col2:
            with st.spinner("Fetching premium currency data..."):
                if use_premium_data and (API_CONFIG['ALPHA_VANTAGE_KEY'] or API_CONFIG['FIXER_KEY']):
                    currency_data = currency_agent.get_real_time_currency_data()
                    
                    if 'bid_price' in currency_data:
                        st.metric(
                            "USD/IDR Rate - Premium", 
                            f"{currency_data['rate']:.2f}",
                            delta=f"{currency_data.get('change_percent', 0):.2f}%"
                        )
                        st.caption(f"Bid: {currency_data.get('bid_price', 0):.2f} | Ask: {currency_data.get('ask_price', 0):.2f}")
                        st.caption(f"Source: {currency_data.get('source', 'unknown').replace('_', ' ').title()}")
                    else:
                        st.metric(
                            "USD/IDR Rate", 
                            f"{currency_data['rate']:.2f}",
                            delta=f"{currency_data.get('change_percent', 0):.2f}%"
                        )
                else:
                    currency_data = currency_agent.get_real_time_currency_data()
                    st.metric(
                        "USD/IDR Rate", 
                        f"{currency_data['rate']:.2f}",
                        delta=f"{currency_data.get('change_percent', 0):.2f}%"
                    )
        
        with col3:
            # Enhanced gold price in IDR calculation
            gold_price_idr = gold_data['price_usd'] * currency_data['rate']
            idr_change = 0
            if 'change_percent' in gold_data and 'change_percent' in currency_data:
                idr_change = gold_data.get('change_percent', 0) + currency_data.get('change_percent', 0)
            
            st.metric(
                "Gold Price (IDR)", 
                f"Rp {gold_price_idr:,.0f}",
                delta=f"{idr_change:.2f}%"
            )
            
            # Data quality indicator
            quality_score = 0
            if gold_data.get('source', '').endswith('premium'):
                quality_score += 1
            if currency_data.get('source', '').endswith('premium'):
                quality_score += 1
            
            quality_labels = ["🔴 Basic", "🟡 Enhanced", "🟢 Premium"]
            st.caption(f"Data Quality: {quality_labels[quality_score]}")
        
        # Enhanced intraday data for premium users
        if use_premium_data and API_CONFIG['ALPHA_VANTAGE_KEY']:
            st.subheader("📈 Intraday Gold Prices (Last 24 Hours)")
            
            with st.spinner("Loading intraday data..."):
                intraday_data = gold_agent.get_intraday_gold_data()
                
                if intraday_data:
                    df_intraday = pd.DataFrame(intraday_data)
                    
                    fig_intraday = go.Figure()
                    fig_intraday.add_trace(go.Scatter(
                        x=df_intraday['timestamp'],
                        y=df_intraday['price'],
                        mode='lines+markers',
                        name='Gold Price (USD)',
                        line=dict(color='gold', width=2),
                        hovertemplate='<b>%{y:.2f} USD</b><br>%{x}<extra></extra>'
                    ))
                    
                    fig_intraday.update_layout(
                        title='Intraday Gold Price Movement',
                        xaxis_title='Time',
                        yaxis_title='Price (USD)',
                        template='plotly_dark',
                        height=400
                    )
                    
                    st.plotly_chart(fig_intraday, use_container_width=True)
                else:
                    st.info("Intraday data not available. Check API limits.")
        
        # Standard historical charts
        st.subheader("📊 Historical Trends")
        historical_data = generate_historical_data(days_to_analyze)
        
        if historical_data:
            fig_gold, fig_rate, fig_gold_idr = create_charts(historical_data)
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_gold, use_container_width=True)
            with col2:
                st.plotly_chart(fig_rate, use_container_width=True)
            
            st.plotly_chart(fig_gold_idr, use_container_width=True)
    
    with tab4:
        st.header("🤖 AI Insights Pro with Qwen3")
        
        # Enhanced AI interface
        if use_qwen3_ai and API_CONFIG['OPENROUTER_API_KEY']:
            st.success("🧠 Qwen3 AI Analysis Enabled - Professional Insights Available")
            
            # Advanced query suggestions
            st.subheader("💬 Ask Qwen3 AI")
            
            advanced_suggestions = [
                "Analyze current gold market conditions and provide investment recommendations",
                "What economic factors are driving USD/IDR movements and how does this affect gold prices?",
                "Generate a risk assessment for gold investment in Indonesian market",
                "Compare current market sentiment with historical patterns for timing decisions",
                "What are the key correlations I should monitor for gold trading in IDR?",
                "Provide a technical analysis of gold price momentum and trend strength"
            ]
            
            col1, col2 = st.columns([3, 1])
            with col1:
                user_query = st.text_input(
                    "Ask Qwen3 for professional analysis:",
                    placeholder="e.g., Should I invest in gold now based on current market conditions?"
                )
            
            with col2:
                st.write("**🎯 Professional Queries:**")
                for i, suggestion in enumerate(advanced_suggestions[:3]):
                    if st.button(f"📊 Query {i+1}", key=f"adv_suggestion_{i}", help=suggestion):
                        user_query = suggestion
                        st.rerun()
        else:
            st.warning("🔴 Qwen3 AI not configured. Add OpenRouter API key for advanced analysis.")
            st.subheader("💬 Basic AI Analysis")
            
            user_query = st.text_input(
                "Ask for basic analysis:",
                placeholder="e.g., What's the trend for gold prices?"
            )
        
        # Generate insights
        historical_data = generate_historical_data(days_to_analyze)
        
        if user_query:
            with st.spinner("🧠 Generating AI insights..."):
                if use_qwen3_ai and API_CONFIG['OPENROUTER_API_KEY']:
                    insights = ai_agent.generate_insights(user_query, historical_data)
                    st.markdown(insights)
                    
                    # Additional analysis options
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("📊 Generate Follow-up Analysis"):
                            follow_up = f"Provide additional details and specific action items for: {user_query}"
                            follow_insights = ai_agent.generate_insights(follow_up, historical_data)
                            st.markdown("### 🔍 Follow-up Analysis")
                            st.markdown(follow_insights)
                    
                    with col2:
                        if st.button("⚠️ Generate Risk Assessment"):
                            risk_query = f"Analyze risks and mitigation strategies for: {user_query}"
                            risk_insights = ai_agent.generate_insights(risk_query, historical_data)
                            st.markdown("### ⚠️ Risk Analysis")
                            st.markdown(risk_insights)
                else:
                    insights = ai_agent.generate_insights(user_query, historical_data)
                    st.markdown(insights)
        
        # Enhanced Knowledge base with categories
        st.subheader("🧠 AI Knowledge Base")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Knowledge", len(retriever.knowledge_base))
        
        with col2:
            qwen3_entries = sum(1 for doc in retriever.knowledge_base if doc['metadata'].get('model') == 'qwen3')
            st.metric("Qwen3 Analyses", qwen3_entries)
        
        with col3:
            recent_entries = sum(1 for doc in retriever.knowledge_base if (datetime.now() - doc['timestamp']).days < 1)
            st.metric("Today's Insights", recent_entries)
    
    with tab5:
        st.header("📋 Professional Market Reports")
        
        if use_qwen3_ai and API_CONFIG['OPENROUTER_API_KEY']:
            st.success("🚀 Premium Report Generation Available")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📈 Generate Market Report", type="primary"):
                    with st.spinner("🧠 Qwen3 generating professional report..."):
                        historical_data = generate_historical_data(days_to_analyze)
                        economic_data = economic_agent.get_market_sentiment()
                        
                        report = ai_agent.generate_market_report(historical_data, economic_data)
                        st.markdown(report)
            
            with col2:
                if st.button("🎯 Generate Investment Strategy"):
                    with st.spinner("🧠 Creating investment strategy..."):
                        strategy_query = "Generate a comprehensive investment strategy for gold in Indonesian market considering current economic conditions, USD/IDR trends, and market sentiment. Include entry points, exit strategies, and risk management."
                        historical_data = generate_historical_data(days_to_analyze)
                        strategy = ai_agent.generate_insights(strategy_query, historical_data)
                        st.markdown("# 🎯 Investment Strategy")
                        st.markdown(strategy)
            
            # Report templates
            st.subheader("📊 Report Templates")
            
            templates = {
                "Daily Market Brief": "Provide a concise daily market brief covering gold prices, USD/IDR movement, key economic events, and trading recommendations for today.",
                "Weekly Outlook": "Generate a weekly market outlook analyzing trends, upcoming economic events, technical indicators, and strategic positioning for the week ahead.",
                "Risk Assessment": "Conduct a comprehensive risk assessment for gold investment including market risks, currency risks, economic risks, and mitigation strategies.",
                "Technical Analysis": "Perform detailed technical analysis of gold prices including trend analysis, support/resistance levels, momentum indicators, and chart patterns."
            }
            
            selected_template = st.selectbox("Choose Report Template:", list(templates.keys()))
            
            if st.button(f"📋 Generate {selected_template}"):
                with st.spinner(f"Generating {selected_template}..."):
                    historical_data = generate_historical_data(days_to_analyze)
                    template_report = ai_agent.generate_insights(templates[selected_template], historical_data)
                    st.markdown(f"# 📋 {selected_template}")
                    st.markdown(template_report)
        
        else:
            st.warning("🔴 Premium reports require Qwen3 AI. Add OpenRouter API key to unlock.")
            st.info("💡 With Qwen3, you can generate professional market reports, investment strategies, and comprehensive analysis.")
    
    # Enhanced remaining tabs (tab2, tab3, tab6) would follow similar pattern...
    # [Previous tab content with premium enhancements]
    
    with tab2:
        st.header("🔍 Enhanced Agent Analysis")
        # [Enhanced version of existing agent analysis with premium data integration]
        
        historical_data = generate_historical_data(days_to_analyze)
        
        if historical_data:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🥇 Premium Gold Analysis")
                gold_analysis = gold_agent.analyze_gold_trends(historical_data)
                if 'error' not in gold_analysis:
                    st.write(f"**Trend:** {gold_analysis['trend'].title()}")
                    st.write(f"**Volatility:** {gold_analysis['volatility']:.2f}")
                    st.write(f"**Average Change:** ${gold_analysis['avg_change']:.2f}")
                    st.write(f"**Price Range:** ${gold_analysis['price_range']['min']:.2f} - ${gold_analysis['price_range']['max']:.2f}")
                    
                    # Show premium data if available
                    if use_premium_data:
                        hist_gold = gold_agent.get_historical_gold_data(days_to_analyze)
                        if hist_gold:
                            st.success(f"✅ Real data: {len(hist_gold)} premium data points")
                        else:
                            st.info("📊 Using enhanced simulation")
            
            with col2:
                st.subheader("💱 Premium Currency Analysis")
                currency_analysis = currency_agent.analyze_currency_trends(historical_data)
                if 'error' not in currency_analysis:
                    st.write(f"**Trend:** USD {currency_analysis['trend'].title()}")
                    st.write(f"**Volatility:** {currency_analysis['volatility']:.2f}")
                    st.write(f"**Average Change:** {currency_analysis['avg_change']:.2f}")
                    st.write(f"**Rate Range:** {currency_analysis['rate_range']['min']:.2f} - {currency_analysis['rate_range']['max']:.2f}")
                    
                    # Show premium currency data
                    if use_premium_data:
                        hist_currency = currency_agent.get_currency_historical_premium(days_to_analyze)
                        if hist_currency:
                            st.success(f"✅ Real data: {len(hist_currency)} premium data points")
                        else:
                            st.info("📊 Using enhanced simulation")
    
    with tab3:
        st.header("📈 Premium Economic Indicators")
        
        # Enhanced Market Sentiment with premium data
        st.subheader("🎯 Advanced Market Sentiment")
        sentiment_data = economic_agent.get_market_sentiment()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            fear_greed = sentiment_data.get('fear_greed_index', 50)
            st.metric("Fear & Greed Index", fear_greed)
            
            # Enhanced interpretation with premium data
            sentiment_label = sentiment_data.get('market_sentiment', 'neutral').replace('_', ' ').title()
            if use_premium_data and API_CONFIG['OPENROUTER_API_KEY']:
                sentiment_color = {"Extreme Fear": "🔴", "Fear": "🟡", "Neutral": "⚪", "Greed": "🟡", "Extreme Greed": "🔴"}
                st.caption(f"{sentiment_color.get(sentiment_label, '⚪')} {sentiment_label}")
            else:
                st.caption(f"Sentiment: {sentiment_label}")
        
        with col2:
            dxy = sentiment_data.get('dxy_index', 102.5)
            dxy_change = sentiment_data.get('dxy_change', 0)
            st.metric("Dollar Index (DXY)", f"{dxy:.2f}", delta=f"{dxy_change:.2f}%")
            
            dollar_strength = sentiment_data.get('dollar_strength', 'neutral').title()
            strength_emoji = {"Strong": "💪", "Weak": "📉", "Neutral": "➡️"}
            st.caption(f"{strength_emoji.get(dollar_strength, '➡️')} Dollar: {dollar_strength}")
        
        with col3:
            vix = sentiment_data.get('vix_index', 20.0)
            st.metric("VIX (Volatility)", f"{vix:.2f}")
            
            volatility_level = sentiment_data.get('market_volatility', 'moderate').title()
            vix_emoji = {"Low": "🟢", "Moderate": "🟡", "High": "🟠", "Extreme": "🔴"}
            st.caption(f"{vix_emoji.get(volatility_level, '🟡')} {volatility_level} Volatility")
        
        with col4:
            # Premium: Add Commodity Index if available
            if use_premium_data and API_CONFIG['ALPHA_VANTAGE_KEY']:
                st.metric("Premium Data", "🟢 Active")
                st.caption("Enhanced accuracy enabled")
            else:
                st.metric("Data Mode", "🔴 Basic")
                st.caption("Add API keys for premium")
        
        # Enhanced Commodity Correlations
        st.subheader("🥈 Advanced Commodity Analysis")
        correlations = economic_agent.get_commodity_correlations()
        
        if correlations:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                gold_silver_corr = correlations.get('gold_silver', 0.8)
                st.metric("Gold vs Silver", f"{gold_silver_corr:.3f}")
                
                # Enhanced interpretation
                if abs(gold_silver_corr) > 0.7:
                    st.caption("🔗 Strong correlation - Consider paired trades")
                elif abs(gold_silver_corr) > 0.3:
                    st.caption("📊 Moderate correlation - Monitor divergence")
                else:
                    st.caption("📉 Weak correlation - Independent movements")
            
            with col2:
                gold_oil_corr = correlations.get('gold_oil', 0.3)
                st.metric("Gold vs Oil", f"{gold_oil_corr:.3f}")
                
                if gold_oil_corr > 0.5:
                    st.caption("⛽ Inflation hedge correlation strong")
                elif gold_oil_corr > 0:
                    st.caption("📈 Positive correlation - Risk-on sentiment")
                else:
                    st.caption("📉 Negative correlation - Safe haven demand")
            
            with col3:
                # Calculate Gold vs Dollar correlation if premium data available
                if use_premium_data:
                    historical_data = generate_historical_data(30)
                    if len(historical_data) > 10:
                        gold_prices = [d.gold_price_usd for d in historical_data]
                        usd_strength = [1/d.usd_idr_rate for d in historical_data]  # Inverted for USD strength
                        gold_usd_corr = np.corrcoef(gold_prices, usd_strength)[0, 1] if len(gold_prices) > 1 else 0
                        
                        st.metric("Gold vs USD Strength", f"{gold_usd_corr:.3f}")
                        
                        if gold_usd_corr < -0.5:
                            st.caption("💵 Strong negative correlation - Classic pattern")
                        elif gold_usd_corr < 0:
                            st.caption("📊 Negative correlation - Normal relationship")
                        else:
                            st.caption("⚠️ Positive correlation - Unusual market conditions")
                else:
                    st.metric("Premium Feature", "🔒 Locked")
                    st.caption("Add Alpha Vantage key to unlock")
        
        # Premium: Economic Calendar Integration
        if use_premium_data and API_CONFIG['FINNHUB_KEY']:
            st.subheader("📅 Economic Calendar (Premium)")
            
            # Simulate economic events (in real implementation, fetch from Finnhub)
            economic_events = [
                {"time": "09:30", "event": "US GDP Preliminary", "impact": "High", "forecast": "2.8%"},
                {"time": "14:00", "event": "Fed Interest Rate Decision", "impact": "High", "forecast": "5.25-5.50%"},
                {"time": "16:30", "event": "US Dollar Index Update", "impact": "Medium", "forecast": "102.5"},
            ]
            
            for event in economic_events:
                col1, col2, col3, col4 = st.columns([1, 3, 1, 1])
                
                with col1:
                    st.write(event["time"])
                with col2:
                    st.write(event["event"])
                with col3:
                    impact_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
                    st.write(f"{impact_color.get(event['impact'], '⚪')} {event['impact']}")
                with col4:
                    st.write(event["forecast"])
        else:
            st.subheader("📅 Economic Calendar")
            st.info("🔑 Add Finnhub API key to view upcoming economic events that may impact gold prices")
    
    with tab6:
        st.header("📉 Advanced Analytics Pro")
        
        historical_data = generate_historical_data(days_to_analyze)
        
        if historical_data:
            df = pd.DataFrame([
                {
                    'Date': data.timestamp,
                    'Gold_USD': data.gold_price_usd,
                    'USD_IDR': data.usd_idr_rate,
                    'Gold_IDR': data.gold_price_idr
                }
                for data in historical_data
            ])
            
            # Enhanced Correlation Analysis
            st.subheader("📊 Advanced Correlation Matrix")
            
            # Add more sophisticated metrics for premium users
            if use_premium_data:
                # Calculate additional metrics
                df['Gold_Returns'] = df['Gold_USD'].pct_change()
                df['USD_IDR_Returns'] = df['USD_IDR'].pct_change()
                df['Gold_IDR_Returns'] = df['Gold_IDR'].pct_change()
                df['Gold_Volatility'] = df['Gold_Returns'].rolling(window=7).std()
                df['USD_Volatility'] = df['USD_IDR_Returns'].rolling(window=7).std()
                
                correlation_cols = ['Gold_USD', 'USD_IDR', 'Gold_IDR', 'Gold_Returns', 'USD_IDR_Returns', 'Gold_Volatility', 'USD_Volatility']
                corr_matrix = df[correlation_cols].corr()
                
                st.success("🟢 Premium Analytics: Enhanced correlation analysis with volatility and returns")
            else:
                correlation_cols = ['Gold_USD', 'USD_IDR', 'Gold_IDR']
                corr_matrix = df[correlation_cols].corr()
            
            fig_corr = px.imshow(
                corr_matrix,
                title="Enhanced Asset Correlation Matrix",
                color_continuous_scale='RdBu_r',
                aspect='auto',
                text_auto=True,
                zmin=-1, zmax=1
            )
            fig_corr.update_layout(height=500)
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Advanced Statistical Analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Performance Metrics")
                
                # Basic metrics
                metrics_data = {}
                for col in ['Gold_USD', 'USD_IDR', 'Gold_IDR']:
                    returns = df[col].pct_change().dropna()
                    if len(returns) > 0:
                        metrics_data[col] = {
                            'Mean Return': f"{returns.mean():.4f}",
                            'Volatility (Daily)': f"{returns.std():.4f}",
                            'Volatility (Annualized)': f"{returns.std() * np.sqrt(252):.2%}",
                            'Sharpe Ratio (Est.)': f"{returns.mean() / returns.std() if returns.std() != 0 else 0:.3f}",
                            'Max Drawdown': f"{(df[col] / df[col].expanding().max() - 1).min():.2%}"
                        }
                
                metrics_df = pd.DataFrame(metrics_data).T
                st.dataframe(metrics_df, use_container_width=True)
                
                if use_premium_data:
                    st.caption("✅ Premium: Enhanced risk metrics with Sharpe ratio and drawdown analysis")
            
            with col2:
                st.subheader("🎯 Trading Signals")
                
                # Generate trading signals for premium users
                if use_premium_data and len(df) > 20:
                    # Calculate moving averages
                    df['MA_5'] = df['Gold_USD'].rolling(window=5).mean()
                    df['MA_20'] = df['Gold_USD'].rolling(window=20).mean()
                    
                    # Generate signals
                    current_price = df['Gold_USD'].iloc[-1]
                    ma_5 = df['MA_5'].iloc[-1]
                    ma_20 = df['MA_20'].iloc[-1]
                    
                    # Trend analysis
                    if ma_5 > ma_20 and current_price > ma_5:
                        trend_signal = "🟢 BULLISH"
                        trend_desc = "Price above moving averages"
                    elif ma_5 < ma_20 and current_price < ma_5:
                        trend_signal = "🔴 BEARISH"  
                        trend_desc = "Price below moving averages"
                    else:
                        trend_signal = "🟡 NEUTRAL"
                        trend_desc = "Mixed signals"
                    
                    st.metric("Trend Signal", trend_signal)
                    st.caption(trend_desc)
                    
                    # Support/Resistance levels
                    recent_high = df['Gold_USD'].rolling(window=10).max().iloc[-1]
                    recent_low = df['Gold_USD'].rolling(window=10).min().iloc[-1]
                    
                    st.write("**Key Levels:**")
                    st.write(f"Resistance: ${recent_high:.2f}")
                    st.write(f"Support: ${recent_low:.2f}")
                    st.write(f"Range: {((recent_high - recent_low) / recent_low * 100):.1f}%")
                    
                    # Volume analysis (if available)
                    if use_qwen3_ai and API_CONFIG['OPENROUTER_API_KEY']:
                        if st.button("🧠 Generate Trading Strategy"):
                            strategy_query = f"""
                            Based on current gold price ${current_price:.2f}, 
                            trend signal {trend_signal}, 
                            resistance at ${recent_high:.2f}, 
                            support at ${recent_low:.2f},
                            generate a specific trading strategy with entry points, stop losses, and targets.
                            """
                            
                            with st.spinner("Qwen3 generating trading strategy..."):
                                strategy = ai_agent.generate_insights(strategy_query, historical_data)
                                st.markdown("### 🎯 AI Trading Strategy")
                                st.markdown(strategy)
                    
                else:
                    st.info("🔑 Premium trading signals available with API keys")
                    st.write("**Available with premium:**")
                    st.write("• Moving average signals")
                    st.write("• Support/resistance levels")  
                    st.write("• AI trading strategies")
                    st.write("• Volume analysis")
            
            # Premium: Advanced Charting
            if use_premium_data:
                st.subheader("📈 Advanced Technical Analysis")
                
                # Create advanced chart with indicators
                fig_advanced = go.Figure()
                
                # Candlestick-style chart (using OHLC simulation)
                fig_advanced.add_trace(go.Scatter(
                    x=df['Date'],
                    y=df['Gold_USD'],
                    mode='lines',
                    name='Gold Price',
                    line=dict(color='gold', width=2)
                ))
                
                # Add moving averages
                if 'MA_5' in df.columns:
                    fig_advanced.add_trace(go.Scatter(
                        x=df['Date'],
                        y=df['MA_5'],
                        mode='lines',
                        name='MA 5',
                        line=dict(color='blue', width=1, dash='dash')
                    ))
                
                if 'MA_20' in df.columns:
                    fig_advanced.add_trace(go.Scatter(
                        x=df['Date'],
                        y=df['MA_20'],
                        mode='lines',
                        name='MA 20',
                        line=dict(color='red', width=1, dash='dash')
                    ))
                
                # Add Bollinger Bands
                if len(df) > 20:
                    bb_period = 20
                    bb_std = 2
                    df['BB_Middle'] = df['Gold_USD'].rolling(window=bb_period).mean()
                    df['BB_Upper'] = df['BB_Middle'] + (df['Gold_USD'].rolling(window=bb_period).std() * bb_std)
                    df['BB_Lower'] = df['BB_Middle'] - (df['Gold_USD'].rolling(window=bb_period).std() * bb_std)
                    
                    fig_advanced.add_trace(go.Scatter(
                        x=df['Date'],
                        y=df['BB_Upper'],
                        mode='lines',
                        name='BB Upper',
                        line=dict(color='gray', width=1),
                        fill=None
                    ))
                    
                    fig_advanced.add_trace(go.Scatter(
                        x=df['Date'],
                        y=df['BB_Lower'],
                        mode='lines',
                        name='BB Lower',
                        line=dict(color='gray', width=1),
                        fill='tonexty',
                        fillcolor='rgba(128,128,128,0.1)'
                    ))
                
                fig_advanced.update_layout(
                    title='Advanced Technical Analysis - Gold Price',
                    xaxis_title='Date',
                    yaxis_title='Price (USD)',
                    template='plotly_dark',
                    height=500
                )
                
                st.plotly_chart(fig_advanced, use_container_width=True)
                st.caption("✅ Premium: Technical indicators include moving averages and Bollinger Bands")
            
            # Enhanced Export Options
            st.subheader("📁 Enhanced Export Options")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Basic CSV export
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="📊 Download Basic CSV",
                    data=csv_data,
                    file_name=f"gold_analysis_basic_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Enhanced CSV with indicators
                if use_premium_data and 'MA_5' in df.columns:
                    enhanced_csv = df.to_csv(index=False)
                    st.download_button(
                        label="🚀 Download Premium CSV",
                        data=enhanced_csv,
                        file_name=f"gold_analysis_premium_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("Premium CSV requires API keys")
            
            with col3:
                # JSON export with metadata
                export_data = {
                    'metadata': {
                        'generated_at': datetime.now().isoformat(),
                        'days_analyzed': days_to_analyze,
                        'premium_features': use_premium_data,
                        'ai_enabled': use_qwen3_ai,
                        'data_sources': ['yahoo_finance'] + 
                                      (['alpha_vantage'] if API_CONFIG['ALPHA_VANTAGE_KEY'] else []) +
                                      (['finnhub'] if API_CONFIG['FINNHUB_KEY'] else []) +
                                      (['fixer_io'] if API_CONFIG['FIXER_KEY'] else []),
                        'total_records': len(df),
                        'analysis_version': '2.0_premium'
                    },
                    'summary_statistics': df.describe().to_dict(),
                    'correlation_matrix': corr_matrix.to_dict(),
                    'data': df.to_dict('records')
                }
                
                json_str = json.dumps(export_data, indent=2, default=str)
                st.download_button(
                    label="📋 Download Full JSON",
                    data=json_str,
                    file_name=f"gold_analysis_full_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json"
                )
        
        else:
            st.error("❌ No data available for advanced analytics. Check your internet connection.")
            
        # API Usage Summary for premium users
        if use_premium_data:
            st.subheader("📊 API Usage Summary")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if API_CONFIG['ALPHA_VANTAGE_KEY']:
                    st.metric("Alpha Vantage", "🟢 Active")
                    st.caption("Professional data quality")
                else:
                    st.metric("Alpha Vantage", "🔴 Not configured")
            
            with col2:
                if API_CONFIG['OPENROUTER_API_KEY']:
                    st.metric("OpenRouter Qwen3", "🟢 Active")
                    st.caption("Advanced AI analysis")
                else:
                    st.metric("OpenRouter Qwen3", "🔴 Not configured")
            
            with col3:
                apis_active = sum([
                    bool(API_CONFIG['ALPHA_VANTAGE_KEY']),
                    bool(API_CONFIG['FIXER_KEY']),
                    bool(API_CONFIG['FINNHUB_KEY']),
                    bool(API_CONFIG['OPENROUTER_API_KEY'])
                ])
                st.metric("APIs Configured", f"{apis_active}/4")
                
                if apis_active == 4:
                    st.caption("🏆 Full premium access")
                elif apis_active >= 2:
                    st.caption("🟡 Partial premium access")
                else:
                    st.caption("🔴 Basic access only")

# Footer with disclaimer
    st.markdown("---")
    
    # Footer content in columns
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.markdown(f"""
        ### 🏆 {APP_NAME}
        **Version:** {APP_VERSION}  
        **Created by:** {APP_CREATOR}  
        **Updated:** {datetime.now().strftime('%Y-%m-%d')}
        """)
    
    with col2:
        st.markdown("""
        ### ⚠️ Disclaimer
        **Investment Risk Notice:** This application provides analytical tools and information for educational purposes only. All investment decisions carry inherent risks, and past performance does not guarantee future results.
        
        **Data Sources:** Real-time data depends on API availability. Always verify critical information from official sources before making investment decisions.
        """)
    
    with col3:
        st.markdown("""
        ### 🔗 Quick Links
        - [Alpha Vantage](https://www.alphavantage.co/)
        - [OpenRouter](https://openrouter.ai/)
        - [Fixer.io](https://fixer.io/)
        - [Finnhub](https://finnhub.io/)
        """)
    
    # Technical info
    st.markdown("---")
    st.caption(f"""
    🔧 **Technical Info:** Running Streamlit {st.__version__} | 
    📊 **Data Sources:** Alpha Vantage, OpenRouter Qwen3, Fixer.io, Finnhub, Yahoo Finance | 
    🧠 **AI Engine:** RAG (Retrieval Augmented Generation) with Multi-Agent Architecture | 
    💾 **Cache Status:** {len(st.session_state) if hasattr(st, 'session_state') else 0} items | 
    ⏰ **Session:** {datetime.now().strftime('%H:%M:%S')} | 
    👨‍💻 **Developer:** {APP_CREATOR}
    """)

if __name__ == "__main__":
    main()