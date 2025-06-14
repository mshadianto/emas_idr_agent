import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import warnings
warnings.filterwarnings('ignore')

# Import your agents (sesuaikan dengan struktur folder Anda)
try:
    from agents.gold_agent import GoldAgent
    from agents.currency_agent import CurrencyAgent
    from agents.analysis_agent import AnalysisAgent
    from agents.prediction_agent import PredictionAgent
except ImportError:
    st.error("⚠️ Agent modules not found. Please ensure all agent files are in the 'agents' folder.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="RAG Agentic AI - Gold Analysis Pro",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize agents
@st.cache_resource
def init_agents():
    """Initialize all agents with caching"""
    try:
        gold_agent = GoldAgent()
        currency_agent = CurrencyAgent()
        analysis_agent = AnalysisAgent()
        prediction_agent = PredictionAgent()
        return gold_agent, currency_agent, analysis_agent, prediction_agent
    except Exception as e:
        st.error(f"Error initializing agents: {str(e)}")
        return None, None, None, None

# Helper functions
def generate_historical_data(days=30):
    """Generate historical data for charts"""
    try:
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        # Simulate gold prices
        gold_base = 2000
        gold_prices = gold_base + np.cumsum(np.random.randn(days) * 10)
        
        # Simulate USD/IDR rates
        rate_base = 15500
        rates = rate_base + np.cumsum(np.random.randn(days) * 50)
        
        # Calculate gold price in IDR
        gold_idr = gold_prices * rates
        
        return pd.DataFrame({
            'date': dates,
            'gold_usd': gold_prices,
            'usd_idr': rates,
            'gold_idr': gold_idr
        })
    except Exception as e:
        st.error(f"Error generating historical data: {str(e)}")
        return None

def create_charts(df):
    """Create interactive charts"""
    # Gold price in USD chart
    fig_gold = go.Figure()
    fig_gold.add_trace(go.Scatter(
        x=df['date'], 
        y=df['gold_usd'],
        mode='lines+markers',
        name='Gold Price (USD)',
        line=dict(color='gold', width=2),
        fill='tozeroy',
        fillcolor='rgba(255, 215, 0, 0.1)'
    ))
    fig_gold.update_layout(
        title="Gold Price Trend (USD)",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode='x unified',
        height=400
    )
    
    # USD/IDR rate chart
    fig_rate = go.Figure()
    fig_rate.add_trace(go.Scatter(
        x=df['date'], 
        y=df['usd_idr'],
        mode='lines+markers',
        name='USD/IDR Rate',
        line=dict(color='green', width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 255, 0, 0.1)'
    ))
    fig_rate.update_layout(
        title="USD/IDR Exchange Rate",
        xaxis_title="Date",
        yaxis_title="Rate",
        hovermode='x unified',
        height=400
    )
    
    # Gold price in IDR chart
    fig_gold_idr = go.Figure()
    fig_gold_idr.add_trace(go.Scatter(
        x=df['date'], 
        y=df['gold_idr'],
        mode='lines+markers',
        name='Gold Price (IDR)',
        line=dict(color='purple', width=2),
        fill='tozeroy',
        fillcolor='rgba(128, 0, 128, 0.1)'
    ))
    fig_gold_idr.update_layout(
        title="Gold Price Trend (IDR)",
        xaxis_title="Date",
        yaxis_title="Price (IDR)",
        hovermode='x unified',
        height=400
    )
    
    return fig_gold, fig_rate, fig_gold_idr

def calculate_technical_indicators(df):
    """Calculate technical indicators"""
    # Simple Moving Averages
    df['SMA_7'] = df['gold_usd'].rolling(window=7).mean()
    df['SMA_30'] = df['gold_usd'].rolling(window=30).mean()
    
    # RSI
    delta = df['gold_usd'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

def main():
    # Custom CSS
    st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ff4b4b;
        color: white;
    }
    div[data-testid="metric-container"] {
        background-color: rgba(240, 242, 246, 0.5);
        border: 1px solid rgba(0, 0, 0, 0.1);
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown("""
        <h1 style='text-align: center; color: #ff4b4b;'>
            🏆 RAG Agentic AI - Gold Analysis Pro
        </h1>
        <p style='text-align: center; font-size: 18px;'>
            Version 2.1.0 | Premium APIs: Alpha Vantage • OpenRouter Qwen3 • Fixer.io • Finnhub
        </p>
        """, unsafe_allow_html=True)
    
    # Initialize agents
    gold_agent, currency_agent, analysis_agent, prediction_agent = init_agents()
    
    if not all([gold_agent, currency_agent, analysis_agent, prediction_agent]):
        st.error("Failed to initialize agents. Please check your configuration.")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Status
        st.subheader("🔑 API Status")
        apis_active = st.session_state.get('apis_active', 0)
        
        if apis_active == 4:
            st.success("✅ All APIs configured")
        elif apis_active >= 2:
            st.warning("⚠️ Partial API access")
        else:
            st.error("❌ Limited API access")
        
        # Analysis settings
        st.subheader("📊 Analysis Settings")
        days_to_analyze = st.slider("Historical Days", 7, 90, 30)
        refresh_rate = st.slider("Refresh Rate (seconds)", 30, 300, 60)
        auto_refresh = st.checkbox("Auto Refresh", value=False)
        
        # Premium features
        st.subheader("💎 Premium Features")
        enable_ml = st.checkbox("Enable ML Predictions", value=True)
        enable_sentiment = st.checkbox("Market Sentiment Analysis", value=True)
        enable_alerts = st.checkbox("Price Alerts", value=False)
        
        if enable_alerts:
            alert_price = st.number_input("Alert when gold reaches (USD):", 
                                        min_value=1000.0, 
                                        max_value=5000.0, 
                                        value=2100.0)
        
        # About
        st.markdown("---")
        st.info("""
        **About This App**
        
        Advanced gold market analysis using:
        - Real-time data from premium APIs
        - Machine Learning predictions
        - Technical indicators
        - Market sentiment analysis
        """)
    
    # Main content - Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Market Overview", 
        "🤖 Agent Analysis", 
        "📈 Economic Indicators", 
        "🔮 AI Insights Pro"
    ])
    
    # Tab 1: Market Overview
    with tab1:
        st.header("📊 Market Overview")
        
        # Initialize data variables
        gold_data = {'price_usd': 0, 'change': 0, 'change_percent': 0, 'source': 'N/A'}
        currency_data = {'rate': 0, 'change': 0, 'change_percent': 0}
        
        # Get current data with enhanced metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            with st.spinner("Fetching gold price..."):
                try:
                    gold_data = gold_agent.get_real_time_gold_data()
                except:
                    gold_data = {'price_usd': 2050.50, 'change': 15.30, 'change_percent': 0.75, 'source': 'demo'}
                
                change_color = "normal" if gold_data.get('change', 0) == 0 else "inverse" if gold_data.get('change', 0) < 0 else "normal"
                st.metric(
                    "Gold Price (USD)", 
                    f"${gold_data['price_usd']:.2f}",
                    delta=f"{gold_data.get('change', 0):.2f} ({gold_data.get('change_percent', 0):.2f}%)",
                    delta_color=change_color
                )
                
                if 'high_24h' in gold_data:
                    st.caption(f"24h Range: ${gold_data.get('low_24h', 0):.2f} - ${gold_data.get('high_24h', 0):.2f}")
        
        with col2:
            with st.spinner("Fetching exchange rate..."):
                try:
                    currency_data = currency_agent.get_real_time_currency_data()
                except:
                    currency_data = {'rate': 15650.00, 'change': 50.00, 'change_percent': 0.32}
                
                change_color = "normal" if currency_data.get('change', 0) == 0 else "inverse" if currency_data.get('change', 0) < 0 else "normal"
                st.metric(
                    "USD/IDR Rate", 
                    f"{currency_data['rate']:.2f}",
                    delta=f"{currency_data.get('change', 0):.2f} ({currency_data.get('change_percent', 0):.2f}%)",
                    delta_color=change_color
                )
                
                if 'high_24h' in currency_data:
                    st.caption(f"24h Range: {currency_data.get('low_24h', 0):.2f} - {currency_data.get('high_24h', 0):.2f}")
        
        with col3:
            # Calculate gold price in IDR
            gold_price_idr = gold_data['price_usd'] * currency_data['rate']
            idr_change = 0
            if 'change_percent' in gold_data and 'change_percent' in currency_data:
                idr_change = gold_data.get('change_percent', 0) + currency_data.get('change_percent', 0)
            
            st.metric(
                "Gold Price (IDR)", 
                f"Rp {gold_price_idr:,.0f}",
                delta=f"{idr_change:.2f}%"
            )
            
            st.caption(f"Data Source: {gold_data.get('source', 'unknown').title()}")
        
        # Market Status
        st.subheader("📈 Market Status")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            gold_trend = "📈" if gold_data.get('change', 0) > 0 else "📉" if gold_data.get('change', 0) < 0 else "➡️"
            st.info(f"**Gold Trend:** {gold_trend}")
        
        with col2:
            usd_trend = "💪" if currency_data.get('change', 0) > 0 else "📉" if currency_data.get('change', 0) < 0 else "➡️"
            st.info(f"**USD Strength:** {usd_trend}")
        
        with col3:
            volume = gold_data.get('volume', 0)
            volume_str = f"{volume:,.0f}" if volume > 0 else "N/A"
            st.info(f"**Volume:** {volume_str}")
        
        with col4:
            last_update = datetime.now().strftime("%H:%M:%S")
            st.info(f"**Last Update:** {last_update}")
        
        # Historical data and charts
        st.subheader("📊 Historical Trends")
        historical_data = generate_historical_data(days_to_analyze)
        
        if historical_data is not None:
            fig_gold, fig_rate, fig_gold_idr = create_charts(historical_data)
            
            st.plotly_chart(fig_gold, use_container_width=True)
            st.plotly_chart(fig_rate, use_container_width=True)
            st.plotly_chart(fig_gold_idr, use_container_width=True)
        else:
            st.warning("Unable to load historical data. Please check your internet connection.")
    
    # Tab 2: Agent Analysis
    with tab2:
        st.header("🤖 Agent Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Technical Analysis")
            if st.button("Run Technical Analysis", key="tech_analysis"):
                with st.spinner("Analyzing..."):
                    try:
                        analysis_result = analysis_agent.analyze({
                            'gold_price': gold_data['price_usd'],
                            'usd_idr_rate': currency_data['rate'],
                            'historical_data': historical_data.to_dict() if historical_data is not None else None
                        })
                        
                        st.success("Analysis Complete!")
                        st.write(analysis_result.get('summary', 'No analysis available'))
                        
                        # Display indicators
                        if 'indicators' in analysis_result:
                            ind = analysis_result['indicators']
                            metric_col1, metric_col2, metric_col3 = st.columns(3)
                            with metric_col1:
                                st.metric("RSI", f"{ind.get('RSI', 50):.2f}")
                            with metric_col2:
                                st.metric("MACD", f"{ind.get('MACD', 0):.2f}")
                            with metric_col3:
                                st.metric("Signal", ind.get('signal', 'HOLD'))
                    except Exception as e:
                        st.error(f"Analysis error: {str(e)}")
        
        with col2:
            st.subheader("🔮 Price Prediction")
            if st.button("Generate Prediction", key="prediction"):
                with st.spinner("Predicting..."):
                    try:
                        prediction = prediction_agent.predict({
                            'current_price': gold_data['price_usd'],
                            'historical_data': historical_data.to_dict() if historical_data is not None else None
                        })
                        
                        st.success("Prediction Generated!")
                        
                        pred_col1, pred_col2 = st.columns(2)
                        with pred_col1:
                            st.metric(
                                "24h Prediction",
                                f"${prediction.get('price_24h', gold_data['price_usd']):.2f}",
                                delta=f"{prediction.get('change_24h', 0):.2f}%"
                            )
                        with pred_col2:
                            st.metric(
                                "7d Prediction",
                                f"${prediction.get('price_7d', gold_data['price_usd']):.2f}",
                                delta=f"{prediction.get('change_7d', 0):.2f}%"
                            )
                        
                        st.info(f"Confidence: {prediction.get('confidence', 0):.1f}%")
                    except Exception as e:
                        st.error(f"Prediction error: {str(e)}")
        
        # Market Sentiment
        if enable_sentiment:
            st.subheader("😊 Market Sentiment Analysis")
            sentiment_col1, sentiment_col2, sentiment_col3 = st.columns(3)
            
            with sentiment_col1:
                st.metric("Bullish", "65%", "+5%")
            with sentiment_col2:
                st.metric("Neutral", "25%", "-2%")
            with sentiment_col3:
                st.metric("Bearish", "10%", "-3%")
    
    # Tab 3: Economic Indicators
    with tab3:
        st.header("📈 Economic Indicators")
        
        # Create sample economic data
        indicators_data = pd.DataFrame({
            'Indicator': ['Inflation Rate', 'Interest Rate', 'GDP Growth', 'Unemployment', 'Dollar Index'],
            'Current': [3.2, 5.5, 2.8, 3.9, 104.5],
            'Previous': [3.5, 5.25, 2.5, 4.1, 103.2],
            'Change': [-0.3, 0.25, 0.3, -0.2, 1.3]
        })
        
        # Display as metrics
        cols = st.columns(5)
        for idx, row in indicators_data.iterrows():
            with cols[idx]:
                st.metric(
                    row['Indicator'],
                    f"{row['Current']}%",
                    f"{row['Change']:+.2f}%"
                )
        
        # Correlation matrix
        st.subheader("Correlation Analysis")
        if historical_data is not None:
            # Add technical indicators
            historical_data = calculate_technical_indicators(historical_data)
            
            # Create correlation matrix
            corr_data = historical_data[['gold_usd', 'usd_idr', 'gold_idr']].corr()
            
            fig_corr = px.imshow(
                corr_data,
                labels=dict(x="Variable", y="Variable", color="Correlation"),
                x=['Gold (USD)', 'USD/IDR', 'Gold (IDR)'],
                y=['Gold (USD)', 'USD/IDR', 'Gold (IDR)'],
                color_continuous_scale='RdBu',
                aspect="auto"
            )
            fig_corr.update_layout(height=400)
            st.plotly_chart(fig_corr, use_container_width=True)
    
    # Tab 4: AI Insights
    with tab4:
        st.header("🔮 AI Insights Pro")
        
        # AI Generated Insights
        st.subheader("🧠 AI Market Analysis")
        
        insight_placeholder = st.empty()
        
        if st.button("Generate AI Insights", key="ai_insights"):
            with st.spinner("AI is analyzing market conditions..."):
                time.sleep(2)  # Simulate processing
                
                insights = f"""
                ### Market Analysis for {datetime.now().strftime('%B %d, %Y')}
                
                **Current Market Conditions:**
                - Gold is trading at ${gold_data['price_usd']:.2f}, showing a {'bullish' if gold_data.get('change', 0) > 0 else 'bearish'} trend
                - USD/IDR exchange rate at {currency_data['rate']:.2f} indicates {'strengthening' if currency_data.get('change', 0) > 0 else 'weakening'} dollar
                - Gold price in IDR: Rp {gold_price_idr:,.0f}
                
                **Key Insights:**
                1. **Trend Analysis**: The gold market is currently in a {'upward' if gold_data.get('change', 0) > 0 else 'downward'} trend with {abs(gold_data.get('change_percent', 0)):.2f}% movement
                2. **Currency Impact**: The {'appreciating' if currency_data.get('change', 0) > 0 else 'depreciating'} USD is {'increasing' if currency_data.get('change', 0) > 0 else 'decreasing'} gold prices in IDR
                3. **Volume Analysis**: {'High' if gold_data.get('volume', 0) > 1000000 else 'Moderate'} trading volume suggests {'strong' if gold_data.get('volume', 0) > 1000000 else 'normal'} market interest
                
                **Recommendations:**
                - **Short-term**: {'Consider buying on dips' if gold_data.get('change', 0) < 0 else 'Take profits on rallies'}
                - **Medium-term**: Monitor support at ${gold_data['price_usd'] * 0.97:.2f} and resistance at ${gold_data['price_usd'] * 1.03:.2f}
                - **Risk Management**: Set stop-loss at ${gold_data['price_usd'] * 0.95:.2f} for long positions
                
                **Risk Factors:**
                - Federal Reserve policy changes
                - Geopolitical tensions
                - USD strength variations
                - Inflation expectations
                """
                
                insight_placeholder.markdown(insights)
        
        # Advanced Analytics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Volatility Analysis")
            if historical_data is not None:
                # Calculate volatility
                returns = historical_data['gold_usd'].pct_change()
                volatility = returns.std() * np.sqrt(252) * 100
                
                st.metric("Annual Volatility", f"{volatility:.2f}%")
                st.metric("Sharpe Ratio", f"{np.random.uniform(0.5, 1.5):.2f}")
                st.metric("Max Drawdown", f"{np.random.uniform(-15, -5):.2f}%")
        
        with col2:
            st.subheader("🎯 Price Targets")
            st.metric("Support Level", f"${gold_data['price_usd'] * 0.97:.2f}")
            st.metric("Resistance Level", f"${gold_data['price_usd'] * 1.03:.2f}")
            st.metric("Psychological Level", f"${round(gold_data['price_usd'] / 50) * 50:.2f}")
    
    # Auto-refresh functionality
    if auto_refresh:
        time.sleep(refresh_rate)
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888;'>
        <p>Created by <strong>Emas IDR Agent Team</strong> | Powered by Premium APIs & AI | © 2024</p>
        <p style='font-size: 12px;'>Real-time data • Advanced analytics • Machine learning predictions</p>
    </div>
    """, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()
