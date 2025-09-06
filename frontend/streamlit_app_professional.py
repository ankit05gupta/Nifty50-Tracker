#!/usr/bin/env python3
"""
Nifty 50 Professional Analytics Dashboard

A comprehensive financial analytics platform for NSE Nifty 50 stocks featuring:
- Real-time market data with professional UI/UX
- AI-powered stock analysis and predictions
- Compact, no-scroll dashboard design
- Comprehensive analytics with expanded views
- Professional trading insights

Author: Financial Analytics Team
Version: 3.0 Professional Compact Edition
Last Updated: January 2025
"""

# ========================================================================================
# IMPORTS AND DEPENDENCIES
# ========================================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sqlite3
import yfinance as yf
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Optional, Tuple, Any
import logging
import sys
import os
import json
from pathlib import Path

# Setup for ML imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

# ========================================================================================
# CONFIGURATION
# ========================================================================================
st.set_page_config(
    page_title="Nifty 50 Professional Analytics", 
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/nifty50-tracker',
        'Report a bug': "https://github.com/yourusername/nifty50-tracker/issues",
        'About': "Professional Nifty 50 Analytics Platform"
    }
)

# Professional styling
CUSTOM_CSS = """
<style>
    /* Remove default Streamlit padding */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }
    
    /* Compact header */
    .stApp > header {
        height: 0;
    }
    
    /* Professional metrics cards */
    .metric-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 0.75rem;
        margin: 0.25rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    /* Compact tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        padding: 0 16px;
        border-radius: 6px;
    }
    
    /* Hide scrollbars */
    .stApp {
        overflow-x: hidden;
    }
    
    /* Compact expanders */
    .streamlit-expanderHeader {
        padding: 0.5rem 1rem;
    }
    
    /* Professional color scheme */
    :root {
        --primary-color: #1f4e79;
        --secondary-color: #2e8b57;
        --success-color: #28a745;
        --danger-color: #dc3545;
        --warning-color: #ffc107;
        --info-color: #17a2b8;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ========================================================================================
# PROFESSIONAL DATABASE MANAGER
# ========================================================================================
class DatabaseManager:
    """Professional database management with enhanced error handling."""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            current_dir = Path(__file__).parent
            self.db_path = current_dir.parent / "data" / "nifty50_stocks.db"
        else:
            self.db_path = Path(db_path)
        
        self.logger = self._setup_logger()
        self._initialize_database()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup professional logging."""
        logger = logging.getLogger(f"DatabaseManager")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _initialize_database(self):
        """Initialize database connection and verify tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                if not tables:
                    self.logger.warning("No tables found in database")
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
    
    def get_connection(self):
        """Get database connection with error handling."""
        try:
            return sqlite3.connect(self.db_path)
        except Exception as e:
            self.logger.error(f"Database connection failed: {e}")
            raise
    
    def get_all_symbols(self) -> List[Dict[str, str]]:
        """Get all stock symbols with enhanced formatting."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT symbol, name, industry 
                    FROM stock_list 
                    ORDER BY symbol
                """)
                results = cursor.fetchall()
                
                return [
                    {
                        'value': row[0],
                        'label': f"{row[0]} - {row[1][:30]}{'...' if len(row[1]) > 30 else ''}",
                        'industry': row[2] if row[2] else 'N/A'
                    }
                    for row in results
                ]
        except Exception as e:
            self.logger.error(f"Error fetching symbols: {e}")
            return []
    
    def get_stock_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive stock information."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM stock_list WHERE symbol = ?", (symbol,))
                result = cursor.fetchone()
                
                if result:
                    columns = [description[0] for description in cursor.description]
                    return dict(zip(columns, result))
                return None
        except Exception as e:
            self.logger.error(f"Error fetching stock info for {symbol}: {e}")
            return None

# ========================================================================================
# PROFESSIONAL MARKET DATA PROVIDER
# ========================================================================================
class MarketDataProvider:
    """Professional market data provider with caching."""
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
    
    def _setup_logger(self) -> logging.Logger:
        """Setup professional logging."""
        logger = logging.getLogger(f"MarketDataProvider")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def get_historical_data(self, symbol: str, period: str = "6mo") -> pd.DataFrame:
        """Get historical data with professional error handling and fallback demo data."""
        cache_key = f"{symbol}_{period}"
        current_time = datetime.now()
        
        # Check cache
        if cache_key in self.cache:
            cached_data, cache_time = self.cache[cache_key]
            if (current_time - cache_time).seconds < self.cache_duration:
                return cached_data
        
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            data = ticker.history(period=period)
            
            if not data.empty:
                # Cache the data
                self.cache[cache_key] = (data, current_time)
                return data
            else:
                self.logger.warning(f"No data found for {symbol}, generating demo data")
                return self._generate_demo_data(symbol, period)
                
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}, generating demo data")
            return self._generate_demo_data(symbol, period)
    
    def _generate_demo_data(self, symbol: str, period: str = "6mo") -> pd.DataFrame:
        """Generate realistic demo chart data when Yahoo Finance fails."""
        import numpy as np
        
        # Determine number of days based on period
        days_map = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365}
        days = days_map.get(period, 90)
        
        # Generate dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Base price from database (get current price)
        try:
            conn = sqlite3.connect('data/nifty50_stocks.db')
            cursor = conn.cursor()
            cursor.execute("SELECT lastPrice FROM stock_list WHERE symbol = ?", (symbol,))
            result = cursor.fetchone()
            base_price = float(result[0]) if result and result[0] else 2500.0
            conn.close()
        except:
            base_price = 2500.0  # Default fallback
        
        # Generate realistic stock data with trends and volatility
        np.random.seed(hash(symbol) % 2**32)  # Consistent data for same symbol
        
        prices = []
        current_price = base_price * 0.9  # Start 10% below current
        
        for i in range(len(dates)):
            # Add some trend (slight upward bias)
            trend = 0.0002
            # Add daily volatility (1-3%)
            volatility = np.random.normal(0, 0.02)
            # Occasional larger moves
            if np.random.random() < 0.05:  # 5% chance of big move
                volatility *= 3
                
            daily_return = trend + volatility
            current_price *= (1 + daily_return)
            prices.append(current_price)
        
        # Create OHLC data
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            # Generate realistic OHLC from close price
            volatility = close * 0.015  # 1.5% intraday range
            high = close + np.random.uniform(0, volatility)
            low = close - np.random.uniform(0, volatility)
            
            if i == 0:
                open_price = close * 0.995  # Slight gap
            else:
                open_price = prices[i-1] * (1 + np.random.normal(0, 0.005))  # Small gap
            
            # Ensure OHLC logic is maintained
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            volume = int(np.random.uniform(100000, 1000000))  # Random volume
            
            data.append({
                'Open': open_price,
                'High': high,
                'Low': low,
                'Close': close,
                'Volume': volume
            })
        
        df = pd.DataFrame(data, index=dates)
        return df

# ========================================================================================
# TECHNICAL ANALYSIS ENGINE
# ========================================================================================
class TechnicalAnalysis:
    """Professional technical analysis with multiple indicators."""
    
    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive technical indicators."""
        if df.empty or len(df) < 20:
            return {}
        
        try:
            # Moving averages
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['EMA_12'] = df['Close'].ewm(span=12).mean()
            df['EMA_26'] = df['Close'].ewm(span=26).mean()
            
            # Bollinger Bands
            df['BB_Middle'] = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
            df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            
            # Current values
            current_price = df['Close'].iloc[-1]
            current_rsi = df['RSI'].iloc[-1]
            current_macd = df['MACD'].iloc[-1]
            current_signal = df['MACD_Signal'].iloc[-1]
            
            # Support and Resistance
            recent_data = df.tail(50)
            support = recent_data['Low'].min()
            resistance = recent_data['High'].max()
            
            # Trend analysis
            trend = "Bullish" if current_price > df['SMA_20'].iloc[-1] > df['SMA_50'].iloc[-1] else "Bearish"
            
            return {
                'current_price': current_price,
                'sma_20': df['SMA_20'].iloc[-1],
                'sma_50': df['SMA_50'].iloc[-1],
                'rsi': current_rsi,
                'macd': current_macd,
                'macd_signal': current_signal,
                'support': support,
                'resistance': resistance,
                'trend': trend,
                'bb_upper': df['BB_Upper'].iloc[-1],
                'bb_lower': df['BB_Lower'].iloc[-1],
                'dataframe': df
            }
            
        except Exception as e:
            st.error(f"Technical analysis error: {e}")
            return {}

# ========================================================================================
# INITIALIZE PROFESSIONAL COMPONENTS
# ========================================================================================
# Create instances
db_manager = DatabaseManager()
market_data = MarketDataProvider()
technical_analyzer = TechnicalAnalysis()

# ML Models directory setup
ML_MODELS_DIR = Path(__file__).parent.parent / "ml_models"

# ========================================================================================
# PROFESSIONAL HEADER AND BRANDING
# ========================================================================================
def create_compact_header():
    """Create a compact, professional header."""
    st.markdown("""
    <div style="background: linear-gradient(90deg, #1f4e79 0%, #2e8b57 100%); 
                padding: 1rem; margin: -1rem -1rem 1rem -1rem; border-radius: 0 0 10px 10px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 style="color: white; margin: 0; font-size: 1.5rem; font-weight: 600;">
                    📈 Nifty 50 Professional Analytics
                </h1>
                <p style="color: #e8f4fd; margin: 0; font-size: 0.9rem;">
                    Real-time market intelligence • AI-powered insights • Professional trading tools
                </p>
            </div>
            <div style="text-align: right;">
                <p style="color: white; margin: 0; font-size: 0.8rem;">
                    Last Updated: {current_time}
                </p>
                <p style="color: #e8f4fd; margin: 0; font-size: 0.7rem;">
                    Live Market Data
                </p>
            </div>
        </div>
    </div>
    """.format(current_time=datetime.now().strftime('%H:%M:%S')), unsafe_allow_html=True)

def create_compact_ticker():
    """Create a compact market ticker."""
    try:
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT symbol, lastPrice, change, pChange 
                FROM stock_list 
                ORDER BY ABS(pChange) DESC 
                LIMIT 10
            """)
            movers = cursor.fetchall()
            
            if movers:
                ticker_items = []
                for symbol, price, change, pchange in movers:
                    color = "#28a745" if change >= 0 else "#dc3545"
                    arrow = "▲" if change >= 0 else "▼"
                    ticker_items.append(
                        f'<span style="color: {color}; margin-right: 2rem;">'
                        f'{symbol}: ₹{price:.2f} {arrow} {pchange:+.2f}%</span>'
                    )
                
                ticker_text = "".join(ticker_items)
                
                st.markdown(f"""
                <div style="background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 6px; 
                           padding: 0.5rem; margin-bottom: 1rem; overflow: hidden; white-space: nowrap;">
                    <marquee behavior="scroll" direction="left" scrollamount="3" style="font-weight: 500;">
                        {ticker_text}
                    </marquee>
                </div>
                """, unsafe_allow_html=True)
    except Exception as e:
        st.info("📊 Market ticker will be available once data is loaded")

# ========================================================================================
# PROFESSIONAL SIDEBAR
# ========================================================================================
def create_professional_sidebar():
    """Create a professional sidebar with expanded analytics."""
    with st.sidebar:
        # Compact branding
        st.markdown("""
        <div style="text-align: center; padding: 0.75rem; background: linear-gradient(45deg, #1f4e79, #2e8b57); 
                    border-radius: 8px; margin-bottom: 1rem;">
            <h3 style="color: white; margin: 0; font-size: 1.1rem;">📊 Analytics Hub</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Theme toggle (compact)
        theme = st.toggle("🌗 Dark Mode", value=st.session_state.get('theme_dark', False))
        st.session_state['theme_dark'] = theme
        
        # Market Overview - Always Expanded
        st.markdown("### 📈 Market Overview")
        try:
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT COUNT(*), AVG(lastPrice), SUM(totalTradedValue),
                           SUM(CASE WHEN change > 0 THEN 1 ELSE 0 END) as gainers,
                           SUM(CASE WHEN change < 0 THEN 1 ELSE 0 END) as losers
                    FROM stock_list
                """)
                result = cursor.fetchone()
                count, avg_price, total_value, gainers, losers = result
                
                # Compact metrics display
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Stocks", count)
                    st.metric("Gainers", gainers)
                with col2:
                    st.metric("Avg Price", f"₹{avg_price:.0f}")
                    st.metric("Losers", losers)
                
                # Market sentiment
                if gainers > losers:
                    sentiment = "🟢 Bullish"
                    sentiment_color = "#28a745"
                elif losers > gainers:
                    sentiment = "🔴 Bearish"
                    sentiment_color = "#dc3545"
                else:
                    sentiment = "⚪ Neutral"
                    sentiment_color = "#6c757d"
                
                st.markdown(f"""
                <div style="text-align: center; padding: 0.5rem; background: {sentiment_color}20; 
                           border-radius: 6px; margin: 0.5rem 0;">
                    <strong style="color: {sentiment_color};">{sentiment}</strong>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Database error: {e}")
        
        st.markdown("---")
        
        # Stock Selection
        st.markdown("### 🔍 Stock Selection")
        all_symbols = db_manager.get_all_symbols()
        
        if not all_symbols:
            st.error("No stocks available")
            return None
        
        selected = st.selectbox(
            "Choose Stock",
            all_symbols,
            index=0,
            format_func=lambda x: x['label'],
            key="stock_selectbox"
        )
        
        return selected['value'] if selected else None

# ========================================================================================
# COMPACT MAIN DASHBOARD
# ========================================================================================
def create_compact_dashboard(symbol: str, company_info: Dict[str, Any]):
    """Create a compact, no-scroll dashboard with all analytics expanded."""
    
    # Company Header - Compact
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #f8fafc 0%, #e3f2fd 100%); 
                padding: 1rem; border-radius: 8px; border-left: 4px solid #1f4e79; margin-bottom: 1rem;">
        <h2 style="color: #1f4e79; margin: 0 0 0.5rem 0; font-size: 1.3rem; font-weight: 600;">
            {company_info['name']} ({company_info['symbol']})
        </h2>
        <div style="display: flex; gap: 1.5rem; font-size: 0.85rem; color: #555;">
            <span><strong>Industry:</strong> {company_info['industry']}</span>
            <span><strong>ISIN:</strong> {company_info['isin']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Metrics Row - Always Visible
    st.markdown("### 💰 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    change = company_info['change']
    pchange = company_info['pChange']
    change_color = "#28a745" if change >= 0 else "#dc3545"
    change_icon = "📈" if change >= 0 else "📉"
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="margin: 0; color: #1f4e79; font-size: 0.9rem;">Current Price</h4>
            <h2 style="margin: 0.3rem 0; color: #333; font-size: 1.3rem;">₹{company_info['lastPrice']:.2f}</h2>
            <p style="margin: 0; color: {change_color}; font-weight: 600; font-size: 0.8rem;">
                {change_icon} {change:+.2f} ({pchange:+.2f}%)
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="margin: 0; color: #1f4e79; font-size: 0.9rem;">Day Range</h4>
            <p style="margin: 0.3rem 0; font-size: 1.1rem; color: #333;">
                ₹{company_info['dayLow']:.2f} - ₹{company_info['dayHigh']:.2f}
            </p>
            <p style="margin: 0; color: #666; font-size: 0.75rem;">Low - High</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="margin: 0; color: #1f4e79; font-size: 0.9rem;">52W Range</h4>
            <p style="margin: 0.3rem 0; font-size: 1.1rem; color: #333;">
                ₹{company_info['yearLow']:.2f} - ₹{company_info['yearHigh']:.2f}
            </p>
            <p style="margin: 0; color: #666; font-size: 0.75rem;">Annual Range</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="margin: 0; color: #1f4e79; font-size: 0.9rem;">Volume</h4>
            <p style="margin: 0.3rem 0; font-size: 1.1rem; color: #333;">
                {company_info['totalTradedVolume']/1000:.1f}K
            </p>
            <p style="margin: 0; color: #666; font-size: 0.75rem;">Traded Volume</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Compact Tabs Layout
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Chart", "🔍 Analysis", "🤖 AI"])
    
    with tab1:
        create_overview_tab(symbol, company_info)
    
    with tab2:
        create_chart_tab(symbol)
    
    with tab3:
        create_analysis_tab(symbol)
    
    with tab4:
        create_ai_tab(symbol, company_info)

def create_overview_tab(symbol: str, company_info: Dict[str, Any]):
    """Create compact overview tab with all metrics visible."""
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 📊 Trading Details")
        
        # Always visible metrics
        metrics_data = [
            ("Open Price", f"₹{company_info['open']:.2f}"),
            ("Previous Close", f"₹{company_info['previousClose']:.2f}"),
            ("Market Cap", f"₹{company_info.get('totalTradedValue', 0)/1e9:.2f} B"),
            ("P/E Ratio", f"{company_info.get('pe', 'N/A')}"),
        ]
        
        for label, value in metrics_data:
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; padding: 0.3rem 0; 
                       border-bottom: 1px solid #eee;">
                <span style="color: #666;">{label}:</span>
                <strong style="color: #333;">{value}</strong>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 📈 Performance Metrics")
        
        performance_data = [
            ("Today's Gain/Loss", f"{company_info['change']:+.2f} ({company_info['pChange']:+.2f}%)"),
            ("Total Traded Value", f"₹{company_info['totalTradedValue']/1e7:.2f} Cr"),
            ("Total Traded Volume", f"{company_info['totalTradedVolume']/1000:.1f}K"),
            ("Face Value", f"₹{company_info.get('faceValue', 'N/A')}"),
        ]
        
        for label, value in performance_data:
            color = "#28a745" if "+" in str(value) else "#dc3545" if "-" in str(value) else "#333"
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; padding: 0.3rem 0; 
                       border-bottom: 1px solid #eee;">
                <span style="color: #666;">{label}:</span>
                <strong style="color: {color};">{value}</strong>
            </div>
            """, unsafe_allow_html=True)

def create_chart_tab(symbol: str):
    """Create compact chart tab with technical indicators."""
    # Get historical data
    df = market_data.get_historical_data(symbol, "3mo")  # Shorter period for faster loading
    
    if df.empty:
        st.error("No chart data available")
        return
    
    # Check if this is demo data (simplified check)
    if len(df) == 90 and df.index[0].date() == (datetime.now() - timedelta(days=90)).date():
        st.info("📊 Demo chart data is being displayed due to data source connectivity issues.")
    
    # Technical analysis
    tech_data = technical_analyzer.calculate_indicators(df)
    
    if tech_data:
        # Compact chart
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Price Chart', 'RSI'),
            vertical_spacing=0.15,
            row_width=[0.7, 0.3]
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name="Price"
            ),
            row=1, col=1
        )
        
        # Moving averages
        fig.add_trace(
            go.Scatter(x=df.index, y=tech_data['dataframe']['SMA_20'], 
                      name="SMA 20", line=dict(color='orange', width=1)),
            row=1, col=1
        )
        
        # RSI
        fig.add_trace(
            go.Scatter(x=df.index, y=tech_data['dataframe']['RSI'], 
                      name="RSI", line=dict(color='purple', width=2)),
            row=2, col=1
        )
        
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        fig.update_layout(
            height=500,  # Compact height
            xaxis_rangeslider_visible=False,
            margin=dict(l=0, r=0, t=50, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Technical summary - Always visible
        st.markdown("#### 🔧 Technical Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            trend_color = "#28a745" if tech_data['trend'] == "Bullish" else "#dc3545"
            st.markdown(f"""
            <div style="text-align: center; padding: 0.5rem; background: {trend_color}20; 
                       border-radius: 6px;">
                <strong style="color: {trend_color};">Trend: {tech_data['trend']}</strong>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            rsi_level = "Overbought" if tech_data['rsi'] > 70 else "Oversold" if tech_data['rsi'] < 30 else "Neutral"
            rsi_color = "#dc3545" if tech_data['rsi'] > 70 else "#28a745" if tech_data['rsi'] < 30 else "#6c757d"
            st.markdown(f"""
            <div style="text-align: center; padding: 0.5rem; background: {rsi_color}20; 
                       border-radius: 6px;">
                <strong style="color: {rsi_color};">RSI: {tech_data['rsi']:.1f} ({rsi_level})</strong>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            support_resistance = f"S: ₹{tech_data['support']:.2f} | R: ₹{tech_data['resistance']:.2f}"
            st.markdown(f"""
            <div style="text-align: center; padding: 0.5rem; background: #f8f9fa; 
                       border-radius: 6px; border: 1px solid #dee2e6;">
                <strong style="color: #333;">{support_resistance}</strong>
            </div>
            """, unsafe_allow_html=True)

def create_analysis_tab(symbol: str):
    """Create compact analysis tab with peer comparison."""
    try:
        # Get company info for industry
        company_info = db_manager.get_stock_info(symbol)
        if not company_info:
            st.error("Company information not available")
            return
        
        # Peer analysis - Always visible
        st.markdown("#### 🏢 Peer Comparison")
        
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT symbol, name, lastPrice, change, pChange, totalTradedValue
                FROM stock_list 
                WHERE industry = ? AND symbol != ?
                ORDER BY totalTradedValue DESC
                LIMIT 5
            """, (company_info['industry'], symbol))
            
            peers = cursor.fetchall()
            
            if peers:
                # Create comparison table
                peer_data = []
                for peer in peers:
                    peer_symbol, peer_name, peer_price, peer_change, peer_pchange, peer_mcap = peer
                    peer_data.append({
                        'Stock': peer_symbol,
                        'Company': peer_name[:20] + '...' if len(peer_name) > 20 else peer_name,
                        'Price': f"₹{peer_price:.2f}",
                        'Change': f"{peer_change:+.2f}",
                        'Change %': f"{peer_pchange:+.2f}%",
                        'Market Cap': f"₹{peer_mcap/1e9:.1f}B" if peer_mcap else "N/A"
                    })
                
                # Display as compact table
                df_peers = pd.DataFrame(peer_data)
                st.dataframe(df_peers, use_container_width=True, height=200)
            else:
                st.info("No peer companies found in the same industry")
        
        # Market comparison
        st.markdown("#### 📊 Market Position")
        col1, col2 = st.columns(2)
        
        with col1:
            # Industry ranking
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT COUNT(*) as total,
                           (SELECT COUNT(*) FROM stock_list s2 
                            WHERE s2.industry = ? AND s2.totalTradedValue > ?) as better_rank
                    FROM stock_list 
                    WHERE industry = ?
                """, (company_info['industry'], company_info.get('totalTradedValue', 0), company_info['industry']))
                
                result = cursor.fetchone()
                if result:
                    total, better_rank = result
                    rank = better_rank + 1
                    st.metric("Industry Rank", f"{rank} of {total}")
        
        with col2:
            # Performance percentile
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT COUNT(*) as total,
                           (SELECT COUNT(*) FROM stock_list WHERE pChange > ?) as better_performers
                    FROM stock_list
                """, (company_info['pChange'],))
                
                result = cursor.fetchone()
                if result:
                    total, better_performers = result
                    percentile = (1 - better_performers/total) * 100 if total > 0 else 0
                    st.metric("Performance Percentile", f"{percentile:.1f}%")
                    
    except Exception as e:
        st.error(f"Analysis error: {e}")

def create_ai_tab(symbol: str, company_info: Dict[str, Any]):
    """Create compact AI analysis tab."""
    st.markdown("#### 🤖 AI Market Intelligence")
    
    try:
        # Import AI model with proper path handling
        try:
            import sys
            import os
            
            # Ensure we can import from the ml_models directory
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)
            
            from ml_models.stock_prediction_model import StockAIAnalyzer
            ai_analyzer = StockAIAnalyzer()
            
            # Get AI analysis
            analysis = ai_analyzer.analyze_stock(symbol)
            
            if analysis:
                # AI recommendation - Always visible
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    recommendation = analysis.get('recommendation', 'HOLD')
                    rec_color = "#28a745" if recommendation == "BUY" else "#dc3545" if recommendation == "SELL" else "#ffc107"
                    st.markdown(f"""
                    <div style="text-align: center; padding: 1rem; background: {rec_color}20; 
                               border-radius: 8px; border: 2px solid {rec_color};">
                        <h3 style="margin: 0; color: {rec_color};">AI Recommendation</h3>
                        <h2 style="margin: 0.5rem 0; color: {rec_color};">{recommendation}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    confidence = analysis.get('confidence', 0.5)
                    confidence_pct = confidence * 100
                    conf_color = "#28a745" if confidence > 0.7 else "#ffc107" if confidence > 0.5 else "#dc3545"
                    st.markdown(f"""
                    <div style="text-align: center; padding: 1rem; background: white; 
                               border-radius: 8px; border: 1px solid #dee2e6;">
                        <h4 style="margin: 0; color: #1f4e79;">Confidence</h4>
                        <h2 style="margin: 0.5rem 0; color: {conf_color};">{confidence_pct:.1f}%</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    target_price = analysis.get('target_price', company_info['lastPrice'])
                    potential = ((target_price - company_info['lastPrice']) / company_info['lastPrice']) * 100
                    pot_color = "#28a745" if potential > 0 else "#dc3545"
                    st.markdown(f"""
                    <div style="text-align: center; padding: 1rem; background: white; 
                               border-radius: 8px; border: 1px solid #dee2e6;">
                        <h4 style="margin: 0; color: #1f4e79;">Target Price</h4>
                        <h2 style="margin: 0.5rem 0; color: {pot_color};">₹{target_price:.2f}</h2>
                        <p style="margin: 0; color: {pot_color}; font-weight: 600;">
                            {potential:+.1f}% potential
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # AI insights - Always visible
                st.markdown("#### 💡 AI Insights")
                insights = analysis.get('insights', [])
                if insights:
                    for i, insight in enumerate(insights[:3]):  # Show top 3 insights
                        icon = "🟢" if i == 0 else "🟡" if i == 1 else "🔵"
                        st.markdown(f"{icon} {insight}")
                else:
                    st.info("AI analysis in progress...")
                    
        except ImportError:
            st.warning("🤖 AI analysis module not available. Please ensure ML models are installed.")
            
            # Alternative basic analysis
            st.markdown("#### 📊 Basic Market Analysis")
            
            # Simple momentum analysis
            change = company_info['change']
            pchange = company_info['pChange']
            
            if pchange > 2:
                momentum = "Strong Bullish"
                momentum_color = "#28a745"
            elif pchange > 0:
                momentum = "Bullish"
                momentum_color = "#28a745"
            elif pchange < -2:
                momentum = "Strong Bearish"
                momentum_color = "#dc3545"
            elif pchange < 0:
                momentum = "Bearish"
                momentum_color = "#dc3545"
            else:
                momentum = "Neutral"
                momentum_color = "#6c757d"
            
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background: {momentum_color}20; 
                       border-radius: 8px; border: 2px solid {momentum_color};">
                <h3 style="margin: 0; color: {momentum_color};">Market Momentum</h3>
                <h2 style="margin: 0.5rem 0; color: {momentum_color};">{momentum}</h2>
            </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"AI analysis error: {e}")

# ========================================================================================
# MAIN APPLICATION
# ========================================================================================
def main():
    """Main application entry point."""
    
    # Create compact header
    create_compact_header()
    
    # Create compact ticker
    create_compact_ticker()
    
    # Create professional sidebar and get selected stock
    selected_symbol = create_professional_sidebar()
    
    if not selected_symbol:
        st.error("❌ Please select a stock from the sidebar")
        st.stop()
    
    # Get company information
    company_info = db_manager.get_stock_info(selected_symbol)
    
    if not company_info:
        st.error(f"❌ No data available for {selected_symbol}")
        st.stop()
    
    # Create compact main dashboard
    create_compact_dashboard(selected_symbol, company_info)
    
    # Professional footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem; color: #666; font-size: 0.8rem;">
        <p>📈 <strong>Nifty 50 Professional Analytics</strong> | Powered by AI & Real-time Market Data</p>
        <p>© 2025 Financial Analytics Team | Professional Trading Platform</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
