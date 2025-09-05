"""
Nifty50-Tracker: Professional Stock Analysis Dashboard
=====================================================

A comprehensive financial analytics platform for NSE Nifty 50 stocks featuring:
- Real-time market data and technical analysis
- AI-powered sentiment analysis and recommendations
- Advanced charting with multiple technical indicators
- Portfolio tracking and risk management tools

Author: Financial Analytics Team
Version: 2.0.0
Last Updated: September 2025
"""

import os
import sys
import json
import time
import random
import sqlite3
import threading
import importlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objs as go
import plotly.express as px
import requests
import yfinance as yf

# Configure Streamlit page
st.set_page_config(
    page_title="Nifty 50 Professional Analytics", 
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ML_MODELS_DIR = os.path.join(PROJECT_ROOT, "ml_models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Add ML models to path
if ML_MODELS_DIR not in sys.path:
    sys.path.append(ML_MODELS_DIR)

# Import custom modules
try:
    from ml_models.stock_prediction_model import StockAIAnalyzer
except ImportError as e:
    st.error(f"Failed to import AI model: {e}")
    StockAIAnalyzer = None

# ========================================================================================
# UTILITY FUNCTIONS
# ========================================================================================

class DatabaseManager:
    """Professional database management class for stock data operations."""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or os.path.join(DATA_DIR, "nifty50_stocks.db")
    
    def get_connection(self) -> sqlite3.Connection:
        """Create and return a database connection."""
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database not found at {self.db_path}")
        return sqlite3.connect(self.db_path)
    
    def get_stock_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Retrieve comprehensive stock information from database."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                base_symbol = symbol.replace('.NS', '').replace('.BO', '')
                
                query = """
                SELECT symbol, name, industry, isin, lastPrice, open, dayHigh, dayLow, 
                       previousClose, change, pChange, yearHigh, yearLow, 
                       totalTradedVolume, totalTradedValue 
                FROM stock_list 
                WHERE symbol = ? LIMIT 1
                """
                
                cursor.execute(query, (base_symbol,))
                row = cursor.fetchone()
                
                if row:
                    columns = [
                        "symbol", "name", "industry", "isin", "lastPrice", "open", 
                        "dayHigh", "dayLow", "previousClose", "change", "pChange", 
                        "yearHigh", "yearLow", "totalTradedVolume", "totalTradedValue"
                    ]
                    return dict(zip(columns, row))
                return None
        except Exception as e:
            st.error(f"Database error: {e}")
            return None
    
    def get_peer_stocks(self, industry: str, exclude_symbol: str = None, limit: int = 5) -> List[Tuple]:
        """Get peer companies in the same industry."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                if exclude_symbol:
                    query = """
                    SELECT symbol, name, lastPrice, pChange 
                    FROM stock_list 
                    WHERE industry = ? AND symbol != ? 
                    ORDER BY totalTradedValue DESC 
                    LIMIT ?
                    """
                    cursor.execute(query, (industry, exclude_symbol, limit))
                else:
                    query = """
                    SELECT symbol, name, lastPrice, pChange 
                    FROM stock_list 
                    WHERE industry = ? 
                    ORDER BY totalTradedValue DESC 
                    LIMIT ?
                    """
                    cursor.execute(query, (industry, limit))
                
                return cursor.fetchall()
        except Exception as e:
            st.error(f"Database error: {e}")
            return []
    
    def get_top_movers(self, limit: int = 8) -> List[Tuple]:
        """Get top moving stocks by percentage change with enhanced filtering."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                # Enhanced query to avoid duplicates and get more diverse results
                cursor.execute("""
                    SELECT DISTINCT symbol, lastPrice, pChange, name
                    FROM stock_list 
                    WHERE symbol IS NOT NULL AND lastPrice > 0 AND pChange IS NOT NULL
                    ORDER BY ABS(pChange) DESC 
                    LIMIT ?
                """, (limit * 2,))  # Get more results to filter
                
                results = cursor.fetchall()
                
                # Remove duplicates and ensure variety
                seen_symbols = set()
                filtered_results = []
                
                for symbol, price, change, name in results:
                    if symbol not in seen_symbols and len(filtered_results) < limit:
                        seen_symbols.add(symbol)
                        filtered_results.append((symbol, price, change, name))
                
                return filtered_results[:limit]
                
        except Exception as e:
            st.error(f"Database error: {e}")
            return []
    
    def get_all_symbols(self) -> List[Dict[str, str]]:
        """Get all available stock symbols with metadata."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT symbol, name, exchange, totalTradedValue 
                    FROM stock_list 
                    ORDER BY totalTradedValue DESC
                """)
                rows = cursor.fetchall()
                
                symbol_options = []
                for symbol, name, exchange, _ in rows:
                    yf_symbol = symbol
                    if not symbol.endswith('.NS') and not symbol.endswith('.BO'):
                        if exchange == 'NSE':
                            yf_symbol = f"{symbol}.NS"
                        elif exchange == 'BSE':
                            yf_symbol = f"{symbol}.BO"
                    
                    label = f"{symbol} - {name} [{exchange}]"
                    symbol_options.append({"label": label, "value": yf_symbol})
                
                return symbol_options
        except Exception as e:
            st.error(f"Database error: {e}")
            return []


class MarketDataProvider:
    """Professional market data provider with caching and error handling."""
    
    @staticmethod
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def get_historical_data(symbol: str, period: str = "6mo") -> pd.DataFrame:
        """Fetch historical price data with professional error handling."""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                st.warning(f"No data available for {symbol}")
                return pd.DataFrame()
            
            return data
        except Exception as e:
            st.error(f"Failed to fetch data for {symbol}: {e}")
            return pd.DataFrame()
    
    @staticmethod
    @st.cache_data(ttl=1800)  # Cache for 30 minutes
    def get_company_news(symbol: str, limit: int = 5) -> List[Dict]:
        """Fetch latest company news with error handling."""
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            processed_news = []
            for item in news[:limit]:
                if isinstance(item, dict) and 'content' in item:
                    content = item['content']
                    title = content.get('title', 'No title available')
                    summary = content.get('summary', 'No summary available')
                    pub_date = content.get('pubDate', '')
                    
                    processed_news.append({
                        'title': title,
                        'summary': summary,
                        'date': pub_date
                    })
            
            return processed_news
        except Exception as e:
            st.warning(f"News data unavailable for {symbol}: {e}")
            return []


class TechnicalAnalysis:
    """Professional technical analysis calculations."""
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        
        return {
            'macd': macd,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: int = 2) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        return {
            'middle': sma,
            'upper': sma + (std * std_dev),
            'lower': sma - (std * std_dev)
        }


# Initialize global instances
db_manager = DatabaseManager()
market_data = MarketDataProvider()

# ========================================================================================
# STREAMLIT CONFIGURATION & PROFESSIONAL UI COMPONENTS
# ========================================================================================

def create_professional_header():
    """Create a professional dashboard header with branding."""
    st.markdown("""
    <div style="background: linear-gradient(90deg, #1f4e79 0%, #2e8b57 100%); padding: 2rem; margin: -1rem -1rem 2rem -1rem; border-radius: 10px;">
        <h1 style="color: white; text-align: center; margin: 0; font-size: 2.5rem; font-weight: 600;">
            📈 Nifty 50 Professional Analytics
        </h1>
        <p style="color: #e8f4f8; text-align: center; margin: 0.5rem 0 0 0; font-size: 1.1rem;">
            Advanced Technical Analysis & AI-Powered Investment Intelligence
        </p>
    </div>
    """, unsafe_allow_html=True)

def render_professional_ticker():
    """Render a clean, professional stock ticker using Streamlit components."""
    movers = db_manager.get_top_movers(limit=8)
    if not movers:
        return
    
    # Create a simple but effective ticker
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f8fafc 0%, #e3f2fd 100%); 
                border: 1px solid #e1e5e9; border-radius: 10px; padding: 1rem; 
                margin: 1rem 0; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
        <div style="display: flex; align-items: center; margin-bottom: 0.8rem;">
            <span style="font-weight: 600; color: #1f4e79; font-size: 1.2rem;">
                📈 Live Market Movers
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create ticker content as a simple scrolling text
    ticker_items = []
    
    for item in movers:
        if len(item) == 4:  # New format with name
            symbol, price, change, name = item
        else:  # Old format compatibility
            symbol, price, change = item
            name = symbol
        
        # Format each item
        arrow = '📈' if change >= 0 else '📉'
        sign = '+' if change >= 0 else ''
        ticker_items.append(f"{arrow} **{symbol}** ₹{price:.2f} ({sign}{change:.2f}%)")
    
    # Join all items with separators
    ticker_text = " • ".join(ticker_items)
    
    # Display as a simple marquee using HTML
    st.markdown(f"""
    <div style="background: white; border-radius: 8px; padding: 0.8rem; 
                border-left: 4px solid #1f4e79; margin: 0.5rem 0;">
        <marquee style="color: #333; font-size: 1.05rem; font-weight: 500;">
            {ticker_text}
        </marquee>
    </div>
    """, unsafe_allow_html=True)

def create_professional_sidebar():
    """Create a professional sidebar with enhanced UI."""
    with st.sidebar:
        # Logo and branding
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: linear-gradient(45deg, #1f4e79, #2e8b57); border-radius: 10px; margin-bottom: 1rem;">
            <h2 style="color: white; margin: 0;">📊 Analytics Hub</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Theme toggle
        theme = st.toggle("🌗 Dark Mode", value=st.session_state.get('theme_dark', False))
        st.session_state['theme_dark'] = theme
        
        # Market statistics
        st.markdown("### 📈 Market Overview")
        try:
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*), AVG(lastPrice), SUM(totalTradedValue) FROM stock_list")
                count, avg_price, total_value = cursor.fetchone()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Stocks", count)
                    st.metric("Avg Price", f"₹{avg_price:.2f}")
                with col2:
                    st.metric("Total Volume", f"₹{total_value/1e7:.2f} Cr")
        except Exception as e:
            st.error(f"Database connection error: {e}")
        
        st.markdown("---")
        
        # Stock selection
        st.markdown("### 🔍 Stock Selection")
        all_symbols = db_manager.get_all_symbols()
        
        if not all_symbols:
            st.error("No stocks available in database")
            return None
        
        selected = st.selectbox(
            "Choose Stock",
            all_symbols,
            index=0,
            format_func=lambda x: x['label'],
            key="stock_selectbox"
        )
        
        return selected['value'] if selected else None

def get_company_info(symbol: str) -> Optional[Dict[str, Any]]:
    """Get company information using the professional database manager."""
    return db_manager.get_stock_info(symbol)

# Legacy compatibility functions
def get_historical_prices(symbol: str, period: str = "6mo") -> pd.DataFrame:
    """Legacy compatibility wrapper for historical data."""
    return market_data.get_historical_data(symbol, period)

def get_peers(industry: str, exclude_symbol: str = None) -> List[Tuple]:
    """Legacy compatibility wrapper for peer data."""
    return db_manager.get_peer_stocks(industry, exclude_symbol)

def get_top_movers(limit: int = 8) -> List[Tuple]:
    """Legacy compatibility wrapper for top movers."""
    results = db_manager.get_top_movers(limit)
    # Convert new format back to old format for compatibility
    return [(item[0], item[1], item[2]) for item in results]

# Initialize the professional UI
create_professional_header()
render_professional_ticker()

# Add market summary section
def render_market_summary():
    """Render a professional market summary below the ticker."""
    try:
        movers = db_manager.get_top_movers(limit=20)  # Get more data for summary
        if not movers:
            return
            
        total_stocks = len(movers)
        gainers = sum(1 for item in movers if item[2] > 0)
        losers = sum(1 for item in movers if item[2] < 0)
        unchanged = total_stocks - gainers - losers
        
        avg_change = sum(item[2] for item in movers) / total_stocks if total_stocks > 0 else 0
        
        # Market sentiment indicator
        if gainers > losers * 1.5:
            sentiment = "🟢 Bullish"
            sentiment_color = "#28a745"
        elif losers > gainers * 1.5:
            sentiment = "🔴 Bearish"
            sentiment_color = "#dc3545"
        else:
            sentiment = "🟡 Neutral"
            sentiment_color = "#ffc107"
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Market Sentiment", sentiment)
        with col2:
            st.metric("Gainers", gainers, f"{(gainers/total_stocks)*100:.1f}%")
        with col3:
            st.metric("Losers", losers, f"{(losers/total_stocks)*100:.1f}%")
        with col4:
            st.metric("Unchanged", unchanged)
        with col5:
            st.metric("Avg Change", f"{avg_change:+.2f}%")
            
    except Exception as e:
        st.warning(f"Market summary unavailable: {e}")

render_market_summary()
selected_stock = create_professional_sidebar()

# ========================================================================================
# MAIN APPLICATION LOGIC
# ========================================================================================

if selected_stock:
    # Get company information
    company_info = get_company_info(selected_stock)
    
    # Additional sidebar controls
    with st.sidebar:
        st.markdown("---")
        st.markdown("### ⚙️ Analysis Settings")
        period = st.selectbox("Analysis Period", ["1mo", "3mo", "6mo", "1y"], index=0)
        
        st.markdown("---")
        st.markdown("### 💸 Portfolio Simulation")
        if 'portfolio' not in st.session_state:
            st.session_state['portfolio'] = {}
            
        col_buy, col_sell = st.columns(2)
        with col_buy:
            buy_qty = st.number_input("Buy Qty", min_value=1, max_value=10000, value=10, key="buy_qty")
            if st.button("📈 Buy", key="buy_btn", use_container_width=True):
                st.session_state['portfolio'][selected_stock] = st.session_state['portfolio'].get(selected_stock, 0) + buy_qty
                st.success(f"✅ Bought {buy_qty} shares")
                
        with col_sell:
            sell_qty = st.number_input("Sell Qty", min_value=1, max_value=10000, value=10, key="sell_qty")
            if st.button("📉 Sell", key="sell_btn", use_container_width=True):
                current = st.session_state['portfolio'].get(selected_stock, 0)
                if sell_qty > current:
                    st.warning("❌ Insufficient shares!")
                else:
                    st.session_state['portfolio'][selected_stock] = current - sell_qty
                    st.success(f"✅ Sold {sell_qty} shares")
        
        holdings = st.session_state['portfolio'].get(selected_stock, 0)
        st.info(f"📊 **Holdings:** {holdings} shares")
        
        st.markdown("---")
        st.markdown("### 📝 Research Notes")
        note_key = f"note_{selected_stock}"
        note = st.text_area("Investment Notes:", value=st.session_state.get(note_key, ""), height=100)
        if st.button("💾 Save Note", key="save_note_btn", use_container_width=True):
            st.session_state[note_key] = note
            st.success("✅ Note saved!")
else:
    st.error("❌ Please select a stock from the sidebar")
    st.stop()

# ========================================================================================
# MAIN DASHBOARD TABS
# ========================================================================================
if company_info:
    # Professional tab configuration
    tab_names = ["📊 Overview", "📈 Technical Chart", "🔍 Peer Analysis", "🤖 AI Intelligence"]
    tabs = st.tabs(tab_names)
    
    # ========================================================================================
    # TAB 1: PROFESSIONAL OVERVIEW
    # ========================================================================================
    with tabs[0]:
        # Company header with enhanced styling
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f8fafc 0%, #e3f2fd 100%); 
                    padding: 1.5rem; border-radius: 12px; border-left: 4px solid #1f4e79; margin-bottom: 1rem;">
            <h2 style="color: #1f4e79; margin: 0 0 0.5rem 0; font-weight: 600;">
                {company_info['name']} ({company_info['symbol']})
            </h2>
            <div style="display: flex; gap: 2rem; font-size: 0.95rem; color: #555;">
                <span><strong>Industry:</strong> {company_info['industry']}</span>
                <span><strong>ISIN:</strong> {company_info['isin']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced price metrics with professional cards
        st.markdown("### 💰 Price & Performance Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        # Current price with change indicator
        change = company_info['change']
        pchange = company_info['pChange']
        change_color = "#28a745" if change >= 0 else "#dc3545"
        change_icon = "📈" if change >= 0 else "📉"
        
        with col1:
            st.markdown(f"""
            <div style="background: white; padding: 1rem; border-radius: 8px; border: 1px solid #dee2e6; text-align: center;">
                <h4 style="margin: 0; color: #1f4e79;">Current Price</h4>
                <h2 style="margin: 0.5rem 0; color: #333;">₹{company_info['lastPrice']:.2f}</h2>
                <p style="margin: 0; color: {change_color}; font-weight: 600;">
                    {change_icon} {change:+.2f} ({pchange:+.2f}%)
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background: white; padding: 1rem; border-radius: 8px; border: 1px solid #dee2e6; text-align: center;">
                <h4 style="margin: 0; color: #1f4e79;">Day Range</h4>
                <p style="margin: 0.5rem 0; font-size: 1.1rem; color: #333;">
                    ₹{company_info['dayLow']:.2f} - ₹{company_info['dayHigh']:.2f}
                </p>
                <p style="margin: 0; color: #666; font-size: 0.9rem;">Low - High</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="background: white; padding: 1rem; border-radius: 8px; border: 1px solid #dee2e6; text-align: center;">
                <h4 style="margin: 0; color: #1f4e79;">52W Range</h4>
                <p style="margin: 0.5rem 0; font-size: 1.1rem; color: #333;">
                    ₹{company_info['yearLow']:.2f} - ₹{company_info['yearHigh']:.2f}
                </p>
                <p style="margin: 0; color: #666; font-size: 0.9rem;">Annual Range</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Additional metrics in expandable sections
        with st.expander("📊 Detailed Trading Metrics", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Open Price", f"₹{company_info['open']:.2f}")
            with col2:
                st.metric("Previous Close", f"₹{company_info['previousClose']:.2f}")
            with col3:
                volume_cr = company_info['totalTradedVolume'] / 1e7 if company_info['totalTradedVolume'] else 0
                st.metric("Volume", f"{volume_cr:.2f} Cr")
            with col4:
                value_cr = company_info['totalTradedValue'] / 1e7 if company_info['totalTradedValue'] else 0
                st.metric("Turnover", f"₹{value_cr:.2f} Cr")
        
        # Market sentiment and key levels
        st.markdown("### 📊 Technical Analysis Summary")
        
        # Calculate some basic technical levels
        current_price = company_info['lastPrice']
        day_high = company_info['dayHigh']
        day_low = company_info['dayLow']
        year_high = company_info['yearHigh']
        year_low = company_info['yearLow']
        
        # Price position analysis
        day_position = ((current_price - day_low) / (day_high - day_low)) * 100 if day_high != day_low else 50
        year_position = ((current_price - year_low) / (year_high - year_low)) * 100 if year_high != year_low else 50
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Day Range Position**")
            st.progress(day_position / 100)
            st.caption(f"Current price is {day_position:.1f}% of today's range")
            
        with col2:
            st.markdown("**52-Week Range Position**")
            st.progress(year_position / 100)
            st.caption(f"Current price is {year_position:.1f}% of annual range")
        
        # Latest news section
        st.markdown("### 📰 Latest News & Updates")
        news_data = market_data.get_company_news(selected_stock)
        
        if news_data:
            for i, news_item in enumerate(news_data[:3]):  # Show top 3 news items
                with st.expander(f"📰 {news_item['title']}", expanded=(i == 0)):
                    st.write(news_item['summary'])
                    if news_item['date']:
                        st.caption(f"Published: {news_item['date']}")
        else:
            st.info("📢 No recent news available for this stock")
    
    # ========================================================================================
    # TAB 2: TECHNICAL CHART ANALYSIS
    # ========================================================================================
    # TAB 2: TECHNICAL CHART ANALYSIS
    # ========================================================================================
    with tabs[1]:
        st.markdown("### 📈 Advanced Technical Analysis")
        
        # Chart configuration controls
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            period = st.selectbox("Analysis Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
        with col2:
            chart_type = st.selectbox("Chart Type", ["Candlestick", "Line", "OHLC"])
        with col3:
            show_volume = st.checkbox("Show Volume", value=True)
        
        # Fetch historical data
        hist_df = market_data.get_historical_data(selected_stock, period)
        
        if not hist_df.empty:
            # Calculate technical indicators
            hist_df['RSI'] = TechnicalAnalysis.calculate_rsi(hist_df['Close'])
            macd_data = TechnicalAnalysis.calculate_macd(hist_df['Close'])
            hist_df['MACD'] = macd_data['macd']
            hist_df['MACD_Signal'] = macd_data['signal']
            hist_df['MACD_Histogram'] = macd_data['histogram']
            
            bollinger_data = TechnicalAnalysis.calculate_bollinger_bands(hist_df['Close'])
            hist_df['BB_Upper'] = bollinger_data['upper']
            hist_df['BB_Middle'] = bollinger_data['middle']
            hist_df['BB_Lower'] = bollinger_data['lower']
            
            hist_df['SMA_20'] = hist_df['Close'].rolling(window=20).mean()
            hist_df['SMA_50'] = hist_df['Close'].rolling(window=50).mean()
            hist_df['EMA_12'] = hist_df['Close'].ewm(span=12).mean()
            hist_df['EMA_26'] = hist_df['Close'].ewm(span=26).mean()
            
            # Create main price chart
            fig = go.Figure()
            
            if chart_type == "Candlestick":
                fig.add_trace(go.Candlestick(
                    x=hist_df.index,
                    open=hist_df['Open'],
                    high=hist_df['High'],
                    low=hist_df['Low'],
                    close=hist_df['Close'],
                    name="Price",
                    increasing_line_color='#26C281',
                    decreasing_line_color='#ED5565'
                ))
            elif chart_type == "Line":
                fig.add_trace(go.Scatter(
                    x=hist_df.index,
                    y=hist_df['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='#1f4e79', width=2)
                ))
            else:  # OHLC
                fig.add_trace(go.Ohlc(
                    x=hist_df.index,
                    open=hist_df['Open'],
                    high=hist_df['High'],
                    low=hist_df['Low'],
                    close=hist_df['Close'],
                    name="OHLC"
                ))
            
            # Add technical indicators
            fig.add_trace(go.Scatter(
                x=hist_df.index, y=hist_df['SMA_20'],
                mode='lines', name='SMA 20',
                line=dict(color='orange', width=1)
            ))
            
            fig.add_trace(go.Scatter(
                x=hist_df.index, y=hist_df['SMA_50'],
                mode='lines', name='SMA 50',
                line=dict(color='purple', width=1)
            ))
            
            # Bollinger Bands
            fig.add_trace(go.Scatter(
                x=hist_df.index, y=hist_df['BB_Upper'],
                mode='lines', name='BB Upper',
                line=dict(color='gray', width=1, dash='dash'),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=hist_df.index, y=hist_df['BB_Lower'],
                mode='lines', name='BB Lower',
                line=dict(color='gray', width=1, dash='dash'),
                fill='tonexty', fillcolor='rgba(128,128,128,0.1)',
                showlegend=False
            ))
            
            # Volume subplot
            if show_volume:
                volume_fig = go.Figure()
                colors = ['red' if hist_df.iloc[i]['Close'] < hist_df.iloc[i]['Open'] else 'green' 
                         for i in range(len(hist_df))]
                
                volume_fig.add_trace(go.Bar(
                    x=hist_df.index,
                    y=hist_df['Volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.7
                ))
                
                volume_fig.update_layout(
                    title="Trading Volume",
                    height=200,
                    showlegend=False,
                    margin=dict(l=0, r=0, t=30, b=0)
                )
            
            # Update main chart layout
            fig.update_layout(
                title=f"{company_info['name']} - Technical Analysis Chart",
                height=600,
                xaxis_rangeslider_visible=False,
                template="plotly_white",
                margin=dict(l=0, r=0, t=50, b=0),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Display charts
            st.plotly_chart(fig, use_container_width=True)
            
            if show_volume:
                st.plotly_chart(volume_fig, use_container_width=True)
            
            # Technical indicators summary
            st.markdown("### 📊 Technical Indicators Summary")
            
            col1, col2, col3 = st.columns(3)
            
            current_rsi = hist_df['RSI'].iloc[-1] if not pd.isna(hist_df['RSI'].iloc[-1]) else 0
            current_macd = hist_df['MACD'].iloc[-1] if not pd.isna(hist_df['MACD'].iloc[-1]) else 0
            current_price = hist_df['Close'].iloc[-1]
            bb_upper = hist_df['BB_Upper'].iloc[-1]
            bb_lower = hist_df['BB_Lower'].iloc[-1]
            
            with col1:
                rsi_color = "🔴" if current_rsi > 70 else "🟢" if current_rsi < 30 else "🟡"
                rsi_signal = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
                st.metric("RSI (14)", f"{current_rsi:.2f}", help=f"{rsi_color} {rsi_signal}")
            
            with col2:
                macd_signal = "Bullish" if current_macd > 0 else "Bearish"
                macd_color = "🟢" if current_macd > 0 else "🔴"
                st.metric("MACD", f"{current_macd:.4f}", help=f"{macd_color} {macd_signal}")
            
            with col3:
                if current_price > bb_upper:
                    bb_signal = "🔴 Above Upper Band"
                elif current_price < bb_lower:
                    bb_signal = "🟢 Below Lower Band"
                else:
                    bb_signal = "🟡 Within Bands"
                st.metric("Bollinger Position", bb_signal)
        
        else:
            st.error("❌ Unable to fetch historical data for technical analysis")
    
    # ========================================================================================
    # TAB 3: PEER ANALYSIS
    # ========================================================================================
    # TAB 3: PEER ANALYSIS
    # ========================================================================================
    with tabs[2]:
        st.markdown("### 🔍 Industry Peer Comparison")
        
        # Get peer companies
        peers = db_manager.get_peer_stocks(company_info["industry"], exclude_symbol=company_info["symbol"])
        
        if peers:
            st.markdown(f"**Industry:** {company_info['industry']}")
            
            # Create peer comparison table
            peer_data = []
            for peer_symbol, peer_name, peer_price, peer_change in peers:
                peer_data.append({
                    "Symbol": peer_symbol,
                    "Company": peer_name,
                    "Price (₹)": f"{peer_price:.2f}",
                    "Change (%)": f"{peer_change:+.2f}%",
                    "Performance": "🟢" if peer_change >= 0 else "🔴"
                })
            
            # Add current stock for comparison
            peer_data.insert(0, {
                "Symbol": f"**{company_info['symbol']}**",
                "Company": f"**{company_info['name']}**",
                "Price (₹)": f"**₹{company_info['lastPrice']:.2f}**",
                "Change (%)": f"**{company_info['pChange']:+.2f}%**",
                "Performance": "🟢" if company_info['pChange'] >= 0 else "🔴"
            })
            
            # Display peer comparison
            peer_df = pd.DataFrame(peer_data)
            st.dataframe(peer_df, use_container_width=True, hide_index=True)
            
            # Peer performance visualization
            st.markdown("### 📊 Peer Performance Comparison")
            
            peer_symbols = [company_info['symbol']] + [peer[0] for peer in peers]
            peer_changes = [company_info['pChange']] + [peer[3] for peer in peers]
            peer_names = [company_info['symbol']] + [peer[0] for peer in peers]
            
            # Create performance chart
            colors = ['#1f4e79' if i == 0 else ('#28a745' if change >= 0 else '#dc3545') 
                     for i, change in enumerate(peer_changes)]
            
            fig_peer = go.Figure(data=[
                go.Bar(
                    x=peer_names,
                    y=peer_changes,
                    marker_color=colors,
                    text=[f"{change:+.2f}%" for change in peer_changes],
                    textposition='auto'
                )
            ])
            
            fig_peer.update_layout(
                title="Daily Performance Comparison (%)",
                height=400,
                showlegend=False,
                template="plotly_white",
                margin=dict(l=0, r=0, t=50, b=0)
            )
            
            st.plotly_chart(fig_peer, use_container_width=True)
            
            # Industry statistics
            st.markdown("### 📈 Industry Statistics")
            
            all_changes = [company_info['pChange']] + [peer[3] for peer in peers]
            avg_change = sum(all_changes) / len(all_changes)
            positive_count = sum(1 for change in all_changes if change >= 0)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Industry Avg Change", f"{avg_change:+.2f}%")
            with col2:
                st.metric("Gainers", f"{positive_count}/{len(all_changes)}")
            with col3:
                market_sentiment = "Bullish" if avg_change > 0 else "Bearish"
                sentiment_color = "🟢" if avg_change > 0 else "🔴"
                st.metric("Market Sentiment", f"{sentiment_color} {market_sentiment}")
            
            # Relative performance analysis
            if company_info['pChange'] > avg_change:
                st.success(f"🎯 **{company_info['symbol']} is outperforming** the industry average by {company_info['pChange'] - avg_change:.2f}%")
            else:
                st.warning(f"⚠️ **{company_info['symbol']} is underperforming** the industry average by {avg_change - company_info['pChange']:.2f}%")
            
        else:
            st.info("📋 No peer companies found in the same industry")
            
        # Additional industry insights
        st.markdown("### 💡 Industry Insights")
        
        with st.expander("📊 Technical Analysis Summary", expanded=True):
            # Calculate some quick technical insights
            current_price = company_info['lastPrice']
            day_high = company_info['dayHigh']
            day_low = company_info['dayLow']
            year_high = company_info['yearHigh']
            year_low = company_info['yearLow']
            
            # Price levels
            resistance_level = day_high
            support_level = day_low
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**📈 Key Levels**")
                st.write(f"• Resistance: ₹{resistance_level:.2f}")
                st.write(f"• Support: ₹{support_level:.2f}")
                st.write(f"• 52W High: ₹{year_high:.2f}")
                st.write(f"• 52W Low: ₹{year_low:.2f}")
            
            with col2:
                st.write("**📊 Price Position**")
                day_position = ((current_price - day_low) / (day_high - day_low)) * 100 if day_high != day_low else 50
                year_position = ((current_price - year_low) / (year_high - year_low)) * 100 if year_high != year_low else 50
                
                st.write(f"• Day Range: {day_position:.1f}%")
                st.write(f"• Year Range: {year_position:.1f}%")
                
                if year_position > 80:
                    st.write("🔥 **Near 52W High**")
                elif year_position < 20:
                    st.write("🔥 **Near 52W Low**")
                else:
                    st.write("📊 **Mid-range Trading**")
    
    # ========================================================================================
    # TAB 4: AI INTELLIGENCE
    # ========================================================================================
    # TAB 4: AI INTELLIGENCE
    # ========================================================================================
    with tabs[3]:
        st.markdown("### 🤖 AI-Powered Investment Intelligence")
        
        with st.spinner("🔍 Analyzing market data with AI..."):
            # Initialize AI analyzer
            try:
                # Add the project root to Python path for imports
                project_root = os.path.dirname(os.path.dirname(__file__))
                if project_root not in sys.path:
                    sys.path.insert(0, project_root)
                
                # Try multiple import methods
                try:
                    from ml_models.stock_prediction_model import StockAIAnalyzer
                except ImportError:
                    # Fallback import method
                    ml_models_dir = os.path.join(os.path.dirname(__file__), "..", "ml_models")
                    sys.path.insert(0, ml_models_dir)
                    from stock_prediction_model import StockAIAnalyzer
                
                ai_analyzer = StockAIAnalyzer()
                
                # Get historical data for AI analysis
                hist_df = market_data.get_historical_data(selected_stock, period="3mo")
                
                if not hist_df.empty:
                    # AI Analysis sections
                    col1, col2 = st.columns(2)
                    
                    # Technical Analysis
                    with col1:
                        st.markdown("#### 📊 Technical Analysis")
                        try:
                            trend_prediction = ai_analyzer.predict_trend(hist_df)
                            
                            # Display trend prediction
                            if trend_prediction == "bullish":
                                st.success("🐂 **BULLISH TREND** detected")
                                st.write("• Technical indicators suggest upward momentum")
                                st.write("• Consider accumulating on dips")
                            elif trend_prediction == "bearish":
                                st.error("🐻 **BEARISH TREND** detected")
                                st.write("• Technical indicators suggest downward pressure")
                                st.write("• Consider booking profits or avoiding fresh longs")
                            else:
                                st.info("📊 **NEUTRAL TREND** detected")
                                st.write("• Market is in consolidation phase")
                                st.write("• Wait for clear directional move")
                            
                        except Exception as e:
                            st.warning(f"Technical analysis unavailable: {e}")
                    
                    # Sentiment Analysis
                    with col2:
                        st.markdown("#### 💭 Market Sentiment")
                        try:
                            sentiment_score = ai_analyzer.analyze_sentiment(company_info['symbol'])
                            
                            if sentiment_score > 0.1:
                                st.success(f"😊 **POSITIVE** sentiment ({sentiment_score:.2f})")
                                st.write("• Social media buzz is positive")
                                st.write("• News sentiment favors the stock")
                            elif sentiment_score < -0.1:
                                st.error(f"😟 **NEGATIVE** sentiment ({sentiment_score:.2f})")
                                st.write("• Social media buzz is negative")
                                st.write("• News sentiment is unfavorable")
                            else:
                                st.info(f"😐 **NEUTRAL** sentiment ({sentiment_score:.2f})")
                                st.write("• Mixed social media sentiment")
                                st.write("• No clear news direction")
                            
                        except Exception as e:
                            st.warning(f"Sentiment analysis unavailable: {e}")
                    
                    # Comprehensive AI Recommendation
                    st.markdown("### 🎯 AI Investment Recommendation")
                    
                    try:
                        recommendation = ai_analyzer.generate_recommendation(
                            hist_df, 
                            company_info['symbol'], 
                            company_info
                        )
                        
                        # Parse recommendation
                        if isinstance(recommendation, dict):
                            action = recommendation.get('action', 'HOLD').upper()
                            confidence = recommendation.get('confidence', 0.5)
                            reasoning = recommendation.get('reasoning', 'No specific reasoning available')
                            target_price = recommendation.get('target_price')
                            stop_loss = recommendation.get('stop_loss')
                            
                            # Display recommendation with styling
                            if action == "BUY":
                                st.success(f"🟢 **{action}** recommendation")
                                rec_color = "#28a745"
                            elif action == "SELL":
                                st.error(f"🔴 **{action}** recommendation")
                                rec_color = "#dc3545"
                            else:
                                st.info(f"🟡 **{action}** recommendation")
                                rec_color = "#ffc107"
                            
                            # Confidence meter
                            st.markdown(f"**Confidence Level:** {confidence:.1%}")
                            st.progress(confidence)
                            
                            # Detailed recommendation card
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
                                        padding: 1.5rem; border-radius: 10px; border-left: 4px solid {rec_color}; margin: 1rem 0;">
                                <h4 style="color: {rec_color}; margin: 0 0 1rem 0;">AI Analysis Summary</h4>
                                <p style="margin: 0; line-height: 1.6; color: #495057;">{reasoning}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Price targets if available
                            if target_price or stop_loss:
                                st.markdown("#### 🎯 Price Targets")
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Current Price", f"₹{company_info['lastPrice']:.2f}")
                                
                                if target_price:
                                    with col2:
                                        potential_gain = ((target_price - company_info['lastPrice']) / company_info['lastPrice']) * 100
                                        st.metric("Target Price", f"₹{target_price:.2f}", f"{potential_gain:+.1f}%")
                                
                                if stop_loss:
                                    with col3:
                                        potential_loss = ((stop_loss - company_info['lastPrice']) / company_info['lastPrice']) * 100
                                        st.metric("Stop Loss", f"₹{stop_loss:.2f}", f"{potential_loss:+.1f}%")
                        
                        else:
                            st.info("🤖 AI recommendation: " + str(recommendation))
                            
                    except Exception as e:
                        st.error(f"AI recommendation unavailable: {e}")
                    
                    # Market Intelligence Dashboard
                    st.markdown("### 📈 Market Intelligence Dashboard")
                    
                    # Key metrics from AI analysis
                    try:
                        # Calculate some advanced metrics
                        volatility = hist_df['Close'].pct_change().std() * 100
                        momentum = ((hist_df['Close'].iloc[-1] - hist_df['Close'].iloc[-10]) / hist_df['Close'].iloc[-10]) * 100
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Volatility", f"{volatility:.2f}%", 
                                    help="Price volatility over analysis period")
                        
                        with col2:
                            momentum_delta = f"{momentum:+.2f}%" if momentum != 0 else "0.00%"
                            st.metric("10D Momentum", momentum_delta,
                                    help="Price momentum over last 10 days")
                        
                        with col3:
                            # Simple trend strength
                            trend_strength = abs(momentum) / volatility if volatility > 0 else 0
                            st.metric("Trend Strength", f"{trend_strength:.2f}",
                                    help="Ratio of momentum to volatility")
                        
                        with col4:
                            # Risk assessment
                            risk_level = "High" if volatility > 3 else "Medium" if volatility > 1.5 else "Low"
                            risk_color = "🔴" if risk_level == "High" else "🟡" if risk_level == "Medium" else "🟢"
                            st.metric("Risk Level", f"{risk_color} {risk_level}",
                                    help="Overall risk assessment")
                        
                    except Exception as e:
                        st.warning(f"Advanced metrics calculation failed: {e}")
                    
                    # Disclaimer
                    st.markdown("---")
                    st.markdown("""
                    <div style="background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px; padding: 1rem; margin: 1rem 0;">
                        <h5 style="color: #856404; margin: 0 0 0.5rem 0;">⚠️ Important Disclaimer</h5>
                        <p style="margin: 0; color: #856404; font-size: 0.9rem;">
                            This AI analysis is for educational purposes only and should not be considered as financial advice. 
                            Please consult with a qualified financial advisor before making investment decisions. 
                            Past performance does not guarantee future results.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                else:
                    st.error("❌ Unable to fetch historical data for AI analysis")
                    
            except Exception as e:
                st.error(f"❌ AI Analysis Error: {e}")
                st.info("💡 The AI analysis module is currently unavailable. Please ensure all dependencies are installed.")

else:
    st.error("❌ No company information available. Please check your stock selection.")
    st.stop()

# ========================================================================================
# FOOTER
# ========================================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem; color: #6c757d; font-size: 0.9rem;">
    <p>📈 <strong>Nifty 50 Professional Analytics</strong> | Powered by AI & Real-time Market Data</p>
    <p>🔒 Data sources: NSE, BSE, Yahoo Finance | Built with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True)