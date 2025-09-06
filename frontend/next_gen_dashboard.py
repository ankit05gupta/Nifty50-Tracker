"""
Next-Generation Professional Stock Analysis Dashboard
===================================================

A comprehensive financial analytics platform that combines multiple data sources
for superior market intelligence and analysis.

Data Sources:
- RapidAPI Indian Stock Exchange (Real-time NSE data)
- Yahoo Finance (Historical data and charts)
- Reddit API (Social sentiment analysis)
- SQLite Database (Local data storage)

Features:
- Multi-source data aggregation
- Real-time market updates
- Advanced sentiment analysis
- Professional technical indicators
- AI-powered recommendations
- Interactive visualizations
- Risk assessment tools

Author: Nifty50 Analytics Team
Version: 3.0 - Next Generation
License: MIT
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sqlite3
from datetime import datetime, timedelta
import warnings
import sys
import os
import time
from typing import Dict, List, Optional, Any
import json

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import our enhanced data provider
from app.enhanced_data_provider import enhanced_data_provider

warnings.filterwarnings('ignore')

# ===============================================================================
# PAGE CONFIGURATION
# ===============================================================================
st.set_page_config(
    page_title="Nifty50 Tracker Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================================================================
# CUSTOM CSS STYLING
# ===============================================================================
st.markdown("""
<style>
    /* Premium Dark Theme */
    .stApp {
        background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%);
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #374151;
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    }
    
    /* Data Source Indicators */
    .data-source {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        margin: 0.2rem;
    }
    
    .rapidapi-source {
        background: linear-gradient(45deg, #ff6b6b, #ff8e8e);
        color: white;
    }
    
    .yfinance-source {
        background: linear-gradient(45deg, #4ecdc4, #44b3a3);
        color: white;
    }
    
    .reddit-source {
        background: linear-gradient(45deg, #ff9500, #ffab33);
        color: white;
    }
    
    .database-source {
        background: linear-gradient(45deg, #6c5ce7, #8777d9);
        color: white;
    }
    
    /* Sentiment Indicators */
    .sentiment-positive {
        color: #10b981;
        font-weight: bold;
    }
    
    .sentiment-negative {
        color: #ef4444;
        font-weight: bold;
    }
    
    .sentiment-neutral {
        color: #f59e0b;
        font-weight: bold;
    }
    
    /* Professional Tables */
    .dataframe {
        background: #1f2937;
        border-radius: 8px;
        border: 1px solid #374151;
    }
    
    /* Advanced Indicators */
    .indicator-bullish {
        background: linear-gradient(45deg, #10b981, #34d399);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: bold;
    }
    
    .indicator-bearish {
        background: linear-gradient(45deg, #ef4444, #f87171);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: bold;
    }
    
    .indicator-neutral {
        background: linear-gradient(45deg, #f59e0b, #fbbf24);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ===============================================================================
# HEADER SECTION
# ===============================================================================
st.markdown("""
<div class="main-header">
    <h1>🚀 Nifty50 Tracker Pro - NSE Edition</h1>
    <p style="font-size: 1.2rem; margin: 0;">NSE (National Stock Exchange) Financial Intelligence Platform</p>
    <div style="margin-top: 1rem;">
        <span class="data-source rapidapi-source">🔴 NSE RapidAPI</span>
        <span class="data-source yfinance-source">📊 NSE Yahoo Finance</span>
        <span class="data-source reddit-source">💬 NSE Sentiment</span>
        <span class="data-source database-source">🗄️ NSE Database</span>
    </div>
    <p style="font-size: 0.9rem; margin-top: 1rem; opacity: 0.8;">
        Exclusively focused on NSE data • BSE data excluded
    </p>
</div>
""", unsafe_allow_html=True)

# ===============================================================================
# SIDEBAR NAVIGATION
# ===============================================================================
st.sidebar.markdown("## 🎛️ Dashboard Controls")

# Navigation
page = st.sidebar.selectbox(
    "📍 NSE Navigation",
    [
        "🏠 NSE Market Overview",
        "📈 NSE Live Analysis", 
        "🔍 NSE Stock Research",
        "💹 NSE Multi-Source Data",
        "🤖 NSE AI Intelligence",
        "📊 NSE Advanced Charts",
        "🌐 NSE Social Sentiment"
    ]
)

# Refresh controls
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔄 NSE Data Refresh")
auto_refresh = st.sidebar.checkbox("Auto Refresh NSE (30s)", value=False)
if st.sidebar.button("🔄 Refresh NSE Data") or auto_refresh:
    st.rerun()

# NSE Data source toggles
st.sidebar.markdown("---")
st.sidebar.markdown("### 📡 NSE Data Sources")
use_rapidapi = st.sidebar.checkbox("🔴 NSE RapidAPI", value=True)
use_yfinance = st.sidebar.checkbox("📊 NSE Yahoo Finance", value=True)
use_reddit = st.sidebar.checkbox("💬 NSE Reddit", value=True)

# ===============================================================================
# UTILITY FUNCTIONS
# ===============================================================================

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_database_stocks():
    """Get NSE stocks from local database."""
    try:
        conn = sqlite3.connect('data/nifty50_stocks.db')
        # Filter to only include NSE stocks and exclude BSE
        df = pd.read_sql_query("""
            SELECT * FROM stock_list 
            WHERE UPPER(exchange) = 'NSE' OR exchange IS NULL
            ORDER BY pChange DESC
        """, conn)
        conn.close()
        
        # Additional filtering to ensure NSE symbols only
        if not df.empty:
            # Remove any symbols that might be BSE-specific
            df = df[~df['symbol'].str.contains('BSE', case=False, na=False)]
            # Ensure symbols are NSE format (no .BSE suffix)
            df = df[~df['symbol'].str.endswith('.BSE', na=False)]
            
        return df
    except Exception as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame()

def format_currency(value):
    """Format currency values."""
    if pd.isna(value) or value == 0:
        return "₹0.00"
    if abs(value) >= 1e7:
        return f"₹{value/1e7:.2f}Cr"
    elif abs(value) >= 1e5:
        return f"₹{value/1e5:.2f}L"
    elif abs(value) >= 1e3:
        return f"₹{value/1e3:.2f}K"
    else:
        return f"₹{value:.2f}"

def format_percentage(value):
    """Format percentage values with color."""
    if pd.isna(value):
        return "0.00%"
    
    color = "green" if value > 0 else "red" if value < 0 else "orange"
    sign = "+" if value > 0 else ""
    return f'<span style="color: {color}; font-weight: bold;">{sign}{value:.2f}%</span>'

def get_sentiment_indicator(score):
    """Get sentiment indicator based on score."""
    if score > 0.2:
        return "🟢 Bullish", "sentiment-positive"
    elif score < -0.2:
        return "🔴 Bearish", "sentiment-negative"
    else:
        return "🟡 Neutral", "sentiment-neutral"

# ===============================================================================
# PAGE ROUTING
# ===============================================================================

if page == "🏠 NSE Market Overview":
    st.markdown("## 🏠 NSE Real-Time Market Overview")
    
    # Get NSE market overview from enhanced provider
    market_overview = enhanced_data_provider.get_market_overview()
    
    # NSE Market status from RapidAPI
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #3b82f6;">📊 NSE Status</h3>
            <p style="font-size: 1.5rem; margin: 0;">OPEN</p>
            <small class="data-source rapidapi-source">NSE RapidAPI</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Get NSE database stats
        db_stocks = get_database_stocks()
        total_stocks = len(db_stocks) if not db_stocks.empty else 0
        
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #10b981;">📈 NSE Stocks</h3>
            <p style="font-size: 1.5rem; margin: 0;">{total_stocks}</p>
            <small class="data-source database-source">NSE Database</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        gainers = len(db_stocks[db_stocks['pChange'] > 0]) if not db_stocks.empty else 0
        
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #10b981;">⬆️ Gainers</h3>
            <p style="font-size: 1.5rem; margin: 0; color: #10b981;">{gainers}</p>
            <small class="data-source database-source">Database</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        losers = len(db_stocks[db_stocks['pChange'] < 0]) if not db_stocks.empty else 0
        
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #ef4444;">⬇️ Losers</h3>
            <p style="font-size: 1.5rem; margin: 0; color: #ef4444;">{losers}</p>
            <small class="data-source database-source">Database</small>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Top performers section
    if not db_stocks.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🚀 Top Gainers")
            top_gainers = db_stocks.nlargest(5, 'pChange')[['symbol', 'name', 'lastPrice', 'pChange']]
            
            for _, stock in top_gainers.iterrows():
                st.markdown(f"""
                <div class="metric-card" style="margin-bottom: 0.5rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong>{stock['symbol']}</strong><br>
                            <small>{stock['name']}</small>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 1.2rem;">₹{stock['lastPrice']:.2f}</div>
                            <div style="color: #10b981; font-weight: bold;">+{stock['pChange']:.2f}%</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 📉 Top Losers")
            top_losers = db_stocks.nsmallest(5, 'pChange')[['symbol', 'name', 'lastPrice', 'pChange']]
            
            for _, stock in top_losers.iterrows():
                st.markdown(f"""
                <div class="metric-card" style="margin-bottom: 0.5rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong>{stock['symbol']}</strong><br>
                            <small>{stock['name']}</small>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 1.2rem;">₹{stock['lastPrice']:.2f}</div>
                            <div style="color: #ef4444; font-weight: bold;">{stock['pChange']:.2f}%</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

elif page == "📈 NSE Live Analysis":
    st.markdown("## 📈 NSE Live Stock Analysis")
    
    # Stock selector
    db_stocks = get_database_stocks()
    if not db_stocks.empty:
        selected_stock = st.selectbox(
            "🎯 Select Stock for Analysis",
            options=db_stocks['symbol'].tolist(),
            format_func=lambda x: f"{x} - {db_stocks[db_stocks['symbol']==x]['name'].iloc[0] if len(db_stocks[db_stocks['symbol']==x]) > 0 else x}"
        )
        
        if selected_stock:
            # Get comprehensive data
            comprehensive_data = enhanced_data_provider.get_comprehensive_stock_data(selected_stock)
            
            # Display data source status
            st.markdown("### 📡 Data Source Status")
            col1, col2, col3 = st.columns(3)
            
            rapidapi_status = comprehensive_data['status'].get('rapidapi', 'unknown')
            yfinance_status = comprehensive_data['status'].get('yfinance', 'unknown')
            reddit_status = comprehensive_data['status'].get('reddit', 'unknown')
            
            with col1:
                status_color = "green" if rapidapi_status == 'success' else "red"
                st.markdown(f"""
                <div class="metric-card">
                    <span class="data-source rapidapi-source">🔴 RapidAPI</span><br>
                    <span style="color: {status_color};">{'✅ Active' if rapidapi_status == 'success' else '❌ Offline'}</span>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                status_color = "green" if yfinance_status == 'success' else "red"
                st.markdown(f"""
                <div class="metric-card">
                    <span class="data-source yfinance-source">📊 Yahoo Finance</span><br>
                    <span style="color: {status_color};">{'✅ Active' if yfinance_status == 'success' else '❌ Offline'}</span>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                status_color = "green" if reddit_status in ['success', 'demo'] else "red"
                reddit_label = "✅ Active" if reddit_status == 'success' else "🔶 Demo" if reddit_status == 'demo' else "❌ Offline"
                st.markdown(f"""
                <div class="metric-card">
                    <span class="data-source reddit-source">💬 Reddit</span><br>
                    <span style="color: {status_color};">{reddit_label}</span>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Stock information from database
            stock_info = db_stocks[db_stocks['symbol'] == selected_stock].iloc[0]
            
            # Main metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>💰 Current Price</h3>
                    <p style="font-size: 2rem; margin: 0;">₹{stock_info['lastPrice']:.2f}</p>
                    <small class="data-source database-source">Database</small>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                change_color = "green" if stock_info['change'] > 0 else "red"
                change_sign = "+" if stock_info['change'] > 0 else ""
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📊 Change</h3>
                    <p style="font-size: 1.5rem; margin: 0; color: {change_color};">{change_sign}₹{stock_info['change']:.2f}</p>
                    <p style="color: {change_color};">({change_sign}{stock_info['pChange']:.2f}%)</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📈 Day High</h3>
                    <p style="font-size: 1.5rem; margin: 0;">₹{stock_info['dayHigh']:.2f}</p>
                    <small>Low: ₹{stock_info['dayLow']:.2f}</small>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                volume_formatted = format_currency(stock_info['totalTradedVolume'])
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📊 Volume</h3>
                    <p style="font-size: 1.5rem; margin: 0;">{volume_formatted}</p>
                    <small class="data-source database-source">Traded Volume</small>
                </div>
                """, unsafe_allow_html=True)
            
            # Reddit Sentiment Analysis
            if comprehensive_data['reddit_sentiment']:
                st.markdown("### 💬 Social Sentiment Analysis")
                
                sentiment_data = comprehensive_data['reddit_sentiment']
                sentiment_score = sentiment_data['sentiment_score']
                sentiment_text, sentiment_class = get_sentiment_indicator(sentiment_score)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>🎭 Sentiment Score</h4>
                        <p class="{sentiment_class}" style="font-size: 1.8rem;">{sentiment_text}</p>
                        <small>Score: {sentiment_score:.2f}</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>📝 Posts Analyzed</h4>
                        <p style="font-size: 1.5rem;">{sentiment_data['post_count']}</p>
                        <small class="data-source reddit-source">Reddit Posts</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>💬 Total Comments</h4>
                        <p style="font-size: 1.5rem;">{sentiment_data['total_comments']}</p>
                        <small>Engagement Level</small>
                    </div>
                    """, unsafe_allow_html=True)

elif page == "🔍 NSE Stock Research":
    st.markdown("## 🔍 NSE Deep Stock Research")
    
    db_stocks = get_database_stocks()
    if not db_stocks.empty:
        # Stock selector with search
        st.markdown("### 🎯 Stock Selection")
        
        # Create a searchable interface
        search_term = st.text_input("🔍 Search stocks...", placeholder="Enter symbol or company name")
        
        if search_term:
            filtered_stocks = db_stocks[
                (db_stocks['symbol'].str.contains(search_term.upper(), na=False)) |
                (db_stocks['name'].str.contains(search_term, case=False, na=False))
            ]
        else:
            filtered_stocks = db_stocks.head(20)  # Show top 20 by default
        
        # Display filtered results
        if not filtered_stocks.empty:
            st.markdown("### 📊 Search Results")
            
            # Create interactive table
            display_cols = ['symbol', 'name', 'lastPrice', 'change', 'pChange', 'dayHigh', 'dayLow']
            
            for _, stock in filtered_stocks.iterrows():
                with st.expander(f"📈 {stock['symbol']} - {stock['name']}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("#### 💰 Price Information")
                        st.write(f"**Current Price:** ₹{stock['lastPrice']:.2f}")
                        st.write(f"**Change:** ₹{stock['change']:.2f} ({stock['pChange']:.2f}%)")
                        st.write(f"**Day Range:** ₹{stock['dayLow']:.2f} - ₹{stock['dayHigh']:.2f}")
                    
                    with col2:
                        st.markdown("#### 📊 Trading Data")
                        st.write(f"**Volume:** {format_currency(stock['totalTradedVolume'])}")
                        st.write(f"**Value:** {format_currency(stock['totalTradedValue'])}")
                        st.write(f"**Previous Close:** ₹{stock['previousClose']:.2f}")
                    
                    with col3:
                        st.markdown("#### 🏢 Company Info")
                        st.write(f"**Industry:** {stock['industry']}")
                        st.write(f"**Exchange:** {stock['exchange']}")
                        st.write(f"**ISIN:** {stock['isin']}")
                    
                    # Get comprehensive data for this stock
                    if st.button(f"📊 NSE Analysis", key=f"nse_analysis_{stock['symbol']}_{hash(stock['name'])}"):
                        comprehensive_data = enhanced_data_provider.get_comprehensive_stock_data(stock['symbol'])
                        
                        # Display historical chart if available
                        if comprehensive_data['historical_data'] is not None:
                            hist_data = comprehensive_data['historical_data']
                            
                            fig = go.Figure()
                            fig.add_trace(go.Candlestick(
                                x=hist_data.index,
                                open=hist_data['Open'],
                                high=hist_data['High'],
                                low=hist_data['Low'],
                                close=hist_data['Close'],
                                name=stock['symbol']
                            ))
                            
                            fig.update_layout(
                                title=f"{stock['symbol']} - 3 Month Chart",
                                xaxis_title="Date",
                                yaxis_title="Price (₹)",
                                template="plotly_dark",
                                height=400
                            )
                            
                            st.plotly_chart(fig, width='stretch')

elif page == "💹 NSE Multi-Source Data":
    st.markdown("## 💹 NSE Multi-Source Data Comparison")
    
    db_stocks = get_database_stocks()
    if not db_stocks.empty:
        selected_stock = st.selectbox(
            "Select stock for multi-source analysis:",
            options=db_stocks['symbol'].tolist()
        )
        
        if selected_stock:
            st.markdown(f"### 🔍 Multi-Source Analysis: {selected_stock}")
            
            # Get data from all sources
            comprehensive_data = enhanced_data_provider.get_comprehensive_stock_data(selected_stock)
            
            # Data Source Status Panel
            st.markdown("#### 📡 Data Source Status")
            status_col1, status_col2, status_col3, status_col4 = st.columns(4)
            
            status = comprehensive_data.get('status', {})
            
            with status_col1:
                rapidapi_status = status.get('rapidapi', 'unknown')
                if rapidapi_status == 'success':
                    st.success("🔴 RapidAPI: Live")
                elif rapidapi_status == 'demo':
                    st.warning("🔴 RapidAPI: Demo Mode")
                else:
                    st.error("🔴 RapidAPI: Offline")
            
            with status_col2:
                yfinance_status = status.get('yfinance', 'unknown')
                if yfinance_status == 'success':
                    st.success("📊 Yahoo Finance: Live")
                else:
                    st.error("📊 Yahoo Finance: Offline")
            
            with status_col3:
                reddit_status = status.get('reddit', 'unknown')
                if reddit_status == 'success':
                    st.success("💬 Reddit: Live")
                elif reddit_status == 'demo':
                    st.warning("💬 Reddit: Demo Mode")
                else:
                    st.error("💬 Reddit: Offline")
            
            with status_col4:
                db_status = "success" if not db_stocks.empty else "failed"
                if db_status == 'success':
                    st.success("🗄️ Database: Active")
                else:
                    st.error("🗄️ Database: Offline")
            
            # Display data from each source
            tab1, tab2, tab3, tab4 = st.tabs(["🔴 RapidAPI", "📊 Yahoo Finance", "💬 Reddit", "🗄️ Database"])
            
            with tab1:
                st.markdown("#### 🔴 RapidAPI Data")
                if comprehensive_data['rapidapi_data']:
                    rapidapi_data = comprehensive_data['rapidapi_data']
                    
                    # Check if this is demo data
                    if rapidapi_data.get('status') == 'demo_enhanced':
                        st.info("⚠️ **Demo Mode**: RapidAPI is currently unavailable. Showing enhanced demo data.")
                        st.markdown("**Note:** " + rapidapi_data.get('note', 'Using fallback data'))
                    
                    # Display key metrics in a formatted way
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Current Price", f"₹{rapidapi_data.get('lastPrice', 0):.2f}")
                        st.metric("Change", f"₹{rapidapi_data.get('change', 0):.2f}")
                    
                    with col2:
                        st.metric("Change %", f"{rapidapi_data.get('pChange', 0):.2f}%")
                        st.metric("Volume", f"{rapidapi_data.get('volume', 0):,}")
                    
                    with col3:
                        st.metric("Day High", f"₹{rapidapi_data.get('dayHigh', 0):.2f}")
                        st.metric("Day Low", f"₹{rapidapi_data.get('dayLow', 0):.2f}")
                    
                    # Show full data
                    with st.expander("📋 Full RapidAPI Data"):
                        # Create a clean copy of the data for display
                        display_data = dict(rapidapi_data)
                        
                        # Import json here to avoid repetition
                        import json
                        
                        # Section 1: Price Information
                        st.markdown("---")
                        st.write("**📊 Price Information**")
                        price_data = {
                            "symbol": display_data.get("symbol"),
                            "exchange": display_data.get("exchange"),
                            "lastPrice": display_data.get("lastPrice"),
                            "change": display_data.get("change"),
                            "pChange": display_data.get("pChange"),
                            "previousClose": display_data.get("previousClose")
                        }
                        st.code(json.dumps(price_data, indent=2), language="json")
                        
                        # Section 2: Trading Information
                        st.markdown("---")
                        st.write("**📈 Trading Information**")
                        trading_data = {
                            "dayHigh": display_data.get("dayHigh"),
                            "dayLow": display_data.get("dayLow"),
                            "open": display_data.get("open"),
                            "volume": display_data.get("volume"),
                            "totalTradedValue": display_data.get("totalTradedValue"),
                            "totalTradedVolume": display_data.get("totalTradedVolume")
                        }
                        st.code(json.dumps(trading_data, indent=2), language="json")
                        
                        # Section 3: Market Metrics
                        st.markdown("---")
                        st.write("**📊 Market Metrics**")
                        metrics_data = {
                            "marketCap": display_data.get("marketCap"),
                            "pe": display_data.get("pe"),
                            "pb": display_data.get("pb"),
                            "eps": display_data.get("eps"),
                            "yearHigh": display_data.get("yearHigh"),
                            "yearLow": display_data.get("yearLow")
                        }
                        st.code(json.dumps(metrics_data, indent=2), language="json")
                        
                        # Section 4: Data Source Information
                        st.markdown("---")
                        st.write("**🔧 Data Source Information**")
                        source_data = {
                            "status": display_data.get("status"),
                            "source": display_data.get("source"),
                            "timestamp": display_data.get("timestamp"),
                            "note": display_data.get("note")
                        }
                        st.code(json.dumps(source_data, indent=2), language="json")
                else:
                    st.error("❌ RapidAPI data not available - Service temporarily offline")
            
            with tab2:
                st.markdown("#### 📊 Yahoo Finance Historical Data")
                if comprehensive_data['historical_data'] is not None:
                    hist_data = comprehensive_data['historical_data']
                    st.dataframe(hist_data.tail(10))
                    
                    # Volume chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=hist_data.index,
                        y=hist_data['Volume'],
                        mode='lines',
                        name='Volume',
                        line=dict(color='cyan')
                    ))
                    fig.update_layout(
                        title="Trading Volume Trend",
                        template="plotly_dark",
                        height=300
                    )
                    st.plotly_chart(fig, width='stretch')
                else:
                    st.warning("Historical data not available")
            
            with tab3:
                st.markdown("#### 💬 Reddit Sentiment Data")
                if comprehensive_data['reddit_sentiment']:
                    sentiment_data = comprehensive_data['reddit_sentiment']
                    
                    # Sentiment visualization
                    sentiment_score = sentiment_data['sentiment_score']
                    
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = sentiment_score,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Sentiment Score"},
                        gauge = {
                            'axis': {'range': [-1, 1]},
                            'bar': {'color': "cyan"},
                            'steps': [
                                {'range': [-1, -0.3], 'color': "red"},
                                {'range': [-0.3, 0.3], 'color': "yellow"},
                                {'range': [0.3, 1], 'color': "green"}
                            ],
                            'threshold': {
                                'line': {'color': "white", 'width': 4},
                                'thickness': 0.75,
                                'value': sentiment_score
                            }
                        }
                    ))
                    fig.update_layout(template="plotly_dark", height=400)
                    st.plotly_chart(fig, width='stretch')
                    
                    # Display sentiment metrics in a clean format
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            label="📊 Sentiment Score",
                            value=f"{sentiment_score:.3f}",
                            delta=None
                        )
                    
                    with col2:
                        st.metric(
                            label="📝 Posts Analyzed",
                            value=sentiment_data.get('post_count', 0),
                            delta=None
                        )
                    
                    with col3:
                        status = sentiment_data.get('status', 'unknown')
                        status_emoji = "✅" if status == 'success' else "🔶" if status == 'demo' else "❌"
                        st.metric(
                            label="📡 Data Source",
                            value=f"{status_emoji} {status.title()}",
                            delta=None
                        )
                    
                    # Additional metrics
                    st.markdown("#### 📈 Reddit Engagement Metrics")
                    
                    metrics_col1, metrics_col2 = st.columns(2)
                    
                    with metrics_col1:
                        avg_score = sentiment_data.get('avg_score', 0)
                        st.write(f"**Average Post Score:** {avg_score:.1f}")
                        
                        total_comments = sentiment_data.get('total_comments', 0)
                        st.write(f"**Total Comments:** {total_comments:,}")
                    
                    with metrics_col2:
                        # Sentiment interpretation
                        if sentiment_score > 0.3:
                            sentiment_text = "🟢 Positive"
                        elif sentiment_score < -0.3:
                            sentiment_text = "🔴 Negative" 
                        else:
                            sentiment_text = "🟡 Neutral"
                        
                        st.write(f"**Sentiment:** {sentiment_text}")
                        
                        # Data freshness
                        st.write(f"**Data Type:** {'Live Reddit Data' if status == 'success' else 'Demo Data'}")
                
                else:
                    st.warning("Reddit sentiment data not available")
            
            with tab4:
                st.markdown("#### 🗄️ Database Information")
                stock_info = db_stocks[db_stocks['symbol'] == selected_stock].iloc[0]
                
                # Create a formatted display
                info_dict = stock_info.to_dict()
                st.json(info_dict)

elif page == "🤖 NSE AI Intelligence":
    st.markdown("## 🤖 NSE AI-Powered Market Intelligence")
    
    # Import AI analyzer
    try:
        from ml_models.stock_prediction_model import StockAIAnalyzer
        ai_analyzer = StockAIAnalyzer()
        
        db_stocks = get_database_stocks()
        if not db_stocks.empty:
            selected_stock = st.selectbox(
                "Select stock for AI analysis:",
                options=db_stocks['symbol'].tolist()
            )
            
            if selected_stock:
                st.markdown(f"### 🧠 AI Analysis: {selected_stock}")
                
                with st.spinner("🤖 Running AI analysis..."):
                    # Get AI analysis
                    ai_analysis = ai_analyzer.analyze_stock(selected_stock)
                    
                    if ai_analysis:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            recommendation = ai_analysis.get('recommendation', 'HOLD')
                            confidence = ai_analysis.get('confidence', 0.5)
                            
                            rec_color = "#10b981" if recommendation == "BUY" else "#ef4444" if recommendation == "SELL" else "#f59e0b"
                            
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3>🎯 AI Recommendation</h3>
                                <p style="font-size: 2rem; color: {rec_color}; margin: 0;">{recommendation}</p>
                                <p>Confidence: {confidence:.1%}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            target_price = ai_analysis.get('target_price')
                            if target_price:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h3>🎯 Target Price</h3>
                                    <p style="font-size: 1.8rem; margin: 0;">₹{target_price:.2f}</p>
                                    <small>AI Prediction</small>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        with col3:
                            stop_loss = ai_analysis.get('stop_loss')
                            if stop_loss:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h3>🛡️ Stop Loss</h3>
                                    <p style="font-size: 1.8rem; margin: 0; color: #ef4444;">₹{stop_loss:.2f}</p>
                                    <small>Risk Management</small>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # AI Insights
                        st.markdown("### 💡 AI Insights")
                        insights = ai_analysis.get('insights', [])
                        for i, insight in enumerate(insights):
                            icon = "🟢" if i == 0 else "🟡" if i == 1 else "🔵"
                            st.markdown(f"{icon} {insight}")
                        
                        # Technical scores visualization
                        st.markdown("### 📊 Analysis Scores")
                        
                        scores = {
                            'Technical': ai_analysis.get('technical_score', 0),
                            'Sentiment': ai_analysis.get('sentiment_score', 0),
                            'Combined': ai_analysis.get('combined_score', 0)
                        }
                        
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=list(scores.keys()),
                            y=list(scores.values()),
                            marker_color=['cyan', 'orange', 'green']
                        ))
                        fig.update_layout(
                            title="AI Analysis Scores",
                            template="plotly_dark",
                            height=400
                        )
                        st.plotly_chart(fig, width='stretch')
    
    except ImportError:
        st.warning("🤖 AI analysis module not available. Please ensure ML models are installed.")

elif page == "📊 NSE Advanced Charts":
    st.markdown("## 📊 NSE Advanced Technical Charts")
    
    db_stocks = get_database_stocks()
    if not db_stocks.empty:
        selected_stock = st.selectbox(
            "Select stock for advanced charting:",
            options=db_stocks['symbol'].tolist()
        )
        
        if selected_stock:
            # Get historical data
            hist_data = enhanced_data_provider.get_historical_data_yfinance(selected_stock, "6mo")
            
            if not hist_data.empty:
                # Chart type selector
                chart_type = st.selectbox(
                    "Chart Type:",
                    ["Candlestick", "OHLC", "Line", "Area"]
                )
                
                # Create the chart
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=(f'{selected_stock} Price Chart', 'Volume'),
                    vertical_spacing=0.1,
                    row_heights=[0.7, 0.3]
                )
                
                # Main price chart
                if chart_type == "Candlestick":
                    fig.add_trace(go.Candlestick(
                        x=hist_data.index,
                        open=hist_data['Open'],
                        high=hist_data['High'],
                        low=hist_data['Low'],
                        close=hist_data['Close'],
                        name=selected_stock
                    ), row=1, col=1)
                
                elif chart_type == "Line":
                    fig.add_trace(go.Scatter(
                        x=hist_data.index,
                        y=hist_data['Close'],
                        mode='lines',
                        name=f'{selected_stock} Close',
                        line=dict(color='cyan', width=2)
                    ), row=1, col=1)
                
                # Volume chart
                fig.add_trace(go.Bar(
                    x=hist_data.index,
                    y=hist_data['Volume'],
                    name='Volume',
                    marker_color='rgba(255, 255, 255, 0.3)'
                ), row=2, col=1)
                
                # Add moving averages
                if len(hist_data) >= 20:
                    ma20 = hist_data['Close'].rolling(window=20).mean()
                    ma50 = hist_data['Close'].rolling(window=50).mean()
                    
                    fig.add_trace(go.Scatter(
                        x=hist_data.index,
                        y=ma20,
                        mode='lines',
                        name='MA20',
                        line=dict(color='orange', width=1)
                    ), row=1, col=1)
                    
                    if len(hist_data) >= 50:
                        fig.add_trace(go.Scatter(
                            x=hist_data.index,
                            y=ma50,
                            mode='lines',
                            name='MA50',
                            line=dict(color='purple', width=1)
                        ), row=1, col=1)
                
                fig.update_layout(
                    title=f"Advanced Chart Analysis - {selected_stock}",
                    template="plotly_dark",
                    height=700,
                    xaxis_rangeslider_visible=False
                )
                
                st.plotly_chart(fig, width='stretch')
                
                # Technical indicators
                st.markdown("### 📈 Technical Indicators")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # RSI calculation
                    delta = hist_data['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    current_rsi = rsi.iloc[-1] if not rsi.empty else 50
                    
                    rsi_status = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
                    rsi_color = "#ef4444" if current_rsi > 70 else "#10b981" if current_rsi < 30 else "#f59e0b"
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>📊 RSI (14)</h4>
                        <p style="font-size: 1.5rem; color: {rsi_color};">{current_rsi:.2f}</p>
                        <small>{rsi_status}</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # MACD calculation (simplified)
                    if len(hist_data) >= 26:
                        ema12 = hist_data['Close'].ewm(span=12).mean()
                        ema26 = hist_data['Close'].ewm(span=26).mean()
                        macd = ema12 - ema26
                        current_macd = macd.iloc[-1]
                        
                        macd_status = "Bullish" if current_macd > 0 else "Bearish"
                        macd_color = "#10b981" if current_macd > 0 else "#ef4444"
                        
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>📈 MACD</h4>
                            <p style="font-size: 1.5rem; color: {macd_color};">{current_macd:.2f}</p>
                            <small>{macd_status}</small>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col3:
                    # Bollinger Bands
                    if len(hist_data) >= 20:
                        sma20 = hist_data['Close'].rolling(window=20).mean()
                        std20 = hist_data['Close'].rolling(window=20).std()
                        upper_band = sma20 + (std20 * 2)
                        lower_band = sma20 - (std20 * 2)
                        
                        current_price = hist_data['Close'].iloc[-1]
                        current_upper = upper_band.iloc[-1]
                        current_lower = lower_band.iloc[-1]
                        
                        if current_price > current_upper:
                            bb_status = "Above Upper"
                            bb_color = "#ef4444"
                        elif current_price < current_lower:
                            bb_status = "Below Lower"
                            bb_color = "#10b981"
                        else:
                            bb_status = "Within Bands"
                            bb_color = "#f59e0b"
                        
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>📊 Bollinger Bands</h4>
                            <p style="font-size: 1.2rem; color: {bb_color};">{bb_status}</p>
                            <small>Upper: ₹{current_upper:.2f}<br>Lower: ₹{current_lower:.2f}</small>
                        </div>
                        """, unsafe_allow_html=True)

elif page == "🌐 NSE Social Sentiment":
    st.markdown("## 🌐 NSE Social Media Sentiment Analysis")
    
    db_stocks = get_database_stocks()
    if not db_stocks.empty:
        # Multi-select for stocks
        selected_stocks = st.multiselect(
            "Select stocks for sentiment analysis:",
            options=db_stocks['symbol'].tolist(),
            default=db_stocks['symbol'].tolist()[:5]
        )
        
        if selected_stocks:
            st.markdown("### 💬 Sentiment Overview")
            
            sentiment_data = []
            
            for stock in selected_stocks:
                with st.spinner(f"Analyzing sentiment for {stock}..."):
                    stock_name = db_stocks[db_stocks['symbol'] == stock]['name'].iloc[0]
                    sentiment = enhanced_data_provider.get_reddit_sentiment(stock, stock_name)
                    
                    sentiment_data.append({
                        'Stock': stock,
                        'Sentiment Score': sentiment['sentiment_score'],
                        'Posts': sentiment['post_count'],
                        'Comments': sentiment['total_comments'],
                        'Status': sentiment['status']
                    })
            
            # Create sentiment DataFrame
            sentiment_df = pd.DataFrame(sentiment_data)
            
            # Sentiment visualization
            fig = go.Figure()
            
            colors = ['green' if score > 0.2 else 'red' if score < -0.2 else 'orange' 
                     for score in sentiment_df['Sentiment Score']]
            
            fig.add_trace(go.Bar(
                x=sentiment_df['Stock'],
                y=sentiment_df['Sentiment Score'],
                marker_color=colors,
                name='Sentiment Score'
            ))
            
            fig.update_layout(
                title="Social Sentiment Comparison",
                xaxis_title="Stock Symbol",
                yaxis_title="Sentiment Score",
                template="plotly_dark",
                height=500
            )
            
            st.plotly_chart(fig, width='stretch')
            
            # Sentiment table
            st.markdown("### 📊 Detailed Sentiment Data")
            st.dataframe(sentiment_df, width='stretch')
            
            # Sentiment insights
            st.markdown("### 💡 Sentiment Insights")
            
            most_positive = sentiment_df.loc[sentiment_df['Sentiment Score'].idxmax()]
            most_negative = sentiment_df.loc[sentiment_df['Sentiment Score'].idxmin()]
            most_discussed = sentiment_df.loc[sentiment_df['Posts'].idxmax()]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>🟢 Most Positive</h4>
                    <p style="font-size: 1.5rem; color: #10b981;">{most_positive['Stock']}</p>
                    <small>Score: {most_positive['Sentiment Score']:.2f}</small>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>🔴 Most Negative</h4>
                    <p style="font-size: 1.5rem; color: #ef4444;">{most_negative['Stock']}</p>
                    <small>Score: {most_negative['Sentiment Score']:.2f}</small>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>💬 Most Discussed</h4>
                    <p style="font-size: 1.5rem; color: #3b82f6;">{most_discussed['Stock']}</p>
                    <small>{most_discussed['Posts']} posts</small>
                </div>
                """, unsafe_allow_html=True)

# ===============================================================================
# FOOTER
# ===============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: #1f2937; border-radius: 10px;">
    <h3>🚀 Nifty50 Tracker Pro - NSE Edition</h3>
    <p>Powered by NSE Data Sources • Real-time NSE Analysis • NSE AI Intelligence</p>
    <div>
        <span class="data-source rapidapi-source">🔴 NSE RapidAPI</span>
        <span class="data-source yfinance-source">📊 NSE Yahoo Finance</span>
        <span class="data-source reddit-source">💬 NSE Reddit</span>
        <span class="data-source database-source">🗄️ NSE Database</span>
    </div>
    <p style="margin-top: 1rem; color: #9ca3af;">
        NSE Edition v3.0 • Exclusively NSE Data • BSE Excluded • Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </p>
</div>
""", unsafe_allow_html=True)

# Auto-refresh mechanism
if auto_refresh:
    time.sleep(30)
    st.rerun()
