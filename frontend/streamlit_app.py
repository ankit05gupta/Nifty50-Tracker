import requests
import threading
import time
import importlib
import plotly.graph_objs as go
def get_historical_prices(symbol, period="6mo"):
    import yfinance as yf
    # yfinance expects .NS/.BO
    df = yf.download(symbol, period=period, progress=False)
    return df

def get_peers(industry, exclude_symbol=None):
    db_path = os.path.join(os.path.dirname(__file__), "..", "data", "nifty50_stocks.db")
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    if exclude_symbol:
        cursor.execute("SELECT symbol, name, lastPrice, pChange FROM stock_list WHERE industry=? AND symbol!=? ORDER BY lastPrice DESC LIMIT 5", (industry, exclude_symbol))
    else:
        cursor.execute("SELECT symbol, name, lastPrice, pChange FROM stock_list WHERE industry=? ORDER BY lastPrice DESC LIMIT 5", (industry,))
    rows = cursor.fetchall()
    conn.close()
    return rows
import pandas as pd
import streamlit as st
import json
import sys
import os
from datetime import datetime
import random

# Add the src directory to the Python path for import
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_dir not in sys.path:
    sys.path.append(src_dir)

from src.enhanced_stock_fetcher import StockDataFetcher

st.set_page_config(page_title="Nifty 50 Technical Analysis", layout="wide")

# --- Live Ticker/Marquee ---
def get_top_movers(limit=8):
    db_path = os.path.join(os.path.dirname(__file__), "..", "data", "nifty50_stocks.db")
    if not os.path.exists(db_path):
        return []
    import sqlite3
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT symbol, lastPrice, pChange FROM stock_list ORDER BY ABS(pChange) DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return rows

def render_ticker():
    movers = get_top_movers()
    if not movers:
        return
    ticker_html = "<marquee style='font-size:1.1em; color:#0072B5; background:#f8fafd; padding:0.3em 0;'>"
    for symbol, price, change in movers:
        color = '#008000' if change >= 0 else '#d00000'
        sign = '+' if change >= 0 else ''
        ticker_html += f"<b>{symbol}</b> ₹{price:.2f} <span style='color:{color};'>{sign}{change:.2f}%</span> &nbsp;|&nbsp; "
    ticker_html += "</marquee>"
    st.markdown(ticker_html, unsafe_allow_html=True)

render_ticker()
st.markdown("<h1 style='text-align: center; color: #0072B5; margin-bottom:0.2em;'>Nifty 50 Interactive Technical Analysis</h1>", unsafe_allow_html=True)

# --- Sidebar: Modern, with quick stats, watchlist, theme toggle ---
with st.sidebar:
    st.image("https://assets.stocksinnews.com/nifty50.png", width='stretch')
    st.markdown("### Select Stock & Period")
    # Theme toggle
    if 'theme_dark' not in st.session_state:
        st.session_state['theme_dark'] = False
    theme = st.toggle("🌗 Dark Mode", value=st.session_state['theme_dark'], key="theme_toggle")
    st.session_state['theme_dark'] = theme
    # Quick stats
    db_path = os.path.join(os.path.dirname(__file__), "..", "data", "nifty50_stocks.db")
    if os.path.exists(db_path):
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*), AVG(lastPrice), SUM(totalTradedValue) FROM stock_list")
        count, avg_price, total_value = cursor.fetchone()
        conn.close()
        st.metric("Stocks Tracked", count)
        st.metric("Avg Price", f"₹{avg_price:.2f}")
        st.metric("Total Traded Value", f"₹{total_value/1e7:.2f} Cr")
    # Watchlist (session-based)
    st.markdown("---")
    st.markdown("#### ⭐ Watchlist")
    if 'watchlist' not in st.session_state:
        st.session_state['watchlist'] = []
    # Add/remove from watchlist
    def add_to_watchlist(symbol):
        if symbol not in st.session_state['watchlist']:
            st.session_state['watchlist'].append(symbol)
    def remove_from_watchlist(symbol):
        if symbol in st.session_state['watchlist']:
            st.session_state['watchlist'].remove(symbol)
    # Load symbols only once for speed
    import sqlite3
    @st.cache_data
    def load_symbols_from_db():
        db_path = os.path.join(os.path.dirname(__file__), "..", "data", "nifty50_stocks.db")
        if not os.path.exists(db_path):
            return []
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # Use totalTradedValue as a proxy for market cap (if market cap not available)
        cursor.execute("SELECT symbol, name, exchange, totalTradedValue FROM stock_list ORDER BY totalTradedValue DESC")
        rows = cursor.fetchall()
        conn.close()
        symbol_options = []
        for symbol, name, exchange, _ in rows:
            yf_symbol = symbol
            if not symbol.endswith('.NS') and not symbol.endswith('.BO'):
                if exchange == 'NSE':
                    yf_symbol = f"{symbol}.NS"
                elif exchange == 'BSE':
                    yf_symbol = f"{symbol}.BO"
            label = f"{symbol} ({name}) [{exchange}]"
            symbol_options.append({"label": label, "value": yf_symbol})
        return symbol_options
    all_symbols = load_symbols_from_db()
    if 'selected_stock_idx' not in st.session_state:
        st.session_state['selected_stock_idx'] = 0

    selected = st.selectbox(
        "Select Stock (NSE/BSE)",
        all_symbols,
        index=st.session_state['selected_stock_idx'],
        format_func=lambda x: x['label'] if isinstance(x, dict) else str(x),
        key="stock_selectbox"
    )
    selected_stock = selected['value'] if isinstance(selected, dict) else selected
    # Update session state index if changed
    st.session_state['selected_stock_idx'] = all_symbols.index(selected) if selected in all_symbols else 0
    # Fetch company info from DB for the selected stock
    def get_company_info(symbol):
        db_path = os.path.join(os.path.dirname(__file__), "..", "data", "nifty50_stocks.db")
        if not os.path.exists(db_path):
            return None
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # Remove .NS/.BO for DB lookup
        base_symbol = symbol.replace('.NS', '').replace('.BO', '')
        cursor.execute("SELECT symbol, name, industry, isin, lastPrice, open, dayHigh, dayLow, previousClose, change, pChange, yearHigh, yearLow, totalTradedVolume, totalTradedValue FROM stock_list WHERE symbol=? LIMIT 1", (base_symbol,))
        row = cursor.fetchone()
        conn.close()
        if row:
            keys = ["symbol", "name", "industry", "isin", "lastPrice", "open", "dayHigh", "dayLow", "previousClose", "change", "pChange", "yearHigh", "yearLow", "totalTradedVolume", "totalTradedValue"]
            return dict(zip(keys, row))
        return None
    company_info = get_company_info(selected_stock)
    period = st.selectbox("Analysis Period", ["1mo", "3mo", "6mo", "1y"], index=0)
    # --- Interactive Buy/Sell Simulation ---
    st.markdown("---")
    st.markdown("#### 💸 Simulated Trade Panel")
    if 'portfolio' not in st.session_state:
        st.session_state['portfolio'] = {}
    col_buy, col_sell = st.columns(2)
    with col_buy:
        buy_qty = st.number_input("Buy Qty", min_value=1, max_value=10000, value=10, key="buy_qty")
        if st.button("Buy", key="buy_btn"):
            st.session_state['portfolio'][selected_stock] = st.session_state['portfolio'].get(selected_stock, 0) + buy_qty
            st.success(f"Bought {buy_qty} shares of {selected_stock}")
    with col_sell:
        sell_qty = st.number_input("Sell Qty", min_value=1, max_value=10000, value=10, key="sell_qty")
        if st.button("Sell", key="sell_btn"):
            current = st.session_state['portfolio'].get(selected_stock, 0)
            if sell_qty > current:
                st.warning("Not enough shares to sell!")
            else:
                st.session_state['portfolio'][selected_stock] = current - sell_qty
                st.success(f"Sold {sell_qty} shares of {selected_stock}")
    st.markdown(f"**Your Holdings:** {st.session_state['portfolio'].get(selected_stock, 0)} shares")
    # --- User Notes ---
    st.markdown("#### 📝 Notes for this Stock")
    note_key = f"note_{selected_stock}"
    note = st.text_area("Add your notes here...", value=st.session_state.get(note_key, ""))
    if st.button("Save Note", key="save_note_btn"):
        st.session_state[note_key] = note
        st.success("Note saved!")

# --- Technical Indicator Toggles (move above chart tab for global scope) ---


if company_info:
    tabs = st.tabs(["Overview", "Chart", "Peers & Analytics"])
    with tabs[0]:
        st.markdown(f"""
            <div style='background-color:#f0f4fa;padding:0.5em 0.8em 0.2em 0.8em;border-radius:8px;margin-bottom:0.5em;'>
                <h3 style='color:#0072B5;margin-bottom:0.1em;font-size:1.2em;'>{company_info['name']}</h3>
                <span style='font-size:0.95em;color:#444;'>Industry: <b>{company_info['industry']}</b> | ISIN: <b>{company_info['isin']}</b></span>
            </div>
        """, unsafe_allow_html=True)
        # Compact metrics in a single row with smaller font
        cols = st.columns(6)
        metric_style = "font-size:1.1em; color:#222; margin-bottom:0.1em;"
        value_style = "font-size:1.3em; font-weight:bold; color:#0072B5;"
        delta_style = "font-size:0.95em; color:#008000; margin-left:2px;"
        # Live
        live_delta = f"{company_info['change']:+.2f} ({company_info['pChange']:+.2f}%)"
        live_delta_color = '#008000' if company_info['change'] >= 0 else '#d00000'
        cols[0].markdown(f"""
            <div style='text-align:center;'>
                <div style='{metric_style}'>Live</div>
                tabs = st.tabs(["Overview", "Chart", "Peers & Analytics"])
                with tabs[0]:
                    # ...existing code for Overview tab...
                    st.markdown(f"""
                        <div style='background-color:#f0f4fa;padding:0.5em 0.8em 0.2em 0.8em;border-radius:8px;margin-bottom:0.5em;'>
                            <h3 style='color:#0072B5;margin-bottom:0.1em;font-size:1.2em;'>{company_info['name']}</h3>
                            <span style='font-size:0.95em;color:#444;'>Industry: <b>{company_info['industry']}</b> | ISIN: <b>{company_info['isin']}</b></span>
                        </div>
                    """, unsafe_allow_html=True)
                    # ...existing code for metrics, OI analysis, etc...
                    # ...existing code for Overview tab end...

                with tabs[1]:
                    st.markdown("### 📈 Price & Volume Chart")
                    # Show indicator checkboxes only in Chart tab
                    st.markdown("#### 📊 Show Technical Indicators")
                    show_sma = st.checkbox("Show SMA/EMA", value=True, key="show_sma")
                    show_rsi = st.checkbox("Show RSI", value=True, key="show_rsi")
                    show_macd = st.checkbox("Show MACD", value=True, key="show_macd")
                    st.markdown("---")
                    st.markdown("#### About Technical Indicators")
                st.info(
            """
            **SMA/EMA:** Moving averages for trend direction  
            **RSI:** Momentum & overbought/oversold  
            **MACD:** Trend strength & reversals  
            """
                )
                    hist_df = get_historical_prices(selected_stock, period=period)
                    if not hist_df.empty:
                        fig = go.Figure()
                        fig.add_trace(go.Candlestick(x=hist_df.index, open=hist_df['Open'], high=hist_df['High'], low=hist_df['Low'], close=hist_df['Close'], name='Price'))
                        fig.add_trace(go.Bar(x=hist_df.index, y=hist_df['Volume'], name='Volume', yaxis='y2', marker_color='rgba(0,114,181,0.2)', opacity=0.4))
                        # Technical overlays
                        if show_sma:
                            if 'Close' in hist_df:
                                fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df['Close'].rolling(20).mean(), mode='lines', name='SMA20', line=dict(color='#FF9900', width=2, dash='dot')))
                                fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df['Close'].ewm(span=12).mean(), mode='lines', name='EMA12', line=dict(color='#00BFFF', width=2, dash='dash')))
                        fig.update_layout(
                            yaxis_title='Price',
                            yaxis2=dict(title='Volume', overlaying='y', side='right', showgrid=False),
                            xaxis_rangeslider_visible=False,
                            height=400,
                            margin=dict(l=10, r=10, t=30, b=10),
                            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                        )
                        st.plotly_chart(fig, width='stretch')
                        # RSI/MACD below chart
                        import plotly.express as px
                        import numpy as np
                        if show_rsi and 'Close' in hist_df:
                            delta = hist_df['Close'].diff()
                            up = delta.clip(lower=0)
                            down = -1 * delta.clip(upper=0)
                            roll_up = up.rolling(14).mean()
                            roll_down = down.rolling(14).mean()
                            rs = roll_up / roll_down
                            rsi = 100 - (100 / (1 + rs))
                            st.line_chart(rsi, height=120, use_container_width=True)
                        if show_macd and 'Close' in hist_df:
                            ema12 = hist_df['Close'].ewm(span=12).mean()
                            ema26 = hist_df['Close'].ewm(span=26).mean()
                            macd = (ema12 - ema26).astype(float)
                            signal = macd.ewm(span=9).mean()
                            macd_hist = macd - signal
                            macd_fig = go.Figure()
                            macd_fig.add_trace(go.Scatter(x=hist_df.index, y=macd, mode='lines', name='MACD', line=dict(color='#0072B5')))
                            macd_fig.add_trace(go.Scatter(x=hist_df.index, y=signal, mode='lines', name='Signal', line=dict(color='#FF9900', dash='dash')))
                            macd_fig.add_trace(go.Bar(x=hist_df.index, y=macd_hist, name='Histogram', marker_color='#B0C4DE', opacity=0.5))
                            macd_fig.update_layout(title='MACD', height=220, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
                            st.plotly_chart(macd_fig, width='stretch')
                    else:
                        st.info("No historical price data available.")
                    fig2 = px.histogram(price_changes, nbins=10, labels={'value':'% Change'}, title='Distribution of % Change')
                    st.plotly_chart(fig2, width='stretch')
                else:
                    st.info("No price change data for peers.")

fetcher = StockDataFetcher()

# Cache the analysis for faster response
@st.cache_data(show_spinner="Fetching and analyzing stock data...")
def get_analysis(symbol, period):
    return fetcher.get_stock_data(symbol, period=period)

# Add function to refresh DB with latest live NSE data
# Add function to refresh DB with latest live NSE data
def refresh_db():
    db_mod = importlib.import_module("app.database")
    db_mod.fetch_live_nse_stocks_and_store()
    db_mod.fetch_live_bse_stocks_and_store()

# Background thread for periodic DB refresh
def background_db_refresher(interval_sec=300):
    while True:
        try:
            refresh_db()
        except Exception as e:
            print(f"[Background DB Refresh] Error: {e}")
        time.sleep(interval_sec)

# Start background refresher only once
if 'db_refresh_thread_started' not in st.session_state:
    thread = threading.Thread(target=background_db_refresher, args=(300,), daemon=True)
    thread.start()
    st.session_state['db_refresh_thread_started'] = True

    tabs = st.tabs(["Overview", "Chart", "Peers & Analytics"])
    with tabs[0]:
        st.markdown(f"""
            <div style='background-color:#f0f4fa;padding:0.5em 0.8em 0.2em 0.8em;border-radius:8px;margin-bottom:0.5em;'>
                <h3 style='color:#0072B5;margin-bottom:0.1em;font-size:1.2em;'>{company_info['name']}</h3>
                <span style='font-size:0.95em;color:#444;'>Industry: <b>{company_info['industry']}</b> | ISIN: <b>{company_info['isin']}</b></span>
            </div>
        """, unsafe_allow_html=True)
        # Compact metrics in a single row with smaller font
        cols = st.columns(6)
        metric_style = "font-size:1.1em; color:#222; margin-bottom:0.1em;"
        value_style = "font-size:1.3em; font-weight:bold; color:#0072B5;"
        delta_style = "font-size:0.95em; color:#008000; margin-left:2px;"
        # Live
        live_delta = f"{company_info['change']:+.2f} ({company_info['pChange']:+.2f}%)"
        live_delta_color = '#008000' if company_info['change'] >= 0 else '#d00000'
        cols[0].markdown(f"""
            <div style='text-align:center;'>
                <div style='{metric_style}'>Live</div>
                <div style='{value_style}'>&#8377;{company_info['lastPrice']:.2f}</div>
                <div style='font-size:0.95em; color:{live_delta_color};'>{live_delta}</div>
            </div>
        """, unsafe_allow_html=True)
        # Open
        cols[1].markdown(f"""
            <div style='text-align:center;'>
                <div style='{metric_style}'>Open</div>
                <div style='{value_style}'>&#8377;{company_info['open']:.2f}</div>
            </div>
        """, unsafe_allow_html=True)
        # PrevCls
        cols[2].markdown(f"""
            <div style='text-align:center;'>
                <div style='{metric_style}'>PrevCls</div>
                <div style='{value_style}'>&#8377;{company_info['previousClose']:.2f}</div>
            </div>
        """, unsafe_allow_html=True)
        # High
        cols[3].markdown(f"""
            <div style='text-align:center;'>
                <div style='{metric_style}'>High</div>
                <div style='{value_style}'>&#8377;{company_info['dayHigh']:.2f}</div>
            </div>
        """, unsafe_allow_html=True)
        # Low
        cols[4].markdown(f"""
            <div style='text-align:center;'>
                <div style='{metric_style}'>Low</div>
                <div style='{value_style}'>&#8377;{company_info['dayLow']:.2f}</div>
            </div>
        """, unsafe_allow_html=True)
        # Vol
        cols[5].markdown(f"""
            <div style='text-align:center;'>
                <div style='{metric_style}'>Vol</div>
                <div style='{value_style}'>{int(company_info['totalTradedVolume'])//1000}K</div>
            </div>
        """, unsafe_allow_html=True)
        # --- Open Interest (OI) Analysis ---
        def fetch_oi(symbol):
            # Remove .NS/.BO for NSE API
            base_symbol = symbol.replace('.NS', '').replace('.BO', '')
            url = f"https://www.nseindia.com/api/option-chain-equities?symbol={base_symbol}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "application/json, text/plain, */*",
                "Referer": "https://www.nseindia.com/",
            }
            session = requests.Session()
            # Get cookies
            session.get("https://www.nseindia.com", headers=headers, timeout=10)
            response = session.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                return None
            data = response.json()
            # Find nearest expiry
            expiry = data['records']['expiryDates'][0]
            oi_data = [item for item in data['records']['data'] if item['expiryDate'] == expiry and 'CE' in item and 'PE' in item]
            if not oi_data:
                return None
            # Sum OI and change in OI for CE and PE
            total_oi = sum(item['CE']['openInterest'] + item['PE']['openInterest'] for item in oi_data)
            total_chg_oi = sum(item['CE']['changeinOpenInterest'] + item['PE']['changeinOpenInterest'] for item in oi_data)
            return {
                'expiry': expiry,
                'total_oi': total_oi,
                'total_chg_oi': total_chg_oi
            }

        oi_result = fetch_oi(selected_stock)
        if oi_result:
            oi_color = '#008000' if oi_result['total_chg_oi'] >= 0 else '#d00000'
            st.markdown(f"""
                <div style='background:#f8fafd;padding:0.5em 0.8em 0.2em 0.8em;border-radius:8px;margin-bottom:0.5em;'>
                    <span style='font-size:1.05em;color:#0072B5;'><b>Open Interest (OI) Analysis</b></span><br>
                    <span style='font-size:0.98em;'>Expiry: <b>{oi_result['expiry']}</b></span><br>
                    <span style='font-size:1.1em;'>Total OI: <b>{oi_result['total_oi']:,}</b></span><br>
                    <span style='font-size:1.1em;color:{oi_color};'>Change in OI: <b>{oi_result['total_chg_oi']:+,}</b></span>
                </div>
            """, unsafe_allow_html=True)
            # Simple interpretation
            if oi_result['total_chg_oi'] > 0 and company_info['change'] > 0:
                st.success("Rising OI with rising price: Bullish sentiment.")
            elif oi_result['total_chg_oi'] > 0 and company_info['change'] < 0:
                st.warning("Rising OI with falling price: Bearish build-up.")
            elif oi_result['total_chg_oi'] < 0 and company_info['change'] > 0:
                st.info("Falling OI with rising price: Short covering.")
            elif oi_result['total_chg_oi'] < 0 and company_info['change'] < 0:
                st.info("Falling OI with falling price: Long unwinding.")
        else:
            st.info("Open Interest data not available for this stock.")
        # Second row for year high/low and traded value
        cols2 = st.columns(3)
        cols2[0].metric("Yr High", f"₹{company_info['yearHigh']:.2f}")
        cols2[1].metric("Yr Low", f"₹{company_info['yearLow']:.2f}")
        cols2[2].metric("Value", f"₹{company_info['totalTradedValue']/1e7:.2f} Cr")
        st.markdown("<div style='margin-bottom:0.5em'></div>", unsafe_allow_html=True)
        st.caption("Switch tabs above for charts, peer comparison, and analytics.")
