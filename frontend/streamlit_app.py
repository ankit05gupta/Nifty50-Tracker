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

# Add the src directory to the Python path for import
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_dir not in sys.path:
    sys.path.append(src_dir)

from src.enhanced_stock_fetcher import StockDataFetcher

st.set_page_config(page_title="Nifty 50 Technical Analysis", layout="wide")
st.markdown("<h1 style='text-align: center; color: #0072B5;'>Nifty 50 Interactive Technical Analysis</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://assets.stocksinnews.com/nifty50.png", width='stretch')
    st.markdown("### Select Stock & Period")
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
    st.markdown("---")
    st.markdown("#### About Technical Indicators")
    st.info("""
- **SMA/EMA:** Moving averages for trend direction  
- **RSI:** Momentum & overbought/oversold  
- **MACD:** Trend strength & reversals  
""")


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
                <div style='{value_style}'>₹{company_info['lastPrice']:.2f}</div>
                <div style='font-size:0.95em; color:{live_delta_color};'>{live_delta}</div>
            </div>
        """, unsafe_allow_html=True)
        # Open
        cols[1].markdown(f"""
            <div style='text-align:center;'>
                <div style='{metric_style}'>Open</div>
                <div style='{value_style}'>₹{company_info['open']:.2f}</div>
            </div>
        """, unsafe_allow_html=True)
        # PrevCls
        cols[2].markdown(f"""
            <div style='text-align:center;'>
                <div style='{metric_style}'>PrevCls</div>
                <div style='{value_style}'>₹{company_info['previousClose']:.2f}</div>
            </div>
        """, unsafe_allow_html=True)
        # High
        cols[3].markdown(f"""
            <div style='text-align:center;'>
                <div style='{metric_style}'>High</div>
                <div style='{value_style}'>₹{company_info['dayHigh']:.2f}</div>
            </div>
        """, unsafe_allow_html=True)
        # Low
        cols[4].markdown(f"""
            <div style='text-align:center;'>
                <div style='{metric_style}'>Low</div>
                <div style='{value_style}'>₹{company_info['dayLow']:.2f}</div>
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

    with tabs[1]:
        st.markdown("### Price & Volume Chart")
        hist_df = get_historical_prices(selected_stock, period=period)
        if not hist_df.empty:
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=hist_df.index, open=hist_df['Open'], high=hist_df['High'], low=hist_df['Low'], close=hist_df['Close'], name='Price'))
            fig.add_trace(go.Bar(x=hist_df.index, y=hist_df['Volume'], name='Volume', yaxis='y2', marker_color='rgba(0,114,181,0.2)', opacity=0.4))
            fig.update_layout(
                yaxis_title='Price',
                yaxis2=dict(title='Volume', overlaying='y', side='right', showgrid=False),
                xaxis_rangeslider_visible=False,
                height=400,
                margin=dict(l=10, r=10, t=30, b=10),
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No historical price data available.")

    with tabs[2]:
        st.markdown("### Peer Comparison (Top 5 by Price)")
        peers = get_peers(company_info['industry'], exclude_symbol=company_info['symbol'])
        if peers:
            peer_df = pd.DataFrame(peers, columns=["Symbol", "Name", "Price", "% Change"])
            st.dataframe(peer_df, use_container_width=True, hide_index=True)
        else:
            st.info("No peer data available for this industry.")

        with st.expander("Analytics: Price Change Distribution (Industry Peers)", expanded=True):
            if peers:
                import numpy as np
                import plotly.express as px
                price_changes = [row[3] for row in peers if row[3] is not None]
                if price_changes:
                    fig2 = px.histogram(price_changes, nbins=10, labels={'value':'% Change'}, title='Distribution of % Change')
                    st.plotly_chart(fig2, use_container_width=True)
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

if st.button("🔄 Refresh Analysis"):
    refresh_db()
    st.cache_data.clear()
    result = get_analysis(selected_stock, period)
else:
    result = get_analysis(selected_stock, period)

if result["success"]:
    ta = result["technical_analysis"]
    # ML Prediction
    ml_result = fetcher.ml_predict_buy_sell(result["data"])  # Use the correct key for the DataFrame

    col1, col2, col3 = st.columns([2, 2, 3])

    with col1:
        st.metric(label="Live Price", value=f"₹{result['current_price']:.2f}", delta=f"{result['price_change']:+.2f} ({result['price_change_percent']:+.2f}%)")
        st.write(f"**SMA20:** {ta.get('sma_20', 'N/A'):.2f}" if isinstance(ta.get('sma_20'), float) and ta.get('sma_20') == ta.get('sma_20') else "SMA20: N/A")
        st.write(f"**SMA50:** {ta.get('sma_50', 'N/A'):.2f}" if isinstance(ta.get('sma_50'), float) and ta.get('sma_50') == ta.get('sma_50') else "SMA50: N/A")
        st.write(f"**SMA200:** {ta.get('sma_200', 'N/A'):.2f}" if isinstance(ta.get('sma_200'), float) and ta.get('sma_200') == ta.get('sma_200') else "SMA200: N/A")
        st.write(f"**EMA12:** {ta.get('ema_12', 'N/A'):.2f}" if isinstance(ta.get('ema_12'), float) and ta.get('ema_12') == ta.get('ema_12') else "EMA12: N/A")
        st.write(f"**EMA26:** {ta.get('ema_26', 'N/A'):.2f}" if isinstance(ta.get('ema_26'), float) and ta.get('ema_26') == ta.get('ema_26') else "EMA26: N/A")
        st.write(f"**EMA50:** {ta.get('ema_50', 'N/A'):.2f}" if isinstance(ta.get('ema_50'), float) and ta.get('ema_50') == ta.get('ema_50') else "EMA50: N/A")

    with col2:
        rsi_value = ta.get("rsi_value")
        st.write(f"**RSI (14):** {rsi_value:.2f}" if isinstance(rsi_value, float) and rsi_value == rsi_value else "RSI: N/A")
        if isinstance(rsi_value, float) and rsi_value == rsi_value:
            if rsi_value > 70:
                buy_pct = 0
                sell_pct = 100
            elif rsi_value < 30:
                buy_pct = 100
                sell_pct = 0
            else:
                buy_pct = int((70 - rsi_value) / 40 * 100)
                sell_pct = int((rsi_value - 30) / 40 * 100)
            st.progress(buy_pct, text=f"Buy: {buy_pct}%")
            st.progress(sell_pct, text=f"Sell: {sell_pct}%")
        else:
            st.warning("Not enough data for RSI calculation.")

        st.write(f"**MACD:** {ta.get('macd_value', 'N/A'):.2f}" if isinstance(ta.get('macd_value'), float) and ta.get('macd_value') == ta.get('macd_value') else "MACD: N/A")
        st.write(f"**MACD Signal:** {ta.get('macd_signal_value', 'N/A'):.2f}" if isinstance(ta.get('macd_signal_value'), float) and ta.get('macd_signal_value') == ta.get('macd_signal_value') else "MACD Signal: N/A")
        st.write(f"**MACD Histogram:** {ta.get('macd_histogram', 'N/A'):.2f}" if isinstance(ta.get('macd_histogram'), float) and ta.get('macd_histogram') == ta.get('macd_histogram') else "MACD Histogram: N/A")

    with col3:
        st.markdown("### Technical Interpretations")
        st.write("**RSI Analysis:**", ta["rsi_analysis"]["status"], ta["rsi_analysis"]["message"])
        st.write("**Moving Average Signals:**")
        for signal in ta["moving_average_signals"]:
            st.write(f"- {signal['signal']}: {signal['message']}")
        st.write("**MACD Analysis:**", ta["macd_analysis"]["status"], ta["macd_analysis"]["message"])

    st.markdown("---")
    st.markdown(f"**Last Updated:** {ta['analysis_date']}")
else:
    st.error(result.get("error", "No data available for this stock."))
