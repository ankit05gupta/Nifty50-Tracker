import streamlit as st
import json
import sys
import os

# Add the app directory to the Python path for import
app_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app"))
if app_dir not in sys.path:
    sys.path.append(app_dir)

from enhanced_stock_fetcher import StockDataFetcher

st.title("Nifty 50 Stock Technical Analysis")

# Load Nifty 50 symbols from JSON
with open(os.path.join(app_dir, "nifty50_symbols_2025.json")) as f:
    nifty_symbols = json.load(f)

fetcher = StockDataFetcher()

selected_stock = st.selectbox("Select Nifty 50 Stock:", nifty_symbols)
period = st.selectbox("Select Period:", ["1mo", "3mo", "6mo", "1y"], index=0)

if st.button("Refresh Analysis"):
    refresh = True
else:
    refresh = False

if refresh or True:  # Always show data on first load
    result = fetcher.get_stock_data(selected_stock, period=period)
    if result["success"]:
        st.subheader(f"{selected_stock} - Live Price & Technicals")
        st.metric(label="Live Price", value=f"₹{result['current_price']:.2f}", delta=f"{result['price_change']:+.2f} ({result['price_change_percent']:+.2f}%)")
        ta = result["technical_analysis"]

        # RSI and Buy/Sell Percentages
        rsi_value = ta["rsi_value"]
        st.write(f"**RSI (14):** {rsi_value:.2f}")

        # Calculate buy/sell percentages based on RSI
        if not isinstance(rsi_value, float) or rsi_value != rsi_value:  # nan check
            st.warning("Not enough data for RSI calculation.")
        else:
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

        # Show all technical indicators
        st.write(f"SMA20: {ta['sma_20']:.2f}")
        st.write(f"SMA50: {ta['sma_50']:.2f}")
        sma_200 = ta.get('sma_200')
        if sma_200 is not None and isinstance(sma_200, float) and sma_200 == sma_200:  # not nan
            st.write(f"SMA200: {sma_200:.2f}")
        else:
            st.write("SMA200: N/A")
        st.write(f"EMA12: {ta['ema_12']:.2f}")

        ema_26 = ta.get('ema_26')
        if ema_26 is not None and isinstance(ema_26, float) and ema_26 == ema_26:  # not nan
            st.write(f"EMA26: {ema_26:.2f}")
        else:
            st.write("EMA26: N/A")

        ema_50 = ta.get('ema_50')
        if ema_50 is not None and isinstance(ema_50, float) and ema_50 == ema_50:  # not nan
            st.write(f"EMA50: {ema_50:.2f}")
        else:
            st.write("EMA50: N/A")
        st.write(f"MACD: {ta['macd_value']:.2f}")
        st.write(f"MACD Signal: {ta['macd_signal_value']:.2f}")
        st.write(f"MACD Histogram: {ta['macd_histogram']:.2f}")

        # Show technical interpretations
        st.write("**RSI Analysis:**", ta["rsi_analysis"]["status"], ta["rsi_analysis"]["message"])
        st.write("**Moving Average Signals:**")
        for signal in ta["moving_average_signals"]:
            st.write(f"- {signal['signal']}: {signal['message']}")
        st.write("**MACD Analysis:**", ta["macd_analysis"]["status"], ta["macd_analysis"]["message"])
    else:
        st.error(result.get("error", "No data available for this stock."))
