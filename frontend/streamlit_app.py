import streamlit as st
import json
import sys
import os

# Add the src directory to the Python path for import
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_dir not in sys.path:
    sys.path.append(src_dir)

from enhanced_stock_fetcher import StockDataFetcher

st.set_page_config(page_title="Nifty 50 Technical Analysis", layout="wide")
st.markdown("<h1 style='text-align: center; color: #0072B5;'>Nifty 50 Interactive Technical Analysis</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://assets.stocksinnews.com/nifty50.png", width='stretch')
    st.markdown("### Select Stock & Period")
    # Load symbols only once for speed
    @st.cache_data
    def load_symbols():
        config_path = os.path.join(os.path.dirname(__file__), "..", "config", "nifty50_symbols_2025.json")
        with open(config_path) as f:
            return json.load(f)
    nifty_symbols = load_symbols()
    selected_stock = st.selectbox("Nifty 50 Stock", nifty_symbols)
    period = st.selectbox("Analysis Period", ["1mo", "3mo", "6mo", "1y"], index=0)
    st.markdown("---")
    st.markdown("#### About Technical Indicators")
    st.info("""
- **SMA/EMA:** Moving averages for trend direction  
- **RSI:** Momentum & overbought/oversold  
- **MACD:** Trend strength & reversals  
""")

fetcher = StockDataFetcher()

# Cache the analysis for faster response
@st.cache_data(show_spinner="Fetching and analyzing stock data...")
def get_analysis(symbol, period):
    return fetcher.get_stock_data(symbol, period=period)

if st.button("🔄 Refresh Analysis"):
    st.cache_data.clear()
    result = get_analysis(selected_stock, period)
else:
    result = get_analysis(selected_stock, period)

if result["success"]:
    ta = result["technical_analysis"]
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
