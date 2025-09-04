import streamlit as st
import yfinance as yf
import talib
import json

with open("nifty50_symbols_2025.json", "r") as f:
    NIFTY50_SYMBOLS = json.load(f)

def get_live_price(symbol):
    ticker = yf.Ticker(symbol)
    data = ticker.history(period="1d", interval="1m")
    if not data.empty:
        return float(data['Close'].iloc[-1])
    else:
        return None

def get_historical_data(symbol, period):
    ticker = yf.Ticker(symbol)
    data = ticker.history(period=period)
    data = data.reset_index()
    return data

def add_indicators(df):
    df['SMA20'] = talib.SMA(df['Close'].values.astype(float), timeperiod=20)
    df['EMA20'] = talib.EMA(df['Close'].values.astype(float), timeperiod=20)
    df['RSI14'] = talib.RSI(df['Close'].values.astype(float), timeperiod=14)
    df['SMA50'] = talib.SMA(df['Close'].values.astype(float), timeperiod=50)
    df['EMA50'] = talib.EMA(df['Close'].values.astype(float), timeperiod=50)
    df['SMA100'] = talib.SMA(df['Close'].values.astype(float), timeperiod=100)
    df['EMA100'] = talib.EMA(df['Close'].values.astype(float), timeperiod=100)
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['Close'].values.astype(float))
    df['ADX'] = talib.ADX(
        df['High'].values.astype(float),
        df['Low'].values.astype(float),
        df['Close'].values.astype(float),
        timeperiod=14
    )
    df['OBV'] = talib.OBV(
        df['Close'].values.astype(float),
        df['Volume'].values.astype(float)
    )
    return df

def get_buy_sell_percentages(df):
    latest_rsi = df['RSI14'].iloc[-1]
    if latest_rsi > 70:
        buy_pct = 0
        sell_pct = 100
    elif latest_rsi < 30:
        buy_pct = 100
        sell_pct = 0
    else:
        buy_pct = int((70 - latest_rsi) / 40 * 100)
        sell_pct = int((latest_rsi - 30) / 40 * 100)
    return buy_pct, sell_pct

st.title("Nifty 50 Tracker – MA/EMA/RSI/MACD/ADX/OBV/Volume & Real-Time Price")

col1, col2 = st.columns(2)

with col1:
    selected_stock = st.selectbox("Select Nifty 50 Stock:", NIFTY50_SYMBOLS)
    period = st.selectbox("Select Period:", ['1mo', '3mo', '6mo', '1y'])

refresh = st.button("Refresh Live Price")

price_placeholder = col2.empty()

df = get_historical_data(selected_stock, period)
df = add_indicators(df)

buy_pct, sell_pct = get_buy_sell_percentages(df)
st.subheader("Current Buy/Sell Sentiment (based on RSI)")
st.progress(buy_pct, text=f"Buy: {buy_pct}%")
st.progress(sell_pct, text=f"Sell: {sell_pct}%")

if refresh or True:
    live_price = get_live_price(selected_stock)
    if live_price is not None:
        price_placeholder.metric(label=f"Live Price for {selected_stock}", value=f"₹{live_price:.2f}")
    else:
        price_placeholder.warning("Live price not available for this symbol.")

st.subheader(f"{selected_stock} - Indicators & Volume ({period})")
st.dataframe(df[['Date', 'Close', 'Volume', 'SMA20', 'EMA20', 'SMA50', 'EMA50', 'SMA100', 'EMA100', 'RSI14', 'MACD', 'MACD_signal', 'MACD_hist', 'ADX', 'OBV']].tail(30))
