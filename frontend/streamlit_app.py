import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import os

# Add app directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from stock_fetcher import StockDataFetcher

# Page configuration
st.set_page_config(
    page_title="Nifty 50 Stock Tracker",
    page_icon="📈",
    layout="wide"
)

# Title
st.title("📈 Nifty 50 Stock Tracker & Trend Analyzer")
st.markdown("---")

# Sidebar
st.sidebar.header("Settings")
selected_stocks = st.sidebar.multiselect(
    "Select Stocks to Track",
    ["RELIANCE.NS", "HDFCBANK.NS", "BHARTIARTL.NS", "TCS.NS", "ICICIBANK.NS"],
    default=["RELIANCE.NS", "HDFCBANK.NS"]
)

time_period = st.sidebar.selectbox(
    "Time Period",
    ["1mo", "3mo", "6mo", "1y"],
    index=1
)

# Main content
if st.button("Fetch Stock Data"):
    if selected_stocks:
        fetcher = StockDataFetcher()
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = {}
        for i, symbol in enumerate(selected_stocks):
            status_text.text(f"Fetching data for {symbol}...")
            results[symbol] = fetcher.get_stock_data(symbol, time_period)
            progress_bar.progress((i + 1) / len(selected_stocks))
        
        status_text.text("Data fetched successfully!")
        
        # Display results
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Current Prices")
            price_data = []
            for symbol, data in results.items():
                if data['success']:
                    price_data.append({
                        'Stock': symbol.replace('.NS', ''),
                        'Price (₹)': f"₹{data['current_price']:.2f}"
                    })
            
            if price_data:
                df_prices = pd.DataFrame(price_data)
                st.table(df_prices)
        
        with col2:
            st.subheader("Price Charts")
            for symbol, data in results.items():
                if data['success']:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=data['data'].index,
                        y=data['data']['Close'],
                        mode='lines',
                        name=symbol.replace('.NS', ''),
                        line=dict(width=2)
                    ))
                    
                    fig.update_layout(
                        title=f"{symbol.replace('.NS', '')} Price Trend",
                        xaxis_title="Date",
                        yaxis_title="Price (₹)",
                        height=300
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please select at least one stock to track.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and Yahoo Finance API")
