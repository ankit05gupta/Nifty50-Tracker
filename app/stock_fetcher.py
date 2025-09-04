import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import json


class StockDataFetcher:
    def __init__(self):
        # Nifty 50 symbols (adding .NS for NSE)
        with open('nifty50_symbols_2025.json') as f:
            self.nifty50_symbols = json.load(f)
    
    # ...existing code...
    def get_stock_data(self, symbol, period="1mo"):
        """Fetch stock data for a given symbol"""
        try:
            data = yf.download(symbol, period=period, progress=False, threads=False, auto_adjust=False)
            if not data.empty and 'Close' in data.columns:
                # Ensure current_price is a float, not a Series
                current_price = float(data['Close'].iloc[-1])
                return {
                    'symbol': symbol,
                    'current_price': current_price,
                    'data': data,
                    'success': True
                }
            else:
                return {'symbol': symbol, 'success': False, 'error': 'No data found'}
        except Exception as e:
            return {'symbol': symbol, 'success': False, 'error': str(e)}
# ...existing code...
    
    def get_multiple_stocks(self, symbols=None):
        """Fetch data for multiple stocks"""
        if symbols is None:
            symbols = self.nifty50_symbols[:5]  # First 5 for testing
        
        results = {}
        for symbol in symbols:
            print(f"Fetching data for {symbol}...")
            results[symbol] = self.get_stock_data(symbol)
        
        return results

# Test the fetcher
if __name__ == "__main__":
    fetcher = StockDataFetcher()
    results = fetcher.get_multiple_stocks()
    
    for symbol, data in results.items():
        if data['success']:
            print(f"{symbol}: Current Price = ₹{data['current_price']:.2f}")
        else:
            print(f"{symbol}: Error - {data.get('error', 'Unknown error')}")