
import yfinance as yf
import json
import os

def fetch_bse_equity_stock_list(symbols=None):
    """
    Fetch live BSE stock data using yfinance for a list of BSE symbols (e.g., SENSEX stocks).
    Returns a list of dicts with fields similar to the NSE scraper.
    """
    # Default to SENSEX 30 if no symbols provided
    if symbols is None:
        symbols = [
            'RELIANCE.BO', 'HDFCBANK.BO', 'BHARTIARTL.BO', 'TCS.BO', 'ICICIBANK.BO',
            'SBIN.BO', 'HINDUNILVR.BO', 'INFY.BO', 'BAJFINANCE.BO', 'ITC.BO',
            'LT.BO', 'MARUTI.BO', 'HCLTECH.BO', 'M&M.BO', 'KOTAKBANK.BO',
            'SUNPHARMA.BO', 'ULTRACEMCO.BO', 'TITAN.BO', 'AXISBANK.BO', 'NTPC.BO',
            'POWERGRID.BO', 'ASIANPAINT.BO', 'NESTLEIND.BO', 'INDUSINDBK.BO', 'JSWSTEEL.BO',
            'TATAMOTORS.BO', 'TATASTEEL.BO', 'DIVISLAB.BO', 'GRASIM.BO', 'ADANIPORTS.BO'
        ]
    stocks = []
    tickers = yf.Tickers(' '.join(symbols))
    for symbol in symbols:
        ticker = tickers.tickers.get(symbol)
        if not ticker:
            continue
        info = ticker.info
        if not info or 'shortName' not in info:
            continue
        stock_info = {
            "symbol": symbol.replace('.BO',''),
            "companyName": info.get("shortName"),
            "industry": info.get("industry", ""),
            "lastPrice": info.get("regularMarketPrice"),
            "open": info.get("regularMarketOpen"),
            "dayHigh": info.get("dayHigh"),
            "dayLow": info.get("dayLow"),
            "previousClose": info.get("regularMarketPreviousClose"),
            "change": info.get("regularMarketChange"),
            "pChange": info.get("regularMarketChangePercent"),
            "yearHigh": info.get("fiftyTwoWeekHigh"),
            "yearLow": info.get("fiftyTwoWeekLow"),
            "totalTradedVolume": info.get("regularMarketVolume"),
            "totalTradedValue": info.get("regularMarketVolume", 0) * info.get("regularMarketPrice", 0),
            "isin": info.get("isin", ""),
        }
        if stock_info["symbol"] and stock_info["companyName"]:
            stocks.append(stock_info)
    print(f"Fetched {len(stocks)} live BSE stocks.")
    if stocks:
        print("Sample stock data:")
        for s in stocks[:3]:
            print(json.dumps(s, indent=2))
    return stocks

if __name__ == "__main__":
    stocks = fetch_bse_equity_stock_list()
    # Save to file for inspection or fallback use
    out_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'live_bse_stock_list.json')
    with open(out_path, 'w') as f:
        json.dump(stocks, f, indent=2)
    print(f"Saved live BSE stock list to {out_path}")
