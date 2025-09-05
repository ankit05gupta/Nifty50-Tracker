import requests
import json
import os

def fetch_nse_equity_stock_list():
    """
    Advanced scraping for live NSE equity stock list using NSE's public API endpoint.
    Returns a list of (symbol, company name) tuples.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/",
        "Connection": "keep-alive",
        "Origin": "https://www.nseindia.com"
    }
    url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050"
    session = requests.Session()
    # Get cookies by visiting the homepage first
    homepage = session.get("https://www.nseindia.com", headers=headers, timeout=10)
    # If homepage fails, raise error
    homepage.raise_for_status()
    # Try up to 2 times in case of 401
    for attempt in range(2):
        response = session.get(url, headers=headers, timeout=10)
        if response.status_code == 401:
            # Refresh cookies and try again
            session.cookies.clear()
            session.get("https://www.nseindia.com", headers=headers, timeout=10)
            continue
        response.raise_for_status()
        break
    else:
        raise Exception("Failed to fetch NSE data: Unauthorized (401)")
    data = response.json()
    stocks = []
    for stock in data.get("data", []):
        # Extract all fields for each stock
        stock_info = {
            "symbol": stock.get("symbol"),
            "companyName": stock.get("meta", {}).get("companyName"),
            "industry": stock.get("meta", {}).get("industry"),
            "lastPrice": stock.get("lastPrice"),
            "open": stock.get("open"),
            "dayHigh": stock.get("dayHigh"),
            "dayLow": stock.get("dayLow"),
            "previousClose": stock.get("previousClose"),
            "change": stock.get("change"),
            "pChange": stock.get("pChange"),
            "yearHigh": stock.get("yearHigh"),
            "yearLow": stock.get("yearLow"),
            "totalTradedVolume": stock.get("totalTradedVolume"),
            "totalTradedValue": stock.get("totalTradedValue"),
            "isin": stock.get("meta", {}).get("isin"),
        }
        if stock_info["symbol"] and stock_info["companyName"]:
            stocks.append(stock_info)
    print(f"Fetched {len(stocks)} live NSE stocks.")
    if stocks:
        print("Sample stock data:")
        for s in stocks[:3]:
            print(json.dumps(s, indent=2))
    return stocks

if __name__ == "__main__":
    stocks = fetch_nse_equity_stock_list()
    # Save to file for inspection or fallback use
    out_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'live_nse_stock_list.json')
    with open(out_path, 'w') as f:
        json.dump(stocks, f, indent=2)
    print(f"Saved live NSE stock list to {out_path}")
