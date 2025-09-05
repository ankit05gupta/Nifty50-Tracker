def fetch_live_bse_stocks_and_store():
    """Fetch live BSE stock data using the yfinance-based scraper and store in stock_list table with all available fields."""
    from app.bse_live_scraper import fetch_bse_equity_stock_list
    stocks = fetch_bse_equity_stock_list()
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Check for required columns, drop and recreate if missing
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='stock_list'")
    exists = cursor.fetchone()
    required = [
        'id','symbol','name','industry','lastPrice','open','dayHigh','dayLow','previousClose','change','pChange','yearHigh','yearLow','totalTradedVolume','totalTradedValue','isin','exchange'
    ]
    recreate = False
    if exists:
        cursor.execute("PRAGMA table_info(stock_list)")
        columns = [row[1] for row in cursor.fetchall()]
        if not all(col in columns for col in required):
            print("Schema mismatch: Dropping and recreating 'stock_list' table with correct columns.")
            cursor.execute("DROP TABLE IF EXISTS stock_list")
            recreate = True
    else:
        recreate = True
    if recreate:
        cursor.execute('''
            CREATE TABLE stock_list (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                name TEXT,
                industry TEXT,
                lastPrice REAL,
                open REAL,
                dayHigh REAL,
                dayLow REAL,
                previousClose REAL,
                change REAL,
                pChange REAL,
                yearHigh REAL,
                yearLow REAL,
                totalTradedVolume INTEGER,
                totalTradedValue REAL,
                isin TEXT,
                exchange TEXT,
                UNIQUE(symbol, exchange)
            )
        ''')
    for stock in stocks:
        cursor.execute('''SELECT id FROM stock_list WHERE symbol=? AND exchange=?''', (stock['symbol'], 'BSE'))
        row = cursor.fetchone()
        if row is None:
            cursor.execute('''INSERT INTO stock_list (symbol, name, industry, lastPrice, open, dayHigh, dayLow, previousClose, change, pChange, yearHigh, yearLow, totalTradedVolume, totalTradedValue, isin, exchange) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (
                    stock['symbol'],
                    stock['companyName'],
                    stock['industry'],
                    stock['lastPrice'],
                    stock['open'],
                    stock['dayHigh'],
                    stock['dayLow'],
                    stock['previousClose'],
                    stock['change'],
                    stock['pChange'],
                    stock['yearHigh'],
                    stock['yearLow'],
                    stock['totalTradedVolume'],
                    stock['totalTradedValue'],
                    stock['isin'],
                    'BSE'
                )
            )
        else:
            cursor.execute('''UPDATE stock_list SET name=?, industry=?, lastPrice=?, open=?, dayHigh=?, dayLow=?, previousClose=?, change=?, pChange=?, yearHigh=?, yearLow=?, totalTradedVolume=?, totalTradedValue=?, isin=? WHERE symbol=? AND exchange=?''',
                (
                    stock['companyName'],
                    stock['industry'],
                    stock['lastPrice'],
                    stock['open'],
                    stock['dayHigh'],
                    stock['dayLow'],
                    stock['previousClose'],
                    stock['change'],
                    stock['pChange'],
                    stock['yearHigh'],
                    stock['yearLow'],
                    stock['totalTradedVolume'],
                    stock['totalTradedValue'],
                    stock['isin'],
                    stock['symbol'],
                    'BSE'
                )
            )
    conn.commit()
    conn.close()
    print(f"Stored {len(stocks)} live BSE stocks in stock_list table.")
def fetch_live_nse_stocks_and_store():
    """Fetch live NSE stock data using the advanced scraper and store in stock_list table with all available fields."""
    from app.nse_live_scraper import fetch_nse_equity_stock_list
    stocks = fetch_nse_equity_stock_list()
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Check for required columns, drop and recreate if missing
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='stock_list'")
    exists = cursor.fetchone()
    required = [
        'id','symbol','name','industry','lastPrice','open','dayHigh','dayLow','previousClose','change','pChange','yearHigh','yearLow','totalTradedVolume','totalTradedValue','isin','exchange'
    ]
    recreate = False
    if exists:
        cursor.execute("PRAGMA table_info(stock_list)")
        columns = [row[1] for row in cursor.fetchall()]
        if not all(col in columns for col in required):
            print("Schema mismatch: Dropping and recreating 'stock_list' table with correct columns.")
            cursor.execute("DROP TABLE IF EXISTS stock_list")
            recreate = True
    else:
        recreate = True
    if recreate:
        cursor.execute('''
            CREATE TABLE stock_list (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                name TEXT,
                industry TEXT,
                lastPrice REAL,
                open REAL,
                dayHigh REAL,
                dayLow REAL,
                previousClose REAL,
                change REAL,
                pChange REAL,
                yearHigh REAL,
                yearLow REAL,
                totalTradedVolume INTEGER,
                totalTradedValue REAL,
                isin TEXT,
                exchange TEXT,
                UNIQUE(symbol, exchange)
            )
        ''')
    for stock in stocks:
        cursor.execute('''SELECT id FROM stock_list WHERE symbol=? AND exchange=?''', (stock['symbol'], 'NSE'))
        row = cursor.fetchone()
        if row is None:
            cursor.execute('''INSERT INTO stock_list (symbol, name, industry, lastPrice, open, dayHigh, dayLow, previousClose, change, pChange, yearHigh, yearLow, totalTradedVolume, totalTradedValue, isin, exchange) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (
                    stock['symbol'],
                    stock['companyName'],
                    stock['industry'],
                    stock['lastPrice'],
                    stock['open'],
                    stock['dayHigh'],
                    stock['dayLow'],
                    stock['previousClose'],
                    stock['change'],
                    stock['pChange'],
                    stock['yearHigh'],
                    stock['yearLow'],
                    stock['totalTradedVolume'],
                    stock['totalTradedValue'],
                    stock['isin'],
                    'NSE'
                )
            )
        else:
            cursor.execute('''UPDATE stock_list SET name=?, industry=?, lastPrice=?, open=?, dayHigh=?, dayLow=?, previousClose=?, change=?, pChange=?, yearHigh=?, yearLow=?, totalTradedVolume=?, totalTradedValue=?, isin=? WHERE symbol=? AND exchange=?''',
                (
                    stock['companyName'],
                    stock['industry'],
                    stock['lastPrice'],
                    stock['open'],
                    stock['dayHigh'],
                    stock['dayLow'],
                    stock['previousClose'],
                    stock['change'],
                    stock['pChange'],
                    stock['yearHigh'],
                    stock['yearLow'],
                    stock['totalTradedVolume'],
                    stock['totalTradedValue'],
                    stock['isin'],
                    stock['symbol'],
                    'NSE'
                )
            )
    conn.commit()
    conn.close()
    print(f"Stored {len(stocks)} live NSE stocks in stock_list table.")
import pandas as pd
import yfinance as yf
import sqlite3
import os
import requests
from sklearn.linear_model import LogisticRegression
import numpy as np

def insert_historical_stock_data():
    """Fetch historical data for all stocks in stock_list and insert with indicators into stocks table."""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT symbol, exchange FROM stock_list')
    stocks = cursor.fetchall()
    conn.close()
    if not stocks:
        print("No stocks found in stock_list table.")
        return
    print(f"Fetching and inserting data for {len(stocks)} stocks...")
    for symbol, exchange in stocks:
        try:
            # For NSE, append .NS; for BSE, append .BO
            yf_symbol = symbol + ('.NS' if exchange == 'NSE' else '.BO')
            df = yf.download(yf_symbol, period='6mo', progress=False)
            if df.empty:
                print(f"No data for {symbol} ({exchange})")
                continue
            # Calculate indicators
            df['sma_20'] = df['Close'].rolling(window=20).mean()
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14, min_periods=14).mean()
            avg_loss = loss.rolling(window=14, min_periods=14).mean()
            rs = avg_gain / avg_loss
            df['rsi'] = 100 - (100 / (1 + rs))
            ema_fast = df['Close'].ewm(span=12).mean()
            ema_slow = df['Close'].ewm(span=26).mean()
            macd = ema_fast - ema_slow
            df['macd'] = macd
            # Insert into DB
            conn = sqlite3.connect(db_path)
            for date, row in df.iterrows():
                cursor = conn.cursor()
                cursor.execute('''INSERT OR REPLACE INTO stocks (symbol, date, open, high, low, close, volume, sma_20, rsi, macd) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (
                        symbol,
                        str(date.date()),
                        float(row['Open']) if not pd.isna(row['Open']) else None,
                        float(row['High']) if not pd.isna(row['High']) else None,
                        float(row['Low']) if not pd.isna(row['Low']) else None,
                        float(row['Close']) if not pd.isna(row['Close']) else None,
                        int(row['Volume']) if not pd.isna(row['Volume']) else None,
                        float(row['sma_20']) if not pd.isna(row['sma_20']) else None,
                        float(row['rsi']) if not pd.isna(row['rsi']) else None,
                        float(row['macd']) if not pd.isna(row['macd']) else None
                    )
                )
            conn.commit()
            conn.close()
            print(f"Inserted data for {symbol} ({exchange})")
        except Exception as e:
            print(f"Error for {symbol} ({exchange}): {e}")

def train_ai_model_on_all_data():
    """Fetch all stock indicator data and train the AI model on it."""
    df = fetch_all_stock_data()
    if df.empty or 'close' not in df.columns:
        print("No data to train model.")
        return None
    features = ['sma_20', 'rsi', 'macd']
    for f in features:
        if f not in df.columns:
            print(f"Missing feature: {f}")
            return None
    df = df.dropna(subset=features + ['close'])
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    X = df[features]
    y = df['target']
    if len(X) < 20:
        print("Not enough data for training.")
        return None
    model = LogisticRegression()
    model.fit(X, y)
    print("AI model trained on all available stock data.")
    return model

def get_db_path():
    db_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(db_dir, exist_ok=True)
    return os.path.join(db_dir, 'nifty50_stocks.db')

def create_stock_table():
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Check if table exists and has correct columns
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='stocks'")
    exists = cursor.fetchone()
    recreate = False
    if exists:
        # Check for required columns
        cursor.execute("PRAGMA table_info(stocks)")
        columns = [row[1] for row in cursor.fetchall()]
        required = ['id','symbol','date','open','high','low','close','volume','sma_20','sma_50','sma_200','ema_12','ema_26','ema_50','rsi','macd','macd_signal','macd_histogram']
        if not all(col in columns for col in required):
            print("Schema mismatch: Dropping and recreating 'stocks' table with correct columns.")
            cursor.execute("DROP TABLE IF EXISTS stocks")
            recreate = True
    else:
        recreate = True
    if recreate:
        cursor.execute('''
            CREATE TABLE stocks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                sma_20 REAL,
                sma_50 REAL,
                sma_200 REAL,
                ema_12 REAL,
                ema_26 REAL,
                ema_50 REAL,
                rsi REAL,
                macd REAL,
                macd_signal REAL,
                macd_histogram REAL
            )
        ''')
    conn.commit()
    conn.close()
# Fetch all NSE stocks (symbol, name)
def fetch_nse_stock_list():
    url = "https://www1.nseindia.com/content/equities/EQUITY_L.csv"
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        lines = response.text.splitlines()
        import csv
        reader = csv.DictReader(lines)
        stocks = []
        for row in reader:
            symbol = row.get('SYMBOL')
            name = row.get('NAME OF COMPANY')
            if symbol is None:
                print(f"Skipping row with missing symbol: {row}")
                continue
            try:
                name_clean = str(name) if name is not None else ''
                name_clean = str(name_clean)
                name_clean = name_clean.replace('\n', '').replace('\r', '').strip()
            except Exception as e:
                print(f"Error cleaning NSE name: {name} (type: {type(name)}), error: {e}")
                name_clean = ''
            stocks.append((symbol, name_clean))
        print(f"Fetched {len(stocks)} NSE stocks.")
        return stocks
    except Exception as e:
        print(f"Error fetching NSE stock list: {e}")
        # Fallback: load from static file
        try:
            import json
            fallback_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_stock_list.json')
            with open(fallback_path, 'r') as f:
                stocks = json.load(f)
            print(f"Loaded {len(stocks)} stocks from fallback sample_stock_list.json")
            return stocks
        except Exception as e2:
            print(f"Error loading fallback stock list: {e2}")
            return []

# Fetch all BSE stocks (symbol, name)
def fetch_bse_stock_list():
    url = "https://api.bseindia.com/BseIndiaAPI/api/GetScripMaster/w?strType=EQ"
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        data = response.json()
        stocks = []
        for item in data.get('Table', []):
            symbol = item.get('SC_CODE')
            name = item.get('SC_NAME')
            if symbol is None:
                print(f"Skipping item with missing symbol: {item}")
                continue
            try:
                name_clean = str(name) if name is not None else ''
                name_clean = str(name_clean)
                name_clean = name_clean.replace('\n', '').replace('\r', '').strip()
            except Exception as e:
                print(f"Error cleaning BSE name: {name} (type: {type(name)}), error: {e}")
                name_clean = ''
            stocks.append((symbol, name_clean))
        print(f"Fetched {len(stocks)} BSE stocks.")
        return stocks
    except Exception as e:
        print(f"Error fetching BSE stock list: {e}")
        return []

def store_stock_list_in_db(stock_list, exchange):
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_list (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            name TEXT,
            exchange TEXT,
            UNIQUE(symbol, exchange)
        )
    ''')
    for symbol, name in stock_list:
        # Try to insert, if exists, update name if changed
        cursor.execute('''SELECT name FROM stock_list WHERE symbol=? AND exchange=?''', (symbol, exchange))
        row = cursor.fetchone()
        if row is None:
            cursor.execute('''INSERT INTO stock_list (symbol, name, exchange) VALUES (?, ?, ?)''', (symbol, name, exchange))
        elif row[0] != name:
            cursor.execute('''UPDATE stock_list SET name=? WHERE symbol=? AND exchange=?''', (name, symbol, exchange))
    conn.commit()
    conn.close()
    print(f"Checked/updated {len(stock_list)} stocks for {exchange} in DB.")

def fetch_and_store_all_stocks():
    nse_stocks = fetch_nse_stock_list()
    bse_stocks = fetch_bse_stock_list()
    store_stock_list_in_db(nse_stocks, 'NSE')
    store_stock_list_in_db(bse_stocks, 'BSE')

def fetch_all_stock_data():
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM stocks", conn)
    conn.close()
    return df

def train_ai_model():
    df = fetch_all_stock_data()
    if df.empty or 'close' not in df.columns:
        print("No data to train model.")
        return None
    # Feature engineering
    features = ['sma_20', 'rsi', 'macd']
    for f in features:
        if f not in df.columns:
            print(f"Missing feature: {f}")
            return None
    df = df.dropna(subset=features + ['close'])
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    X = df[features]
    y = df['target']
    if len(X) < 20:
        print("Not enough data for training.")
        return None
    model = LogisticRegression()
    model.fit(X, y)
    print("AI model trained on stock data.")
    return model

def update_stock_indicators_with_ai():
    model = train_ai_model()
    if model is None:
        print("Model not trained. Skipping update.")
        return
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM stocks", conn)
    features = ['sma_20', 'rsi', 'macd']
    if not all(f in df.columns for f in features):
        print("Missing features in DB.")
        return
    df = df.dropna(subset=features)
    if df.empty:
        print("No data to update.")
        return
    X = df[features]
    preds = model.predict(X)
    df['ai_signal'] = np.where(preds > 0, 'Buy', 'Sell')
    # Store predictions back to DB (optional: create new table or update existing)
    df[['id', 'ai_signal']].to_sql('ai_signals', conn, if_exists='replace', index=False)
    conn.close()
    print("AI signals updated in database.")

if __name__ == "__main__":
    create_stock_table()
    print("✅ Database and table created at:", get_db_path())
    # Always fetch and store stock list first
    fetch_and_store_all_stocks()
    # Optionally train and update AI signals
    update_stock_indicators_with_ai()
    # Insert historical stock data with indicators
    insert_historical_stock_data()
    # Train AI model on all available data
    train_ai_model_on_all_data()
    from app.database import fetch_and_store_all_stocks
    fetch_and_store_all_stocks()
