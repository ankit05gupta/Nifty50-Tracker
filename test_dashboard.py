#!/usr/bin/env python3
"""
Quick test to verify the dashboard components work
"""
import sys
import os
from pathlib import Path

# Add the frontend directory to path
sys.path.append(str(Path(__file__).parent / "frontend"))

# Test database connection
try:
    import sqlite3
    db_path = Path(__file__).parent / "data" / "nifty50_stocks.db"
    
    print(f"Testing database connection...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Test symbol list
    cursor.execute("SELECT COUNT(*) FROM stock_list")
    count = cursor.fetchone()[0]
    print(f"✓ Total stocks in database: {count}")
    
    # Test first few stocks
    cursor.execute("SELECT symbol, name, lastPrice FROM stock_list LIMIT 5")
    stocks = cursor.fetchall()
    print(f"✓ Sample stocks:")
    for symbol, name, price in stocks:
        print(f"  - {symbol}: {name} @ ₹{price}")
    
    # Test market overview data
    cursor.execute("""
        SELECT COUNT(*), AVG(lastPrice),
               SUM(CASE WHEN change > 0 THEN 1 ELSE 0 END) as gainers,
               SUM(CASE WHEN change < 0 THEN 1 ELSE 0 END) as losers
        FROM stock_list
    """)
    total, avg_price, gainers, losers = cursor.fetchone()
    print(f"✓ Market Overview:")
    print(f"  - Total stocks: {total}")
    print(f"  - Average price: ₹{avg_price:.2f}")
    print(f"  - Gainers: {gainers}")
    print(f"  - Losers: {losers}")
    
    conn.close()
    print(f"✓ Database test completed successfully!")
    print(f"\n🎯 Dashboard should now be working at: http://localhost:8501")
    
except Exception as e:
    print(f"❌ Database test failed: {e}")
