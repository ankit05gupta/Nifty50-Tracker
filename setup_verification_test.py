# Test Script for Nifty 50 Stock Tracker Setup
# Run this script to verify your setup is working correctly

print("=" * 50)
print("NIFTY 50 STOCK TRACKER - SETUP VERIFICATION")
print("=" * 50)

# Test 1: Python Version
import sys
print(f"✓ Python Version: {sys.version.split()[0]}")

# Test 2: Required Libraries
try:
    import pandas as pd
    print("✓ Pandas imported successfully")
except ImportError as e:
    print(f"✗ Pandas import failed: {e}")

try:
    import yfinance as yf
    print("✓ yfinance imported successfully")
except ImportError as e:
    print(f"✗ yfinance import failed: {e}")

try:
    import streamlit as st
    print("✓ Streamlit imported successfully")
except ImportError as e:
    print(f"✗ Streamlit import failed: {e}")

try:
    import plotly.graph_objects as go
    print("✓ Plotly imported successfully")
except ImportError as e:
    print(f"✗ Plotly import failed: {e}")

# Test 3: SQLite Database
try:
    import sqlite3
    print(f"✓ SQLite version: {sqlite3.sqlite_version}")
    
    # Test database connection
    conn = sqlite3.connect(':memory:')  # In-memory database for testing
    cursor = conn.cursor()
    cursor.execute("SELECT 1")
    result = cursor.fetchone()
    conn.close()
    print("✓ SQLite database connection test successful")
except Exception as e:
    print(f"✗ SQLite test failed: {e}")

# Test 4: Sample Stock Data Fetch
print("\n" + "=" * 30)
print("TESTING STOCK DATA FETCH")
print("=" * 30)

try:
    ticker = yf.Ticker("RELIANCE.NS")
    data = ticker.history(period="5d")
    
    if not data.empty:
        latest_price = data['Close'].iloc[-1]
        print(f"✓ Successfully fetched RELIANCE stock data")
        print(f"  Latest Close Price: ₹{latest_price:.2f}")
        print(f"  Data Points: {len(data)} days")
    else:
        print("✗ No data received for RELIANCE")
        
except Exception as e:
    print(f"✗ Stock data fetch failed: {e}")

# Test 5: File System Permissions
try:
    import os
    test_file = "setup_test_temp.txt"
    with open(test_file, 'w') as f:
        f.write("Test file for setup verification")
    
    # Read the file back
    with open(test_file, 'r') as f:
        content = f.read()
    
    # Clean up
    os.remove(test_file)
    print("✓ File system read/write permissions working")
    
except Exception as e:
    print(f"✗ File system test failed: {e}")

# Summary
print("\n" + "=" * 50)
print("SETUP VERIFICATION COMPLETE")
print("=" * 50)
print("If all tests show ✓, your setup is ready!")
print("If any tests show ✗, please review the installation steps.")
print("\nNext steps:")
print("1. Create your project structure")
print("2. Run 'streamlit run frontend/streamlit_app.py'")
print("3. Start building your stock tracker!")
