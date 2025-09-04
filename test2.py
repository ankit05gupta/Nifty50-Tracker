# Create a sample code file for testing the setup
sample_test_code = '''# Test Script for Nifty 50 Stock Tracker Setup
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
print("\\n" + "=" * 30)
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
print("\\n" + "=" * 50)
print("SETUP VERIFICATION COMPLETE")
print("=" * 50)
print("If all tests show ✓, your setup is ready!")
print("If any tests show ✗, please review the installation steps.")
print("\\nNext steps:")
print("1. Create your project structure")
print("2. Run 'streamlit run frontend/streamlit_app.py'")
print("3. Start building your stock tracker!")
'''

# Save the test script
with open("setup_verification_test.py", "w", encoding="utf-8") as f:
    f.write(sample_test_code)

print("Setup verification script created: setup_verification_test.py")
print("\\nAfter completing the setup steps, run this command to verify everything works:")
print("python setup_verification_test.py")

# Create a quick start script
quick_start_script = '''#!/bin/bash
# Quick Start Script for Nifty 50 Stock Tracker

echo "🚀 Starting Nifty 50 Stock Tracker Setup..."

# Check if Python is installed
if command -v python3 &> /dev/null; then
    echo "✓ Python3 found"
    PYTHON_CMD=python3
elif command -v python &> /dev/null; then
    echo "✓ Python found"  
    PYTHON_CMD=python
else
    echo "✗ Python not found! Please install Python first."
    exit 1
fi

# Create project directory
echo "📁 Creating project structure..."
mkdir -p nifty50-tracker/{app,frontend,data,ml_models,tests}
cd nifty50-tracker

# Create virtual environment
echo "🔧 Creating virtual environment..."
$PYTHON_CMD -m venv venv

# Activate virtual environment (instructions)
echo "✨ Virtual environment created!"
echo "Please run the following commands:"
echo ""

if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "  venv\\Scripts\\activate"
else
    echo "  source venv/bin/activate"
fi

echo "  pip install -r requirements.txt"
echo "  python setup_verification_test.py"
echo ""
echo "🎉 Setup foundation ready! Follow the detailed guide for complete setup."
'''

# Save the quick start script
with open("quick_start.sh", "w") as f:
    f.write(quick_start_script)

print("\\nQuick start script created: quick_start.sh")
print("Make it executable with: chmod +x quick_start.sh")

# Create a Windows batch version too
windows_script = '''@echo off
echo 🚀 Starting Nifty 50 Stock Tracker Setup...

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ✗ Python not found! Please install Python first.
    pause
    exit /b 1
)
echo ✓ Python found

REM Create project directory
echo 📁 Creating project structure...
mkdir nifty50-tracker\\app 2>nul
mkdir nifty50-tracker\\frontend 2>nul  
mkdir nifty50-tracker\\data 2>nul
mkdir nifty50-tracker\\ml_models 2>nul
mkdir nifty50-tracker\\tests 2>nul
cd nifty50-tracker

REM Create virtual environment
echo 🔧 Creating virtual environment...
python -m venv venv

echo ✨ Virtual environment created!
echo Please run the following commands:
echo.
echo   venv\\Scripts\\activate
echo   pip install -r requirements.txt
echo   python setup_verification_test.py
echo.
echo 🎉 Setup foundation ready! Follow the detailed guide for complete setup.
pause
'''

with open("quick_start.bat", "w", encoding="utf-8") as f:
    f.write(sample_test_code)

print("Windows batch script created: quick_start.bat")

print("\\n" + "="*60)
print("FILES CREATED FOR EASY SETUP:")
print("="*60)
print("1. setup_verification_test.py - Test your installation")
print("2. quick_start.sh - Linux/Mac quick setup")
print("3. quick_start.bat - Windows quick setup")
print("4. setup-guide-beginners.md - Complete step-by-step guide")
print("\\nThese files will help beginners get started quickly!")