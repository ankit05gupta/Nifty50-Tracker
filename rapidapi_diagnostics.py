#!/usr/bin/env python3
"""
RapidAPI Diagnostics and Fallback Enhancement
==============================================
Diagnose RapidAPI issues and implement better fallback systems
"""

import requests
import time
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_rapidapi_connectivity():
    """Test RapidAPI connectivity and diagnose issues."""
    print("🔍 RapidAPI Diagnostics")
    print("=" * 40)
    
    # Test configuration
    api_key = "88e0cf9e52mshda9fcddad46d339p1c5795jsn23bdadc1df8a"
    host = "indian-stock-exchange-api2.p.rapidapi.com"
    
    headers = {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": host
    }
    
    # Test different endpoints
    endpoints = [
        ("Market Status", "market_status", {"exchange": "NSE"}),
        ("Stock Quote", "stock", {"symbol": "RELIANCE", "exchange": "NSE"}),
        ("Nifty 50", "nifty50", {}),
        ("Basic Info", "", {})  # Root endpoint
    ]
    
    for name, endpoint, params in endpoints:
        print(f"\n🔍 Testing {name}")
        print("-" * 30)
        
        try:
            if endpoint:
                url = f"https://{host}/{endpoint}"
            else:
                url = f"https://{host}/"
            
            print(f"URL: {url}")
            print(f"Params: {params}")
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            print(f"Status Code: {response.status_code}")
            print(f"Response Size: {len(response.content)} bytes")
            
            if response.status_code == 200:
                print("✅ Success!")
                data = response.json()
                print(f"Data keys: {list(data.keys()) if isinstance(data, dict) else 'Not dict'}")
            elif response.status_code == 404:
                print("❌ Endpoint not found (404)")
            elif response.status_code == 422:
                print("❌ Rate limited or invalid request (422)")
            elif response.status_code == 403:
                print("❌ Access forbidden - Check API key (403)")
            else:
                print(f"⚠️ Unexpected status: {response.status_code}")
                print(f"Response: {response.text[:200]}...")
        
        except Exception as e:
            print(f"❌ Connection error: {e}")
        
        time.sleep(2)  # Rate limiting
    
    print("\n" + "=" * 40)
    print("💡 Diagnosis Summary:")
    print("📊 Expected Results:")
    print("   ✅ 200: API working normally")
    print("   ⚠️ 422: Rate limited (common)")
    print("   ❌ 404: Endpoint changed/unavailable")
    print("   ❌ 403: API key issue")
    
    print("\n🔧 Fallback Strategy:")
    print("   1. Enhanced demo data generation")
    print("   2. Yahoo Finance as primary source")
    print("   3. Cached data for unavailable APIs")
    print("   4. User notifications about data sources")

def generate_enhanced_demo_data():
    """Generate enhanced demo data for RapidAPI replacement."""
    print("\n🎭 Enhanced Demo Data Generation")
    print("=" * 40)
    
    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta
    
    # NSE stocks for demo
    nse_stocks = [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "HINDUNILVR",
        "ICICIBANK", "KOTAKBANK", "BHARTIARTL", "ITC", "SBIN"
    ]
    
    demo_data = {}
    
    for symbol in nse_stocks:
        # Generate realistic stock data
        base_price = np.random.uniform(500, 3000)
        change_pct = np.random.uniform(-3, 3)
        change = base_price * (change_pct / 100)
        
        demo_data[symbol] = {
            "symbol": symbol,
            "exchange": "NSE",
            "lastPrice": round(base_price, 2),
            "change": round(change, 2),
            "pChange": round(change_pct, 2),
            "dayHigh": round(base_price * 1.02, 2),
            "dayLow": round(base_price * 0.98, 2),
            "volume": np.random.randint(1000000, 10000000),
            "value": np.random.randint(100000000, 1000000000),
            "marketCap": np.random.randint(50000000000, 500000000000),
            "pe": round(np.random.uniform(15, 35), 2),
            "status": "demo_enhanced"
        }
    
    print("✅ Enhanced demo data generated for NSE stocks")
    print(f"📊 Sample data for RELIANCE:")
    for key, value in demo_data["RELIANCE"].items():
        print(f"   {key}: {value}")
    
    return demo_data

if __name__ == "__main__":
    print("🚀 RapidAPI Diagnostics & Enhancement")
    print("=" * 50)
    
    # Test API connectivity
    test_rapidapi_connectivity()
    
    # Generate enhanced demo data
    enhanced_demo = generate_enhanced_demo_data()
    
    print("\n🎯 Recommendations:")
    print("1. Use Yahoo Finance as primary data source")
    print("2. Implement enhanced demo data for RapidAPI fallback")
    print("3. Add user notifications about data source status")
    print("4. Cache successful API responses longer")
    print("5. Implement exponential backoff for rate limits")
    
    print(f"\n🚀 Dashboard available at: http://localhost:8505")
    print("📈 Yahoo Finance provides reliable NSE data with .NS suffix")
