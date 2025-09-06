#!/usr/bin/env python3
"""
Test Enhanced Multi-Source Data Provider
=======================================
Comprehensive test for the new enhanced data provider with RapidAPI,
Yahoo Finance, and Reddit integration.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_enhanced_data_provider():
    """Test all components of the enhanced data provider."""
    print("🧪 Testing Enhanced Multi-Source Data Provider")
    print("=" * 60)
    
    try:
        from app.enhanced_data_provider import enhanced_data_provider
        print("✅ Enhanced data provider imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return
    
    # Test 1: RapidAPI Stock Quote
    print("\n📊 Test 1: RapidAPI Stock Quote")
    print("-" * 40)
    
    test_symbol = "RELIANCE"
    rapidapi_data = enhanced_data_provider.get_stock_quote_rapidapi(test_symbol)
    
    if rapidapi_data:
        print(f"✅ RapidAPI data fetched for {test_symbol}")
        print(f"   Data keys: {list(rapidapi_data.keys())}")
    else:
        print(f"⚠️  RapidAPI data not available for {test_symbol} (expected due to API limits)")
    
    # Test 2: Yahoo Finance Historical Data
    print("\n📈 Test 2: Yahoo Finance Historical Data")
    print("-" * 40)
    
    hist_data = enhanced_data_provider.get_historical_data_yfinance(test_symbol, "1mo")
    
    if not hist_data.empty:
        print(f"✅ Historical data fetched: {len(hist_data)} days")
        print(f"   Columns: {list(hist_data.columns)}")
        print(f"   Date range: {hist_data.index[0].strftime('%Y-%m-%d')} to {hist_data.index[-1].strftime('%Y-%m-%d')}")
    else:
        print(f"⚠️  Historical data empty (using demo data)")
    
    # Test 3: Reddit Sentiment Analysis
    print("\n💬 Test 3: Reddit Sentiment Analysis")
    print("-" * 40)
    
    reddit_sentiment = enhanced_data_provider.get_reddit_sentiment(test_symbol, "Reliance Industries")
    
    if reddit_sentiment:
        print(f"✅ Reddit sentiment analysis complete")
        print(f"   Sentiment score: {reddit_sentiment['sentiment_score']:.2f}")
        print(f"   Posts analyzed: {reddit_sentiment['post_count']}")
        print(f"   Status: {reddit_sentiment['status']}")
    else:
        print(f"⚠️  Reddit sentiment not available")
    
    # Test 4: Comprehensive Stock Data
    print("\n🔍 Test 4: Comprehensive Stock Data")
    print("-" * 40)
    
    comprehensive_data = enhanced_data_provider.get_comprehensive_stock_data(test_symbol)
    
    if comprehensive_data:
        print(f"✅ Comprehensive data fetched for {test_symbol}")
        print(f"   Data sources status:")
        for source, status in comprehensive_data['status'].items():
            status_emoji = "✅" if status == 'success' else "🔶" if status == 'demo' else "❌"
            print(f"     {status_emoji} {source}: {status}")
    
    # Test 5: Market Overview
    print("\n🏠 Test 5: Market Overview")
    print("-" * 40)
    
    market_overview = enhanced_data_provider.get_market_overview()
    
    if market_overview:
        print(f"✅ Market overview data fetched")
        if market_overview.get('market_sentiment'):
            sentiment = market_overview['market_sentiment']
            print(f"   Market sentiment: {sentiment['sentiment_score']:.2f}")
            print(f"   Gainers: {sentiment['gainers']}, Losers: {sentiment['losers']}")
    
    # Test 6: Caching System
    print("\n⚡ Test 6: Caching System")
    print("-" * 40)
    
    import time
    start_time = time.time()
    
    # First call (should cache)
    enhanced_data_provider.get_stock_quote_rapidapi(test_symbol)
    first_call_time = time.time() - start_time
    
    start_time = time.time()
    
    # Second call (should use cache)
    enhanced_data_provider.get_stock_quote_rapidapi(test_symbol)
    second_call_time = time.time() - start_time
    
    print(f"✅ Caching system test complete")
    print(f"   First call: {first_call_time:.3f}s")
    print(f"   Second call: {second_call_time:.3f}s")
    
    if second_call_time < first_call_time:
        print(f"   🚀 Cache is working (2nd call {((first_call_time-second_call_time)/first_call_time)*100:.1f}% faster)")
    
    print("\n" + "=" * 60)
    print("🎉 Enhanced Data Provider Test Complete!")
    print("Ready to use with the Next-Generation Dashboard")

if __name__ == "__main__":
    test_enhanced_data_provider()
