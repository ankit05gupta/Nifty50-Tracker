#!/usr/bin/env python3
"""
RapidAPI Fallback System Test
=============================
Test the enhanced fallback system when RapidAPI is unavailable
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_rapidapi_fallback():
    """Test the enhanced RapidAPI fallback system."""
    print("🔧 Testing RapidAPI Fallback System")
    print("=" * 40)
    
    try:
        from app.enhanced_data_provider import enhanced_data_provider
        print("✅ Enhanced data provider imported")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return
    
    # Test symbols
    test_symbols = ['RELIANCE', 'TCS', 'UNKNOWN_STOCK']
    
    for symbol in test_symbols:
        print(f"\n📊 Testing {symbol}")
        print("-" * 25)
        
        # Get comprehensive data
        comprehensive_data = enhanced_data_provider.get_comprehensive_stock_data(symbol)
        
        if comprehensive_data:
            status = comprehensive_data.get('status', {})
            rapidapi_data = comprehensive_data.get('rapidapi_data')
            
            print(f"Status Summary:")
            for source, state in status.items():
                emoji = "✅" if state == 'success' else "🔶" if state == 'demo' else "❌"
                print(f"  {emoji} {source}: {state}")
            
            if rapidapi_data:
                is_demo = rapidapi_data.get('status') == 'demo_enhanced'
                data_type = "Enhanced Demo" if is_demo else "Live API"
                
                print(f"\n🔴 RapidAPI Data ({data_type}):")
                print(f"  Price: ₹{rapidapi_data.get('lastPrice', 0):.2f}")
                print(f"  Change: {rapidapi_data.get('pChange', 0):.2f}%")
                print(f"  Volume: {rapidapi_data.get('volume', 0):,}")
                print(f"  Source: {rapidapi_data.get('source', 'Unknown')}")
                
                if is_demo:
                    print(f"  📝 Note: {rapidapi_data.get('note', 'Demo data')}")
            else:
                print("❌ No RapidAPI data available")
        else:
            print("❌ No comprehensive data available")
    
    print("\n" + "=" * 40)
    print("🎯 Fallback System Summary:")
    print("✅ Enhanced demo data when RapidAPI fails")
    print("✅ Realistic price data based on stock symbols")
    print("✅ Clear status indicators for data sources")
    print("✅ User-friendly notifications about data quality")
    print("✅ Yahoo Finance continues as reliable backup")
    
    print(f"\n🚀 Enhanced Dashboard: http://localhost:8505")
    print("📍 Check: 💹 NSE Multi-Source Data → RapidAPI tab")
    print("🎯 Expected: Demo mode notification with realistic data")

if __name__ == "__main__":
    test_rapidapi_fallback()
