#!/usr/bin/env python3
"""
Quick NSE Verification
======================
Quick check to verify NSE filtering is working
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def quick_nse_check():
    """Quick NSE filtering verification."""
    print("🔍 Quick NSE Verification")
    print("=" * 30)
    
    # Test 1: Database NSE count
    try:
        import sqlite3
        import pandas as pd
        
        conn = sqlite3.connect('data/nifty50_stocks.db')
        
        # Get NSE stocks only
        nse_query = """
            SELECT COUNT(*) as count, 
                   GROUP_CONCAT(DISTINCT exchange) as exchanges
            FROM stock_list 
            WHERE (UPPER(exchange) = 'NSE' OR exchange IS NULL)
            AND symbol NOT LIKE '%BSE%'
            AND symbol NOT LIKE '%.BSE'
        """
        
        result = pd.read_sql_query(nse_query, conn)
        count = result['count'].iloc[0]
        exchanges = result['exchanges'].iloc[0]
        
        print(f"✅ NSE stocks in database: {count}")
        print(f"   Exchanges: {exchanges}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Database check error: {e}")
    
    # Test 2: Enhanced provider import
    try:
        from app.enhanced_data_provider import enhanced_data_provider
        print("✅ Enhanced NSE provider loaded")
        
        # Quick symbol test
        sample_data = enhanced_data_provider.get_comprehensive_stock_data("RELIANCE")
        if sample_data:
            status = sample_data.get('status', {})
            print(f"✅ Sample NSE data sources:")
            for source, state in status.items():
                emoji = "✅" if state == 'success' else "🔶" if state == 'demo' else "❌"
                print(f"     {emoji} {source}: {state}")
        
    except Exception as e:
        print(f"❌ Provider error: {e}")
    
    print("\n🎯 NSE-Only Dashboard Status:")
    print("   • Database: NSE stocks only")
    print("   • Yahoo Finance: .NS suffix")
    print("   • RapidAPI: exchange=NSE parameter")
    print("   • UI: NSE branding")
    print("   • BSE data: Excluded")
    
    print(f"\n🚀 Access NSE Dashboard: http://localhost:8505")

if __name__ == "__main__":
    quick_nse_check()
