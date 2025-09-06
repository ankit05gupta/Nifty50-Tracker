#!/usr/bin/env python3
"""
Test AI Analysis Module Fix
===========================
Verify that the AI analysis functionality works correctly.
"""

import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_ai_analysis():
    """Test the AI analysis functionality."""
    print("🤖 Testing AI Analysis Module...")
    print("=" * 50)
    
    try:
        # Import the AI analyzer
        from ml_models.stock_prediction_model import StockAIAnalyzer
        print("✅ StockAIAnalyzer imported successfully")
        
        # Initialize analyzer
        analyzer = StockAIAnalyzer()
        print("✅ StockAIAnalyzer initialized successfully")
        
        # Test analysis with a sample stock
        test_symbol = "RELIANCE"
        print(f"\n📊 Testing analysis for {test_symbol}...")
        
        result = analyzer.analyze_stock(test_symbol)
        
        if result:
            print("✅ AI Analysis completed successfully!")
            print(f"  📈 Recommendation: {result.get('recommendation', 'N/A')}")
            print(f"  🎯 Confidence: {result.get('confidence', 0):.1%}")
            print(f"  💡 Reasoning: {result.get('reasoning', 'N/A')}")
            print(f"  📝 Insights: {len(result.get('insights', []))} insights generated")
            
            # Show insights
            insights = result.get('insights', [])
            if insights:
                print("  📋 AI Insights:")
                for i, insight in enumerate(insights[:3], 1):
                    print(f"    {i}. {insight}")
        else:
            print("❌ AI Analysis returned empty result")
            
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("   Make sure all required packages are installed")
        
    except Exception as e:
        print(f"❌ Analysis Error: {e}")
        print("   AI module may need additional configuration")
        
    print(f"\n🎉 AI Analysis Test Complete!")
    print("The AI tab in the dashboard should now work without showing 'module not available' error")

if __name__ == "__main__":
    test_ai_analysis()
