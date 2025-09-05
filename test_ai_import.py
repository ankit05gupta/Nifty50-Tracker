"""
Test script to verify AI model import and functionality
"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

try:
    from ml_models.stock_prediction_model import StockAIAnalyzer
    print("✅ Successfully imported StockAIAnalyzer")
    
    # Test initialization
    ai_analyzer = StockAIAnalyzer()
    print("✅ Successfully initialized StockAIAnalyzer")
    
    # Test methods
    print(f"✅ Model type: {ai_analyzer.model_type}")
    print(f"✅ Confidence threshold: {ai_analyzer.confidence_threshold}")
    
    # Test sentiment analysis
    sentiment = ai_analyzer.analyze_sentiment("RELIANCE")
    print(f"✅ Sentiment analysis test: {sentiment}")
    
    print("\n🎉 All AI model tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
