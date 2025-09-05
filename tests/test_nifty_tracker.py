"""
Test suite for Nifty 50 Stock Tracker

Basic tests to ensure the package functionality works correctly.
"""

import unittest
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nifty_tracker import StockDataFetcher, get_db_path

class TestStockFetcher(unittest.TestCase):
    """Test the StockDataFetcher functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.fetcher = StockDataFetcher()
    
    def test_fetcher_initialization(self):
        """Test that the fetcher initializes correctly"""
        self.assertIsNotNone(self.fetcher)
        self.assertIsInstance(self.fetcher.nifty50_symbols, list)
        self.assertGreater(len(self.fetcher.nifty50_symbols), 0)
    
    def test_technical_indicators_calculation(self):
        """Test that technical indicators can be calculated"""
        import pandas as pd
        import numpy as np
        
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        sample_data = pd.DataFrame({
            'Close': np.random.randn(100).cumsum() + 100,
            'Open': np.random.randn(100).cumsum() + 100,
            'High': np.random.randn(100).cumsum() + 102,
            'Low': np.random.randn(100).cumsum() + 98,
            'Volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        # Test SMA calculation
        sma_20 = self.fetcher.calculate_sma(sample_data, 20)
        self.assertIsNotNone(sma_20)
        
        # Test EMA calculation  
        ema_12 = self.fetcher.calculate_ema(sample_data, 12)
        self.assertIsNotNone(ema_12)
        
        # Test RSI calculation
        rsi = self.fetcher.calculate_rsi(sample_data, 14)
        self.assertIsNotNone(rsi)
    
    def test_database_path(self):
        """Test database path functionality"""
        db_path = get_db_path()
        self.assertIsNotNone(db_path)
        self.assertTrue(db_path.endswith('.db'))

class TestConfigFiles(unittest.TestCase):
    """Test configuration files exist"""
    
    def test_nifty50_symbols_exist(self):
        """Test that Nifty 50 symbols configuration exists"""
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'nifty50_symbols_2025.json')
        self.assertTrue(os.path.exists(config_path))

if __name__ == '__main__':
    print("🧪 Running Nifty 50 Stock Tracker Tests")
    print("=" * 50)
    
    # Run the tests
    unittest.main(verbosity=2)