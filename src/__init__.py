"""
Nifty 50 Tracker - Core Application Module

This package contains the core functionality for the Nifty 50 stock tracker application.
"""

__version__ = "1.0.0"
__author__ = "Nifty50-Tracker Team"

# Make key classes available at package level
from .enhanced_stock_fetcher import StockDataFetcher
from .stock_fetcher import StockDataFetcher as SimpleStockDataFetcher

__all__ = ["StockDataFetcher", "SimpleStockDataFetcher"]
