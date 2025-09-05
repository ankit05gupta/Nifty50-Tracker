"""
Nifty 50 Stock Tracker Package

A comprehensive stock tracking and technical analysis package for Nifty 50 stocks.
"""

from .stock_fetcher import StockDataFetcher
from .database import (
    get_db_path,
    create_stock_table,
    fetch_and_store_all_stocks
)

__version__ = "1.0.0"
__author__ = "Nifty50-Tracker"

__all__ = [
    'StockDataFetcher',
    'get_db_path', 
    'create_stock_table',
    'fetch_and_store_all_stocks'
]