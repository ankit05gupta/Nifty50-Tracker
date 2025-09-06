#!/usr/bin/env python3
"""
Database Test Script
"""
import sqlite3
import os
from pathlib import Path

# Set up database path
db_path = Path(__file__).parent / "data" / "nifty50_stocks.db"
print(f"Checking database at: {db_path}")
print(f"Database exists: {db_path.exists()}")

if db_path.exists():
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        print(f"Tables found: {tables}")
        
        if tables:
            # Check stock_list table
            cursor.execute("SELECT COUNT(*) FROM stock_list")
            count = cursor.fetchone()[0]
            print(f"Total stocks in database: {count}")
            
            if count > 0:
                # Check column structure first
                cursor.execute("PRAGMA table_info(stock_list)")
                columns = cursor.fetchall()
                print("Column structure:")
                for col in columns:
                    print(f"  {col}")
                
                # Show sample data with correct column names
                try:
                    cursor.execute("SELECT * FROM stock_list LIMIT 3")
                    sample_data = cursor.fetchall()
                    print("Sample data (first 3 rows):")
                    for row in sample_data:
                        print(f"  {row}")
                except Exception as e:
                    print(f"Error fetching sample data: {e}")
            else:
                print("No data found in stock_list table")
        
        conn.close()
        
    except Exception as e:
        print(f"Database error: {e}")
else:
    print("Database file not found!")
    
    # Check if data directory exists
    data_dir = Path(__file__).parent / "data"
    print(f"Data directory exists: {data_dir.exists()}")
    if data_dir.exists():
        files = list(data_dir.glob("*"))
        print(f"Files in data directory: {files}")
