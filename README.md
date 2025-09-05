# Nifty 50 Tracker

A comprehensive stock tracking and analysis application for Nifty 50 stocks with technical indicators and interactive dashboard.

## Project Structure

```
Nifty50-Tracker/
├── src/                          # Core application code
│   ├── __init__.py
│   ├── enhanced_stock_fetcher.py # Advanced stock data fetching with technical analysis
│   ├── stock_fetcher.py         # Simple stock data fetching
│   └── main.py                  # Main application entry point
├── frontend/                     # Streamlit web application
│   └── streamlit_app.py         # Interactive dashboard
├── config/                       # Configuration files
│   ├── __init__.py
│   └── nifty50_symbols_2025.json # Nifty 50 stock symbols
├── scripts/                      # Utility scripts
│   ├── __init__.py
│   ├── database_setup.py        # Database initialization
│   └── setup_generator.py       # Setup file generation
├── docs/                         # Documentation assets
│   ├── README.md
│   └── nifty50_setup_flowchart.png
├── data/                         # Database and data files
│   └── nifty50_stocks.db
├── tests/                        # Test files
│   └── test1.py
├── requirements.txt              # Python dependencies
├── setup_verification_test.py    # Setup verification script
├── quick_start.sh               # Unix setup script
└── quick_start.bat              # Windows setup script
```

## Features

- Real-time stock data fetching from Yahoo Finance
- Technical analysis with SMA, EMA, RSI, and MACD indicators
- Interactive Streamlit dashboard
- SQLite database for data storage
- Comprehensive setup verification

## Quick Start

1. **Setup Environment**
   ```bash
   # Linux/Mac
   ./quick_start.sh
   
   # Windows
   quick_start.bat
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Setup**
   ```bash
   python setup_verification_test.py
   ```

4. **Initialize Database**
   ```bash
   python scripts/database_setup.py
   ```

5. **Run Dashboard**
   ```bash
   streamlit run frontend/streamlit_app.py
   ```

## Development

The project follows a modular structure:

- **src/** - Contains the core business logic and data fetching capabilities
- **frontend/** - Web interface built with Streamlit
- **config/** - Configuration files and data assets
- **scripts/** - Utility scripts for setup and maintenance

## Technical Indicators

- **SMA (Simple Moving Average)** - 20, 50, 200 periods
- **EMA (Exponential Moving Average)** - 12, 26, 50 periods  
- **RSI (Relative Strength Index)** - 14 periods
- **MACD (Moving Average Convergence Divergence)** - 12, 26, 9 periods

## Requirements

- Python 3.7+
- Internet connection for real-time data
- Dependencies listed in requirements.txt