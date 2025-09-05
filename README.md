# 📈 Nifty 50 Stock Tracker

A comprehensive Python-based stock tracking and technical analysis application for Nifty 50 stocks. Features an interactive Streamlit dashboard with real-time data fetching, technical indicators, and educational content for beginners.

## 🌟 Features

### 📊 Technical Analysis
- **Moving Averages**: SMA (20, 50, 200) and EMA (12, 26, 50)
- **RSI Indicator**: 14-period Relative Strength Index with buy/sell signals
- **MACD Analysis**: Moving Average Convergence Divergence with signal interpretation
- **Real-time Price Data**: Live stock prices with change indicators

### 🎓 Educational Content
- Detailed explanations of each technical indicator
- Signal interpretations for beginners
- Interactive visualizations and progress bars
- Educational tooltips and analysis descriptions

### 🚀 Easy Setup
- Cross-platform setup scripts (Windows & Linux/Mac)
- Comprehensive setup verification system
- One-command installation process
- Detailed error handling and troubleshooting

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- Internet connection for stock data fetching

### Quick Start

#### Option 1: Use Setup Scripts (Recommended)

**For Linux/Mac:**
```bash
chmod +x quick_start.sh
./quick_start.sh
```

**For Windows:**
```cmd
quick_start.bat
```

#### Option 2: Manual Setup

1. **Clone the repository:**
```bash
git clone https://github.com/ankit05gupta/Nifty50-Tracker.git
cd Nifty50-Tracker
```

2. **Create virtual environment:**
```bash
python -m venv venv

# Activate virtual environment
# For Linux/Mac:
source venv/bin/activate
# For Windows:
venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Verify setup:**
```bash
python setup_verification_test.py
```

## 🚦 Usage

### 1. Start the Interactive Dashboard
```bash
streamlit run frontend/streamlit_app.py
```

This will open a web browser with the interactive dashboard where you can:
- Select any Nifty 50 stock
- Choose analysis period (1 month, 3 months, 6 months, 1 year)
- View real-time technical analysis
- Get buy/sell recommendations

### 2. Database Setup (Optional)
```bash
python app/database.py
```

### 3. Setup Verification
```bash
python setup_verification_test.py
```

## 📁 Project Structure

```
Nifty50-Tracker/
├── app/
│   ├── __init__.py
│   ├── database.py              # SQLite database setup
│   ├── enhanced_stock_fetcher.py # Main stock analysis engine
│   ├── main.py                  # Application entry point
│   └── stock_fetcher.py         # Stock data fetching utilities
├── frontend/
│   └── streamlit_app.py         # Interactive web dashboard
├── data/                        # Data storage directory
├── tests/
│   └── test1.py                 # Test files
├── requirements.txt             # Python dependencies
├── nifty50_symbols_2025.json   # Nifty 50 stock symbols
├── setup_verification_test.py   # Setup verification script
├── quick_start.sh              # Linux/Mac setup script
├── quick_start.bat             # Windows setup script
└── README.md                   # This file
```

## 🔧 Technical Indicators Explained

### Simple Moving Average (SMA)
- **Purpose**: Smooths price fluctuations to show trend direction
- **Signals**: 
  - Price above SMA = Uptrend (Buy/Hold)
  - Price below SMA = Downtrend (Sell)
  - Golden Cross (SMA50 > SMA200) = Strong buy signal

### Exponential Moving Average (EMA)
- **Purpose**: More responsive to recent price changes than SMA
- **Usage**: Better for short-term trading and quick trend changes

### Relative Strength Index (RSI)
- **Range**: 0-100
- **Signals**:
  - RSI > 70: Overbought (Consider selling)
  - RSI < 30: Oversold (Consider buying)
  - RSI 30-70: Normal trading range

### MACD (Moving Average Convergence Divergence)
- **Components**: MACD Line, Signal Line, Histogram
- **Signals**:
  - MACD crosses above Signal = Buy signal
  - MACD crosses below Signal = Sell signal
  - MACD above 0 = Overall uptrend

## 📋 Supported Stocks

The application supports all Nifty 50 stocks including:
- RELIANCE, HDFCBANK, BHARTIARTL, TCS, ICICIBANK
- SBIN, HINDUNILVR, INFY, BAJFINANCE, ITC
- And 40+ more Nifty 50 constituents

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**: Run `python setup_verification_test.py` to check dependencies
2. **Network Issues**: Ensure internet connection for stock data fetching
3. **Permission Errors**: Check file system permissions for data directory

### Setup Verification
The included setup verification script checks:
- ✅ Python version compatibility
- ✅ All required libraries installation
- ✅ Database connectivity
- ✅ Stock data fetching capability
- ✅ File system permissions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Yahoo Finance API for stock data
- Streamlit for the interactive dashboard framework
- TA-Lib for technical analysis calculations
- Plotly for advanced visualizations

## 📞 Support

If you encounter any issues or have questions:
1. Check the troubleshooting section above
2. Run the setup verification script
3. Review the error messages for specific guidance
4. Open an issue on GitHub for additional support

---

**Happy Trading! 📈**

*Disclaimer: This tool is for educational purposes only. Always do your own research before making investment decisions.*
