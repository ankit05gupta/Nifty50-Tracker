"""
Enhanced Stock Data Fetcher with Technical Indicators and Educational Content

This module provides comprehensive stock analysis with technical indicators
and educational explanations for beginners learning about stock trading.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # For non-blocking backend

# Try to import TA-Lib for optimized calculations
try:
    import talib
    TALIB_AVAILABLE = True
    print("✅ TA-Lib available - Using optimized calculations")
except ImportError:
    TALIB_AVAILABLE = False
    print("⚠️  TA-Lib not installed - Using pandas calculations")
    print("   Install with: pip install TA-Lib (for faster performance)")

class StockDataFetcher:
    """
    Enhanced stock data fetcher with technical analysis capabilities
    
    Features:
    - Fetches data for all Nifty 50 stocks
    - Calculates SMA, EMA, RSI, and MACD indicators
    - Provides educational explanations for each indicator
    - Interprets signals automatically
    - Generates buy/sell recommendations
    """
    
    def __init__(self):
        """Initialize the fetcher with Nifty 50 symbols and educational content"""
        # Load Nifty 50 symbols
        try:
            with open('nifty50_symbols_2025.json') as f:
                    self.nifty50_symbols = json.load(f)
                    print(f"✅ Loaded {len(self.nifty50_symbols)} Nifty 50 symbols")
        except FileNotFoundError:
            # Fallback list if file not found
            self.nifty50_symbols = [
                'RELIANCE.NS', 'HDFCBANK.NS', 'BHARTIARTL.NS', 'TCS.NS', 'ICICIBANK.NS',
                'SBIN.NS', 'HINDUNILVR.NS', 'INFY.NS', 'BAJFINANCE.NS', 'ITC.NS',
                'LT.NS', 'MARUTI.NS', 'HCLTECH.NS', 'M&M.NS', 'KOTAKBANK.NS',
                'SUNPHARMA.NS', 'ULTRACEMCO.NS', 'TITAN.NS', 'AXISBANK.NS', 'NTPC.NS'
            ]
            print("⚠️  JSON file not found - Using fallback symbols")
            
            # Create the JSON file for future use
            with open('nifty50_symbols_2025.json', 'w') as f:
                json.dump(self.nifty50_symbols, f, indent=2)
            print("✅ Created nifty50_symbols_2025.json file")
        
        # Educational content about technical indicators
        self.indicator_explanations = {
            'SMA': {
                'name': 'Simple Moving Average',
                'description': 'The average closing price over a specific number of days. It smooths out price fluctuations to show the overall trend direction.',
                'formula': 'SMA = (Price1 + Price2 + ... + PriceN) / N',
                'example': 'If RELIANCE traded at ₹2400, ₹2410, ₹2420, ₹2430, ₹2440 over 5 days, SMA = ₹2420',
                'periods': [20, 50, 200],
                'signals': {
                    'bullish': 'Price above SMA = Uptrend (Good time to hold/buy)',
                    'bearish': 'Price below SMA = Downtrend (Consider selling)',
                    'golden_cross': 'SMA(50) crosses above SMA(200) = Strong buy signal',
                    'death_cross': 'SMA(50) crosses below SMA(200) = Strong sell signal'
                }
            },
            'EMA': {
                'name': 'Exponential Moving Average', 
                'description': 'Like SMA but gives more weight to recent prices, making it more responsive to current market conditions.',
                'formula': 'EMA = (Close × Multiplier) + (Previous EMA × (1 - Multiplier))',
                'advantage': 'Reacts faster to price changes than SMA, better for short-term trading',
                'periods': [12, 26, 50],
                'signals': {
                    'bullish': 'Price above EMA = Short-term bullish momentum',
                    'bearish': 'Price below EMA = Short-term bearish momentum',
                    'crossover': 'EMA(12) above EMA(26) = Buy signal for MACD'
                }
            },
            'RSI': {
                'name': 'Relative Strength Index',
                'description': 'Measures how fast and how much a stock price is changing. It oscillates between 0 and 100.',
                'formula': 'RSI = 100 - (100 / (1 + (Average Gain / Average Loss)))',
                'period': 14,
                'ranges': {
                    'overbought': '70-100: Stock may be overpriced, consider selling',
                    'normal': '30-70: Stock in normal trading range, hold position',
                    'oversold': '0-30: Stock may be underpriced, consider buying'
                },
                'divergence': 'When price makes new highs but RSI doesn\'t = Potential reversal'
            },
            'MACD': {
                'name': 'Moving Average Convergence Divergence',
                'description': 'Shows the relationship between two moving averages and helps identify trend changes.',
                'components': {
                    'MACD Line': 'EMA(12) - EMA(26)',
                    'Signal Line': '9-day EMA of MACD Line',
                    'Histogram': 'MACD Line - Signal Line'
                },
                'signals': {
                    'bullish_crossover': 'MACD crosses above Signal line = Buy signal',
                    'bearish_crossover': 'MACD crosses below Signal line = Sell signal',
                    'zero_line': 'MACD above 0 = Overall uptrend, below 0 = downtrend'
                }
            }
        }
    
    def calculate_sma(self, data, period=20):
        """Calculate Simple Moving Average"""
        if TALIB_AVAILABLE:
            # Ensure input is 1D float array
            return talib.SMA(data['Close'].values.astype(float).ravel(), timeperiod=period)
        else:
            return data['Close'].rolling(window=period).mean()

    def calculate_ema(self, data, period=20):
        """Calculate Exponential Moving Average"""
        if TALIB_AVAILABLE:
            return talib.EMA(data['Close'].values.astype(float).ravel(), timeperiod=period)
        else:
            return data['Close'].ewm(span=period).mean()

    def calculate_rsi(self, data, period=14):
        """Calculate Relative Strength Index"""
        if TALIB_AVAILABLE:
            return talib.RSI(data['Close'].values.astype(float).ravel(), timeperiod=period)
        else:
            # Manual RSI calculation using pandas
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi

    def calculate_macd(self, data, fast=12, slow=26, signal=9):
        """Calculate MACD indicator"""
        if TALIB_AVAILABLE:
            macd_line, signal_line, histogram = talib.MACD(
                data['Close'].values.astype(float).ravel(),
                fastperiod=fast,
                slowperiod=slow,
                signalperiod=signal
            )
            return macd_line, signal_line, histogram
        else:
            ema_fast = data['Close'].ewm(span=fast).mean()
            ema_slow = data['Close'].ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            histogram = macd_line - signal_line
            return macd_line, signal_line, histogram
    
    def add_technical_indicators(self, df):
        """Add all technical indicators to the dataframe (in-place for speed)"""
        # Moving Averages
        df['SMA_20'] = self.calculate_sma(df, 20)
        df['SMA_50'] = self.calculate_sma(df, 50)
        df['SMA_200'] = self.calculate_sma(df, 200)
        df['EMA_12'] = self.calculate_ema(df, 12)
        df['EMA_26'] = self.calculate_ema(df, 26)
        df['EMA_50'] = self.calculate_ema(df, 50)
        # RSI
        df['RSI'] = self.calculate_rsi(df, 14)
        # MACD
        macd_line, signal_line, histogram = self.calculate_macd(df)
        df['MACD'] = macd_line
        df['MACD_Signal'] = signal_line
        df['MACD_Histogram'] = histogram
        return df
    
    def interpret_rsi(self, rsi_value):
        """Interpret RSI value and provide trading suggestion"""
        if pd.isna(rsi_value):
            return {
                'status': '❓ Insufficient data',
                'message': 'Need more historical data to calculate RSI',
                'action': 'Wait'
            }
        elif rsi_value >= 70:
            return {
                'status': f'🔴 OVERBOUGHT',
                'message': f'RSI: {rsi_value:.1f} - Stock may be overpriced',
                'action': 'Consider SELLING or wait for pullback'
            }
        elif rsi_value <= 30:
            return {
                'status': f'🟢 OVERSOLD',
                'message': f'RSI: {rsi_value:.1f} - Stock may be underpriced', 
                'action': 'Consider BUYING opportunity'
            }
        else:
            return {
                'status': f'🟡 NEUTRAL',
                'message': f'RSI: {rsi_value:.1f} - Normal trading range',
                'action': 'HOLD current position'
            }
    
    def interpret_moving_averages(self, current_price, sma_20, sma_50, ema_12):
        """Interpret moving average signals"""
        signals = []
        
        # SMA signals
        if not pd.isna(sma_20):
            if current_price > sma_20:
                signals.append({
                    'type': 'SMA_20',
                    'signal': '🟢 BULLISH',
                    'message': f'Price (₹{current_price:.2f}) above SMA(20) (₹{sma_20:.2f})'
                })
            else:
                signals.append({
                    'type': 'SMA_20', 
                    'signal': '🔴 BEARISH',
                    'message': f'Price (₹{current_price:.2f}) below SMA(20) (₹{sma_20:.2f})'
                })
        
        # Golden Cross / Death Cross
        if not pd.isna(sma_20) and not pd.isna(sma_50):
            if sma_20 > sma_50:
                signals.append({
                    'type': 'MA_Cross',
                    'signal': '🟢 GOLDEN CROSS',
                    'message': 'SMA(20) above SMA(50) - Strong uptrend'
                })
            else:
                signals.append({
                    'type': 'MA_Cross',
                    'signal': '🔴 DEATH CROSS', 
                    'message': 'SMA(20) below SMA(50) - Potential downtrend'
                })
        
        # EMA short-term signal
        if not pd.isna(ema_12):
            if current_price > ema_12:
                signals.append({
                    'type': 'EMA_12',
                    'signal': '🟢 Short-term bullish',
                    'message': f'Price above EMA(12) (₹{ema_12:.2f})'
                })
            else:
                signals.append({
                    'type': 'EMA_12',
                    'signal': '🔴 Short-term bearish',
                    'message': f'Price below EMA(12) (₹{ema_12:.2f})'
                })
        
        return signals
    
    def interpret_macd(self, macd, macd_signal, macd_hist):
        """Interpret MACD signals"""
        if pd.isna(macd) or pd.isna(macd_signal):
            return {
                'status': '❓ Insufficient data',
                'message': 'Need more data for MACD calculation',
                'action': 'Wait'
            }
        
        if macd > macd_signal:
            if macd_hist > 0:
                return {
                    'status': '🟢 BULLISH MOMENTUM',
                    'message': f'MACD ({macd:.2f}) above Signal ({macd_signal:.2f})',
                    'action': 'Upward momentum confirmed'
                }
            else:
                return {
                    'status': '🟡 WEAKENING BULLISH',
                    'message': 'MACD above signal but histogram declining',
                    'action': 'Monitor for potential reversal'
                }
        else:
            if macd_hist < 0:
                return {
                    'status': '🔴 BEARISH MOMENTUM', 
                    'message': f'MACD ({macd:.2f}) below Signal ({macd_signal:.2f})',
                    'action': 'Downward momentum confirmed'
                }
            else:
                return {
                    'status': '🟡 WEAKENING BEARISH',
                    'message': 'MACD below signal but histogram rising',
                    'action': 'Potential upward reversal forming'
                }
    
    def get_stock_data(self, symbol, period="3mo"):
        """
        Fetch stock data with complete technical analysis
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE.NS')
            period: Time period ('1mo', '3mo', '6mo', '1y', '2y')
        
        Returns:
            Dictionary with stock data and technical analysis
        """
        try:
            print(f"📡 Fetching {symbol} data for {period}...")
            
            # Download stock data
            data = yf.download(symbol, period=period, progress=False, threads=False, auto_adjust=False)
            
            if data.empty or 'Close' not in data.columns:
                return {
                    'symbol': symbol,
                    'success': False, 
                    'error': 'No data available for this symbol/period'
                }
            
            # Add technical indicators
            data_with_indicators = self.add_technical_indicators(data)
            
            # Get latest values
            current_price = float(data['Close'].iloc[-1])
            latest_rsi = data_with_indicators['RSI'].iloc[-1]
            latest_sma_20 = data_with_indicators['SMA_20'].iloc[-1]
            latest_sma_50 = data_with_indicators['SMA_50'].iloc[-1]
            latest_ema_12 = data_with_indicators['EMA_12'].iloc[-1]
            latest_macd = data_with_indicators['MACD'].iloc[-1]
            latest_macd_signal = data_with_indicators['MACD_Signal'].iloc[-1]
            latest_macd_hist = data_with_indicators['MACD_Histogram'].iloc[-1]
            
            # Generate interpretations
            rsi_analysis = self.interpret_rsi(latest_rsi)
            ma_signals = self.interpret_moving_averages(
                current_price, latest_sma_20, latest_sma_50, latest_ema_12
            )
            macd_analysis = self.interpret_macd(latest_macd, latest_macd_signal, latest_macd_hist)
            
            # Calculate price changes
            prev_close = float(data['Close'].iloc[-2]) if len(data) > 1 else current_price
            price_change = current_price - prev_close
            price_change_pct = (price_change / prev_close) * 100
            
            return {
                'symbol': symbol,
                'success': True,
                'current_price': current_price,
                'price_change': price_change,
                'price_change_percent': price_change_pct,
                'data': data_with_indicators,
                'technical_analysis': {
                    # Raw values
                    'rsi_value': latest_rsi,
                    'sma_20': latest_sma_20,
                    'sma_50': data_with_indicators['SMA_50'].iloc[-1],
                    'sma_200': data_with_indicators['SMA_200'].iloc[-1],  # <-- Add this line
                    'ema_12': latest_ema_12,
                    'ema_26': data_with_indicators['EMA_26'].iloc[-1],    # <-- Add this line
                    'ema_50': data_with_indicators['EMA_50'].iloc[-1],    # <-- Add this line
                    'macd_value': latest_macd,
                    'macd_signal_value': latest_macd_signal,
                    'macd_histogram': latest_macd_hist,
                    
                    # Interpretations
                    'rsi_analysis': rsi_analysis,
                    'moving_average_signals': ma_signals,
                    'macd_analysis': macd_analysis,
                    
                    # Meta data
                    'data_points': len(data),
                    'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            }
            
        except Exception as e:
            return {
                'symbol': symbol,
                'success': False,
                'error': f'Error fetching data: {str(e)}'
            }
    
    def get_multiple_stocks(self, symbols=None, max_stocks=10):
        """
        Analyze multiple stocks with technical indicators (batch fetch for speed)
        """
        if symbols is None:
            symbols = self.nifty50_symbols[:max_stocks]
        print(f"🚀 Analyzing {len(symbols)} stocks with technical indicators...")
        # Batch fetch all data at once using yfinance
        data = yf.download(' '.join(symbols), period="3mo", group_by='ticker', progress=False, threads=True, auto_adjust=False)
        results = {}
        for i, symbol in enumerate(symbols, 1):
            # Reduce print statements for speed
            # print(f"[{i:2d}/{len(symbols)}] {symbol}")
            try:
                symbol_data = data[symbol] if symbol in data else None
                if symbol_data is None or symbol_data.empty or 'Close' not in symbol_data.columns:
                    results[symbol] = {
                        'symbol': symbol,
                        'success': False,
                        'error': 'No data available for this symbol/period'
                    }
                    continue
                symbol_data = self.add_technical_indicators(symbol_data)
                current_price = float(symbol_data['Close'].iloc[-1])
                latest_rsi = symbol_data['RSI'].iloc[-1]
                latest_sma_20 = symbol_data['SMA_20'].iloc[-1]
                latest_sma_50 = symbol_data['SMA_50'].iloc[-1]
                latest_ema_12 = symbol_data['EMA_12'].iloc[-1]
                latest_macd = symbol_data['MACD'].iloc[-1]
                latest_macd_signal = symbol_data['MACD_Signal'].iloc[-1]
                latest_macd_hist = symbol_data['MACD_Histogram'].iloc[-1]
                rsi_analysis = self.interpret_rsi(latest_rsi)
                ma_signals = self.interpret_moving_averages(current_price, latest_sma_20, latest_sma_50, latest_ema_12)
                macd_analysis = self.interpret_macd(latest_macd, latest_macd_signal, latest_macd_hist)
                prev_close = float(symbol_data['Close'].iloc[-2]) if len(symbol_data) > 1 else current_price
                price_change = current_price - prev_close
                price_change_pct = (price_change / prev_close) * 100
                results[symbol] = {
                    'symbol': symbol,
                    'success': True,
                    'current_price': current_price,
                    'price_change': price_change,
                    'price_change_percent': price_change_pct,
                    'data': symbol_data,
                    'technical_analysis': {
                        'rsi_value': latest_rsi,
                        'sma_20': latest_sma_20,
                        'sma_50': symbol_data['SMA_50'].iloc[-1],
                        'sma_200': symbol_data['SMA_200'].iloc[-1],
                        'ema_12': latest_ema_12,
                        'ema_26': symbol_data['EMA_26'].iloc[-1],
                        'ema_50': symbol_data['EMA_50'].iloc[-1],
                        'macd_value': latest_macd,
                        'macd_signal_value': latest_macd_signal,
                        'macd_histogram': latest_macd_hist,
                        'rsi_analysis': rsi_analysis,
                        'moving_average_signals': ma_signals,
                        'macd_analysis': macd_analysis,
                        'data_points': len(symbol_data),
                        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                }
            except Exception as e:
                results[symbol] = {
                    'symbol': symbol,
                    'success': False,
                    'error': f'Error fetching data: {str(e)}'
                }
        print("✅ Analysis complete!")
        return results
    
    def print_detailed_analysis(self, symbol, analysis_result):
        """Print detailed analysis for a single stock and plot price/indicator graph"""
        if not analysis_result['success']:
            print(f"❌ {symbol}: {analysis_result.get('error', 'Unknown error')}")
            return
        
        # Header
        price_change = analysis_result['price_change']
        change_pct = analysis_result['price_change_percent']
        change_arrow = "📈" if price_change >= 0 else "📉"
        
        print(f"\n{change_arrow} {symbol.replace('.NS', '')} - ₹{analysis_result['current_price']:.2f}")
        print(f"   Change: {price_change:+.2f} ({change_pct:+.2f}%)")
        print("-" * 50)
        
        ta = analysis_result['technical_analysis']
        
        # RSI Analysis
        rsi = ta['rsi_analysis']
        print(f"🎯 {rsi['status']}: {rsi['message']}")
        print(f"   Action: {rsi['action']}")
        
        # Moving Average Signals
        print("\n📊 Moving Average Signals:")
        for signal in ta['moving_average_signals']:
            print(f"   {signal['signal']}: {signal['message']}")
        
        # MACD Analysis  
        macd = ta['macd_analysis']
        print(f"\n⚡ {macd['status']}: {macd['message']}")
        print(f"   Trend: {macd['action']}")
        
        # Plot price and indicators
        df = analysis_result['data']
        plt.figure(figsize=(10,6))
        plt.plot(df.index, df['Close'], label='Close Price', color='black')
        plt.plot(df.index, df['SMA_20'], label='SMA 20', linestyle='--')
        plt.plot(df.index, df['SMA_50'], label='SMA 50', linestyle='--')
        plt.plot(df.index, df['EMA_12'], label='EMA 12', linestyle=':')
        plt.plot(df.index, df['EMA_26'], label='EMA 26', linestyle=':')
        plt.title(f'{symbol.replace(".NS", "")} Price & Moving Averages')
        plt.xlabel('Date')
        plt.ylabel('Price (₹)')
        plt.legend()
        plt.tight_layout()
        fname = f'{symbol.replace(".NS", "")}_price_ma.png'
        plt.savefig(fname)
        print(f"Price/MA chart saved as {fname}")
        
        # Plot RSI and MACD
        fig, ax1 = plt.subplots(figsize=(10,5))
        ax1.plot(df.index, df['RSI'], label='RSI', color='purple')
        ax1.axhline(70, color='red', linestyle='--', linewidth=0.7)
        ax1.axhline(30, color='green', linestyle='--', linewidth=0.7)
        ax1.set_ylabel('RSI')
        ax1.set_xlabel('Date')
        ax1.set_title(f'{symbol.replace(".NS", "")} RSI & MACD')
        ax2 = ax1.twinx()
        ax2.plot(df.index, df['MACD'], label='MACD', color='blue')
        ax2.plot(df.index, df['MACD_Signal'], label='MACD Signal', color='orange')
        ax2.bar(df.index, df['MACD_Histogram'], label='MACD Histogram', color='grey', alpha=0.3)
        ax2.set_ylabel('MACD')
        fig.legend(loc='upper left')
        plt.tight_layout()
        fname2 = f'{symbol.replace(".NS", "")}_rsi_macd.png'
        plt.savefig(fname2)
        print(f"RSI/MACD chart saved as {fname2}")
    
    def print_portfolio_summary(self, results):
        """Print a comprehensive portfolio summary with pie charts and bar graphs"""
        print("\n" + "="*90)
        print("📊 NIFTY 50 TECHNICAL ANALYSIS PORTFOLIO SUMMARY")
        print("="*90)
        
        successful_analyses = 0
        buy_candidates = []
        sell_candidates = []
        hold_candidates = []
        bullish_momentum = []
        bearish_momentum = []
        price_changes = []
        price_labels = []
        
        # Analyze all results
        for symbol, data in results.items():
            if data['success']:
                successful_analyses += 1
                ta = data['technical_analysis']
                clean_symbol = symbol.replace('.NS', '')
                
                # RSI-based categorization
                rsi_status = ta['rsi_analysis']['status']
                if 'OVERSOLD' in rsi_status:
                    buy_candidates.append(clean_symbol)
                elif 'OVERBOUGHT' in rsi_status:
                    sell_candidates.append(clean_symbol)
                else:
                    hold_candidates.append(clean_symbol)
                
                # MACD momentum
                macd_status = ta['macd_analysis']['status']
                if 'BULLISH' in macd_status:
                    bullish_momentum.append(clean_symbol)
                elif 'BEARISH' in macd_status:
                    bearish_momentum.append(clean_symbol)
                
                # Price change for bar chart
                price_changes.append(data['price_change_percent'])
                price_labels.append(clean_symbol)
        
        # Print summary statistics
        print(f"✅ Successfully analyzed: {successful_analyses}/{len(results)} stocks")
        print(f"📅 Analysis completed: {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}")
        
        # Pie chart for RSI-based recommendations
        pie_labels = ['Buy (RSI<30)', 'Sell (RSI>70)', 'Hold (RSI 30-70)']
        pie_sizes = [len(buy_candidates), len(sell_candidates), len(hold_candidates)]
        pie_colors = ['#4CAF50', '#F44336', '#FFEB3B']
        plt.figure(figsize=(6,6))
        plt.pie(pie_sizes, labels=pie_labels, colors=pie_colors, autopct='%1.1f%%', startangle=140)
        plt.title('RSI-Based Recommendations')
        plt.tight_layout()
        plt.savefig('portfolio_rsi_pie.png')
        print("Pie chart saved as portfolio_rsi_pie.png")
        
        # Bar chart for price changes
        if price_changes:
            plt.figure(figsize=(12,5))
            bars = plt.bar(price_labels, price_changes, color=['#4CAF50' if x>=0 else '#F44336' for x in price_changes])
            plt.xticks(rotation=90)
            plt.ylabel('Price Change (%)')
            plt.title('Stock Price Change (%)')
            plt.tight_layout()
            plt.savefig('portfolio_price_change_bar.png')
            print("Bar chart saved as portfolio_price_change_bar.png")
        
        # MACD momentum analysis
        print(f"\n🎯 RSI-BASED RECOMMENDATIONS:")
        print("-" * 45)
        if buy_candidates:
            print(f"🟢 BUY Opportunities (RSI < 30): {len(buy_candidates)} stocks")
            print(f"   {', '.join(buy_candidates[:10])}{'...' if len(buy_candidates) > 10 else ''}")
        
        if sell_candidates:
            print(f"🔴 SELL Signals (RSI > 70): {len(sell_candidates)} stocks") 
            print(f"   {', '.join(sell_candidates[:10])}{'...' if len(sell_candidates) > 10 else ''}")
        
        if hold_candidates:
            print(f"🟡 HOLD Positions (RSI 30-70): {len(hold_candidates)} stocks")
            print(f"   {', '.join(hold_candidates[:10])}{'...' if len(hold_candidates) > 10 else ''}")
        
        # MACD momentum analysis
        print(f"\n⚡ MOMENTUM ANALYSIS (MACD):")
        print("-" * 35)
        if bullish_momentum:
            print(f"📈 Bullish Momentum: {len(bullish_momentum)} stocks")
            print(f"   {', '.join(bullish_momentum[:10])}{'...' if len(bullish_momentum) > 10 else ''}")
        
        if bearish_momentum:
            print(f"📉 Bearish Momentum: {len(bearish_momentum)} stocks")
            print(f"   {', '.join(bearish_momentum[:10])}{'...' if len(bearish_momentum) > 10 else ''}")
        
        # Market sentiment
        total_analyzed = len(buy_candidates) + len(sell_candidates) + len(hold_candidates)
        if total_analyzed > 0:
            bullish_pct = (len(buy_candidates) / total_analyzed) * 100
            bearish_pct = (len(sell_candidates) / total_analyzed) * 100;
            
            print(f"\n📊 OVERALL MARKET SENTIMENT:")
            print("-" * 35)
            if bullish_pct > 40:
                print(f"🟢 BULLISH Market ({bullish_pct:.1f}% oversold stocks)")
            elif bearish_pct > 40:
                print(f"🔴 BEARISH Market ({bearish_pct:.1f}% overbought stocks)")
            else:
                print(f"🟡 NEUTRAL Market (Balanced conditions)")
        
        print(f"\n" + "="*90)
        print("⚠️  IMPORTANT DISCLAIMER")
        print("="*90)
        print("• This analysis is for educational purposes only")
        print("• Always do your own research before investing")
        print("• Consider multiple factors beyond technical indicators")
        print("• Use proper risk management and position sizing")
        print("• Past performance doesn't guarantee future results")
    
    def print_educational_content(self):
        """Print comprehensive educational content about technical indicators"""
        print("\\n" + "="*90)
        print("📚 TECHNICAL ANALYSIS EDUCATION FOR BEGINNERS")
        print("="*90)
        
        print("\\n🎯 What is Technical Analysis?")
        print("Technical analysis studies price patterns and indicators to predict future")
        print("price movements. It's based on the idea that price reflects all available")
        print("information and that history tends to repeat itself.")
        
        # Explain each indicator
        for indicator_key, info in self.indicator_explanations.items():
            print(f"\\n{'='*60}")
            print(f"📊 {info['name']} ({indicator_key})")
            print("="*60)
            
            print(f"\\n🔍 What it is:")
            print(f"   {info['description']}")
            
            if 'formula' in info:
                print(f"\\n🧮 Formula:")
                print(f"   {info['formula']}")
            
            if 'example' in info:
                print(f"\\n💡 Example:")
                print(f"   {info['example']}")
            
            if 'periods' in info:
                print(f"\\n⏰ Common periods: {', '.join(map(str, info['periods']))} days")
            
            if 'ranges' in info:
                print(f"\\n📏 Interpretation ranges:")
                for range_name, description in info['ranges'].items():
                    print(f"   • {range_name.title()}: {description}")
            
            if 'signals' in info:
                print(f"\\n🚦 Trading signals:")
                for signal_type, description in info['signals'].items():
                    print(f"   • {description}")
            
            if 'components' in info:
                print(f"\\n🔧 Components:")
                for component, description in info['components'].items():
                    print(f"   • {component}: {description}")
        
        # Trading tips
        print(f"\\n{'='*90}")
        print("🎓 BEGINNER TRADING TIPS")
        print("="*90)
        
        tips = [
            "🔴 SELL Signals to watch:",
            "   • RSI > 70 (overbought condition)",
            "   • Price breaks below key moving averages", 
            "   • MACD crosses below signal line",
            "   • Volume spikes on down days",
            "",
            "🟢 BUY Signals to watch:",
            "   • RSI < 30 (oversold condition)",
            "   • Price breaks above moving averages",
            "   • MACD crosses above signal line", 
            "   • Golden cross (SMA 50 > SMA 200)",
            "",
            "⚠️  GOLDEN RULES:",
            "   • Never rely on just ONE indicator",
            "   • Always use 2-3 indicators for confirmation",
            "   • Consider overall market trend",
            "   • Set stop-losses to limit losses",
            "   • Start with paper trading to practice",
            "   • Never invest more than you can afford to lose"
        ]
        
        for tip in tips:
            print(tip)


# Example usage and testing functions
def run_comprehensive_analysis():
    """Run a comprehensive analysis of Nifty 50 stocks"""
    print("🚀 NIFTY 50 COMPREHENSIVE TECHNICAL ANALYSIS")
    print("="*80)
    
    # Initialize fetcher
    fetcher = StockDataFetcher()
    
    # Show educational content first
    fetcher.print_educational_content()
    
    # Ask user what they