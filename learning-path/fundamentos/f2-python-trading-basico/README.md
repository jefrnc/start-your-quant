> 🇪🇸 [Leer en Español](README.es.md) | 🇺🇸 **English**

# F2: Basic Python Trading

**Fundamental Module 2 - Duration: 3-4 hours**

## Module Objectives

After completing this module you will be able to:
- Install and configure your trading environment
- Download real data from any stock
- Create professional price charts
- Calculate basic technical indicators
- Write your first analysis script

## Complete Setup (One Time Only)

### Step 1: Verify Python
```bash
# Open terminal/cmd and verify version
python --version
# Should show: Python 3.8 or higher

# If you don't have Python:
# Windows: Download from python.org
# Mac: brew install python
# Linux: sudo apt install python3 python3-pip
```

### Step 2: Install Essential Libraries
```bash
# Install all libraries at once
pip install yfinance pandas matplotlib seaborn numpy jupyter

# Verify installation
python -c "import yfinance, pandas, matplotlib; print('Everything installed correctly!')"
```

### Step 3: Configure Environment
```bash
# Create folder for your projects
mkdir my-quant-trading
cd my-quant-trading

# Optional: create virtual environment
python -m venv quant_env
# Activate: quant_env\\Scripts\\activate (Windows) or source quant_env/bin/activate (Mac/Linux)
```

## Your First Data Download

### Script 1: Basic Data

```python
# file: basic_download.py
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def download_stock(symbol, period='1y'):
    """
    Downloads stock data from Yahoo Finance

    Args:
        symbol (str): Stock symbol (e.g., 'AAPL', 'MSFT')
        period (str): Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')

    Returns:
        DataFrame: OHLCV data for the stock
    """
    try:
        # Download data
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)

        if data.empty:
            print(f"No data found for {symbol}")
            return None

        print(f"Downloaded {len(data)} days of data for {symbol}")
        return data

    except Exception as e:
        print(f"Error downloading {symbol}: {e}")
        return None

def show_basic_info(data, symbol):
    """Shows basic information about the data"""

    print(f"\nBASIC INFORMATION FOR {symbol}")
    print("=" * 40)

    # General information
    print(f"Period: {data.index[0].date()} to {data.index[-1].date()}")
    print(f"Total days: {len(data)}")

    # Prices
    current_price = data['Close'].iloc[-1]
    initial_price = data['Close'].iloc[0]
    total_change = (current_price / initial_price - 1) * 100

    print(f"\nPRICES:")
    print(f"  Initial price: ${initial_price:.2f}")
    print(f"  Current price: ${current_price:.2f}")
    print(f"  Total change: {total_change:+.2f}%")

    # Statistics
    print(f"\nSTATISTICS:")
    print(f"  Maximum price: ${data['High'].max():.2f}")
    print(f"  Minimum price: ${data['Low'].min():.2f}")
    print(f"  Average volume: {data['Volume'].mean():,.0f}")

    # Volatility
    daily_returns = data['Close'].pct_change().dropna()
    daily_volatility = daily_returns.std() * 100
    annual_volatility = daily_volatility * (252 ** 0.5)  # 252 trading days per year

    print(f"  Daily volatility: {daily_volatility:.2f}%")
    print(f"  Annual volatility: {annual_volatility:.2f}%")

# Test with Apple
if __name__ == "__main__":
    symbol = 'AAPL'
    data = download_stock(symbol, '1y')

    if data is not None:
        show_basic_info(data, symbol)
```

**Exercise 2.1**: Run this script and then:
1. Change the symbol to 'TSLA' and compare the statistics
2. Try with different periods ('6mo', '2y')
3. Note which stock is more volatile

## Creating Professional Charts

### Script 2: Visualizations

```python
# file: trading_charts.py
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np

# Configure chart style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def basic_price_chart(data, symbol):
    """Creates a basic price chart"""

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot closing price
    ax.plot(data.index, data['Close'], linewidth=2, label=f'{symbol} Close Price')

    # Customize
    ax.set_title(f'{symbol} - Close Price', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price ($)', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    # Date format on X axis
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

def simple_candlestick_chart(data, symbol, days=60):
    """Creates a simplified candlestick chart"""

    # Only show last X days
    recent_data = data.tail(days)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                   gridspec_kw={'height_ratios': [3, 1]})

    # Price chart (simulating candlesticks with lines)
    for i, (date, row) in enumerate(recent_data.iterrows()):
        color = 'green' if row['Close'] > row['Open'] else 'red'

        # High-low line
        ax1.plot([date, date], [row['Low'], row['High']], color='black', linewidth=0.5)

        # Candle "body"
        ax1.plot([date, date], [row['Open'], row['Close']], color=color, linewidth=3)

    ax1.set_title(f'{symbol} - Candlestick (Last {days} days)', fontsize=16)
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Volume chart
    colors = ['green' if close > open else 'red'
              for close, open in zip(recent_data['Close'], recent_data['Open'])]

    ax2.bar(recent_data.index, recent_data['Volume'], color=colors, alpha=0.7)
    ax2.set_ylabel('Volume', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Date formatting
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def multi_stock_comparison(symbols, period='6mo'):
    """Compares multiple normalized stocks"""

    fig, ax = plt.subplots(figsize=(12, 8))

    for symbol in symbols:
        try:
            data = yf.download(symbol, period=period)
            if not data.empty:
                # Normalize to 100 at start
                normalized_price = (data['Close'] / data['Close'].iloc[0]) * 100
                ax.plot(normalized_price.index, normalized_price,
                       linewidth=2, label=symbol)
        except:
            print(f"Error with {symbol}")

    ax.set_title('Return Comparison (Base 100)', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Normalized Value', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=100, color='black', linestyle='--', alpha=0.5)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def correlation_heatmap(symbols, period='1y'):
    """Creates a correlation heatmap between stocks"""

    # Download data for all symbols
    portfolio_data = {}

    for symbol in symbols:
        try:
            data = yf.download(symbol, period=period)
            if not data.empty:
                portfolio_data[symbol] = data['Close'].pct_change().dropna()
        except:
            print(f"Error downloading {symbol}")

    # Create returns DataFrame
    df_returns = pd.DataFrame(portfolio_data)

    # Calculate correlation matrix
    correlation = df_returns.corr()

    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='RdYlBu_r', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})

    plt.title('Returns Correlation Matrix', fontsize=16)
    plt.tight_layout()
    plt.show()

    return correlation

# Test visualizations
if __name__ == "__main__":
    # Download data
    data = yf.download('AAPL', period='1y')

    # Create charts
    print("Creating basic chart...")
    basic_price_chart(data, 'AAPL')

    print("Creating candlestick...")
    simple_candlestick_chart(data, 'AAPL')

    print("Comparing multiple stocks...")
    tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'NVDA']
    multi_stock_comparison(tech_stocks)

    print("Creating correlation heatmap...")
    correlations = correlation_heatmap(tech_stocks)
    print("\nCorrelations:")
    print(correlations.round(2))
```

**Exercise 2.2**:
1. Run all charts
2. Change the symbols to stocks that interest you
3. What do you observe in the correlations?

## Basic Technical Indicators

### Script 3: Essential Indicators

```python
# file: basic_indicators.py

def calculate_moving_averages(data, periods=[20, 50, 200]):
    """Calculates simple moving averages"""

    for period in periods:
        column = f'SMA_{period}'
        data[column] = data['Close'].rolling(window=period).mean()

    return data

def calculate_rsi(data, period=14):
    """Calculates Relative Strength Index"""

    delta = data['Close'].diff()

    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Calculate moving averages of gains and losses
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    # RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

def calculate_bollinger_bands(data, period=20, std_dev=2):
    """Calculates Bollinger Bands"""

    sma = data['Close'].rolling(window=period).mean()
    std = data['Close'].rolling(window=period).std()

    data['BB_Upper'] = sma + (std * std_dev)
    data['BB_Lower'] = sma - (std * std_dev)
    data['BB_Middle'] = sma

    # Relative position within bands
    data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])

    return data

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculates MACD (Moving Average Convergence Divergence)"""

    ema_fast = data['Close'].ewm(span=fast).mean()
    ema_slow = data['Close'].ewm(span=slow).mean()

    data['MACD'] = ema_fast - ema_slow
    data['MACD_Signal'] = data['MACD'].ewm(span=signal).mean()
    data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']

    return data

def calculate_volatility(data, period=20):
    """Calculates realized volatility"""

    returns = data['Close'].pct_change()
    data['Volatility'] = returns.rolling(window=period).std() * np.sqrt(252) * 100

    return data

def complete_technical_analysis(symbol, period='1y'):
    """Complete technical analysis of a stock"""

    # Download data
    data = yf.download(symbol, period=period)

    if data.empty:
        print(f"No data found for {symbol}")
        return None

    # Calculate all indicators
    data = calculate_moving_averages(data)
    data['RSI'] = calculate_rsi(data)
    data = calculate_bollinger_bands(data)
    data = calculate_macd(data)
    data = calculate_volatility(data)

    # Create dashboard charts
    create_technical_dashboard(data, symbol)

    # Current analysis
    analyze_current_situation(data, symbol)

    return data

def create_technical_dashboard(data, symbol):
    """Creates dashboard with multiple indicators"""

    fig, axes = plt.subplots(4, 1, figsize=(15, 16))

    # 1. Price with moving averages and Bollinger
    ax1 = axes[0]
    ax1.plot(data.index, data['Close'], label='Price', linewidth=2)
    ax1.plot(data.index, data['SMA_20'], label='SMA 20', alpha=0.7)
    ax1.plot(data.index, data['SMA_50'], label='SMA 50', alpha=0.7)
    ax1.fill_between(data.index, data['BB_Upper'], data['BB_Lower'],
                     alpha=0.2, label='Bollinger Bands')
    ax1.set_title(f'{symbol} - Price and Moving Averages', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. RSI
    ax2 = axes[1]
    ax2.plot(data.index, data['RSI'], color='purple', linewidth=2)
    ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought')
    ax2.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold')
    ax2.axhline(y=50, color='gray', linestyle='-', alpha=0.5)
    ax2.set_title('RSI (Relative Strength Index)', fontsize=14)
    ax2.set_ylabel('RSI')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)

    # 3. MACD
    ax3 = axes[2]
    ax3.plot(data.index, data['MACD'], label='MACD', linewidth=2)
    ax3.plot(data.index, data['MACD_Signal'], label='Signal', linewidth=2)
    ax3.bar(data.index, data['MACD_Histogram'], alpha=0.3, label='Histogram')
    ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax3.set_title('MACD', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Volatility
    ax4 = axes[3]
    ax4.plot(data.index, data['Volatility'], color='orange', linewidth=2)
    ax4.set_title('Realized Volatility (Annualized)', fontsize=14)
    ax4.set_ylabel('Volatility (%)')
    ax4.set_xlabel('Date')
    ax4.grid(True, alpha=0.3)

    # Date formatting
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def analyze_current_situation(data, symbol):
    """Analyzes the current technical situation"""

    # Most recent data
    last_row = data.iloc[-1]
    current_price = last_row['Close']

    print(f"\nCURRENT TECHNICAL ANALYSIS - {symbol}")
    print("=" * 50)

    # Price vs Moving Averages
    print("TREND:")
    if current_price > last_row['SMA_20'] > last_row['SMA_50']:
        print("  Strong uptrend (price > SMA20 > SMA50)")
    elif current_price > last_row['SMA_20']:
        print("  Weak uptrend (price > SMA20)")
    elif current_price < last_row['SMA_20'] < last_row['SMA_50']:
        print("  Strong downtrend (price < SMA20 < SMA50)")
    else:
        print("  Sideways or changing trend")

    # RSI
    current_rsi = last_row['RSI']
    print(f"\nRSI: {current_rsi:.1f}")
    if current_rsi > 70:
        print("  Overbought zone - possible correction")
    elif current_rsi < 30:
        print("  Oversold zone - possible bounce")
    else:
        print("  RSI in neutral zone")

    # Bollinger Bands
    bb_pos = last_row['BB_Position']
    print(f"\nBollinger Bands Position: {bb_pos:.2f}")
    if bb_pos > 0.8:
        print("  Near upper band - possible resistance")
    elif bb_pos < 0.2:
        print("  Near lower band - possible support")
    else:
        print("  In normal band range")

    # MACD
    macd_current = last_row['MACD']
    signal_current = last_row['MACD_Signal']
    print(f"\nMACD: {macd_current:.4f}")
    if macd_current > signal_current:
        print("  MACD above signal - positive momentum")
    else:
        print("  MACD below signal - negative momentum")

    # Volatility
    vol_current = last_row['Volatility']
    vol_average = data['Volatility'].tail(60).mean()
    print(f"\nVolatility: {vol_current:.1f}% (60d Average: {vol_average:.1f}%)")
    if vol_current > vol_average * 1.5:
        print("  High volatility - greater risk")
    elif vol_current < vol_average * 0.7:
        print("  Low volatility - quiet market")
    else:
        print("  Normal volatility")

# Run complete analysis
if __name__ == "__main__":
    symbol = 'AAPL'
    print(f"Analyzing {symbol}...")

    data = complete_technical_analysis(symbol)

    if data is not None:
        print("\nComplete analysis finished!")
        print(f"Data available from {data.index[0].date()} to {data.index[-1].date()}")
```

**Exercise 2.3**:
1. Run the complete analysis for AAPL
2. Switch to another stock (TSLA, MSFT, etc.)
3. Compare the current technical analyses
4. Which seems more "buyable" according to the indicators?

## Final Module Project

### Script 4: Your First Analysis System

```python
# file: my_first_system.py

class StockAnalyzer:
    """
    Your first quantitative analysis system
    """

    def __init__(self):
        self.results = {}

    def analyze_stock(self, symbol, period='6mo'):
        """Fully analyzes a stock"""

        print(f"\nAnalyzing {symbol}...")

        # Download data
        data = yf.download(symbol, period=period)

        if data.empty:
            print(f"No data for {symbol}")
            return None

        # Calculate indicators
        data = calculate_moving_averages(data)
        data['RSI'] = calculate_rsi(data)
        data = calculate_bollinger_bands(data)
        data = calculate_macd(data)
        data = calculate_volatility(data)

        # Calculate metrics
        result = self.calculate_metrics(data, symbol)
        self.results[symbol] = result

        return result

    def calculate_metrics(self, data, symbol):
        """Calculates key metrics"""

        # Current data
        current = data.iloc[-1]

        # Return
        total_return = (current['Close'] / data['Close'].iloc[0] - 1) * 100

        # Trend (score 0-100)
        trend_score = 0
        if current['Close'] > current['SMA_20']:
            trend_score += 25
        if current['SMA_20'] > current['SMA_50']:
            trend_score += 25
        if current['Close'] > current['SMA_50']:
            trend_score += 25
        if data['Close'].tail(5).mean() > data['Close'].tail(10).mean():
            trend_score += 25

        # RSI Score (50 = neutral, 0 = extreme oversold, 100 = extreme overbought)
        rsi_score = min(100, max(0, current['RSI']))

        # Momentum Score (MACD)
        momentum_score = 50  # Neutral base
        if current['MACD'] > current['MACD_Signal']:
            momentum_score += 25
        if current['MACD'] > 0:
            momentum_score += 15
        if data['MACD'].tail(3).mean() > data['MACD'].tail(6).mean():
            momentum_score += 10
        momentum_score = min(100, momentum_score)

        # Overall quality score
        overall_score = (trend_score * 0.4 +
                        (100 - abs(rsi_score - 50)) * 0.3 +
                        momentum_score * 0.3)

        return {
            'symbol': symbol,
            'current_price': current['Close'],
            'period_return': total_return,
            'trend_score': trend_score,
            'momentum_score': momentum_score,
            'current_rsi': current['RSI'],
            'current_volatility': current['Volatility'],
            'overall_score': overall_score,
            'recommendation': self.generate_recommendation(overall_score, current['RSI'])
        }

    def generate_recommendation(self, overall_score, rsi):
        """Generates recommendation based on scores"""

        if overall_score > 75 and 30 < rsi < 70:
            return "STRONG BUY"
        elif overall_score > 60 and 25 < rsi < 75:
            return "WEAK BUY"
        elif overall_score < 25 or rsi > 80 or rsi < 20:
            return "AVOID"
        else:
            return "NEUTRAL"

    def analyze_portfolio(self, symbols):
        """Analyzes multiple stocks"""

        print("Starting portfolio analysis...")

        for symbol in symbols:
            self.analyze_stock(symbol)

        # Create report
        self.generate_report()

    def generate_report(self):
        """Generates final report"""

        if not self.results:
            print("No results to report")
            return

        # Convert to DataFrame
        df = pd.DataFrame(self.results).T

        # Sort by overall score
        df = df.sort_values('overall_score', ascending=False)

        print("\n" + "="*80)
        print("FINAL ANALYSIS REPORT")
        print("="*80)

        print(f"\nTOP 3 RECOMMENDATIONS:")
        for i, (symbol, row) in enumerate(df.head(3).iterrows(), 1):
            print(f"{i}. {symbol}: {row['recommendation']} (Score: {row['overall_score']:.1f})")

        print(f"\nPERIOD RETURNS:")
        for symbol, row in df.iterrows():
            print(f"{symbol}: {row['period_return']:+.2f}%")

        print(f"\nRISK ANALYSIS (Volatility):")
        for symbol, row in df.iterrows():
            risk_level = "HIGH" if row['current_volatility'] > 30 else "MEDIUM" if row['current_volatility'] > 20 else "LOW"
            print(f"{symbol}: {row['current_volatility']:.1f}% ({risk_level})")

        # Create comparison chart
        self.comparison_chart()

        return df

    def comparison_chart(self):
        """Creates a comparative chart"""

        df = pd.DataFrame(self.results).T

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Overall Score
        axes[0,0].bar(df.index, df['overall_score'],
                     color=['green' if x > 60 else 'orange' if x > 40 else 'red' for x in df['overall_score']])
        axes[0,0].set_title('Overall Score')
        axes[0,0].set_ylabel('Score (0-100)')
        axes[0,0].tick_params(axis='x', rotation=45)

        # Returns
        colors = ['green' if x > 0 else 'red' for x in df['period_return']]
        axes[0,1].bar(df.index, df['period_return'], color=colors)
        axes[0,1].set_title('Period Return')
        axes[0,1].set_ylabel('Return (%)')
        axes[0,1].tick_params(axis='x', rotation=45)

        # RSI
        axes[1,0].bar(df.index, df['current_rsi'])
        axes[1,0].axhline(y=70, color='red', linestyle='--', alpha=0.7)
        axes[1,0].axhline(y=30, color='green', linestyle='--', alpha=0.7)
        axes[1,0].set_title('Current RSI')
        axes[1,0].set_ylabel('RSI')
        axes[1,0].tick_params(axis='x', rotation=45)

        # Volatility
        axes[1,1].bar(df.index, df['current_volatility'])
        axes[1,1].set_title('Volatility')
        axes[1,1].set_ylabel('Volatility (%)')
        axes[1,1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()

# Run complete system
if __name__ == "__main__":
    # Create analyzer
    analyzer = StockAnalyzer()

    # List of stocks to analyze
    portfolio = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMZN', 'META', 'NFLX']

    # Analyze portfolio
    results = analyzer.analyze_portfolio(portfolio)

    print("\nComplete analysis finished!")
    print("You now have your first quantitative analysis system running!")
```

**Final Exercise 2.4**:
1. Run the complete system with the default stocks
2. Change the list to stocks that interest you
3. Analyze the results: do you trust the recommendations?
4. What would you change in the scoring logic?

## Module Checkpoint

### Installation
- [ ] Python working correctly
- [ ] All libraries installed
- [ ] You can download data without errors

### Skills
- [ ] Download data from any stock
- [ ] Create professional price charts
- [ ] Calculate basic technical indicators
- [ ] Interpret RSI, MACD, Bollinger Bands

### Code
- [ ] All scripts run without errors
- [ ] Your analysis system produces results
- [ ] You understand the basic code logic
- [ ] You can modify symbols and parameters

### Mindset
- [ ] You feel comfortable running code
- [ ] You understand that charts tell stories
- [ ] You see the value of automating analysis
- [ ] You are excited to create strategies

## Next Module

**F3: Advanced Technical Indicators**
- More professional indicators
- Advanced interpretation
- Combining signals
- Quality filters

**You can now officially call yourself a "Python Trader"!**

---

**Ready for more advanced indicators?** -> [F3: Technical Indicators](../f3-indicadores-tecnicos/README.md)
