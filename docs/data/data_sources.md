> 🇪🇸 [Leer en Español](data_sources.es.md) | 🇺🇸 **English**

# Historical Data Sources

## Overview

The quality of your data determines the quality of your backtesting. Here are the sources I currently use, each with their pros and cons.

## Yahoo Finance

### Features
- **Free** for EOD data
- Global stock coverage
- Data from the 1960s for many tickers
- Simple Python API with `yfinance`

### Installation and Usage
```python
pip install yfinance
```

```python
import yfinance as yf
import pandas as pd

# Daily data
ticker = yf.Ticker("AAPL")
data = ticker.history(start="2023-01-01", end="2023-12-31")

# Multiple tickers
tickers = ["AAPL", "MSFT", "GOOGL"]
data = yf.download(tickers, start="2023-01-01", end="2023-12-31")

# Intraday data (last 60 days max)
intraday = ticker.history(period="1d", interval="1m")
```

### Limitations
- Only reliable EOD data
- Intraday limited to 60 days
- Does not include dark pool or odd lots
- Splits and dividends sometimes incorrect

### When to Use
- Initial backtesting of swing strategies
- Quick idea screening
- Long historical data

## Polygon.io

### Features
- **Professional API** with tick-by-tick data
- WebSocket for real-time data
- Complete history since 2003
- Includes dark pools, odd lots, conditions

### Setup
```python
pip install polygon-api-client
```

```python
from polygon import RESTClient
import os

client = RESTClient(api_key=os.getenv("POLYGON_API_KEY"))

# Aggregates (bars)
aggs = client.get_aggs(
    ticker="AAPL",
    multiplier=5,
    timespan="minute",
    from_="2023-01-01",
    to="2023-12-31"
)

# Trades (tick data)
trades = client.list_trades(
    ticker="AAPL",
    timestamp="2023-06-01"
)

# Quote data (NBBO)
quotes = client.list_quotes(
    ticker="AAPL",
    timestamp="2023-06-01"
)
```

### Plans and Pricing (2024)
- **Basic**: $0/month - 2 years historical, EOD, 5 calls/min
- **Starter**: $29/month - 5 years historical, 15-min delayed, WebSockets
- **Developer**: $79/month - 10 years historical, 15-min delayed, trades data
- **Advanced**: $199/month - 20+ years historical, real-time data, quotes

> **Note**: Prices for individual users. Professional plans have different costs.

### When to Use
- Intraday strategies that need precision
- Microstructure analysis
- Backtesting with realistic fills

## Interactive Brokers TWS

### Features
- Real-time data included with account
- Robust API for automation
- Limited but free history
- Direct connection for live trading

### Configuration with ib_insync
```python
pip install ib_insync
```

```python
from ib_insync import *

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)  # Paper: 7497, Live: 7496

# Contract
contract = Stock('AAPL', 'SMART', 'USD')

# Historical data
bars = ib.reqHistoricalData(
    contract,
    endDateTime='',
    durationStr='30 D',
    barSizeSetting='1 min',
    whatToShow='TRADES',
    useRTH=True
)

# Convert to DataFrame
df = util.df(bars)

# Real-time bars
def onBarUpdate(bars, hasNewBar):
    print(bars[-1])

bars = ib.reqRealTimeBars(contract, 5, 'TRADES', True)
bars.updateEvent += onBarUpdate
```

### Limitations
- Maximum 1 year of minute data
- Strict rate limits
- Requires TWS open

### When to Use
- Automated trading in production
- Paper trading with real data
- Verification of other data sources

## DAS Trader

### Features
- Professional day trading platform
- Complete Level 2
- Hotkeys and automation
- Integration with multiple brokers

### Exporting Data
```python
# DAS saves logs in CSV format
import pandas as pd
import glob

# Read executed trades
trades_files = glob.glob('C:/DAS/Trades/*.csv')
trades = pd.concat([pd.read_csv(f) for f in trades_files])

# Process for analysis
trades['Time'] = pd.to_datetime(trades['Time'])
trades['PnL'] = trades['ExitPrice'] - trades['EntryPrice']
```

### Python Integration
```python
# Use DAS API (requires DAS Trader Pro)
import win32com.client

das = win32com.client.Dispatch("DAS.Application")
das.SendOrder("BUY", "AAPL", 100, "MARKET")
```

### When to Use
- Manual execution with post-trade analysis
- Combine manual execution with quant analysis
- Strategy testing before automating

## QuantConnect

### Features
- **Complete cloud platform**
- Data included (equities, options, futures, forex, crypto)
- Professional backtesting engine
- Direct deploy to live trading

### Algorithm Example
```python
class MyAlgorithm(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2023, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(100000)
        
        # Add securities
        self.spy = self.AddEquity("SPY", Resolution.Minute)
        
        # Indicators
        self.sma = self.SMA("SPY", 20, Resolution.Daily)
        
    def OnData(self, data):
        if not self.sma.IsReady:
            return
            
        if data["SPY"].Price > self.sma.Current.Value:
            self.SetHoldings("SPY", 1.0)
        else:
            self.Liquidate("SPY")
```

### Advantages
- No need to manage infrastructure
- Clean, adjusted data
- Extensive community and examples

### When to Use
- Complex multi-asset strategies
- When you don't want to manage data
- Quick transition from backtest to live

## Flash Research

### Features
- Focused on **market microstructure**
- Tape reading analysis
- Institutional footprint identification
- Options flow data

### Use Cases
```python
# Conceptual example - Flash Research provides insights, not raw data
insights = {
    'institutional_accumulation': ['AAPL', 'MSFT'],
    'unusual_options_activity': [
        {'ticker': 'NVDA', 'strike': 500, 'volume': 10000}
    ],
    'dark_pool_prints': [
        {'ticker': 'TSLA', 'size': 500000, 'price': 250.50}
    ]
}

# Use these insights to filter universe
universe = screen_stocks(insights['institutional_accumulation'])
```

### When to Use
- Technical signal confirmation
- Identify institutional accumulation
- Options flow for directionality

## My Current Stack

```python
# config.py
DATA_SOURCES = {
    'historical': 'polygon',      # For precise backtesting
    'realtime': 'ibkr_tws',      # For execution
    'screening': 'yahoo',         # For quick ideas
    'research': 'quantconnect',   # For complex strategies
    'insights': 'flash_research'  # For confirmation
}

# data_manager.py
class DataManager:
    def __init__(self):
        self.polygon = PolygonClient()
        self.yahoo = YahooClient()
        self.ibkr = IBKRClient()
        
    def get_data(self, ticker, source='auto'):
        if source == 'auto':
            # Logic to choose best source
            if self.need_intraday:
                return self.polygon.get_data(ticker)
            else:
                return self.yahoo.get_data(ticker)
```

## Monthly Costs

| Source | Cost | What You Get |
|--------|-------|-----------------|
| Yahoo Finance | $0 | EOD data, basic screening |
| Polygon.io | $79 | 10 years intraday, websocket |
| IBKR | $10 + comms | Real-time, execution |
| DAS Trader | $150 | Pro platform + data |
| QuantConnect | $0-$200 | Backtest + live deployment |
| Flash Research | Variable | Market intelligence |

**Total**: ~$300-400/month for professional setup

## Tips for Getting Started

1. **Start free**: Yahoo Finance + IBKR paper trading
2. **First upgrade**: Polygon.io Developer ($79)
3. **When you're consistent**: Add DAS or similar
4. **To scale**: QuantConnect for multiple strategies

## Data Quality Checklist

```python
def validate_data(df):
    checks = {
        'no_gaps': df.index.is_monotonic_increasing,
        'no_nulls': not df.isnull().any().any(),
        'volume_positive': (df['Volume'] >= 0).all(),
        'prices_positive': (df[['Open', 'High', 'Low', 'Close']] > 0).all().all(),
        'high_low_valid': (df['High'] >= df['Low']).all(),
        'ohlc_valid': (
            (df['High'] >= df[['Open', 'Close']].max(axis=1)).all() &
            (df['Low'] <= df[['Open', 'Close']].min(axis=1)).all()
        )
    }
    
    return pd.Series(checks)
```

## Next Step

Continue with [Data Types](data_types.md) to understand the differences between EOD, intraday, and tick data.