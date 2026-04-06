> 🇪🇸 [Leer en Español](data_types.es.md) | 🇺🇸 **English**

# Data Types: EOD, Intraday, and Tick

## End-of-Day (EOD) Data

### What Is It?
Data with a single point per day: Open, High, Low, Close, Volume (OHLCV).

```python
# Example EOD data
date         open    high    low     close   volume
2024-01-15   175.00  178.50  174.25  177.80  45000000
2024-01-16   177.80  179.00  176.00  178.25  42000000
```

### When to Use
- Swing trading (holding days/weeks)
- Long-term trend analysis
- Initial idea screening
- Position strategy backtesting

### Pros and Cons
✅ **Pros:**
- Free or cheap
- Easy to handle
- Less noise
- Fast backtests

❌ **Cons:**
- Not useful for day trading
- Loses intraday information
- Can't optimize entries/exits

### Example Code
```python
import yfinance as yf
import pandas as pd

# Get EOD data
ticker = 'AAPL'
eod_data = yf.download(ticker, start='2023-01-01', end='2024-01-01')

# Calculate simple metrics
eod_data['SMA20'] = eod_data['Close'].rolling(20).mean()
eod_data['Daily_Range'] = ((eod_data['High'] - eod_data['Low']) / eod_data['Low'] * 100)
eod_data['Gap'] = (eod_data['Open'] / eod_data['Close'].shift(1) - 1) * 100
```

## Intraday Data (Minute Bars)

### What Is It?
OHLCV for specific intervals: 1min, 5min, 15min, etc.

```python
# Example 5-min bars
datetime              open    high    low     close   volume
2024-01-15 09:30:00  175.00  175.50  174.95  175.20  500000
2024-01-15 09:35:00  175.20  175.80  175.10  175.75  450000
2024-01-15 09:40:00  175.75  176.00  175.50  175.55  380000
```

### When to Use
- Day trading
- Precise entries/exits
- Intraday patterns (VWAP, breakouts)
- Intraday risk management

### Common Resolutions
```python
RESOLUTIONS = {
    'scalping': '1min',
    'day_trading': '5min',
    'swing_entries': '15min',
    'trend_confirmation': '60min'
}
```

### Data Handling
```python
# With Polygon.io
from polygon import RESTClient
client = RESTClient("YOUR_API_KEY")

# 5-minute bars
bars = client.get_aggs(
    ticker="AAPL",
    multiplier=5,
    timespan="minute",
    from_="2024-01-15",
    to="2024-01-15"
)

# Convert to DataFrame
df = pd.DataFrame(bars)
df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('datetime', inplace=True)

# Calculate VWAP
df['cum_vol'] = df['volume'].cumsum()
df['cum_vol_price'] = (df['close'] * df['volume']).cumsum()
df['vwap'] = df['cum_vol_price'] / df['cum_vol']
```

## Tick Data

### What Is It?
Every individual transaction with exact timestamp.

```python
# Example tick data
timestamp              price   size  exchange  conditions
2024-01-15 09:30:00.123  175.00  100   NYSE     ['regular']
2024-01-15 09:30:00.125  175.01  500   NASDAQ   ['regular']
2024-01-15 09:30:00.127  175.00  200   ARCA     ['odd_lot']
```

### When to Use
- High frequency trading
- Microstructure analysis
- Block/dark pool detection
- Exact slippage analysis

### Considerations
- **Size**: 1GB+ per day for liquid stocks
- **Processing**: You need optimized code
- **Cost**: $100-500+/month for quality data

### Working with Tick Data
```python
# Example with Polygon tick data
trades = client.list_trades(
    ticker="AAPL",
    timestamp="2024-01-15",
    limit=50000
)

# Process for analysis
tick_df = pd.DataFrame(trades)
tick_df['timestamp'] = pd.to_datetime(tick_df['sip_timestamp'], unit='ns')

# Detect large prints
large_prints = tick_df[tick_df['size'] >= 10000]

# Analyze by exchange
exchange_volume = tick_df.groupby('exchange')['size'].sum()

# Create time bars from ticks
def create_time_bars(ticks, bar_size='5T'):
    ticks.set_index('timestamp', inplace=True)
    bars = ticks.resample(bar_size).agg({
        'price': ['first', 'max', 'min', 'last'],
        'size': 'sum'
    })
    bars.columns = ['open', 'high', 'low', 'close', 'volume']
    return bars
```

## Practical Comparison

| Type | Size/Day | Cost | Use Case | Latency |
|------|----------|------|----------|---------|
| EOD | 1 row | $0 | Swing/Position | N/A |
| 1-min | 390 rows | $20-50 | Day trading | 1 min |
| Tick | 100k-1M rows | $100+ | HFT/Analysis | Real-time |

## Data Aggregation

### From Tick to Minute
```python
def aggregate_ticks_to_bars(ticks, bar_type='time', bar_size=60):
    if bar_type == 'time':
        # Time bars (every 60 seconds)
        bars = ticks.resample(f'{bar_size}S').agg({
            'price': ['first', 'max', 'min', 'last'],
            'size': 'sum'
        })
    
    elif bar_type == 'volume':
        # Volume bars (every N shares)
        bars = aggregate_volume_bars(ticks, bar_size)
    
    elif bar_type == 'dollar':
        # Dollar bars (every $N traded)
        ticks['dollar_vol'] = ticks['price'] * ticks['size']
        bars = aggregate_dollar_bars(ticks, bar_size)
    
    return bars
```

### Volume Bars (Advanced)
```python
def create_volume_bars(ticks, volume_per_bar=100000):
    bars = []
    current_bar = {'volume': 0, 'high': 0, 'low': float('inf')}
    
    for _, tick in ticks.iterrows():
        current_bar['volume'] += tick['size']
        current_bar['high'] = max(current_bar['high'], tick['price'])
        current_bar['low'] = min(current_bar['low'], tick['price'])
        
        if current_bar['volume'] >= volume_per_bar:
            bars.append(current_bar)
            current_bar = {'volume': 0, 'high': 0, 'low': float('inf')}
    
    return pd.DataFrame(bars)
```

## Data Quality

### Validation Checklist
```python
def validate_intraday_data(df):
    issues = []
    
    # 1. Temporal gaps
    expected_bars = pd.date_range(
        start=df.index[0].replace(hour=9, minute=30),
        end=df.index[0].replace(hour=16, minute=0),
        freq='1min'
    )
    missing = expected_bars.difference(df.index)
    if len(missing) > 0:
        issues.append(f"Missing {len(missing)} bars")
    
    # 2. Negative or zero prices
    if (df[['open', 'high', 'low', 'close']] <= 0).any().any():
        issues.append("Zero or negative prices found")
    
    # 3. High/Low consistency
    invalid_hl = df['high'] < df['low']
    if invalid_hl.any():
        issues.append(f"{invalid_hl.sum()} bars with high < low")
    
    # 4. Suspicious volume
    if (df['volume'] == 0).sum() > len(df) * 0.1:
        issues.append("Too many zero volume bars")
    
    return issues
```

## My Personal Approach

```python
# I use different types depending on the strategy
DATA_CONFIG = {
    'gap_scanner': {
        'type': 'EOD',
        'source': 'yahoo',
        'reason': 'Only need overnight gap %'
    },
    'vwap_trading': {
        'type': '1min',
        'source': 'polygon',
        'reason': 'VWAP accuracy + entry timing'
    },
    'tape_reading': {
        'type': 'tick',
        'source': 'polygon_websocket',
        'reason': 'See order flow in real time'
    }
}
```

## Practical Tips

1. **Start with EOD**, it's free and sufficient for learning
2. **Upgrade to 5-min** when you do day trading
3. **Tick data** only if you do HFT or deep analysis
4. **Store locally** data you use frequently
5. **Timestamp timezone** always in Eastern (NYSE time)

## Efficient Storage

```python
# Save efficiently
df.to_parquet('data/AAPL_2024_1min.parquet')  # Better than CSV
df.to_hdf('data/ticks.h5', key='AAPL')  # For large datasets

# Read efficiently
df = pd.read_parquet('data/AAPL_2024_1min.parquet')
df = pd.read_hdf('data/ticks.h5', key='AAPL', 
                 where='timestamp >= "2024-01-15" & timestamp < "2024-01-16"')
```

## Next Step

Now that you understand data types, let's move on to [Data Cleaning](data_cleaning.md) to ensure your data is reliable.